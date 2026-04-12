import pandas as pd
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data_transformer import VolatilitySmileDataset

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

def seed_torch(seed=124):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dpmm_train(iv_train, spline_basis, D, K = 2):

    seed_torch()
    
    # Initialize volatility smile dataset
    dataset_dpmm_train = VolatilitySmileDataset(spline_basis, iv_train)
    # Initialize dataloader
    dataloader_train = DataLoader(dataset_dpmm_train, batch_size=len(dataset_dpmm_train), shuffle=True)

    # Initialize prior means
    prior_mean_init = dataset_dpmm_train.get_prior_mean_init(method='kmeans', K=K)

    # Initialize DPMM
    dpmm = DPMM(
        K=K, 
        D=D, 
        num_samples=len(dataset_dpmm_train),
        prior_mean_init=prior_mean_init
    )

    # Train DPMM
    trainer = pl.Trainer(
        max_epochs=100, 
        accelerator="auto",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False
    )
    trainer.fit(dpmm, train_dataloaders=dataloader_train)

    return dpmm

def dpmm_forecast(dpmm, iv_test, spline_transformer, dates):
    
    # Initialize volatility smile dataset
    spline_basis = spline_transformer.get_basis().astype(np.float32)
    dataset_dpmm_test = VolatilitySmileDataset(spline_basis, iv_test)
    # Initialize dataloader
    dataloader_test = DataLoader(dataset_dpmm_test, batch_size=len(dataset_dpmm_test), shuffle=False)
    
    # Compute posterior probabilities
    dpmm.eval()
    with torch.no_grad():
        full_batch = next(iter(dataloader_test))
        posterior_probabilities = dpmm.get_posterior_probabilities(full_batch['x'], full_batch['y'])
        index_panic_cluster = dpmm.get_index_panic_cluster(spline_transformer)

    df_regimes = pd.DataFrame(
        posterior_probabilities, 
        index=dates, 
        columns=[f'cluster_{k}' for k in range(posterior_probabilities.shape[1])]
    )
    # Add and impute missing trading days
    return df_regimes, index_panic_cluster


class DPMM(pl.LightningModule):
    """
    Dirichlet Process Mixture Model - Variational inference
    """
    def __init__(self, K, D, num_samples, delta=2.0, lr=0.01, prior_mean_init=None):
        super().__init__()
        self.save_hyperparameters()
        
        # Cluster truncation
        self.K = K
        # Dimension B-spline
        self.D = D
        # Number of samples
        self.N = num_samples

        self.delta = delta
        self.lr = lr
        
        # Spline coefficients
        self.q_mu = nn.Parameter(torch.zeros(self.K, self.D))
        
        # Prior initialization
        if prior_mean_init is not None:
            self.register_buffer('prior_means', prior_mean_init)
            with torch.no_grad():
                self.q_mu.copy_(self.prior_means)
        else:
            self.register_buffer('prior_means', torch.zeros(self.K, self.D))

        # q_logvar: (K, D) - Uncertainty about the shape
        self.q_logvar = nn.Parameter(torch.ones(self.K, self.D) * -4.0)

        # DP stick breaking weights
        self.stick_alpha = nn.Parameter(torch.randn(self.K - 1)) 
        self.stick_beta = nn.Parameter(torch.randn(self.K - 1)) 

        # Precision
        self.log_tau = nn.Parameter(torch.tensor(3.0))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, batch, batch_idx):
        loss = self(batch['x'], batch['y'])
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def forward(self, batch_X, batch_Y):
        batch_size = batch_X.shape[0]

        # Implied volatility
        Y_hat = torch.matmul(batch_X, self.q_mu.T) 
        
        # Residuals
        residuals = batch_Y.unsqueeze(-1) - Y_hat
        
        # Huber Loss
        abs_error = torch.abs(residuals)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        huber_loss = (quadratic ** 2) + (2 * self.delta * linear)
        
        # Pointwise log-likelihood
        ll_pointwise = 0.5 * self.log_tau - 0.5 * torch.exp(self.log_tau) * huber_loss
        
        # Total log-likelihood per smile
        ll_per_smile = torch.sum(ll_pointwise, dim=1) 
        
        # Cluster weights
        log_pi = self.get_stick_weights()
        
        # Posterior probabilities: log P(z|x) = log P(x|z) + log P(z)
        log_rho = ll_per_smile + log_pi.unsqueeze(0)
        phi = F.softmax(log_rho, dim=1)

        # Likelihood
        likelihood_term = torch.sum(phi * log_rho)
        # Entropy
        entropy_term = -torch.sum(phi * torch.log(phi + 1e-10))
        
        # KL scaling
        kl_scale = self.N / batch_size
        
        # KL divergence
        kl_sticks = self.kl_beta(F.softplus(self.stick_alpha), F.softplus(self.stick_beta))
        
        # Regularization
        kl_curves = 0.5 * torch.sum((self.q_mu - self.prior_means)**2 * torch.exp(-self.q_logvar) + self.q_logvar)
        
        # ELBO
        elbo = likelihood_term + entropy_term - (kl_scale * kl_sticks) - (kl_scale * 0.001 * kl_curves)
        return -elbo

    def get_stick_weights(self):
        # DP stick breaking weights
        a = F.softplus(self.stick_alpha)
        b = F.softplus(self.stick_beta)
        dig_a = torch.digamma(a)
        dig_b = torch.digamma(b)
        dig_sum = torch.digamma(a + b)
        E_log_v = dig_a - dig_sum
        E_log_1_v = dig_b - dig_sum
        
        E_log_pi = torch.zeros(self.K, device=self.device)
        cum_log_1_v = torch.cumsum(E_log_1_v, dim=0)
        cum_log_1_v_shifted = torch.cat([torch.zeros(1, device=self.device), cum_log_1_v[:-1]])
        
        E_log_pi[:-1] = E_log_v + cum_log_1_v_shifted
        E_log_pi[-1] = cum_log_1_v[-1]
        return E_log_pi

    def kl_beta(self, alpha, beta):
        # KL divergence between Beta(alpha, beta) and Beta(1, alpha_prior)
        prior_alpha = torch.tensor(1.0, device=self.device)
        prior_beta = torch.tensor(1.0, device=self.device)
        
        ln_B_q = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
        ln_B_p = torch.lgamma(prior_alpha) + torch.lgamma(prior_beta) - torch.lgamma(prior_alpha + prior_beta)
        
        dig_alpha = torch.digamma(alpha)
        dig_beta = torch.digamma(beta)
        dig_sum = torch.digamma(alpha + beta)
        
        return torch.sum((ln_B_p - ln_B_q) + (alpha - prior_alpha)*(dig_alpha - dig_sum) + (beta - prior_beta)*(dig_beta - dig_sum))
    
    @torch.no_grad()
    def get_posterior_probabilities(self, batch_X, batch_Y):
        """
        Return posterior probability of each regime
        """
        batch_size = batch_X.shape[0]
        Y_hat = torch.matmul(batch_X, self.q_mu.T)
        residuals = batch_Y.unsqueeze(-1) - Y_hat
        
        abs_error = torch.abs(residuals)
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = abs_error - quadratic
        huber_loss = (quadratic ** 2) + (2 * self.delta * linear)
        
        ll_pointwise = 0.5 * self.log_tau - 0.5 * torch.exp(self.log_tau) * huber_loss
        ll_per_smile = torch.sum(ll_pointwise, dim=1) 
        
        log_pi = self.get_stick_weights()
        log_rho = ll_per_smile + log_pi.unsqueeze(0)
        return F.softmax(log_rho, dim=1)
    
    @torch.no_grad()
    def get_index_panic_cluster(self, spline_transformer):
        # ATM grid index
        atm_index = spline_transformer.atm_index

        # Grid for plotting
        plot_grid = spline_transformer.moneyness_grid

        # B-spline basis for plotting grid
        basis_grid = torch.tensor(spline_transformer(plot_grid), dtype=torch.float32)

        # Compute implied volatility curve from learned coefficients
        coeffs = self.q_mu.cpu()
        curves = torch.matmul(basis_grid.cpu(), coeffs.T)

        return int(torch.argsort(curves[atm_index,:])[-1])