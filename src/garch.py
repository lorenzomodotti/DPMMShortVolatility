import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter

class GARCH:
    """
    Implement a GARCH-X model with recursion Var(t) = omega + alpha*log_returns(t-1)^2 + beta*Var(t-1) + gamma*X(t-1)
    and Gaussian log-likelihood: L = -0.5 * sum( log(2pi) + log(sigma2) + epsilon^2/sigma2 )
    """

    def __init__(self):
        self.params = None
        
    def log_likelihood(self, params, log_returns, x):
        """
        GARCH log-likelihood
        """
        mu, omega, alpha, beta, gamma = params
        
        # Ensure stationarity (alpha + beta < 1) and positivity
        if alpha + beta >= 0.999 or omega <= 0 or alpha < 0 or beta < 0 or gamma < 0:
            return np.inf

        # Error
        epsilon = log_returns - mu
        epsilon_sq = epsilon ** 2
        
        # Lagged data
        eps_sq_lag = np.concatenate(([0.0], epsilon_sq[:-1]))
        x_lag = np.concatenate(([0.0], x[:-1]))
        
        # Input term: omega + alpha*eps(t-1)^2 + gamma*X(t-1)
        process_input = omega + (alpha * eps_sq_lag) + (gamma * x_lag)
        
        # Initialize the first variance to the sample variance
        process_input[0] = np.var(log_returns)

        # GARCH recursion
        sigma2 = lfilter([1.0], [1.0, -beta], process_input)
        
        # Ensure positivity
        sigma2 = np.maximum(sigma2, 1e-6)
            
        # Gaussian log-likelihood
        ll = -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + (epsilon_sq / sigma2))
        return -np.sum(ll)

    def fit(self, log_returns, x):
        """
        Fit GARCH model
        """
        # Initial guess
        initial_params = [0.0, 0.01, 0.1, 0.8, 0.05]
        
        # Constraints & Bounds
        bounds = ((None,None), (1e-6, None), (0, 1), (0, 1), (0, None))
        
        # Optimization
        result = minimize(
            self.log_likelihood,
            initial_params, 
            args=(log_returns, x),
            method='L-BFGS-B', 
            bounds=bounds,
            tol=1e-6
        )
        self.params = result.x
        return self
    
    def forecast(self, current_sigma2, proxy_variance_ratio, time_horizon=21):
        """
        Forecasts future volatility path with fitted model
        """
        mu, omega, alpha, beta, gamma = self.params
        
        # Long-run clamped persistence
        persistence = min(alpha + beta + (gamma * proxy_variance_ratio), 0.999)
        
        # Long run variance
        lrv = omega / (1 - persistence)
        
        # Geometric series factor
        geom_factor = (persistence * (1 - persistence**time_horizon)) / (1 - persistence)
        
        # Total variance
        total_variance = (time_horizon * lrv) + (current_sigma2 - lrv) * geom_factor
        
        # Annualized volatility
        return np.sqrt(total_variance * (252 / time_horizon))

    def get_conditional_variance_path(self, log_returns, x):
        """
        Reconstruct the historical path of volatility
        """
        mu, omega, alpha, beta, gamma = self.params
        
        # Error
        epsilon = log_returns - mu
        epsilon_sq = epsilon**2
        
        # Lagged data
        eps_sq_lag = np.concatenate(([0.0], epsilon_sq[:-1]))
        x_lag = np.concatenate(([0.0], x[:-1]))
        
        # Input term
        process_input = omega + (alpha * eps_sq_lag) + (gamma * x_lag)
        
        # Initialize the first variance to the sample variance
        process_input[0] = np.var(log_returns)
        
        # GARCH recursion
        sigma2 = lfilter([1.0], [1.0, -beta], process_input)
        
        return sigma2
    
    def get_ratio(self, log_returns, x):
        """
        Return ratio of realized variance and predicted variance
        """
        variance_path_train = self.get_conditional_variance_path(log_returns, x)
        return np.median(x / variance_path_train)