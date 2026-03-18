import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d, BSpline
from sklearn.cluster import KMeans, AgglomerativeClustering
from torch.utils.data import Dataset
from src.utils import LOGGER

class SplineTransformer:
    def __init__(self, log_moneyness, quantiles, degree = 3):
        """
        Implement B-spline Transformer for Volatility over Log-Moneyness grid
        """
        # Polynomial degree
        self.degree = degree
        
        # Construct knot grid based on data quantiles to cover data uniformly 
        grid_points = np.quantile(log_moneyness, quantiles)
        self.moneyness_grid = np.unique(np.sort(np.append(grid_points, 0.0)))

        # Index of ATM (moneyness = 1, log_moneyness = 0) in grid
        self.atm_index = np.searchsorted(self.moneyness_grid, 0.0)
        
        # Inner knots
        self.inner_knots = self.moneyness_grid[1:-1]
        # Minimum
        self.lower_bound = self.moneyness_grid[0]
        # Maximum 
        self.upper_bound = self.moneyness_grid[-1]
        
        # Knots
        self.knots = np.concatenate(([self.lower_bound] * degree, 
                                     self.inner_knots, 
                                     [self.upper_bound] * degree))
        
        # Number of basis elements
        self.num_basis = len(self.knots) - (degree + 1)

        # Identity matrix for basis construction
        self.eye_coeffs = np.eye(self.num_basis)
        
        # BSpline object
        self.bspline_obj = BSpline(self.knots, self.eye_coeffs, self.degree)

        LOGGER.info(f"Spline transformer initialized: \n >D={self.num_basis} basis elements \n >Grid={self.moneyness_grid}")
    
    def get_moneyness_grid(self):
        return self.moneyness_grid
    
    def get_basis(self):
        return self(torch.tensor(self.moneyness_grid, dtype=torch.float32))
    
    def get_num_basis(self):
        return self.num_basis

    def __call__(self, x):
        """
        Compute and returns the B-spline basis matrix for the input x
        """
        x_np = np.asarray(x)
        original_shape = x_np.shape
        x_flat = x_np.flatten()
        
        basis_matrix = self.bspline_obj(x_flat)
    
        output_shape = original_shape + (self.num_basis,)
        return basis_matrix.reshape(output_shape)


class VolatilitySmileDataset(Dataset):
    def __init__(self, df, spline_transformer, target_time_to_expiry=30):
        """
        Torch Dataset for DPMM with volatility smiles
        """
        # Time to expiry (days)
        self.target_time_to_expiry = target_time_to_expiry
        # Time to expiry (years)
        self.target_t_years = self.target_time_to_expiry / 365.0
        
        # Moneyness grid and basis
        self.moneyness_grid = spline_transformer.get_moneyness_grid().astype(np.float32)
        self.x_np = spline_transformer.get_basis().astype(np.float32)
        self.x = torch.from_numpy(self.x_np)

        # Index of ATM (moneyness = 1, log_moneyness = 0) in grid
        self.atm_index = np.searchsorted(self.moneyness_grid, 0.0)

        # Clean dataframe
        df_clean = (df
                    .sort_values(['date', 'time_to_expiry', 'log_moneyness'])
                    .drop_duplicates(['date', 'time_to_expiry', 'log_moneyness'])
                )
        self.dates = df_clean['date'].unique()
        T = len(self.dates)
        
        # Initialize volatility
        self.y = torch.zeros((T, len(self.moneyness_grid)), dtype=torch.float32)
        
        # Interpolate volatility at target time to expiry
        self._initialize_y(df_clean)

        LOGGER.info(f"Dataset initialized: T={T} trading days")

    def _get_variance_on_grid(self, group_expiry):
        """
        Fit spline to raw data for a specific expiry and evaluates on grid
        """
        # Raw data
        k_obs = group_expiry['log_moneyness'].values
        vol_obs = group_expiry['vol'].values
        expiry = group_expiry['time_to_expiry'].iloc[0]
        
        # Convert annualized volatility to total variance
        variances = (vol_obs ** 2) * (expiry / 365.0)
        
        # Fit cubic spline
        cubic_spline_var = interp1d(k_obs, variances, kind='cubic', fill_value='extrapolate')
        
        # Evaluate on the standard moneyness grid
        grid_variances = cubic_spline_var(self.moneyness_grid)
        
        # Variance cannot be negative
        return np.maximum(grid_variances, 1e-9)
    
    def _get_volatility_at_time_to_expiry(self, group, tolerance_days = 2):
        # Available expiries for the trading day
        expiries = np.sort(group['time_to_expiry'].unique())

        # Check for expiries near the target
        dist = np.abs(expiries - self.target_time_to_expiry)
        min_dist_idx = np.argmin(dist)

        if dist[min_dist_idx] <= tolerance_days:
            # If target time to expiry is close enough to data no interpolation is needed
            best_expiry = expiries[min_dist_idx]
            var_vector = self._get_variance_on_grid(group[group['time_to_expiry'] == best_expiry])
        else:
            # Find insertion point
            idx = np.searchsorted(expiries, self.target_time_to_expiry)
            
            # Expiry before target
            t_before = expiries[idx - 1]
            # Expiry after target
            t_after = expiries[idx]

            # Get variance vectors
            var_before = self._get_variance_on_grid(group[group['time_to_expiry'] == t_before])
            var_after  = self._get_variance_on_grid(group[group['time_to_expiry'] == t_after])
            
            # Interpolation weight
            w = (self.target_time_to_expiry - t_before) / (t_after - t_before)
            # Linear interpolation in total variance
            var_vector = var_before + w * (var_after - var_before)
        
        # Conversion to annualized volatility
        return np.sqrt(np.maximum(var_vector, 1e-9) / self.target_t_years)

    def _initialize_y(self, df):
        """
        Interpolate volatility smiles to the target time-to-expiry
        """
        # Group by date
        grouped = df.groupby('date')
        for t, (date, group) in enumerate(grouped):
            vol_vector = self._get_volatility_at_time_to_expiry(group)
            self.y[t, :] = torch.from_numpy(vol_vector).float()
    
    def _quantile_init(self, K):
        """
        Initializes cluster centers using Least Squares
        """
        all_smiles = self.y.numpy()
        daily_avg_vols = all_smiles.mean(axis=1)
        quantiles = np.linspace(0.05, 0.95, K)
        
        prior_means_list = []
        
        for q in quantiles:
            target_vol = np.quantile(daily_avg_vols, q)
            # Find day closest to this quantile volatility
            idx = (np.abs(daily_avg_vols - target_vol)).argmin()
            y_sample = all_smiles[idx]
            # Robust Solve: X * Beta = y
            coeffs, _, _, _ = np.linalg.lstsq(self.x_np, y_sample, rcond=None)
            prior_means_list.append(torch.tensor(coeffs, dtype=torch.float32))
            
        return torch.stack(prior_means_list)
    
    def _quantile_init(self, K):
        """
        Initializes cluster centers using Least Squares
        """
        all_smiles = self.y.numpy()
        daily_avg_vols = all_smiles.mean(axis=1)
        quantiles = np.linspace(0.05, 0.95, K)
        
        prior_means_list = []
        
        for q in quantiles:
            target_vol = np.quantile(daily_avg_vols, q)
            # Find day closest to this quantile volatility
            idx = (np.abs(daily_avg_vols - target_vol)).argmin()
            y_sample = all_smiles[idx]
            # Robust Solve: X * Beta = y
            coeffs, _, _, _ = np.linalg.lstsq(self.x_np, y_sample, rcond=None)
            prior_means_list.append(torch.tensor(coeffs, dtype=torch.float32))
            
        return torch.stack(prior_means_list)

    def _kmeans_init(self, K):
        """
        Initializes cluster centers using KMeans on spline coefficients
        """

        all_smiles = self.y.numpy()

        x_pinv = np.linalg.pinv(self.x_np) 
        
        daily_coeffs = all_smiles @ x_pinv.T 
        
        kmeans = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(daily_coeffs)
        
        return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    
    def _hierarchical_init(self, K):
        """
        Initializes cluster centers using Hierarchical Agglomerative Clustering on spline coefficients
        """
        all_smiles = self.y.numpy()

        x_pinv = np.linalg.pinv(self.x_np) 
        
        daily_coeffs = all_smiles @ x_pinv.T 
        
        hc = AgglomerativeClustering(n_clusters=K, linkage='ward')
        labels = hc.fit_predict(daily_coeffs)
        
        centroids = []
        for k in range(K):
            cluster_members = daily_coeffs[labels == k]
            centroids.append(cluster_members.mean(axis=0))
            
        return torch.tensor(np.stack(centroids), dtype=torch.float32)
    
    def _random_init(self, K):
        """
        Initializes cluster centers by randomly sampling K days from the history
        """
        all_smiles = self.y.numpy()
        
        x_pinv = np.linalg.pinv(self.x_np) 

        daily_coeffs = all_smiles @ x_pinv.T 
        
        T = daily_coeffs.shape[0]
        
        random_indices = np.random.choice(T, size=K, replace=False)
        
        random_centroids = daily_coeffs[random_indices]
        
        return torch.tensor(random_centroids, dtype=torch.float32)
    
    def get_prior_mean_init(self, method, K):
        match method:
            case 'quantile':
                return self._quantile_init(K)
            case 'kmeans':
                return self._kmeans_init(K)
            case 'hierarchical':
                return self._hierarchical_init(K)
            case 'random':
                return self._random_init(K)
            
    def get_atm_iv(self):
        return pd.Series(
            100*self.y[:,self.atm_index],
            index = self.dates,
            name = 'atm_iv'
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'y': self.y[idx],
            'x': self.x 
        }

def compute_hf_daily_realized_vol(df_intraday, df_daily):
    """
    Compute daily realized volatility from high-frequency (5min) intraday data
    """
    # Realized intraday variance (sum of squares)
    df_intraday_variance = df_intraday['log_return'].pow(2).groupby(df_intraday.index.date).sum().reset_index()
    df_intraday_variance.columns = ['date', 'intraday_variance']
    df_intraday_variance['date'] = pd.to_datetime(df_intraday_variance['date'])
    # Realized overnight variance
    df_overnight_variance = (np.log(df_daily['open'] / df_daily['close'].shift(1))).pow(2).reset_index()
    df_overnight_variance.columns = ['date', 'overnight_variance']
    df_overnight_variance['date'] = pd.to_datetime(df_overnight_variance['date'])
    # Total variance
    df = pd.merge(df_intraday_variance, df_overnight_variance, how='inner', on='date')
    df['variance'] = df['intraday_variance'] + df['overnight_variance']
    # Annualized volatility
    df['vol'] = df.eval("sqrt(variance * 252)")
    df.set_index('date', inplace=True)
    df.dropna(inplace=True)
    return df

def get_training_data_parkinson_vol(df_intraday, df_daily):
    """
    Return training data for linear regression model of realized volatility ~ Parkinson volatility
    """
    # Compute realized volatility with intraday data
    df_realized_vol = compute_hf_daily_realized_vol(df_intraday, df_daily)
    # Join dataframes on date
    df = pd.merge(df_realized_vol, df_daily, how='inner', on='date')
    df.reset_index(inplace=True)
    # Training dataset
    x = df['parkinson_vol'].values.reshape(-1, 1)
    y = df['vol'].values
    return x, y, df['date'].values

def get_garch_dataset(df_daily, lr):
    """
    Return dataset for GARCH
    """
    # Clean data
    df_clean = df_daily[['date', 'log_return', 'parkinson_vol']].dropna().copy()
    # Scale log-returns
    log_returns = 100 * df_clean['log_return'].values
    # Extract Parkinson volatility
    parkinson_vol = df_clean['parkinson_vol'].values.reshape(-1, 1)
    # Predict realized variance with trained regression model
    var_hat = (lr.predict(parkinson_vol) ** 2)
    return log_returns, var_hat, df_clean['date'].values