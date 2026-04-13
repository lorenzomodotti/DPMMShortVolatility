import numpy as np
import pandas as pd
import torch
from scipy.interpolate import BSpline, interp1d
from sklearn.cluster import KMeans, AgglomerativeClustering
from torch.utils.data import Dataset

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

        # Create indices for ATM, and OTM put/call
        target_atm_val = 0.0
        target_call_val = np.quantile(log_moneyness, quantiles[1])
        target_put_val = np.quantile(log_moneyness, quantiles[-2])
        self.atm_index = np.searchsorted(self.moneyness_grid, target_atm_val)
        self.call_index = np.searchsorted(self.moneyness_grid, target_call_val)
        self.put_index = np.searchsorted(self.moneyness_grid, target_put_val)
        
        # Inner knots
        self.inner_knots = self.moneyness_grid[1:-1]
        # Minimum
        self.lower_bound = self.moneyness_grid[0]
        # Maximum 
        self.upper_bound = self.moneyness_grid[-1]
        
        # Knots
        self.knots = np.concatenate(([self.lower_bound] * degree, self.inner_knots, [self.upper_bound] * degree))
        
        # Number of basis elements
        self.num_basis = len(self.knots) - (degree + 1)

        # Identity matrix for basis construction
        self.eye_coeffs = np.eye(self.num_basis)
        
        # BSpline object
        self.bspline_obj = BSpline(self.knots, self.eye_coeffs, self.degree)
    
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
    

class IVSmoother:

    def __init__(self, moneyness_grid, target_time_to_expiry = 30):
        """
        Interpolate implied volatility over expiries to get smooth values at target expiry
        """

        # Time to expiry (years)
        self.target_time_to_expiry = target_time_to_expiry / 365.0

        # Moneyness grid and basis
        self.moneyness_grid = moneyness_grid

        # Index of ATM (moneyness = 1, log_moneyness = 0) in grid
        self.atm_index = np.searchsorted(self.moneyness_grid, 0.0)

    def _get_variance_on_grid(self, group_expiry):
        """
        Fit spline to raw total variance for a specific expiry and evaluates on grid
        """
        # Raw data
        k_obs = group_expiry['log_moneyness'].values
        vol_obs = group_expiry['iv'].values
        expiry = group_expiry['time_to_expiry'].iloc[0]
        
        # Convert annualized volatility to total variance
        variances = (vol_obs ** 2) * expiry
        
        # Fit cubic spline
        cubic_spline_var = interp1d(k_obs, variances, kind='cubic', fill_value='extrapolate')
        
        # Evaluate on the standard moneyness grid
        grid_variances = cubic_spline_var(self.moneyness_grid)
        
        # Variance cannot be negative
        return np.maximum(grid_variances, 1e-9)
    
    def _get_volatility_at_time_to_expiry(self, group, tolerance_years = 0.006):
        """
        Interpolate volatility over expiries
        """
        # Available expiries for the trading day
        expiries = np.sort(group['time_to_expiry'].unique())

        # Check for expiries near the target
        dist = np.abs(expiries - self.target_time_to_expiry)
        min_dist_idx = np.argmin(dist)

        if dist[min_dist_idx] <= tolerance_years:
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
        return np.sqrt(np.maximum(var_vector, 1e-9) / self.target_time_to_expiry)
    
    def get_iv(self, df):
        """
        Return smooth implied volatility smiles and ATM implied volatility for each day
        """
        # Clean data
        df = (df
                .reset_index()
                .sort_values(['date', 'time_to_expiry', 'log_moneyness'])
                .drop_duplicates(['date', 'time_to_expiry', 'log_moneyness'])
            )
        
        # Initialize tensor
        T = df['date'].nunique()
        iv = torch.zeros((T, len(self.moneyness_grid)), dtype=torch.float32)
        trading_days = []
        
        # Smooth IV in each day
        grouped = df.groupby('date')
        for t, (date, group) in enumerate(grouped):
            vol_vector = self._get_volatility_at_time_to_expiry(group)
            iv[t, :] = torch.from_numpy(vol_vector).float()
            trading_days.append(date)

        # ATM IV
        atm_iv = pd.Series(
            iv[:,self.atm_index],
            index = trading_days,
            name = 'atm_iv'
        )

        return iv, atm_iv


class VolatilitySmileDataset(Dataset):

    def __init__(self, x, y):
        self.x_np = x
        self.x = torch.from_numpy(x)
        self.y = y

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
            
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'y': self.y[idx],
            'x': self.x 
        }