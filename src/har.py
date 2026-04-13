import pandas as pd
import numpy as np
import statsmodels.api as sm

def har_train(df_realized_vol_train, horizon = 21):
    """
    Train HAR
    """
    
    har = HAR(horizon = horizon)
    har.fit(df_realized_vol_train)
    return har

def har_forecast(har, df_realized_vol_test):
    """
    Compute average volatility over the future time horizon
    """
    
    har_vol_forecast = pd.Series(
        har.forecast(df_realized_vol_test),
        index = df_realized_vol_test.index,
        name = 'vol_forecast'
    )

    return har_vol_forecast

class HAR:
    """
    Implement a Heterogeneous AutoRegressive (HAR) model
    """

    def __init__(self, horizon):
        self.horizon = horizon
        self.model = None

    def _get_train_data(self, df):

        data = df.copy()

        # Daily component: today volatility
        data['vol_daily'] = data['vol']
        # Weekly component: rolling average of the last 5 days, including today
        data['vol_weekly'] = data['vol'].rolling(window=5).mean()
        # Monthly component: rolling average of the last 21 days, including today
        data['vol_monthly'] = data['vol'].rolling(window=21).mean()

        # Target: 21-day forward rolling mean, shifted backwards to align with today
        data['y'] = data['vol'].rolling(window=self.horizon).mean().shift(-self.horizon)

        data = data.dropna()

        X = data[['vol_daily', 'vol_weekly', 'vol_monthly']]
        # Add constant
        X = sm.add_constant(X)
        
        y = data['y']

        return X, y
    
    def _get_test_data(self, df):

        data = df.copy()

        # Daily component: today's volatility
        data['vol_daily'] = data['vol']
        # Weekly component: rolling average of the last 5 days, including today
        data['vol_weekly'] = data['vol'].rolling(window=5).mean()
        # Monthly component: rolling average of the last 21 days, including today
        data['vol_monthly'] = data['vol'].rolling(window=21).mean()

        data = data.dropna()

        X = data[['vol_daily', 'vol_weekly', 'vol_monthly']]
        # Add constant
        X = sm.add_constant(X)

        return X
    
    def fit(self, df, summary = False):
        
        # Prepare data
        X, y = self._get_train_data(df)

        # Fit model
        self.model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': self.horizon})

        if summary:
            print(self.model.summary())


    def forecast(self, df):

        if self.model is not None:

            # Prepare data
            X = self._get_test_data(df)

            # Forecast
            y_hat = self.model.predict(X)

            return y_hat