import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import timedelta
from src.pricing import bs_price, compute_sharpe_ratio, compute_sortino_ratio

@dataclass
class Tranche:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    expiration_date: pd.Timestamp
    strike: float
    entry_iv: float
    entry_price: float
    quantity: int
    is_active: bool = True
    mtm_history: list = None
    
    def __post_init__(self):
        self.mtm_history = []

class Backtester:
    def __init__(
            self, 
            initial_capital, 
            slippage_pct, 
            fixed_fee, 
            leverage, 
            horizon, 
            margin_requirement
        ):
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.fixed_fee = fixed_fee
        self.leverage = leverage
        self.horizon = horizon
        self.margin_requirement = margin_requirement
        
        self.trade_count = 0
        self.open_tranches = []
        self.closed_tranches = []
        self.trading_days = []
        self.equity_curve = []
        self.daily_pnl_log = []

    def get_synthetic_straddle_prices(self, F, K, T, r, iv, spread_ratio):
        """
        Calculate synthetic mid, bid, and ask for an ATM Straddle (call + put)
        """
        # Mid for both legs
        call_mid = bs_price(F, K, T, r, iv, True)
        put_mid = bs_price(F, K, T, r, iv, False)
        straddle_mid = call_mid + put_mid
        
        # Bid/Ask from spread ratio
        half_spread = (straddle_mid * spread_ratio) / 2
        
        bid = max(0.01, straddle_mid - half_spread)
        ask = max(0.01, straddle_mid + half_spread)
        
        return bid, straddle_mid, ask

    def _get_exit_date(self, entry_date, days=30):
        """
        Get exit date accounting for weekends: if Saturday (5) or Sunday (6), anticipate to Friday (4)
        """
        target_date = entry_date + timedelta(days=days)
        if target_date.weekday() == 5:
            return target_date - timedelta(days=1)
        elif target_date.weekday() == 6:
            return target_date - timedelta(days=2)
        return target_date
    
    def _liquidate(self, df, current_equity):
        """
        Close all open tranches
        """
        if not self.open_tranches:
            return current_equity
            
        last_row = df.iloc[-1]
        final_date = last_row['date']
        terminal_adjustment = 0
        
        for tranche in self.open_tranches[:]:
            # Time to expiry
            time_remaining = max(0.0001, (tranche.exit_date - final_date).days / 365.0)
            
            # Get prices
            bid, mid, ask = self.get_synthetic_straddle_prices(
                last_row['forward'], tranche.strike, time_remaining,
                last_row['r_annual'], last_row['atm_iv'], last_row['spread_ratio']
            )
            
            # Exit price including slippage
            exit_price_total = ask * (1 + self.slippage_pct)
            # half_spread = (mid * last_row['spread_ratio']) / 2
            # exit_price_total = mid + (half_spread * self.slippage_pct)
            
            # Calculate the final realized PnL for this tranche
            final_realized_pnl = ((tranche.entry_price - exit_price_total) * tranche.quantity * 100) - (self.fixed_fee * tranche.quantity)
            
            prev_mtm = tranche.mtm_history[-1] if tranche.mtm_history else 0
            terminal_adjustment += (final_realized_pnl - prev_mtm)
        
            tranche.is_active = False
            self.closed_tranches.append(tranche)
            self.open_tranches.remove(tranche)

        current_equity += terminal_adjustment
        self.equity_curve[-1] = current_equity
        
        return current_equity

    def run(self, df, trading_signal, liquidate=False):
        current_equity = self.initial_capital

        for i, row in df.iterrows():
            today = row['date']
            self.trading_days.append(today)

            # Interest on cash balance
            interest_earned = current_equity * row['r_daily']
            
            # Update open tranches
            daily_mtm_change = 0
            for tranche in self.open_tranches[:]:
                
                time_remaining = max(0, (tranche.expiration_date - today).days) / 365.0
                
                if today >= tranche.exit_date:
                    # Close position
                    
                    # Time to expiry
                    time_remaining = max(0, (tranche.expiration_date - today).days) / 365.0
                    
                    # Get prices
                    bid, mid, ask = self.get_synthetic_straddle_prices(
                        row['forward'], tranche.strike, time_remaining, 
                        row['r_annual'], row['atm_iv'], row['spread_ratio']
                    )

                    # Exit price including slippage
                    exit_price = ask * (1 + self.slippage_pct)
                    # half_spread = (mid * row['spread_ratio']) / 2
                    # exit_price = mid + (half_spread * self.slippage_pct)
                    
                    # Total PnL for the tranche
                    realized_pnl = ((tranche.entry_price - exit_price) * tranche.quantity * 100) - (self.fixed_fee * tranche.quantity)
                    
                    prev_mtm = tranche.mtm_history[-1] if tranche.mtm_history else 0
                    daily_mtm_change += (realized_pnl - prev_mtm)
                    
                    tranche.is_active = False
                    self.closed_tranches.append(tranche)
                    self.open_tranches.remove(tranche)
                else:
                    # Update MtM with current price

                    # Get price
                    bid, mid, ask = self.get_synthetic_straddle_prices(
                        row['forward'], tranche.strike, time_remaining, 
                        row['r_annual'], row['atm_iv'], row['spread_ratio']
                    )
                    current_tranche_pnl = (tranche.entry_price - ask) * tranche.quantity * 100
                    prev_mtm = tranche.mtm_history[-1] if tranche.mtm_history else 0
                    
                    daily_mtm_change += (current_tranche_pnl - prev_mtm)
                    tranche.mtm_history.append(current_tranche_pnl)

            # Update equity curve and alpha pnl
            total_daily_change = daily_mtm_change + interest_earned
            current_equity += total_daily_change
            self.daily_pnl_log.append(total_daily_change)
            self.equity_curve.append(current_equity)
            

            # Open new tranches
            if trading_signal[today] == 1:
                
                # Dollar amount of margin requirement
                margin_requirement_per_straddle = (row['close'] * 100) * self.margin_requirement
                # Dollar amount to allocate
                daily_allocation = (current_equity * self.leverage) / self.horizon
                # Number of contracts to buy
                num_contracts = int(daily_allocation // margin_requirement_per_straddle)
                
                if num_contracts > 0:
                    # Get price
                    bid, mid, ask = self.get_synthetic_straddle_prices(
                        row['forward'], row['close'], 14/365.0, 
                        row['r_annual'], row['atm_iv'], row['spread_ratio']
                    )
                    # Enter price including slippage
                    entry_price = bid * (1 - self.slippage_pct)
                    # half_spread = (mid * row['spread_ratio']) / 2
                    # entry_price = mid - (half_spread * self.slippage_pct)
                    
                    # Open new tranche
                    new_tranche = Tranche(
                        entry_date=today,
                        exit_date=self._get_exit_date(today, self.horizon),
                        expiration_date=self._get_exit_date(today, 14),
                        strike=row['close'],
                        entry_iv=row['atm_iv'],
                        entry_price=entry_price,
                        quantity=num_contracts
                    )
                    self.open_tranches.append(new_tranche)
                    self.trade_count += 1

        if liquidate:
            current_equity = self._liquidate(df, current_equity)

    def get_metrics(self, risk_free_rate):

        alpha_returns = pd.Series(self.daily_pnl_log) / self.initial_capital
        
        returns = pd.Series(
            self.equity_curve,
            index = self.trading_days
        ).pct_change().dropna()

        risk_free_rate = risk_free_rate.loc[returns.index]
        
        metrics = {
            "total_return": (self.equity_curve[-1] / self.initial_capital) - 1,
            "max_drawdown": (pd.Series(self.equity_curve) / pd.Series(self.equity_curve).cummax() - 1).min(),
            "alpha_sharpe_ratio": (alpha_returns.mean() / alpha_returns.std()) * np.sqrt(252),
            "sharpe_ratio": compute_sharpe_ratio(returns, risk_free_rate) if len(returns) > 0 else 0,
            "sortino_ratio": compute_sortino_ratio(returns, risk_free_rate) if len(returns) > 0 else 0,
            "turnover": len(self.closed_tranches) * 2,
            "trade_count": self.trade_count
        }
        return metrics
    
    
    def get_equity_curve(self, name = 'equity_curve'):
        equity_curve = pd.Series(
            self.equity_curve,
            index = self.trading_days,
            name = name
        )

        daily_returns = pd.Series(
            self.equity_curve,
            index = self.trading_days,
            name = name
        ).pct_change().dropna()

        return equity_curve, daily_returns
