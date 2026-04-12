import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import timedelta
from src.pricing import bs_price

@dataclass
class Tranche:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    strike: float
    entry_iv: float
    entry_price: float
    quantity: int
    is_active: bool = True
    mtm_history: list = None
    
    def __post_init__(self):
        self.mtm_history = []

class Backtester:
    def __init__(self, initial_capital=1_000_000, slippage_pct=0.01, fixed_fee=1.0, leverage=1.0, horizon=21):
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self.fixed_fee = fixed_fee
        self.leverage = leverage
        self.horizon = horizon
        
        self.trade_count = 0
        self.open_tranches = []
        self.closed_tranches = []
        self.trading_days = []
        self.equity_curve = []
        self.daily_pnl_log = []

    def get_synthetic_prices(self, F, K, T, r, iv, spread_ratio):
        """
        Calculates Synthetic Mid, Bid, and Ask based on interpolated IV 
        and the Spread Ratio from nearby actual options.
        """
        # 1. Get Synthetic Mid via Black-76
        mid_price = bs_price(F, K, T, r, iv, True)
        
        # 2. Recover Bid/Ask from Spread Ratio
        # Spread Ratio = (Actual Ask - Actual Bid) / Actual Mid
        half_spread = (mid_price * spread_ratio) / 2
        
        bid = max(0.01, mid_price - half_spread)
        ask = max(0.01, mid_price + half_spread)
        
        return bid, mid_price, ask

    def _get_exit_date(self, entry_date, days=30):
        """Logic to handle weekend expirations (Close on Friday)."""
        target_date = entry_date + timedelta(days=days)
        # If Saturday (5) or Sunday (6), move to Friday (4)
        if target_date.weekday() == 5:
            return target_date - timedelta(days=1)
        elif target_date.weekday() == 6:
            return target_date - timedelta(days=2)
        return target_date
    
    def _liquidate(self, df, current_equity):
        """
        Closes all open tranches at the final row's synthetic Ask price.
        Ensures validation metrics account for the cost of closing the portfolio.
        """
        if not self.open_tranches:
            return current_equity
            
        last_row = df.iloc[-1]
        final_date = last_row['date']
        terminal_adjustment = 0
        
        for tranche in self.open_tranches[:]:
            # How much time is left until its natural expiry?
            time_remaining = max(0.0001, (tranche.exit_date - final_date).days / 365.0)
            
            # We must BUY BACK (at the ASK) to liquidate a short position
            _, _, synth_ask = self.get_synthetic_prices(
                last_row['forward'], tranche.strike, time_remaining,
                last_row['r_annual'], last_row['atm_iv'], last_row['spread_ratio']
            )
            
            # Exit price includes slippage and commissions
            exit_price_total = synth_ask * (1 + self.slippage_pct)
            
            # Calculate the final realized PnL for this tranche
            final_realized_pnl = ((tranche.entry_price - exit_price_total) * tranche.quantity * 100) \
                                - (self.fixed_fee * tranche.quantity)
            
            # The equity curve is already marked to Mid. 
            # We need to add the difference between 'Final Realized' and 'Last Mid-based MtM'
            prev_mtm = tranche.mtm_history[-1] if tranche.mtm_history else 0
            terminal_adjustment += (final_realized_pnl - prev_mtm)
            
            # Cleanup
            tranche.is_active = False
            self.closed_tranches.append(tranche)
            self.open_tranches.remove(tranche)

        # Apply the final adjustment to the last point in the equity curve
        current_equity += terminal_adjustment
        self.equity_curve[-1] = current_equity
        
        return current_equity

    def run(self, df, trading_signal, liquidate=False):
        current_equity = self.initial_capital

        for i, row in df.iterrows():
            today = row['date']
            self.trading_days.append(today)

            # 1. Interest on cash balance
            interest_earned = current_equity * row['r_daily']
            
            # 2. Update existing positions (MtM and Exits)
            daily_mtm_change = 0
            for tranche in self.open_tranches[:]:
                time_remaining = max(0, (tranche.exit_date - today).days) / 365.0
                
                if today >= tranche.exit_date:
                    # EXIT: Use ASK price (to buy back the short) + slippage + fees
                    _, _, synth_ask = self.get_synthetic_prices(
                        row['forward'], tranche.strike, 0.0001,
                        row['r_annual'], row['atm_iv'], row['spread_ratio']
                    )
                    exit_price = synth_ask * (1 + self.slippage_pct)
                    
                    # Total Dollar PnL for the tranche
                    realized_pnl = ((tranche.entry_price - exit_price) * tranche.quantity * 100) \
                                   - (self.fixed_fee * tranche.quantity)
                    
                    prev_mtm = tranche.mtm_history[-1] if tranche.mtm_history else 0
                    daily_mtm_change += (realized_pnl - prev_mtm)
                    
                    tranche.is_active = False
                    self.closed_tranches.append(tranche)
                    self.open_tranches.remove(tranche)
                else:
                    # MtM: Mark to Synthetic MID
                    _, current_mid, _ = self.get_synthetic_prices(
                        row['forward'], tranche.strike, time_remaining, 
                        row['r_annual'], row['atm_iv'], row['spread_ratio']
                    )
                    current_tranche_pnl = (tranche.entry_price - current_mid) * tranche.quantity * 100
                    prev_mtm = tranche.mtm_history[-1] if tranche.mtm_history else 0
                    
                    daily_mtm_change += (current_tranche_pnl - prev_mtm)
                    tranche.mtm_history.append(current_tranche_pnl)

            # 3. Update Equity before potentially entering new trades
            total_daily_change = daily_mtm_change + interest_earned
            current_equity += total_daily_change
            self.daily_pnl_log.append(total_daily_change)
            self.equity_curve.append(current_equity)
            

            # 4. New Entry Logic
            if trading_signal[today] == 1:
                # Laddered Notional Sizing
                notional_per_contract = row['close'] * 100
                daily_allocation = (current_equity * self.leverage) / self.horizon
                num_contracts = int(daily_allocation // notional_per_contract)
                
                if num_contracts > 0:
                    synth_bid, _, _ = self.get_synthetic_prices(
                        row['forward'], row['close'], 30/365.0,
                        row['r_annual'], row['atm_iv'], row['spread_ratio']
                    )
                    # We receive the BID - slippage
                    entry_price = synth_bid * (1 - self.slippage_pct)
                    
                    new_tranche = Tranche(
                        entry_date=today,
                        exit_date = self._get_exit_date(today, 30),
                        strike=row['close'],
                        entry_iv=row['atm_iv'],
                        entry_price=entry_price,
                        quantity=num_contracts
                    )
                    self.open_tranches.append(new_tranche)
                    self.trade_count += 1

        if liquidate:
            current_equity = self._liquidate(df, current_equity)

    
    def compute_sharpe_ratio(self, returns, risk_free_rate = 0.0):
        excess_returns = returns - risk_free_rate
        if excess_returns.std() == 0:
            return 0
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    

    def compute_sortino_ratio(self, returns, risk_free_rate = 0.0):
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) < 2 or downside_returns.std() == 0:
            return 0
        return (excess_returns.mean() / downside_returns.std()) * np.sqrt(252)


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
            "sharpe_ratio": self.compute_sharpe_ratio(returns, risk_free_rate) if len(returns) > 0 else 0,
            "sortino_ratio": self.compute_sortino_ratio(returns, risk_free_rate) if len(returns) > 0 else 0,
            "turnover": len(self.closed_tranches) * 2,
            "trade_count": self.trade_count
        }
        return metrics
    
    
    def get_equity_curve(self, name = 'equity_curve'):
        return pd.Series(
            self.equity_curve,
            index = self.trading_days,
            name = name
        )