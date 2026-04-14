# Dirichlet Process Mixture Model (DPMM) for Latent Market Regimes
## Bayesian tail-risk filtering in systematic volatility trading

### Context
In asset management and quantitative finance, capturing the variance risk premium, namely the spread between implied and realized volatility, is a well-known source of returns. Since market participants are willing to pay a premium for downside protection, acting as an insurer generates consistent yield. This dynamic is foundational to short-volatility strategies but leaves portfolios vulnerable to sudden market crashes.

In a naive short-volatility strategy an investor receives and holds volatility premium on a regular basis (e.g. daily or monthly) regardless of the broader financial and macroeconomic environment. This strategy is exposed to massive tail risk because during market turmoils volatility investments incur disproportional losses that offset the persistent returns accrued.

To mitigate tail risk, it is possible to rely on backward- and forward-looking filters, or hedging. However, these technical filters are often lagging indicators that fail to react quickly enough to abrupt regime changes, and continuously buying insurance erodes the collected premium.

The proposed probabilistic framework dynamically adapts to market conditions by predicting and identifying stable and panic latent regimes, allowing to maximize the collected volatility premium while minimizing tail risk and drawdowns.

### Theoretical Framework & Methodology

To mitigate tail risk while collecting volatility premium, the proposed rule-based strategy executes a trade only if the market is not in a fear regime and the volatility premium has a large enough edge to offset costs.

#### DPMM
To identify latent market regimes, the framework employs a non-parametric Bayesian approach to cluster implied volatility smiles.  
The volatility smile $Y_t$ for day $t$ is the graph of implied volatility at 30-day expiry $\text{IV}_t$ as a function of log-moneyness $l$
```math
    Y_t = \{ (l, y) : l \in \mathbb{R}, \  y = \text{IV}_t(l) \}.
```
Consider a B-spline basis matrix $X$ constructed on grid knots $\{ j_i \}_i$ of log-moneyness based on empirical quantiles to ensure uniform data coverage, and defined as
```math
    X_{i,0}(\cdot) = \mathbb{1}_{[j_i, j_{i+1}]}(\cdot) \qquad \qquad X_{i,j}(\cdot) = \frac{\cdot - j_i}{j_{i+j} - j_i} X_{i,j-1}(\cdot) + \frac{j_{i+j+1} - \cdot}{j_{i+j+1} - j_{i+1}} X_{i+1,j-1}(\cdot), \quad j>0.
```
Given the fixed B-spline basis matrix $X$, the volatility smile is modeled as a random draw from a Huber distribution to ensure smoothness and robustness against outliers
```math
    Y_t \, | \, \beta_t; X,\tau \stackrel{\text{iid}}{\sim} \text{Huber}(X\beta_t, \tau),
```
with $\tau$ an hyperparameter.
The spline coefficients $\beta_t$ active for day $t$ are drawn from a discrete probability measure
```math
    \beta_t \stackrel{\text{iid}}{\sim} P.
```
A Dirichlet Process (DP) prior governs the mixture weights
```math
    P \sim \text{DP}(\alpha, G_0).
```
The base measure for the Dirichlet Process prior is chosen as a multivariate Normal distribution with prior mean initialized from the data
```math
    G_0 = \text{Norm}(\mu, I).
```
Under this specification, the full likelihood for the volatility smile $Y_t$ over the discrete grid knots $\{ j_1,\ldots,j_M \}$ of log-moneyness is
```math
    p(Y_t \, | \, \beta_t, \pi ; X, \tau) = \sum_{k=1}^K \pi_k \prod_{i=1}^M p(y_{t}(j_i) \, | \beta_t; X, \tau),
```
where $\pi = (\pi_1,\ldots,\pi_K)$ are the cluster weights from the DPMM prior.

After training the DPMM, the posterior probability that day $t$ belongs to regime $k$ is
```math
    \hat\phi_{t,k} = \frac{\exp\{\ell_{t,k}\}}{\sum_{j=1}^K \exp\{\ell_{t,j}\}}, \qquad \ell_{t,k} = \log \hat\pi_k + \log p(Y_t \, | \, \hat\beta_t, \hat\pi ; X, \tau).
```
Given the posterior probabilities and the estimated centroids of volatility smiles, the posterior mean volatility smile for day $t$ can be recovered as
```math
    \hat{Y_t} = \sum_{k=1}^K \hat\phi_{t,k} X\hat\beta_{t,k}.
```
Using this posterior mean volatility smile, it is possible to define a "fear score" based on the level of ATM implied volatility, the skew of the smile, and the curvature of the smile:
```math
    \text{\psi}_t = 0.2 * L_t + 0.6 * S_t + 0.2 * C_t,
```
where
```math
    L_t = \frac{\hat{Y_{t, \text{ATM}}}}{\bar{Y_{t, \text{ATM}}}} \qquad S_t = \frac{\hat{Y_{t, \text{Put}}}-\hat{Y_{t, \text{ATM}}}}{\hat{Y_{t, \text{ATM}}}}, \qquad C_t = \frac{\hat{Y_{t, \text{Put}}}+\hat{Y_{t, \text{Call}}}-2\hat{Y_{t, \text{ATM}}}}{\hat{Y_{t, \text{ATM}}}},
```
where $\hat{Y_{t, \text{ATM}}}$ is the value of ATM implied volatility, $\hat{Y_{t, \text{Put}}}$ is the value of deep OTM puts, and $\hat{Y_{t, \text{Call}}}$ is the value of deep OTM calls. Intuitively, $L_t$ quantifies how expensive the market is relative to its recent history, $S_t$ quantifies the price investors are paying for downside protection, and $C_t$ quantifies convexity of the smile.

---



---

#### HAR
To forecast realized volatility over the holding horizon $H$, the framework employs a Heterogeneous Auto Regressive (HAR) model.  
Given daily realized volatility $s_t$ for day $t$ computed from high-frequency intraday log-returns $\{r_{t,n}\}_n$:
```math
    s_t = \sqrt{252} \sqrt{ \sum_{n} r_{t,n}^2 + \left( \ln \left( \frac{\text{open}_t}{\text{close}_t} \right) \right)^2 },
```
regressors for the HAR model include the average past weekly and monthly volatility, and are computed as:
```math
    s_{t, \text{w}} = \frac{1}{5}\sum_{h=1}^5 s_{t+1-h}, \qquad s_{t, \text{m}} = \frac{1}{21}\sum_{h=1}^21 s_{t+1-h},
```
while the target is the average future volatility over over the holding horizon $T$:
```math
    s_{t, H} = \frac{1}{H}\sum_{h=1}^H s_{t+h}.
```
The HAR specification
```math
    s_{t, T} = \beta_0 + \beta_1 s_t + \beta_2 s_{t, \text{w}} + \beta_3 s_{t, \text{m}} + \varepsilon_t
```
can be fitted by OLS with Heteroskedasticity- and Autocorrelation-Consistent (HAC) standard errors.

---



---
### Trading Strategy
The proposed systematic short volatility strategy on the SPY enters a trade on day $t$ only if:
- there is a large enough edge on volatility premium: $\text{IV}_t - \hat\s_{t,21} - (\text{spread_ratio}_t \text{IV}_t) > \delta_{\sigma}$
- there is a low market fear score: $\psi_t < \delta_{\psi}$
- the underlying instrument does not have negative momentum: $\text{SPY}_t >= \text{EMA}(\text{SPY}, 10)$

A trade consists of:
- selling ATM straddles on the SPY with 14 days to expiration
- holding the position for 7 days
- buying back the ATM straddles

The number of contracts to buy is determined accounting for the holding period, margin requirement, and leverage.
For each transaction, the backtester accounts for a fixed fee and charges the full spread (buy at ask, sell at bid).

---
### Results & Performance
The data comprises:
- 1-month US Treasury Rate for option pricing and backtesting
- S&P 500 (SPY) option chain for DPMM volatility smile modeling
- S&P 500 (SPY) intraday price for HAR volatility prediction
- S&P 500 (SPY) daily price and dividends for option pricing and backtesting
and spans the horizon between February 2019 and March 2026.

The proposed short straddle strategy is compared with the following ablations:
- short straddle, no rules
- short straddle, volatility edge rule only
- short straddle, market fear rule only

The strategies are evaluated using Walk Forward Optimization:
- DPMM and HAR models are trained on a training partition covering 1.5 years
- the strategy parameters $\delta_{\sigma}$ and $\delta_{\psi}$ are optimized on a validation partition covering 4 months
- the strategy is evaluated with the optimized parameters on a test partition covering 4 months
Each partition is disjoint, and after completing one full pass (training, optimization, testing) each partition is shifted forward by the length of the test partition.

The proposed strategy drastically mitigates risk while capturing returns. By reducing maximum drawdown by 57% and retaining 70% of total returns compared to the naive short straddle with no rules, the strategy significantly improves capital preservation while maintaining a competitive return profile. Morever, the combination of DPMM (fear) and HAR (edge) based rules allows to generate persistent alpha while also reducing trading frequency, thus isolating precise and profitable trading signals. 

---

<div align="center">

| Strategy | Total Return | Sharpe (Alpha) | Sharpe Ratio | Sortino Ratio | Max Drawdown | Trades |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| No Rule | $14.85\%$ | $0.0420$ | $0.3182$ | $0.4214$ | $-53.93\%$ | $435$ |
| Edge Rule | $10.36\%$ | $0.7319$ | $0.2061$ | $0.1655$ | $-31.65\%$ | $281$ |
| Fear Rule | $14.73\%$ | $0.3044$ | $0.2532$ | $0.1991$ | $-29.26\%$ | $294$ |
| Edge+Fear Rule | $10.36\%$ | $0.7863$ | $0.1809$ | $0.1244$ | $-23.15\%$ | $230$ |

</div>


