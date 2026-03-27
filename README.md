# Dirichlet Process Mixture Model (DPMM) for Latent Market Regimes
## Bayesian tail-risk filtering in systematic volatility trading

### Context
In asset management and quantitative finance, capturing the variance risk premium, namely the spread between implied and realized volatility, is a well-known source of returns. Since market participants are willing to pay a premium for downside protection, acting as an insurer generates consistent yield. This dynamic is foundational to short-volatility strategies but leaves portfolios vulnerable to sudden market crashes.

In a naive short-volatility strategy an investor receives and holds volatility premium on a regular basis (e.g. monthly) regardless of the broader financial and macroeconomic environment. This strategy is exposed to massive tail risk because during market turmoils volatility investments incur disproportional losses that offset the persistent returns accrued.

To mitigate tail risk, it is possible to rely on backward- and forward-looking filters, or hedging. However, these technical filters are often lagging indicators that fail to react quickly enough to abrupt regime changes, and continuously buying insurance erodes the collected premium.

The proposed probabilistic framework dynamically adapts to market conditions by predicting and identifying stable and panic latent regimes, allowing to maximize the collected volatility premium while minimizing tail risk and drawdowns.

### Theoretical Framework & Methodology

To mitigate tail risk while collecting volatility premium, the proposed rule-based strategy executes a trade only if there is a high probability of being in a normal market regime, and a large enough edge on the volatility premium.

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

---

<p align="center">
<img width="889" height="428" alt="image" src="https://github.com/user-attachments/assets/0a28bbce-4c12-4d6a-80e6-3b4ee93e434e" />
</p>

<p align="center">
<img width="1490" height="1629" alt="image" src="https://github.com/user-attachments/assets/63d53e2f-ee11-4942-892f-494508b8aca7" />
</p>

---

#### GARCH-X
To forecast the volatility premium, the framework employs a GARCH-X model.  
Log-returns $r_t$ are assumed to evolve as
```math
    r_t = \mu + \epsilon_t,
```
where $\epsilon_t = \sigma_t z_t$ with $z_t \sim \text{Norm}(0,1)$ and variance defined by the GARCH-X recursion
```math
    \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2 + \gamma S_{t-1}.
```
The exogenous variable $S$ is a variance proxy based on Parkinson volatility $P$
```math
    S_t = (b_0 + b_1 P_t)^2, \qquad P_t = \sqrt{ \frac{1}{4 \ln(2)} \left( \ln\left(\frac{\text{high}_t}{\text{low}_t}\right) \right)^2 },
```
where the coefficients $b_0, b_1$ are estimated by regressing high-frequency realized variance $s_t^2$ on Parkinson volatility
```math
    s_t^2 = b_0 + b_1 P_t.
```
Daily realized volatility $s_t$ for day $t$ is computed from high-frequency intraday log-returns $\{r_{t,n}\}_n$ as
```math
    s_t = \sqrt{252} \sqrt{ \sum_{n} r_{t,n}^2 + \left( \ln \left( \frac{\text{open}_t}{\text{close}_t} \right) \right)^2 }.
```
Given the estimated GARCH-X parameters $\hat\omega,\hat\alpha,\hat\beta,\hat\gamma$ and the long-run persistance
```math
    \hat\rho = \hat\alpha + \hat\beta + \hat\gamma \, \text{median}(\{S_t\}_t),
```
volatility is forecasted over a the future time horizon of $H$ days as
```math
    \hat\sigma_{t,H} = \sqrt{\frac{252}{H}} \sqrt{H \frac{\hat\omega}{1-\hat\rho} + \left(\sigma_t^2 - \frac{\hat\omega}{1-\hat\rho}\right) \frac{\hat\rho(1-\hat\rho^H)}{1-\hat\rho} }.
```
---
<p align="center">
<img width="1185" height="696" alt="image" src="https://github.com/user-attachments/assets/57d5a795-3f60-4cf6-8dcd-a1fbf9bb98b1" />
</p>
---

### Results & Performance
The data comprises:
- 4-Week Treasury Bill Secondary Market Rate (DTB4WK) for Sharpe Ratio calculation
- S&P 500 (SPY) option chain for DPMM volatility smile modeling
- S&P 500 (SPY) intraday price for GARCH volatility prediction
- S&P 500 (SPY) daily price for backtesting

with the models trained on data from January 2008 to April 2021, and strategies backtested using daily market data from May 2021 to December 2025.  
The strategies compared are:
- Long SPY strategy: buy and hold SPY
- Naive Short Volatility strategy: daily delta-hedged short volatility position on SPY 30-day options
- Conditional Short Volatility strategy: daily delta-hedged short volatility position on SPY 30-day options with trade executed on day $t$ only if
    - there is a large enough edge on volatility premium: $\text{IV}_t - \hat\sigma_{t,21} > \delta_{\sigma}$
    - there is a high probability of being in a normal market regime: $\hat\phi_{t,\text{Normal}} > \delta_{\phi}$

For backtesting we employ $\delta_{\sigma} = 1.0$ and $\delta_{\phi} = 0.75$. Compared to the baselines, the conditional Bayesian strategy drastically mitigates risk while capturing uncorrelated returns:
- the strategy successfully predicts panic regimes, safely exiting trading positions, thus completely avoiding the volatility spike that made the naive strategy go bankrupt 
- the strategy suffers a maximum drawdown of -26.0%, slighlty larger than the S&P 500's -25.0%, proving the ability to mitigate tail-risk
- the strategy generates genuine alpha by achieving a total cumulative return of 159.08% (against the 56.12% of the SPY baseline) and a Sharpe Ratio of 1.10 (against 0.47)
- the strategy exhibits uncorrelated returns from equities, with the rolling 60-day correlation to the SPY averaging 0.05

---

<div align="center">
    
| Strategy | Returns | Sharpe Ratio |
| :--- | :---: | :---: |
| Long SPY | $56.124\%$ | $0.472$ |
| Naive Short Volatility | $-111.785\%$ | $-0.528$ |
| Conditional Short Volatility | $159.076\%$ | $1.099$ |

</div>

---

<p align="center">
<img width="1186" height="695" alt="image" src="https://github.com/user-attachments/assets/a556f370-c868-4e96-8e24-7dd71dc949e9" />
</p>

---

<p align="center">
<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/2e8f8ef1-6515-4cdc-8aac-fdc964b65067" />
</p>

<p align="center">
<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/e2642e19-04e9-443f-8a0e-c9a6e11295e1" />
</p>

<p align="center">
<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/c55f0557-09bd-40d9-879e-589095f01b9e" />
</p>

---

<p align="center">
<img width="1186" height="489" alt="image" src="https://github.com/user-attachments/assets/2c7569f4-8977-4cdb-a5a4-b153a6ee4bee" />
</p>

---

<p align="center">
<img width="1042" height="374" alt="image" src="https://github.com/user-attachments/assets/b330604f-1032-4c3c-b56b-9372866309f9" />
</p>

<p align="center">
<img width="1037" height="374" alt="image" src="https://github.com/user-attachments/assets/711f3345-fea4-448e-9b06-b2d00420692e" />
</p>

<p align="center">
<img width="1042" height="374" alt="image" src="https://github.com/user-attachments/assets/5b38348e-dca7-46a0-a077-8939e0276664" />
</p>
