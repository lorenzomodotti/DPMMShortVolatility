## Dirichlet Process Volatility Modeling and Trading
### Bayesian volatility regime-switching and tail-risk filtering

This repository implements a systematic short-volatility trading strategy designed to capture the variance risk premium while dynamically filtering out tail-risk events. 

The framework leverages a Dirichlet Process Mixture Model to probabilistically cluster latent market regimes from implied volatility smiles, paired with a GARCH-X model for conditional variance forecasting.