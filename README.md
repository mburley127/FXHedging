## Comparative Analysis of FX Hedging Models
This repository contains implementations of various hedging models in Python. The models included are:

1. **Direct Heding Model**
   - `FX_DirectHedge`: Contains the implementation of the Direct Hedging model.

   This model involves taking a position in the forex market that directly offsets the risk of the primary position. The hedge ratio is computed by running a linear regression between the returns of two currency pairs. It aims to reduce the exposure by directly countering the potential losses from the long position with the short position.

2. **Correlated Hedging Model**
   - `FX_CorrelatedHedge`: Contains the implementation of the Correlated Hedging model.

   This model leverages the historical correlation between two forex assets to mitigate risk and enhance returns. The hedge ratio is determined based on the correlation coefficient between the two pairs, allowing for positions to be adjusted according to how closely the currency pairs move together. This method seeks to exploit the relationship between pairs to offset risks.

3. **Beta Hedging Model**
   - `FX_BetaHedge`: Contains the implementation of the Beta Hedging model.

   This model uses the beta coefficient to determine the hedge ratio, which measures the sensitivity of an asset's returns relative to a benchmark. The hedge ratio is computed through a linear regression between the base and benchmark currency pairs. This method adjusts the hedge based on how strongly the base pair moves in relation to the benchmark pair.

4. **Options Hedging Model**
   - `FX_OptionsHedge`: Contains the implementation of the Options Hedging model.

   This model employs the Black-Scholes Merton (BSM) model to price options and compute the necessary greeks, with a particular focus on delta ($\Delta$). Delta hedging involves adjusting the position in the underlying asset to offset the risk of the option position. The model allows for a more sophisticated hedging technique that uses financial derivatives to manage risk.Input basic info here

5. **Results**
   - `results`: Contains the complete write-up and full comparative analysis of results.

   This directory includes the detailed results and analysis of the four hedging models. It provides insights into the performance metrics such as total returns, percentage returns, mean returns, standard deviation, and Sharpe ratio, allowing for a comprehensive comparison of the hedging strategies.
