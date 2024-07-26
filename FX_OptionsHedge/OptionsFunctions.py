### Library Imports
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from scipy.stats import norm
warnings.filterwarnings("ignore")


### Function to Import Rate Data
def import_fx_data(tickers, start_date, end_date):
    data = pd.DataFrame()
    if isinstance(tickers, str):
        tickers = [tickers]
        
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start = start_date, end = end_date)['Adj Close']
        
    # Reset index to make headings in the same row
    data.reset_index(inplace = True)
    # Convert values in date col to dt
    data['Date'] = pd.to_datetime(data['Date'])

    return data


### Function to Get Most Recent Stock Price (S_0) and Set Strike (K)
def set_price_and_strike(pair, start_date, end_date, option_type):
    # Function Call to get FX Data
    forex_data = import_fx_data(pair, start_date, end_date) 
    # Get the Spot Price (Most recent FX rate value)
    S_0 = forex_data[pair].iloc[-1]  # Extract the last value from the pair column (ex. 'USDEUR=X')
    # Set Strike Price 2% above (call) or 2% below (put) S_0
    if option_type == 'call':
        K = 1.02*S_0
    elif option_type == 'put':
        K = 0.98*S_0

    return forex_data, S_0, K


### Plot FX Rate Data
def plot_forex(df, forex_pairs):
    # Create Loop to Plot Each Individual FX Pair
    for pair in forex_pairs:
        plt.figure(figsize = (6, 4))

        plt.plot(df['Date'], df[pair]) # Need to adjust to spit out multiple graphs for each ticker
        plt.title(pair + ' Closing Rates Over Time')
        plt.xlabel('Date')
        plt.ylabel('FX Rates')

        plt.xticks(rotation = 45) # Rotate x-axis labels for better readability
        plt.grid(True)
        plt.tight_layout()

    return plt.show()


### Function to Compute Daily Returns
def daily_returns(fx_data):
    # Ensure 'Date' is set as the index for proper computation
    fx_data.set_index('Date', inplace = True)

    # Compute the daily returns of the forex data
    rets = fx_data.pct_change().dropna()

    # Reset index to make date a column again
    rets.reset_index(inplace = True)

    return rets


### Function to compute sigma (volatility)
def compute_sigma(returns):
    if 'Date' in returns.columns:
        returns = returns.drop(columns = 'Date')
    # Sigma computation with removed date col
    sigma = np.std(returns) * np.sqrt(252)  # Annualize the volatility
    return sigma.iloc[-1]


## Function to Build Black Scholes (BSM) Model
def black_scholes(S_0, K, r, r_f, T, sigma, option_type):
    # Compute d_1 and d_2
    d_1 = (np.log(S_0 / K) + (r - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    #print(d_1)
    d_2 = d_1 - sigma * np.sqrt(T)
    #print(d_2)
    
    # Compute Option Premium and Greeks
    if option_type == 'call':
        option_price = S_0 * np.exp(-r_f * T) * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)
        delta = np.exp(-r_f * T) * norm.cdf(d_1)
        theta = (-S_0 * sigma * np.exp(-r_f * T) * norm.pdf(d_1) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d_2)
                 + r_f * S_0 * np.exp(-r_f * T) * norm.cdf(d_1))
        rho = K * T * np.exp(-r * T) * norm.cdf(d_2)

    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d_2) - S_0 * np.exp(-r_f * T) * norm.cdf(-d_1)
        delta = -np.exp(-r_f * T) * norm.cdf(-d_1)
        theta = (-S_0 * sigma * np.exp(-r_f * T) * norm.pdf(d_1) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(-d_2)
                 + r_f * S_0 * np.exp(-r_f * T) * norm.cdf(-d_1))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d_2)

    else:
        raise ValueError('Error: Incorrect Option Type, must input call or put')

    # Calculate Remaining Greeks
    gamma = np.exp(-r_f * T) * norm.pdf(d_1) / (S_0 * sigma * np.sqrt(T))
    vega = S_0 * np.exp(-r_f * T) * np.sqrt(T) * norm.pdf(d_1)

    # Probabilities
    prob_ITM_call = norm.cdf(d_1)
    prob_ITM_put = norm.cdf(-d_1)
    prob_exercise_call = norm.cdf(d_2)
    prob_exercise_put = norm.cdf(-d_2)

    # Return Option Premium and Greeks
    return option_price, delta, gamma, theta, rho, vega, prob_ITM_call, prob_ITM_put, prob_exercise_call, prob_exercise_put


### Function to Implement the Hedge (Compute Hedge Position and hedged returns)
def hedge_pos_and_rets(fx_rets, delta, long_pos):
    if 'Date' in fx_rets.columns:
        fx_rets = fx_rets.drop(columns='Date')
    
    hedge_pos = -delta * long_pos
    hedged_rets = fx_rets + hedge_pos
    return hedge_pos, hedged_rets


### Calculate Hedged/Unhedged Returns
def calculate_hedged_unhedged(pair, long_pos, returns, option_delta):
    # Remove Date column as needed
    if 'Date' in returns.columns:
        returns = returns.set_index('Date')
    
    # Select the returns for the specified pair
    pair_returns = returns[pair]

    # Compute daily returns for the long position
    long_rets = pair_returns * long_pos

    # Compute Hedge Position: standard option hedge
    hedge_pos = option_delta * long_pos

    # Compute the returns of the hedged position
    hedged_rets = long_rets - hedge_pos * pair_returns

    # Compute cumulative returns for the hedged and unhedged positions
    hedged_cumulative_rets = (1 + hedged_rets / long_pos).cumprod() * (long_pos + abs(hedge_pos))
    unhedged_cumulative_rets = (1 + pair_returns).cumprod() * long_pos

    return hedge_pos, hedged_cumulative_rets, unhedged_cumulative_rets


### Function to Plot Hedged vs. Unhedged Returns
def plot_hedged_returns(hedged_cumulative_rets, unhedged_cumulative_rets):
    plt.figure(figsize=(10, 6))
    
    # Convert cumulative returns to percentage
    hedged_cumulative_rets_pct = ((hedged_cumulative_rets / hedged_cumulative_rets.iloc[0]) - 1) * 100
    unhedged_cumulative_rets_pct = ((unhedged_cumulative_rets / unhedged_cumulative_rets.iloc[0]) - 1) * 100

    plt.plot(hedged_cumulative_rets.index, hedged_cumulative_rets_pct, label='Hedged Returns (%)')
    plt.plot(unhedged_cumulative_rets.index, unhedged_cumulative_rets_pct, label='Unhedged Returns (%)')
    plt.xlabel('Periods')
    plt.ylabel('Cumulative Returns (%)')
    plt.title('Hedged vs. Unhedged Cumulative Returns for FX Option Hedging Strategy')
    plt.legend()
    plt.grid(True)

    plt.show()


### Function to Compute Performance Metrics
def performance_metrics(long_pos, hedge_pos, hedged_cumulative_rets, unhedged_cumulative_rets, returns):
    # Calculate total initial investments
    total_investment_hedged = long_pos + abs(hedge_pos)
    total_investment_unhedged = long_pos

    # Drop na values for proper analysis
    hedged_cumulative_rets = hedged_cumulative_rets.dropna()
    unhedged_cumulative_rets = unhedged_cumulative_rets.dropna()

    # Compute Hedged/Unhedged Total Returns (Current - Start)
    hedged_total_rets = (hedged_cumulative_rets.iloc[-1] - total_investment_hedged) / total_investment_hedged * 100
    unhedged_total_rets = (unhedged_cumulative_rets.iloc[-1] - total_investment_unhedged) / total_investment_unhedged * 100

    # Compute Hedged/Unhedged Annualized Returns
    hedged_ann_rets = ((hedged_cumulative_rets.iloc[-1] / total_investment_hedged)**(252 / len(returns)) - 1) * 100
    unhedged_ann_rets = ((unhedged_cumulative_rets.iloc[-1] / total_investment_unhedged)**(252 / len(returns)) - 1) * 100

    # Compute Hedged/Unhedged Mean Returns (annualized)
    hedged_mean = np.mean(hedged_cumulative_rets.pct_change().dropna()) * 252
    unhedged_mean = np.mean(unhedged_cumulative_rets.pct_change().dropna()) * 252

    # Compute Hedged/Unhedged Standard Deviation of Returns (annualized)
    hedged_stddev = np.std(hedged_cumulative_rets.pct_change().dropna()) * np.sqrt(252)
    unhedged_stddev = np.std(unhedged_cumulative_rets.pct_change().dropna()) * np.sqrt(252)

    # Compute the Hedged/Unhedged Sharpe Ratio
    hedged_sharpe_ratio = hedged_mean / hedged_stddev
    unhedged_sharpe_ratio = unhedged_mean / unhedged_stddev

    # Enhanced readability using formatted strings
    print(f"\nPerformance Metrics for Hedged and Unhedged Positions")
    print(f"{'Metric':<30} {'Hedged':>15} {'Unhedged':>15}")
    print("="*60)
    print(f"{'Total Returns (%)':<30} {hedged_total_rets:>15.4f} {unhedged_total_rets:>15.4f}")
    print(f"{'Annualized Returns (%)':<30} {hedged_ann_rets:>15.4f} {unhedged_ann_rets:>15.4f}")
    print(f"{'Mean Return':<30} {hedged_mean:>15.4f} {unhedged_mean:>15.4f}")
    print(f"{'Standard Deviation':<30} {hedged_stddev:>15.4f} {unhedged_stddev:>15.4f}")
    print(f"{'Sharpe Ratio':<30} {hedged_sharpe_ratio:>15.4f} {unhedged_sharpe_ratio:>15.4f}")

    # Explanation of Sharpe Ratio in print statements
    print(f"\nThe Sharpe ratio measures the risk-adjusted return of an investment.")
    print(f"In this instance, the hedged strategy has a Sharpe ratio of {hedged_sharpe_ratio:.4f},")
    print(f"which means for every unit of risk, the hedged portfolio is generating {hedged_sharpe_ratio:.4f} units of return above the risk-free rate.")
    print(f"The unhedged strategy has a Sharpe ratio of {unhedged_sharpe_ratio:.4f},")
    print(f"indicating that for every unit of risk, the unhedged portfolio is generating {unhedged_sharpe_ratio:.4f} units of return above the risk-free rate.")
    print(f"A higher Sharpe ratio typically indicates a more favorable risk-adjusted return.")
