### Library Imports
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import warnings
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


### Plot FX Data
def plot_forex(df, forex_pairs):
    # Create Loop to Plot Each Individual FX Pair
    for pair in forex_pairs:
        plt.figure(figsize = (6, 4))

        plt.plot(df['Date'], df[pair]) # Need to adjust to spit out multiple graphs for each ticker
        plt.title(pair + ' Closing Rates Over Time')
        plt.xlabel('Date')
        plt.ylabel('FX Rates')

        plt.xticks(rotation = 45)  # Rotate x-axis labels for better readability
        plt.grid(True)
        plt.tight_layout()

    return plt.show()


### Correlation Matrix - Heat Map
def corr_matrix(df):
    df = df.drop(['Date'], axis = 1)
    correlation_matrix = df.corr()
    # Create the day Data heatmap using seaborn
    plt.figure(figsize = (6, 6))
    sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f', linewidths = 0.5)
    plt.title('Correlation Matrix of Forex Codes')
    plt.show()


### Compute Stacked Correlation
def stacked_correlations(df):
    # Compute the correlation matrix of the DataFrame
    corr_matrix = df.corr()
    # Stack the correlation matrix to convert it into a Series
    stacked_corr = corr_matrix.stack()
    # Filter out correlation values = 1 (self-correlation)
    filtered_corr = stacked_corr[stacked_corr < 1]
    # Sort the correlations in descending order
    sorted_corr = filtered_corr.sort_values(ascending = False)

    return sorted_corr


### Calculate daily returns of Most Correlated Pairs
def daily_rets_most_corr(forex_data):
    # Compute stacked correlation and get the most correlated pairs
    stacked_corr = stacked_correlations(forex_data)
    most_corr_pairs = list(stacked_corr.index[0])

    # Get FX data for the most correlated pairs
    most_corr_fx = forex_data[['Date'] + most_corr_pairs]

    # Compute the daily returns of the most correlated pairs
    most_corr_fx.set_index('Date', inplace = True)
    returns = most_corr_fx.pct_change().dropna()

    # Reset Headers
    returns.reset_index(inplace = True)

    return most_corr_pairs, returns


### Compute Hedge Ratio
def compute_hedge_ratio(pairs, returns):
    # Run linear regression to compute the hedge ratio
    rets1 = returns[pairs[0]]
    rets2 = returns[pairs[1]]
    # Add a constant to the independent variable
    rets1 = sm.add_constant(rets1)
    # Fit the regression model
    model = sm.OLS(rets2, rets1).fit()
    # Extract the hedge ratio
    hedge_ratio = model.params[pairs[0]]

    return hedge_ratio


### Define and Calculate Hedged and Unhedged Returns
def calculate_hedged_unhedged(long_pos, pairs, returns, hedge_ratio):
    # Compute Short Position
    short_pos = -hedge_ratio * long_pos

    # Compute daily returns for the long and short positions
    long_rets = returns[pairs[0]]
    short_rets = -hedge_ratio * returns[pairs[1]]

    # Compute the returns of the hedged and unhedged positions
    hedged_rets = long_rets + short_rets
    unhedged_rets = long_rets

    # Compute cumulative returns for the hedged and unhedged positions
    hedged_cumulative_rets = (1 + hedged_rets).cumprod() * long_pos
    unhedged_cumulative_rets = (1 + unhedged_rets).cumprod() * long_pos

    return short_pos, hedged_cumulative_rets, unhedged_cumulative_rets


### Plot Hedged vs. Unhedged Returns
def plot_hedged_returns(hedged_cumulative_rets, unhedged_cumulative_rets):
    plt.figure(figsize = (8, 5))
    plt.plot(hedged_cumulative_rets.index, hedged_cumulative_rets, label = 'Hedged Returns')
    plt.plot(unhedged_cumulative_rets.index, unhedged_cumulative_rets, label = 'Unhedged Returns')
    plt.xlabel('Periods')
    plt.ylabel('Cumulative Returns')
    plt.title('Hedged vs. Unhedged Cumulative Returns for Correlated FX Pairs')
    plt.legend()

    return plt.show()


### Function to Compute Performance Metrics
def performance_metrics(long_pos, hedged_cumulative_rets, unhedged_cumulative_rets, returns):
    # Drop na values for proper analysis
    hedged_cumulative_rets.dropna()
    unhedged_cumulative_rets.dropna()

    # Compute Hedged/Unhedged Total Returns (Current - Start)
    hedged_total_rets = hedged_cumulative_rets.iloc[-1] - long_pos
    unhedged_total_rets = unhedged_cumulative_rets.iloc[-1] - long_pos

    # Compute Hedged/Unhedged Annualized Returns
    hedged_ann_rets = (1 + hedged_total_rets)**(252 / len(returns)) - 1
    unhedged_ann_rets = (1 + unhedged_total_rets)**(252 / len(returns)) - 1

    # Compute Hedged/Unhedged Mean Returns
    hedged_mean = np.mean(hedged_cumulative_rets)
    unhedged_mean = np.mean(unhedged_cumulative_rets)

    # Compute Hedged/Unhedged Mean Returns
    hedged_stddev = np.std(hedged_cumulative_rets)
    unhedged_stddev = np.std(unhedged_cumulative_rets)

    # Compute the Hedged/Unhedged Sharpe Ratio
    hedged_sharpe_ratio = (hedged_mean / long_pos) / hedged_stddev * np.sqrt(252)
    unhedged_sharpe_ratio = (unhedged_mean / long_pos) / unhedged_stddev * np.sqrt(252)

    # Enhanced readability using formatted strings
    print(f"{'Metric':<30} {'Hedged':>15} {'Unhedged':>15}")
    print("="*60)
    print(f"{'Total Returns':<30} {hedged_total_rets:>15.4f} {unhedged_total_rets:>15.4f}")
    print(f"{'Annualized Returns':<30} {hedged_ann_rets:>15.4f} {unhedged_ann_rets:>15.4f}")
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