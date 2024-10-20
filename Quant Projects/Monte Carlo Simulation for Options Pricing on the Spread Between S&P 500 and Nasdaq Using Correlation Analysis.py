import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf

sp500 = yf.download('^GSPC', start='2010-01-01', end='2024-10-19')
nasdaq = yf.download('^IXIC', start='2010-01-01', end='2024-10-19')

data = pd.DataFrame({'SP500_Close': sp500['Close'],'Nasdaq_Close': nasdaq['Close']})

data['SP500_Returns'] = np.log(data['SP500_Close'] / data['SP500_Close'].shift(1))
data['Nasdaq_Returns'] = np.log(data['Nasdaq_Close'] / data['Nasdaq_Close'].shift(1))

data.dropna(inplace=True)

correlation = data['SP500_Returns'].corr(data['Nasdaq_Returns'])
covariance = data['SP500_Returns'].cov(data['Nasdaq_Returns'])
print(f"Correlation: {correlation}")
print(f"Covariance: {covariance}")

annualized_volatility_sp500 = data['SP500_Returns'].std() * np.sqrt(252)
annualized_volatility_nasdaq = data['Nasdaq_Returns'].std() * np.sqrt(252)
print(f"Annualized Volatility S&P 500: {annualized_volatility_sp500}")
print(f"Annualized Volatility Nasdaq: {annualized_volatility_nasdaq}")

num_simulations = 10000
time_horizon = 252  
sp500_initial = data['SP500_Close'].iloc[-1]
nasdaq_initial = data['Nasdaq_Close'].iloc[-1]
correlation_matrix = np.array([[1, correlation], [correlation, 1]])
chol_decomp = np.linalg.cholesky(correlation_matrix)

def simulate_price_paths(S0, mu, sigma, T, steps, correlation_matrix):
    dt = T / steps
    prices = np.zeros((steps + 1, 2))
    prices[0] = [S0[0], S0[1]]
    
    for t in range(1, steps + 1):
        z = np.random.normal(size=2)
        correlated_random = chol_decomp @ z
        prices[t, 0] = prices[t-1, 0] * np.exp((mu[0] - 0.5 * sigma[0]**2) * dt + sigma[0] * np.sqrt(dt) * correlated_random[0])
        prices[t, 1] = prices[t-1, 1] * np.exp((mu[1] - 0.5 * sigma[1]**2) * dt + sigma[1] * np.sqrt(dt) * correlated_random[1])
    
    return prices

simulated_prices = np.zeros((num_simulations, time_horizon + 1, 2))

for i in range(num_simulations):
    simulated_prices[i] = simulate_price_paths(
        [sp500_initial, nasdaq_initial],
        [0, 0],  
        [annualized_volatility_sp500, annualized_volatility_nasdaq],
        1, 
        time_horizon,
        correlation_matrix
    )

strike_price = 0
option_prices = np.maximum(simulated_prices[:, -1, 0] - simulated_prices[:, -1, 1] - strike_price, 0)

risk_free_rate = 0.02  
discount_factor = np.exp(-risk_free_rate * 1)
option_price_estimate = discount_factor * np.mean(option_prices)
standard_error = np.std(option_prices) / np.sqrt(num_simulations)

print(f"Option Price Estimate: {option_price_estimate}")
print(f"Standard Error: {standard_error}")


plt.figure(figsize=(12, 6))
for i in range(5):
    plt.plot(simulated_prices[i, :, 0], label=f'Simulation {i+1} S&P 500')
    plt.plot(simulated_prices[i, :, 1], label=f'Simulation {i+1} Nasdaq')
plt.legend()
plt.title("Simulated Price Paths for S&P 500 and Nasdaq")
plt.xlabel("Time Steps")
plt.ylabel("Index Level")
plt.show()
