import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Define the ticker symbol for Bank Nifty
symbol = '^NSEBANK'

# Retrieve intraday data from Yahoo Finance for the last 10 years
data = yf.download(symbol, period='10y', interval='1d', group_by='ticker')

# Compute the intraday price movements
prices = data['Close']
returns = np.log(prices) - np.log(prices.shift(1))
avg_return = returns.mean()
std_return = returns.std()

plt.title('Intraday Price Movement for Bank Nifty')
plt.xlabel('Intraday Price Movement')
plt.ylabel('Frequency')
plt.hist(returns, bins=50, density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-(np.log(x)-avg_return)**2/(2*std_return**2))/(x*std_return*np.sqrt(2*np.pi))
plt.plot(x, p, 'k', linewidth=2)
plt.show()
