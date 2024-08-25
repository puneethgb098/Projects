import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.stats import norm


def bs_call(S, K, r, T, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(S, K, r, T, C):
    implied_vol = lambda sigma: abs(bs_call(S, K, r, T, sigma) - C)
    return optimize.minimize(implied_vol, x0=0.5).x[0]

S = 2400                          #the underlying asset price
K = np.linspace(2100, 2600, num=20) #the strike price #The strike prices are generated using the NumPy linspace function with start at 50, 
#stop at 150, and num as 21, which means the strike prices will be evenly spaced between 50 and 150 with 21 total strikes.
r = 7.31                         #the risk-free interest rate
T = 0.083                          # the time to maturity in years
market_prices = np.array([15.41, 12.13, 9.15, 6.42, 4.18, 2.55, 1.5, 0.85, 0.47, 0.25, 0.12, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

implied_vols = []
for i in range(len(K)):
    implied_vols.append(implied_volatility(S, K[i], r, T, market_prices[i]))

plt.plot(K, implied_vols, 'bo')
plt.xlabel('Strike Price')
plt.ylabel('Implied Volatility')
plt.title('Volatility Smile')
plt.show()
