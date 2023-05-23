import numpy as np
from scipy.stats import norm

# Define parameters
S0 = 2400  # initial stock price
K1 = 2500   # strike price of call option
K2 = 2500    # strike price of put option
T = 0.3      # time to expiration (in years)
r = 7   # risk-free interest rate
sigma = 0.2   # volatility

# Define Black-Scholes formula for option pricing
def bs_price(S, K, T, r, sigma, option_type):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    elif option_type == 'put':
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return price

# Calculate delta and gamma values for call option
C1 = bs_price(S0, K1, T, r, sigma, 'call')
delta_C1 = norm.cdf((np.log(S0/K1) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T)))
gamma_C1 = norm.pdf((np.log(S0/K1) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))) / (S0*sigma*np.sqrt(T))

# Calculate delta and gamma values for put option
P2 = bs_price(S0, K2, T, r, sigma, 'put')
delta_P2 = norm.cdf((np.log(S0/K2) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))) - 1
gamma_P2 = norm.pdf((np.log(S0/K2) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))) / (S0*sigma*np.sqrt(T))

# Construct zero gamma portfolio
n_call = -1 * (delta_P2 / delta_C1)   # sell call options
n_put = 1                            # buy put options
cash_position = n_call*C1 + n_put*P2

# Verify that the portfolio has zero gamma
gamma_portfolio = n_call*gamma_C1 + n_put*gamma_P2
print('Gamma of portfolio:', gamma_portfolio)
