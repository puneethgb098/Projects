import numpy as np
from scipy.stats import norm

#define variables
r = 0.068
S = 2500
K = 2600
T = 30/265
sigma = 0.30

def blackScholes(r, S, K, T, sigma, type="c"):
    "Calculate BS price of call/put"
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    try:
        if type == "c":
            price = S*norm.cdf(d1, 0, 1) - K*np.exp(-r*T)*norm.cdf(d2, 0, 1)
        elif type == "p":
            price = K*np.exp(-r*T)*norm.cdf(-d2, 0, 1) - S*norm.cdf(-d1, 0, 1)
        return price
    except:
        print("Please confirm option type, either 'c' for Call or 'p' for Put!")

option_type = input("Enter option type (c for Call, p for Put): ")
option_type = option_type.lower()

if option_type == 'c' or option_type == 'p':
    option_price = blackScholes(r, S, K, T, sigma, option_type)
    print("Option Price:", option_price)
else:
    print("Invalid option type entered!")
