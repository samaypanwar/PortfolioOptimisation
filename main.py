#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[21]:


Tickers = [
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOG",
    "FB",
    "TSLA",
    "BABA",
    "NVDA",
    "INTC",
    "CRM",
    "AMD",
    "BIO",
    "BIIB",
    "CVS",
    "PODD",
    "HZNP",
    "QDEL",
]

Stocks = yf.download(
    tickers=Tickers,
    start="2015-01-01",
    end="2020-12-31",
    progress=True,
    auto_adjust=True,
)["Close"]

SP500 = yf.download(
    tickers="^GSPC",
    start="2015-01-01",
    end="2020-12-31",
    progress=True,
    auto_adjust=True,
)["Close"]

TS10 = (
    yf.download(
        tickers="^TNX",
        start="2015-01-01",
        end="2020-12-31",
        progress=True,
        auto_adjust=True,
    )["Close"]
    / 100
)

StocksNormal = (Stocks - Stocks.mean()) / Stocks.std()
SP500Normal = (SP500 - SP500.mean()) / SP500.std()
StocksLogReturns = np.log(Stocks / Stocks.shift(1))[1:]
SP500LogReturns = np.log(SP500 / SP500.shift(1))[1:]


# In[22]:


fig, axes = plt.subplots(2, 1)
axes[0].plot(Stocks)
axes[0].set_title("Stonks", size=14)
axes[1].plot(StocksLogReturns)
plt.show()


# In[23]:


StocksCov = StocksLogReturns.cov()
D, S = np.linalg.eigh(StocksCov)
Eigenportfolio = S[:, -1] / np.sum(S[:, -1])  # Normalize to sum to 1
EigenportfolioLargest = pd.DataFrame(
    data=Eigenportfolio,
    columns=["Investment Weight"],
    index=list(StocksLogReturns.columns),
)


# In[24]:


EigenportfolioLargest.plot(kind="bar", alpha=0.8)


# In[25]:


fig, axes = plt.subplots(2)
axes[0].plot(SP500)
axes[1].plot(TS10)
fig.suptitle("Market Index and Risk Free Rate", size=20)
plt.show()


# In[26]:


portfolioData = np.dot(StocksLogReturns.values, EigenportfolioLargest.values)
portfolioReturns = pd.DataFrame(
    data=portfolioData.cumsum() * 100,
    index=StocksLogReturns.index,
    columns=["Portfolio Returns"],
)


# In[27]:


plt.plot(portfolioReturns)
plt.title("Portfolio Returns in %", size=18)
plt.show()


# In[28]:


marketReturnsCumsum = SP500LogReturns.cumsum() * 100
plt.plot(marketReturnsCumsum, label="Market Returns")
plt.title("Market Returns in %", size=18)
plt.show()


# In[30]:


ExcessReturnsofPortfolio = np.subtract(portfolioData[11:].T, TS10.values)[0]
ExcessReturnsofMarket = SP500LogReturns[11:].values - TS10.values
regressionData = pd.DataFrame(
    {
        "Excess Returns of Portfolio": ExcessReturnsofPortfolio,
        "Excess Returns of Market": ExcessReturnsofMarket,
    },
    index=StocksLogReturns.index[11:],
)
regressionData.head()


# In[31]:


regressionData.plot()


# In[32]:


X = sm.add_constant(regressionData["Excess Returns of Market"].values)
Y = regressionData["Excess Returns of Portfolio"].values
result = sm.OLS(Y, X).fit()


# In[33]:


result.summary()


# In[35]:


jensenAlpha = result.params[0]
print("\n ------- CONGRATULATIONS ! ! ---------\n")
print("Your Portfolio has a Jensen Alpha of {}".format(round(jensenAlpha, 6)))


# In[ ]:
