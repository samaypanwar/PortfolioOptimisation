import pandas as pd
from statsmodels import graphics
import statsmodels.api as sm
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import warnings
from tqdm import tqdm_notebook as tqdm
from scipy.stats import shapiro
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model as ARCH

warnings.filterwarnings("ignore")
plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = 15, 5
plt.rcParams["lines.linewidth"] = 1


def graphics(data):

    fig, axes = plt.subplots(nrows=1, ncols=3)
    fig.suptitle("Graphical Analysis")

    axes[0].hist(data, bins=20, alpha=0.8)
    axes[0].set_title("Histogram")

    sns.boxplot(y=data["Close"], ax=axes[1], orient="vertical")
    axes[1].set_title("Boxplot")

    sm.qqplot(data["Close"], ax=axes[2], line="q")
    axes[2].set_title("Q-Q Plot against a normal distribution")

    plt.show()


def check_stationarity(data, alpha=0.05):

    print("\n------Checking for Stationarity------ \n")
    testResults = adfuller(data)
    pvalue = testResults[1]

    if pvalue >= alpha:
        print("We do not Reject the Null Hypothesis")
        print("The data can not be assumed to be Stationary")

        return True

    else:
        print("We reject the Null Hypothesis")
        print("The data can be assumed to be Stationary")

        return False


def shapiro_normality(data, alpha=0.05):

    print("\n------Testing for Normality------ \n")
    _, pvalue = shapiro(data)

    if pvalue >= alpha:
        print("We do not Reject the Null Hypothesis")
        print("The data can be assumed to be Normally distributed")
        return True
    else:
        print("We reject the Null Hypothesis")
        print("The data can not be assumed to be Normally distributed")
        return False


def check_autocorrelation(data):

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle("Autocorrelation Plots", size=20)
    plot_acf(data, ax=axes[0])
    plot_pacf(data, ax=axes[1])
    plt.show()


def find_best_arima(data):

    """We want to find the order that fits our ARIMA model the best. We use
    Grid Search and go through each and every combination which is O(n2)
    """

    pvalues = [value for value in range(10)]
    qvalues = [value for value in range(10)]
    d = 1
    minAIC, bestOrder, bestTrend = float("inf"), None, None

    for p in tqdm(pvalues):
        for q in qvalues:
            for trend in ["c", "ct", "ctt"]:

                order = (p, d, q)
                try:
                    model = ARIMA(endog=data, order=order, trend=trend)
                    result = model.fit()
                    AIC = result.aic

                    if AIC < minAIC:
                        bestTrend = trend
                        minAIC = AIC
                        bestOrder = order

                    else:
                        continue
                except:
                    continue

    bestResult = ARIMA(endog=data, order=bestOrder, trend=bestTrend).fit()

    return bestResult, bestOrder


def plot_forecast(predictedArray, testSet, algo):

    fig, axes = plt.subplots()
    axes.fill_between(
        x=predictedArray.index,
        y1=predictedArray.Confidence_Interval_1,
        y2=predictedArray.Confidence_Interval_2,
        alpha=0.1,
        color="orange",
    )
    axes.plot(predictedArray.Prediction, color="red")
    axes.plot(testSet, color="blue")
    axes.set_title("Forecast of USD/JPY by {}".format(algo))
    axes.legend((predictedArray, testSet), ("Forecast", "True values"))

    plt.show()