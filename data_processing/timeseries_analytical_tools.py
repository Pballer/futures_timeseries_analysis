"""Module for timeseries analytical tools."""
import pandas as pd
import numpy as np
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


def test_stationarity(timeseries, window):
    """Run Dickey-Fuller test to check if data is stationary."""
    # Determing rolling statistics
    rolmean = timeseries.rolling(window=window).mean()
    rolstd = timeseries.rolling(window=window).std()

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries.iloc[window:], color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

def seasonal_charts(close, freq):
    """Generate stats models seasonal charts."""
    decomposition = seasonal_decompose(close, freq=freq)
    fig = plt.figure()
    fig = decomposition.plot();
    fig.set_size_inches(15, 8)
    return decomposition


def find_optimal_pdq(data, train_delta, start_train=None):
    """Brute force optimal pdq."""
    train_date = data.index.max() - (train_delta * data.index.freq)

    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    test_results = []

    for param in tqdm(pdq):
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(data.loc[start_train:train_date],
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                result_string = 'ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic)
                test_results.append([result_string, results.aic])

            except Exception as e:
                print('exception in:', pdq, seasonal_pdq)
                print(e)
                continue
    results_df = pd.DataFrame(test_results, columns=['key', 'aic'])
    print(results_df.loc[results_df.aic.idxmin()])
    return results_df


def plot_act_pacf(close, freq, lags=24):
    """Plot ACT and PACF charts."""
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    #plot the ACF
    fig = sm.graphics.tsa.plot_acf(close.diff(freq).dropna(), lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    #plot the PACF
    fig = sm.graphics.tsa.plot_pacf(close.diff(freq).dropna(), lags=lags, ax=ax2)


def sarimax_plot(data, order, trend, seasonal_order, train_delta, predict_delta, start_date=None, filename=None):
    """Build sarimax model and plot results."""
    train_date = data.index.max() - pd.DateOffset(months=12)
    mod = sm.tsa.statespace.SARIMAX(
        data.loc[start_date:train_date],
        trend=trend,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        seasonal_order=seasonal_order,
    )
    res = mod.fit()
    print(res.summary())

    prediction = res.predict(res.nobs, res.nobs + predict_delta)

    data.loc[start_date:].plot(figsize=(15, 6), label='Futures Price')
    ax = prediction.plot(figsize=(15, 6), c='orange', label='Predicted Price')
    conf_interval = res.get_forecast(predict_delta + 1).conf_int(alpha=.2)
    plt.fill_between(conf_interval.index,
                     conf_interval['lower close'],
                     conf_interval['upper close'],
                     color='b',
                     alpha=.1)
    ax.set_ylabel('Price (Cents per Bushel)')
    ax.set_xlabel('Date')
    ax.legend()
    if filename:
        # To capture x label.
        plt.subplots_adjust(bottom=0.22)
        plt.savefig(filename, transparent=True)

    return res
