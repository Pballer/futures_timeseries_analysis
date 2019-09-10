"""Module for loading raw MRCI data and pre-processing it for analysis."""

import pandas as pd
import numpy as np
import glob
import os
import re

SOYBEANS = 'Soybeans(CBOT)'
CORN = 'Corn(CBOT)'


def get_raw_mrci_file_list(path='../data'):
    """Get list of raw MRCI files."""
    mrci_eod_filter = os.path.join(path, '*.csv')
    mrci_eod_files = glob.glob(mrci_eod_filter)
    return mrci_eod_files


def get_front_month(contract_name, data):
    """Create continuous price by taking the nearest contract to expire."""
    front_month = data.loc[data.future_name == contract_name]
    front_month = front_month.loc[front_month.groupby('date')['month'].idxmin()]
    front_month.sort_values(by='date', inplace=True)
    front_month.set_index('date', inplace=True)
    front_month = front_month.asfreq('B').fillna(method='ffill')
    return front_month


def get_monthly_average(close):
    """Reduce price to monthly average."""
    average = close.resample('m').mean()
    return average


def format_monthly_data(data, contract_name):
    """Format specific commodity to monthly price."""
    futures = get_front_month(CORN, data)
    futures_monthly = get_monthly_average(futures.close)
    file_name = re.sub('[() ]', '', SOYBEANS)
    futures_monthly.to_pickle('clean_data/{}_monthly.pkl'.format((contract_name)))


def concat_raw_data_files(mrci_eod_files):
    """Concat list of raw mrci files."""
    converters = {'month': lambda d: pd.to_datetime(d.replace('\r\n', ''), format='%b%y'),
                  'date': lambda d: pd.to_datetime(d, format='%y%m%d'),
                  'close': lambda c: np.double(c.replace('~', '.'))
                  }

    col_names = ['future_name', 'month', 'date', 'close']
    futures_all = pd.concat((pd.read_csv(f,
                                         usecols=[1, 2, 3, 7],
                                         header=1,
                                         names=col_names,
                                         converters=converters,
                                         )
                             for f in mrci_eod_files),
                            ignore_index=True)
    return futures_all


def format_raw_mrci_data():
    """Format raw mrci data to prepare it for analysis."""
    mrci_eod_files = get_raw_mrci_file_list()

    futures_all = concat_raw_data_files(mrci_eod_files)

    futures_all = futures_all.drop_duplicates()
    futures_all.to_pickle('../clean_data/all_close.pkl')

    format_monthly_data(futures_all, CORN)
    format_monthly_data(futures_all, SOYBEANS)

    return futures_all
