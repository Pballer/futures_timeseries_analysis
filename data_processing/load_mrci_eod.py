"""Module for loading daily close prices for all futures contracts."""
from bs4 import BeautifulSoup
import pandas as pd
import requests
from tqdm import tqdm
from format_mrci_data import format_raw_mrci_data

COLUMN_NAMES = ['future_name', 'Mth', 'Date', 'Open', 'High', 'Low', 'Close', 'Change', 'Volume', 'Open Int', 'Change']


def get_mrci_hloc_eod(url):
    """Load MRCI webpage and parse the main table."""
    mrci_req = requests.get(url)
    mrci_soup = BeautifulSoup(mrci_req.content)

    futures_table = mrci_soup.select('table tr')

    futures_eod = []
    for table_row in futures_table:
        try:
            future_name = table_row.find('th').get_text()
        except:
            pass

        row_data = [data.get_text() for data in table_row.find_all('td')]
        if len(row_data) == 10:
            futures_eod.append([future_name] + row_data)

    return futures_eod


if __name__ == '__main__':
    base_url = 'https://www.mrci.com/ohlc/{year}/{yymmdd}.php'
    for date in tqdm(pd.date_range(pd.datetime(2012, 10, 3), pd.datetime.today())):
        url = base_url.format(year=date.year, yymmdd=date.strftime('%y%m%d'))
        print(url)
        data = get_mrci_hloc_eod(url)
        futures_eod = pd.DataFrame(data, columns=COLUMN_NAMES)
        if not futures_eod.empty:
            file_name = 'data/mrci_{yymmdd}.csv'.format(yymmdd=date.strftime('%y%m%d'))
            futures_eod.to_csv(file_name)

    futures_all = format_raw_mrci_data()
