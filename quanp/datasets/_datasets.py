from pathlib import Path
from typing import Optional
import logging, lxml, os, requests, shutil, sys, time, urllib, warnings

from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import itertools
import glob

from pandas.io.json import json_normalize
from anndata import AnnData

from .. import logging as logg, _utils
from .._compat import Literal
from .._settings import settings
from ..readwrite import read
from ._utils import check_datasetdir_exists, check_logdir_exists

# # import program settings
# sys.path.append('../../settings')  
from ..local_settings import TOS_API_KEY

HERE = Path(__file__).parent

def blobs(
    n_variables: int = 11,
    n_centers: int = 5,
    cluster_std: float = 1.0,
    n_observations: int = 640,
) -> AnnData:
    """\
    Gaussian Blobs.
    Parameters
    ----------
    n_variables
        Dimension of feature space.
    n_centers
        Number of cluster centers.
    cluster_std
        Standard deviation of clusters.
    n_observations
        Number of observations. By default, this is the same observation number
        as in :func:`scanpy.datasets.krumsiek11`.
    Returns
    -------
    Annotated data matrix containing a observation annotation 'blobs' that
    indicates cluster identity.
    """
    import sklearn.datasets

    X, y = sklearn.datasets.make_blobs(
        n_samples=n_observations,
        n_features=n_variables,
        centers=n_centers,
        cluster_std=cluster_std,
        random_state=0,
    )
    return AnnData(X, obs=dict(blobs=y.astype(str)))


########################################################################################

@check_datasetdir_exists
def wiki_sp500_members_update(update_filepath) -> pd.DataFrame:
    """\
    A function to update Standard & Poor 500 listed companies according to wikipedia
    @ https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
    """
    template_filename = HERE / 'sp500.csv'
    verbosity_save = settings.verbosity
    settings.verbosity = 'error'  # suppress output...
    settings.verbosity = verbosity_save

    df = pd.read_csv(template_filename)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True )
    df.set_index('date', inplace=True, drop=True)

    df_update = pd.read_csv(update_filepath)
    df_update['date'] = pd.to_datetime(df_update['date'])
    df_update = df_update.sort_values(by=['date'], ascending=True)
    df_update.set_index('date', inplace=True, drop=True)

    df_dict = {}
    for item in df.index:
        df_dict[item.strftime('%Y-%m-%d')] = df.at[item, 'tickers'].split(',')
    
    last_date = df.index[-1:].strftime('%Y-%m-%d')[0]
    df_update.index = df_update.index.strftime('%Y-%m-%d')

    for dt in df_update.index.unique():
        df_dict[dt] = df_dict[last_date].copy()
    
        try:
            if df_update.at[dt, 'ticker_add']:
                tick = df_update.at[dt, 'ticker_add']
                df_dict[dt].append(tick)
                print('Added', df_update.at[dt, 'ticker_add'])
            
        except:
            for tick in df_update.at[dt, 'ticker_add']:
                if tick is not np.nan:
                    df_dict[dt].append(tick)
                    print('Added', tick)
                
        try:
            if df_update.at[dt, 'ticker_drop']:
                df_dict[dt].remove(df_update.at[dt, 'ticker_drop'])
                print('Dropped', df_update.at[dt, 'ticker_drop'])

        except:
            for tick in df_update.at[dt, 'ticker_drop']:
                if tick is not np.nan:
                    df_dict[dt].remove(tick)
                    print('Dropped', tick)
                
        last_date = dt

    df_final = pd.DataFrame(columns=['tickers'])

    for key in df_dict.keys():
        df_final.at[key, 'tickers'] = ','.join(df_dict[key])

    df_final.to_csv(HERE / 'sp500_updated.csv')
    print('Updated sp500 list were saved as sp500_updated.csv at {}.'.format(HERE))

    return df_final


@check_datasetdir_exists
@check_logdir_exists
def download_tickers_price_history():
    """
    A function to download price history of Standard & Poor 500 listed companies based 
    on the updated sp500 members. SPY will also be downloaded as the
    benchmark.
    """
    # setting logger 
    logging.basicConfig(filename= settings.logdir / 'price_history_download.txt', 
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. the input csv directory
    csv_dir = HERE / 'sp500_updated.csv'

    # 2. create output directory if not exist
    output_dir = settings.datasetdir / 'daily'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    else:
        os.makedirs(output_dir)

    # the daily eod price
    try:
        df_meta = pd.read_csv(csv_dir, index_col=0, parse_dates=[0])
        df_meta_selected = df_meta.loc[df_meta.index > pd.Timestamp('2000-8-05')]
        tickers_list = list(df_meta_selected['tickers'].values)
        tickers_list_2d = [ls.split(',') for ls in tickers_list]
        tickers = list(itertools.chain.from_iterable(tickers_list_2d))
        tickers = set(tickers + ['SPY'])
        print('Total tickers to download:', len(tickers))

    except KeyError as ke:
        print(KeyError, 'Please provide a csv that contain a column', 
                        'named {}'.format(ke))
        logging.info('Please provide a csv that contain a column \
                        named {}'.format(ke))
        exit(1)


    # define current datetime in miliseconds
    current_milli_time = lambda: int(round(time.time() * 1000))

    # define payload
    payload = {'apikey': TOS_API_KEY,
                'periodType': 'year',
                'frequencyType': 'daily',
                'frequency': '1',
                'period': '20',
                'endDate': current_milli_time()}

    # Match with NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date='1996-01-01', end_date='2022-12-31')

    tickers_failed = []

    # loop and process through each provided ticker (including SPY)
    for ticker in tqdm(tickers, desc='Downloading and processing price history...', 
                                unit='ticker', 
                                position=0):

        # define the endpoint
        endpoint = "https://api.tdameritrade.com/v1/marketdata/{}/pricehistory".format(ticker)

        # request and process data
        content = requests.get(url = endpoint, params = payload)
        data = content.json()

        try:

            # dataframing
            df_tmp = pd.DataFrame.from_dict(data['candles'])
            df_tmp['datetime'] = pd.to_datetime(df_tmp['datetime'], unit='ms').dt.date
            df_tmp = df_tmp[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            df_tmp.set_index('datetime', inplace=True, drop=True)
            df_tmp = df_tmp[df_tmp.index.astype(str).isin(schedule.index.astype(str))]
        
            # Reindex the df to accommodate missing values at certain NYSE calendar
            schedule_order = schedule.loc[df_tmp.index[0]:df_tmp.index[-1]]
            df_tmp = df_tmp.reindex(schedule_order.index) 
        
            # save the outputs
            if not len(df_tmp) < 2:
                df_tmp.to_csv(output_dir / '{}.csv'.format(ticker), sep=',', mode='w')
                # print('Downloaded data up to {} for {}'.format(df_tmp.index[-1:][0], ticker))
            
            else:
                print('Total price history collected for {} is < 2 - download \
                     discarded'.format(ticker))
                tickers_failed.append(ticker)                

        except KeyError as ke:
            # print(KeyError, 'column {} not found in the processed dataframe.'.format(ke), 
            #     'Possible reason may be the ticker provided not found in TOS database.')
            tickers_failed.append(ticker)

    logging.info('column {} not found in the processed dataframe. Possible \
                reason may be the ticker provided not found in TOS \
                database.'.format(tickers_failed))


    # list all downloaded tickers 
    downloaded_tickers = os.listdir(output_dir)

    # # filter out downloaded tickers (only for use when data downloaded)
    downloaded_tickers = set([os.path.splitext(f)[0] for f in os.listdir(output_dir)])
    tickers_failed = tickers - downloaded_tickers

    # matching the sp500 members for analyses
    df_dict = {}
    for item in df_meta_selected.index:
        df_dict[item.strftime('%Y-%m-%d')] = df_meta_selected.at[item, 'tickers'].split(',')
    
    for ticker in tickers_failed:
        for key in df_dict.keys():
            try:
                df_dict[key].remove(ticker)
            except:
                None

    df_final = pd.DataFrame(columns=['tickers'])

    for key in df_dict.keys():
        df_final.at[key, 'tickers'] = ','.join(df_dict[key])

    df_final = df_final.sort_index()
    saved_filename = 'sp500_updated_clean.csv'
    df_final.to_csv(HERE / saved_filename)

    print('sp500_updated_clean.csv has been save at {}'.format(HERE))
    logging.info('sp500_updated_clean.csv has been save at {}'.format(HERE))

    return None


@check_datasetdir_exists
@check_logdir_exists
def download_tickers_price_history_fromlist(ls_tickers):
    """
    A function to download price history of Standard & Poor 500 listed companies based 
    on the updated sp500 members. SPY will also be downloaded as the
    benchmark.
    """
    # setting logger 
    logging.basicConfig(filename= settings.logdir / 'price_history_download.txt', 
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')


    # 1. create output directory if not exist
    output_dir = settings.datasetdir / 'daily'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    else:
        os.makedirs(output_dir)


    # define current datetime in miliseconds
    current_milli_time = lambda: int(round(time.time() * 1000))

    # define payload
    payload = {'apikey': TOS_API_KEY,
                'periodType': 'year',
                'frequencyType': 'daily',
                'frequency': '1',
                'period': '20',
                'endDate': current_milli_time()}

    # Match with NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date='1996-01-01', end_date='2022-12-31')

    tickers_failed = []

    # loop and process through each provided ticker (including SPY)
    for ticker in tqdm(set(ls_tickers), desc='Downloading and processing price history...', 
                                unit='ticker', 
                                position=0):

        # define the endpoint
        endpoint = "https://api.tdameritrade.com/v1/marketdata/{}/pricehistory".format(ticker)

        # request and process data
        content = requests.get(url = endpoint, params = payload)
        data = content.json()

        try:

            # dataframing
            df_tmp = pd.DataFrame.from_dict(data['candles'])
            df_tmp['datetime'] = pd.to_datetime(df_tmp['datetime'], unit='ms').dt.date
            df_tmp = df_tmp[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            df_tmp.set_index('datetime', inplace=True, drop=True)
            df_tmp = df_tmp[df_tmp.index.astype(str).isin(schedule.index.astype(str))]
        
            # Reindex the df to accommodate missing values at certain NYSE calendar
            schedule_order = schedule.loc[df_tmp.index[0]:df_tmp.index[-1]]
            df_tmp = df_tmp.reindex(schedule_order.index) 
        
            # save the outputs
            if not len(df_tmp) < 2:
                df_tmp.to_csv(output_dir / '{}.csv'.format(ticker), sep=',', mode='w')
                # print('Downloaded data up to {} for {}'.format(df_tmp.index[-1:][0], ticker))
            
            else:
                print('Total price history collected for {} is < 2 - download \
                     discarded'.format(ticker))
                tickers_failed.append(ticker)                

        except KeyError as ke:
            # print(KeyError, 'column {} not found in the processed dataframe.'.format(ke), 
            #     'Possible reason may be the ticker provided not found in TOS database.')
            tickers_failed.append(ticker)

    logging.info('column {} not found in the processed dataframe. Possible \
                reason may be the ticker provided not found in TOS \
                database.'.format(tickers_failed))

    return None



@check_datasetdir_exists
@check_logdir_exists
def download_tickers_fundamental() -> pd.DataFrame:
    """
    A function to download cross-sectional fundamentals of 
    the Standard & Poor 500 listed companies based on the updated sp500 members. 
    """
    # setting logger 
    logging.basicConfig(filename= settings.logdir / 'fundamental_download.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. check presence of working directory and metadata file
    working_dir = settings.datasetdir / 'metadata'
    if not os.path.exists(working_dir):
        print('metadata directory and sp500_metadata.csv not found. \
              Please initialize first by quanp.datasets.get_wiki_sp500_metadata()')
        exit(1)

    # 2. the input csv directory
    csv_dir = working_dir / 'sp500_metadata.csv'

    # the daily eod price
    try:
        df_meta = pd.read_csv(csv_dir, index_col=0)
        tickers = list(df_meta.index)
        print('Total tickers to download:', len(tickers))

    except KeyError as ke:
        print(KeyError, 'Please provide a csv that contain a column', 
                        'named {}'.format(ke))
        logging.info('Please provide a csv that contain a column \
                        named {}'.format(ke))
        exit(1)

    # create list for download failed tickers
    tickers_failed = []

    # create empty dataframe for downloaded tickers
    df_fundamental = pd.DataFrame()

    # loop and process through each provided ticker
    for ticker in tqdm(tickers, desc='Downloading fundamentals...', unit='ticker', position=0):

        # define payload
        payload = {'apikey': TOS_API_KEY,
                    'symbol': ticker,
                    'projection': 'fundamental'}

        # define the endpoint
        endpoint = "https://api.tdameritrade.com/v1/instruments"

        # request and process data
        content = requests.get(url = endpoint, params = payload)
        data = content.json()

        try:
            # dataframing and appending data
            df_tmp = pd.DataFrame.from_dict(data[ticker])
            df_tmp_fundamental = df_tmp[['fundamental']]
            df_tmp_fundamental.columns = [ticker]
            df_fundamental[ticker] = df_tmp_fundamental[ticker]
                   
        except KeyError as ke:
            print(KeyError, 'column {} not found in the processed dataframe.'.format(ke), 
                'Possible reason may be the ticker provided not found in TOS database.')
            tickers_failed.append(ticker)

    logging.info('column {} not found in the processed dataframe. Possible \
                reason may be the ticker provided not found in TOS \
                database.'.format(tickers_failed))

    df_meta_merged = pd.merge(df_meta, df_fundamental.T, how='inner', 
                                                         left_index=True,
                                                         right_index=True)

    saved_filename = 'sp500_metadata_fundamentalAdded.csv'
    df_meta_merged.to_csv(working_dir / saved_filename)

    print('Fundamentals appended sp500_metadata file has been save as {}'.format(str(working_dir) + '/' + saved_filename))
    logging.info('Fundamentals appended sp500_metadata file has been save as {}'.format(str(working_dir) + '/' + saved_filename))

    return df_meta_merged

@check_datasetdir_exists
@check_logdir_exists
def get_wiki_sp500_metadata() -> pd.DataFrame:
    """\
    A function to download Standard & Poor 500 listed companies according to wikipedia
    @ https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
    """
    verbosity_save = settings.verbosity
    settings.verbosity = 'error'  # suppress output...
    settings.verbosity = verbosity_save

    # create output directory if not exist
    output_dir = settings.datasetdir / 'metadata'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get html file
    htmlFile = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")

    # parsing html file to beautisoup object
    soup = BeautifulSoup(htmlFile.text, 'lxml')

    # get table content
    data = []
    table = soup.find('table', attrs={'id': 'constituents'})
    table_body = table.find('tbody')

    rows = table_body.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        if cols:
            data.append([ele for ele in cols if ele]) # Get rid of empty values

    # list column headers
    tableHeader = [ele.text.strip() for ele in table_body.find_all('th')]

    # Dataframing
    saved_filename = 'sp500_metadata.csv'
    df = pd.DataFrame(data, columns=tableHeader)
    df.set_index('Symbol', inplace=True, drop=True)
    df.to_csv(output_dir / saved_filename)

    print('The metadata file has been initialized and saved as {}'.format(str(output_dir) + '/' + saved_filename))
    logging.info('The metadata file has been initialized and saved as {}'.format(str(output_dir) + '/' + saved_filename))

    return df


# A function to prepare eod price for each ticker in the list
@check_datasetdir_exists
@check_logdir_exists
def process_eod_price(ls_tickers, startdate='2020-11-17', enddate='2020-12-18'):
    
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date='2000-01-01', end_date='2025-12-31')
    
    csv_dir = settings.datasetdir / 'daily'
    processed_dir = settings.datasetdir / 'processed' / startdate / 'daily'
    df_eod = pd.DataFrame()

    if startdate in schedule.index:
        if enddate in schedule.index:

            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir)
    
            startdate = pd.to_datetime(startdate, utc=True)
            enddate = pd.to_datetime(enddate, utc=True)
        
            for ticker in ls_tickers:
                
                try:
                    # read raw data
                    df = pd.read_csv(csv_dir / '{}.csv'.format(ticker), index_col=0)
                    df.index = pd.to_datetime(df.index, utc=True)
     
                    # only include those data with the startdate and enddate data
                    if startdate in set(df.index):
                
                        if enddate in set(df.index):
                            df_eod[ticker] = df.loc[startdate:enddate]['close']
                        else:
                            print(ticker + ' was excluded as its last tick was collected from ' \
                                + str(df.index[-1]))     
                        
                    else:
                        print(ticker + ' was excluded as its first tick was collected from ' \
                              + str(df.index[0]))

                except:
                    print(ticker + ' was excluded as the ticker csv was not found in the  ' \
                          + str(csv_dir))                   
                    
        else:
            print('process_eod_price() function not performed. Please provide an enddate \
               that is a NYSE trading day.')
            logging.info('process_eod_price() function not performed. Please provide an enddate \
                     that is a NYSE trading day.')
            exit(1)                    
                    

        df_eod.index = schedule = pd.to_datetime(nyse.schedule(start_date=startdate, 
                                                               end_date=enddate).index, utc=True)
        df_eod.to_csv(processed_dir / 'eod_price.csv')
        
    else:
        
        print('process_eod_price() function not performed. Please provide a startdate \
               that is a NYSE trading day.')
        logging.info('process_eod_price() function not performed. Please provide a startdate \
                     that is a NYSE trading day.')
        exit(1)
    
    return df_eod


# A function to prepare eod price for each ticker in the list
@check_datasetdir_exists
@check_logdir_exists
def process_eod_price_volume(ls_tickers, startdate='2020-11-17', enddate='2020-12-18'):
    
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date='2000-01-01', end_date='2025-12-31')
    
    csv_dir = settings.datasetdir / 'daily'
    processed_dir = settings.datasetdir / 'processed' / startdate / 'daily'
    df_eod = pd.DataFrame()

    if startdate in schedule.index:
        if enddate in schedule.index:

            if not os.path.exists(processed_dir):
                os.makedirs(processed_dir)
    
            startdate = pd.to_datetime(startdate, utc=True)
            enddate = pd.to_datetime(enddate, utc=True)
        
            for ticker in ls_tickers:
                
                try:
                    # read raw data
                    df = pd.read_csv(csv_dir / '{}.csv'.format(ticker), index_col=0)
                    df.index = pd.to_datetime(df.index, utc=True)
     
                    # only include those data with the startdate and enddate data
                    if startdate in set(df.index):
                
                        if enddate in set(df.index):
                            df_eod['{}_close'.format(ticker)] = df.loc[startdate:enddate]['close']
                            df_eod['{}_volume'.format(ticker)] = df.loc[startdate:enddate]['volume']
                        else:
                            print(ticker + ' was excluded as its last tick was collected from ' \
                                + str(df.index[-1]))     
                        
                    else:
                        print(ticker + ' was excluded as its first tick was collected from ' \
                              + str(df.index[0]))

                except:
                    print(ticker + ' was excluded as the ticker csv was not found in the  ' \
                          + str(csv_dir))                   
                    
        else:
            print('process_eod_price() function not performed. Please provide an enddate \
               that is a NYSE trading day.')
            logging.info('process_eod_price() function not performed. Please provide an enddate \
                     that is a NYSE trading day.')
            exit(1)                    
                    

        df_eod.index = schedule = pd.to_datetime(nyse.schedule(start_date=startdate, 
                                                               end_date=enddate).index, utc=True)
        df_eod.to_csv(processed_dir / 'eod_price_volume.csv')
        
    else:
        
        print('process_eod_price() function not performed. Please provide a startdate \
               that is a NYSE trading day.')
        logging.info('process_eod_price() function not performed. Please provide a startdate \
                     that is a NYSE trading day.')
        exit(1)
    
    return df_eod