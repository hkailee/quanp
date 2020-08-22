#!/Users/leehongkai/anaconda/envs/trading/bin/python
__author__ = 'Hong Kai LEE'
version = '1.0'

###########################################################
# Import packages used commonly
import logging, os, requests, sys, time
import pandas as pd
import pandas_market_calendars as mcal
from pandas.io.json import json_normalize
import numpy as np
import itertools
import glob

# import program settings
sys.path.append('../../settings')  
from local import *
###########################################################

# 1: Checks if in proper number of arguments are passed gives instructions on proper use.
def argsCheck(numArgs):
	if len(sys.argv) < numArgs:
		print('To start the program, please provide a csv directory',
                '(absolute or relative path) with a column \'ticker\'')
		print('Usage: {} [csv directory] [output directory]'.format(sys.argv[0]))
		print('Examples: {} ../../Data/SP500_wiki_20200726.csv', 
                '../../Data'.format(sys.argv[0]))
		exit(1) # Aborts program. (exit(1) indicates that an error occurred)

argsCheck(3)

# 2. Logging events...
logging.basicConfig(filename= '../../Log/error_price_history_download.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# 3. the input csv directory
csv_dir = os.path.abspath(sys.argv[1])

# 4. create output directory if not exist
output_dir = os.path.abspath(sys.argv[2]) + '/daily'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# the daily eod price
try:
    df_meta = pd.read_csv(csv_dir, index_col=0, parse_dates=[0])
    df_meta_selected = df_meta.loc[df_meta.index > pd.Timestamp('2000-8-05', tz='utc')]
    tickers_list = list(df_meta_selected['tickers'].values)
    tickers_list_2d = [ls.split(',') for ls in tickers_list]
    tickers = list(itertools.chain.from_iterable(tickers_list_2d))
    tickers = set(tickers)
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

# loop and process through each provided ticker
for ticker in tickers:

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
        df_tmp.to_csv(output_dir + '/{}.csv'.format(ticker), sep=',', mode='w')
        print('Downloaded data up to {} for {}'.format(df_tmp.index[-1:][0], ticker))

    except KeyError as ke:
        print(KeyError, 'column {} not found in the processed dataframe.'.format(ke), 
            'Possible reason may be the ticker provided not found in TOS database.')
        tickers_failed.append(ticker)
        logging.info('column {} not found in the processed dataframe. Possible \
            reason may be the ticker provided not found in TOS database.'.format(ke))


# list all downloaded tickers 
downloaded_tickers = os.listdir(output_dir)

# collect list of csv with at least one na or negative value
ls_csv_with_na_or_neg = []
for csv in downloaded_tickers:

    df = pd.read_csv(output_dir + '/' + csv, index_col=0)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    if df.isnull().values.any() or df.lt(0).any().any():
        ls_csv_with_na_or_neg.append(csv)


# remove csv with na or negative value
for csv in ls_csv_with_na_or_neg:
    os.remove(output_dir + '/' +  csv)


# # filter out downloaded tickers (only for use when data downloaded)
# downloaded_tickers = set([os.path.splitext(f)[0] for f in os.listdir(output_dir)])
# tickers_failed = tickers - downloaded_tickers

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
saved_filename = 'sp500_20200801_v2.csv'
df_final.to_csv(os.path.dirname(os.path.abspath(csv_dir)) + '/' + saved_filename)

print('Updated metadata file has been save as {}'.format(saved_filename))
logging.info('Updated metadata file has been save as {}'.format(saved_filename))
    


