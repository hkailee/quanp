#!/Users/leehongkai/anaconda/envs/trading/bin/python
__author__ = 'Hong Kai LEE'
version = '1.0'

###########################################################
# Import packages used commonly
import os, requests, sys, time, urllib
import pandas as pd
from splinter import Browser
from pandas import json_normalize

# In-house utility script
from utils.housekeeping import argsCheck_main
###########################################################

# Housekeeping
argsCheck_main(2)
working_dir = '/home/ec2-user/robolee/'

# the intraday price
tickers = sys.argv[1]

# define the endpoint
endpoint = r"https://api.tdameritrade.com/v1/marketdata/quotes"

# define payload
payload = {'apikey': 'IPVNKJZGPDE5VINELGX3PCA6GWAJDX40',
            'symbol': tickers}

# retrieve data
content = requests.get(url = endpoint, params = payload)
data = content.json()

# create quote directory
if not os.path.exists(working_dir + 'quote'):
    os.makedirs(working_dir + 'quote')

# save or append quote
for ticker in tickers.split(","):
    df = json_normalize(data[ticker])
    if not os.path.exists(working_dir + 'quote/{}.csv'.format(ticker)):
        df.to_csv(working_dir + 'quote/{}.csv'.format(ticker), index=False, mode='w', header=True)
    else:
        df.to_csv(working_dir + 'quote/{}.csv'.format(ticker), index=False, mode='a', header=False)
