#!/home/ec2-user/anaconda3/envs/robolee_py36/bin/python
__author__ = 'Hong Kai LEE'
version = '1.0'

import sys
from datetime import datetime, time
from crontab import CronTab
import pandas as pd
import pandas_market_calendars as mcal

# In-house utility script
from utils.housekeeping import argsCheck_main

# Housekeeping
argsCheck_main(2)

# the tickers
tickers = sys.argv[1]

nyse = mcal.get_calendar('NYSE')

today = str(datetime.now().date())

schedule = nyse.schedule(start_date='2020-01-25', end_date='2020-12-31')
schedule['open_hour'] = pd.DataFrame(schedule)['market_open'].dt.hour
schedule['open_minute'] = pd.DataFrame(schedule)['market_open'].dt.minute

if today in schedule.index:        

    hr= schedule.loc[today, 'open_hour']
    minut = schedule.loc[today, 'open_minute']

    # define cron
    cron = CronTab(user='ec2-user')

    # first hour
    job_1st_hr = cron.new(command='/home/ec2-user/robolee/dl_quote.py {}'.format(tickers), comment='DLjob')
    job_1st_hr.hour.on(hr)
    job_1st_hr.minute.on(minut,minut+5,minut+10,minut+15,minut+20,minut+25)

    # subsequent hours
    job_subsequent_hr = cron.new(command='/home/ec2-user/robolee/dl_quote.py {}'.format(tickers), comment='DLjob')
    job_subsequent_hr.hour.on(hr+1,hr+2,hr+3,hr+4,hr+5,hr+6)
    job_subsequent_hr.minute.on(0,5,10,15,20,25,30,35,40,45,50,55)

    # last hour
    job_last_hr = cron.new(command='/home/ec2-user/robolee/dl_quote.py {}'.format(tickers), comment='DLjob')
    job_last_hr.hour.on(hr+7)
    job_last_hr.minute.on(0)

    for job in cron:
        print(job)
        
    # cron.remove_all(comment='DLjob') 
    cron.write()
