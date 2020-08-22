#!/home/ec2-user/anaconda3/envs/robolee_py36/bin/python
__author__ = 'Hong Kai LEE'
version = '1.0'

import sys
from datetime import datetime, time
from crontab import CronTab

# In-house utility script
from utils.housekeeping import argsCheck_main

# Housekeeping
argsCheck_main(2)

# the tickers
tickers = sys.argv[1]

# define and start cron
cron = CronTab(user='ec2-user')

# add job
job_add = cron.new(command='/home/ec2-user/robolee/crontab_dl_quote.py {}'.format(tickers), comment='DL_core_job')
job_add.hour.on(11) ##### --->>>>To change back to 11
job_add.minute.on(0) ##### --->>>>To change back to 0 only
job_add.dow.on(1,2,3,4,5) ##### --->>>>To change back to 1-5

# remove job
job_remove = cron.new(command='/home/ec2-user/robolee/crontab_dl_quote_stopJob.py', comment='DL_core_job')
job_remove.hour.on(22) ##### --->>>>To change back to 21
job_remove.minute.on(0) ##### --->>>>To change back to 0 only
job_remove.dow.on(1,2,3,4,5)

for job in cron:
    print(job)
        
# cron.remove_all(comment='DLjob') 
cron.write()
