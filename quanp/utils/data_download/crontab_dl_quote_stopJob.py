#!/home/ec2-user/anaconda3/envs/robolee_py36/bin/python
__author__ = 'Hong Kai LEE'
version = '1.0'

import sys
from datetime import datetime, time
from crontab import CronTab

# define and start cron
cron = CronTab(user='ec2-user')

for job in cron:
    print('Removed daily job: {}'.format(job))
        
cron.remove_all(comment='DLjob') 
# cron.remove_all()  # Uncomment to stop all jobs  <<<<=============

cron.write()
