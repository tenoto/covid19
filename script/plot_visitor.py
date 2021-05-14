#!/usr/bin/env python

import pandas as pd
import numpy as np 
from datetime import date, datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib


date_start = '2020-02-01'
date_end = '2021-09-01'

df = pd.read_csv('script/data/visitor_arrivals.txt',
	skiprows=0,header=1,thousands=r',',delim_whitespace=True)
df['time'] = pd.to_datetime(df['yyyy-mm'])

plt.style.use('script/matplotlibrc_vaccine.txt')

figsize_x = 16
figsize_y = 9
fig = plt.figure(figsize=(figsize_x,figsize_y),tight_layout=True)
ax1 = fig.add_subplot(111)
ax1.fill_between(df['time'],df['visitor']/1e+4,
	step="post",color='#E67E22',alpha=1.0)
ax1.fill_between(
	[pd.to_datetime('2021-07-01'),pd.to_datetime('2021-08-01')],
	[9.0,9.0],
	[0.0,0.0],
	step="post",color='r',alpha=1.0)
ax1.set_xlim(date_start,date_end)
ax1.set_ylim(0.0,20)
ax1.set_title('日本政府観光局のデータによる、日本への入国者数')
ax1.set_xlabel('年月')
ax1.set_ylabel('月ごとの訪日外客数 (万人)',labelpad=20)
plt.text(pd.to_datetime('2021-07-01')-pd.DateOffset(25),10,
	'オリンピック時の9万人?', #backgroundcolor='white',
	ha='center', va='center',fontsize=25)
fig.savefig("fig/japan_visitor.pdf")
fig.savefig("fig/japan_visitor.jpg",dpi=300)
