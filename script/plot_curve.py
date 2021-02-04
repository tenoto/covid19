#!/usr/bin/env python

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def get_running_average_data(x,y,window_size):
	"""
	x : x axis numpy array
	window_size : number of used data points for averaging (integer)
	x_runave : running averaged x axis numpy array
	y_runave : running averaged y axis numpy array 
	"""
	if x.__class__.__name__ != 'ndarray':
		print("Error: input x should be ndarray.")
		quit()
	if y.__class__.__name__ != 'ndarray':
		print("Error: input y should be ndarray.")
		quit()  
	convolution_core = np.ones(window_size)/float(window_size)
	y_runave = np.convolve(y,convolution_core,mode='valid')
	if window_size % 2 == 1:
		gap_size = int((window_size - 1) / 2 )
		x_runave = x[gap_size:len(x)-gap_size]
	else:
		gap_size = int( window_size / 2 )    
		x_runave = x[gap_size:len(x)-gap_size+1]
	return x_runave,y_runave

df_positive = pd.read_csv('data/pcr_positive_daily.csv',header=0,names=['date','positive'])
df_positive['date'] = pd.to_datetime(df_positive['date'])
#print(df)

df_tested = pd.read_csv('data/pcr_tested_daily.csv',header=0,names=['date','tested'])
df_tested['date'] = pd.to_datetime(df_tested['date'])

df = pd.merge(df_positive,df_tested,how='outer',on='date')

fig = plt.figure(figsize=(18,5))
ax1 = fig.add_subplot(111)
ax1.step(df['date'],df['positive'],'o-',where='mid',markersize=0,
	label='Daily positive cases of the COVID-19 PCR tests in Japan')
#plt.xscale('log')
#plt.yscale('log')
ax1.set_xlim('2020-01-01','2021-03-31')
ax1.set_ylim(0.0,8000.0)
ax1.set_xlabel('Date')
ax1.set_ylabel('Number of positive cases')
ax1.legend(loc='upper left',borderaxespad=1,fontsize=10,ncol=2)
fig.patch.set_alpha(0.0)
ax1.patch.set_alpha(0.0) 
fig.savefig("fig/covid19_pcr_positive_daily.pdf")

elapsed_date = df['date'] - df['date'][0]
elapsed_day = np.array(elapsed_date.dt.days)
elapsed_day_ave, positive_ave = get_running_average_data(
	elapsed_day,
	np.array(df['positive']),
	window_size=7)
date_ave = df['date'][0] + pd.to_timedelta(elapsed_day_ave, unit='D')
ax1.axvspan(pd.to_datetime('2020-04-07'), pd.to_datetime('2020-05-25'), 
	color="gray", alpha=0.3)
ax1.axvspan(pd.to_datetime('2021-01-08'), pd.to_datetime('2021-03-07'), 
	color="gray", alpha=0.3)
ax1.plot(date_ave, positive_ave,'-',linewidth=2)
#ax1.step(df['date'], df['tested'],'o-',where='mid',markersize=0,
#	label='Daily tested number')
fig.savefig("fig/covid19_pcr_positive_daily_ave.pdf")

df['positive_ratio'] = 100.0 * df['positive']/df['tested']
elapsed_day_ave, positive_ratio_ave = get_running_average_data(
	elapsed_day,
	np.array(df['positive_ratio']),
	window_size=7)

ax2 = ax1.twinx()  
#color = 'tab:blue'
ax2.set_ylabel('Ratio of positive cases',color='r')  # we already handled the x-label with ax1
ax2.step(date_ave, positive_ratio_ave,'o-',where='mid',markersize=0,
	label='Daily ratio of positive cases',color='r')
ax2.set_ylim(0,100.0)
ax2.tick_params(axis='y',color='r')

plt.tight_layout()
fig.savefig("fig/covid19_pcr_positive_daily_ave_ratio.pdf")
