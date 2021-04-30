#!/usr/bin/env python

import pandas as pd
import numpy as np 
from datetime import date, datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import japanize_matplotlib

### Common parameters

figsize_x = 16
figsize_y = 9
date_start = '2021-02-15'
#date_end = '2021-04-29'
date_end = date.today()
normalization = 10000

total_population = 12557*10000
# 2021-01-01
# https://www.stat.go.jp/data/jinsui/pdf/202101.pdf

plt.style.use('script/matplotlibrc_vaccine.txt')

### Reaf data 

# https://www.kantei.go.jp/jp/headline/kansensho/vaccine.html
df_iryo = pd.read_excel('data/IRYO-vaccination_data.xlsx',engine='openpyxl',
	skiprows=4,skipfooter=9,usecols=[0,1,2,3,4],header=None,
	names=['date','null','#total','#1st','#2nd'],thousands=',',
	dtype={'date':str,'null':str,'#total':int,'#1st':int,'#2nd':int})
df_iryo['date'] = pd.to_datetime(df_iryo['date'])
#print(df_iryo)

df_korei = pd.read_excel('data/KOREI-vaccination_data.xlsx',engine='openpyxl',
	skiprows=4,skipfooter=2,usecols=[0,1,2,3,4],header=None,
	names=['date','null','#total','#1st','#2nd'],thousands=',',
	dtype={'date':str,'null':str,'#total':int,'#1st':int,'#2nd':int})
df_korei['date'] = pd.to_datetime(df_korei['date'])
#print(df_korei)

# https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/vaccine_sesshujisseki.html
df_mhlw = pd.read_excel('data/vaccine_mhlw.xls',skiprows=[38],
	names=['date','#total','#1st','#2nd','#place'],thousands=',',
	dtype={'date':str,'#total':int,'#1st':int,'#2nd':int,'#place':int})
df_mhlw['date'] = pd.to_datetime(df_mhlw['date'])
#print(df_mhlw)

### Sum

date_array = np.arange(np.datetime64(date_start),np.datetime64(date_end))
numtot_array = np.zeros(len(date_array))
num1st_array = np.zeros(len(date_array))
num2nd_array = np.zeros(len(date_array))

for i in range(len(date_array)):
	date_str = date_array[i]

	flag_mhlw = (df_mhlw['date'] == date_str)
	if len(df_mhlw['#1st'][flag_mhlw]) == 1:
		#print(date,int(df_mhlw['#1st'][flag_mhlw]))
		num1st_array[i] = num1st_array[i] + int(df_mhlw['#1st'][flag_mhlw])
		numtot_array[i] = numtot_array[i] + int(df_mhlw['#1st'][flag_mhlw])		
	if len(df_mhlw['#2nd'][flag_mhlw]) == 1:
		print(date_str,int(df_mhlw['#2nd'][flag_mhlw]))
		num2nd_array[i] = num2nd_array[i] + int(df_mhlw['#2nd'][flag_mhlw])
		numtot_array[i] = numtot_array[i] + int(df_mhlw['#2nd'][flag_mhlw])		

	flag_iryo = (df_iryo['date'] == date_str)
	if len(df_iryo['#1st'][flag_iryo]) == 1:
		#print(date,int(df_iryo['#1st'][flag_iryo]))
		num1st_array[i] = num1st_array[i] + int(df_iryo['#1st'][flag_iryo])
		numtot_array[i] = numtot_array[i] + int(df_iryo['#1st'][flag_iryo])		
	if len(df_iryo['#2nd'][flag_iryo]) == 1:
		print(date_str,int(df_iryo['#2nd'][flag_iryo]))
		num2nd_array[i] = num2nd_array[i] + int(df_iryo['#2nd'][flag_iryo])
		numtot_array[i] = numtot_array[i] + int(df_iryo['#2nd'][flag_iryo])		

	flag_korei = (df_korei['date'] == date_str)
	if len(df_korei['#1st'][flag_korei]) == 1:
		#print(date,int(df_korei['#1st'][flag_korei]))
		num1st_array[i] = num1st_array[i] + int(df_korei['#1st'][flag_korei])
		numtot_array[i] = numtot_array[i] + int(df_korei['#1st'][flag_korei])		
	if len(df_korei['#2nd'][flag_korei]) == 1:
		print(date_str,int(df_korei['#2nd'][flag_korei]))
		num2nd_array[i] = num2nd_array[i] + int(df_korei['#2nd'][flag_korei])
		numtot_array[i] = numtot_array[i] + int(df_korei['#2nd'][flag_korei])		


### Plot data

################
# Daily 1st 
################

fig = plt.figure(figsize=(figsize_x,figsize_y),tight_layout=True)
ax1 = fig.add_subplot(111)
"""
ax1.step(df_mhlw['date'],df_mhlw['#total'],'o-',where='mid',markersize=3,
	label='Number of daily covid-19 vaccine shots')
ax1.step(df_iryo['date'],df_iryo['#total'],'o-',where='mid',markersize=3,
	label='Number of daily covid-19 vaccine shots')
ax1.step(df_korei['date'],df_korei['#total'],'o-',where='mid',markersize=3,
	label='Number of daily covid-19 vaccine shots')
"""
ax1.fill_between(pd.to_datetime(date_array),num1st_array/normalization,
	step="mid",color='#F08080',alpha=0.5)
ax1.step(pd.to_datetime(date_array),num1st_array/normalization,
	'-',where='mid',markersize=0,label='合計(医療従事者+高齢者)',
	color='r')
#ax1.plot(pd.to_datetime(date_array),num1st_array/normalization,color='r')
#ax1.plot(df_mhlw['date'],df_mhlw['#1st']/normalization,'o-',markersize=8)
#ax1.plot(df_iryo['date'],df_iryo['#1st']/normalization,'o-',markersize=8,
#	label='医療従事者')
ax1.plot(df_korei['date'],df_korei['#1st']/normalization,'o-',
	markersize=8,label='高齢者')
#plt.yscale('log')
ax1.set_xlim(date_start,date_end)
ax1.set_xlabel('日付')
ax1.set_ylabel('接種人数 (万人)',labelpad=20)
myFmt = mdates.DateFormatter('%m/%d')
ax1.xaxis.set_major_formatter(myFmt)
ax1.legend(loc='upper left',borderaxespad=1,fontsize=20,ncol=2,
	title='１回目ワクチン接種者数\n (1日毎, %s時点)' % date.today())
fig.patch.set_alpha(0.0)
ax1.patch.set_alpha(0.0) 
plt.gca().set_ylim(bottom=0)
fig.savefig("fig/covid19_vaccine_daily_1st.pdf")
fig.savefig("fig/covid19_vaccine_daily_1st.jpg",dpi=300)


################
# Daily 2nd 
################

fig = plt.figure(figsize=(figsize_x,figsize_y),tight_layout=True)
ax1 = fig.add_subplot(111)
ax1.fill_between(pd.to_datetime(date_array),num2nd_array/normalization,
	step="mid",color='#F08080',alpha=0.5)
ax1.step(pd.to_datetime(date_array),num2nd_array/normalization,
	'-',where='mid',markersize=0,label='合計(医療従事者+高齢者)',
	color='r')
ax1.plot(df_korei['date'],df_korei['#2nd']/normalization,'o-',
	markersize=8,label='高齢者')
#plt.yscale('log')
ax1.set_xlim(date_start,date_end)
ax1.set_xlabel('日付 (2021年)')
ax1.set_ylabel('接種人数 (万人)',labelpad=20)
myFmt = mdates.DateFormatter('%m/%d')
ax1.xaxis.set_major_formatter(myFmt)
ax1.legend(loc='upper left',borderaxespad=1,fontsize=20,ncol=2,
	title='２回目ワクチン接種者数\n (1日毎, %s時点)' % date.today())
fig.patch.set_alpha(0.0)
ax1.patch.set_alpha(0.0) 
plt.gca().set_ylim(bottom=0)
fig.savefig("fig/covid19_vaccine_daily_2nd.pdf")
fig.savefig("fig/covid19_vaccine_daily_2nd.jpg",dpi=300)


################
# Accumulation 1st
################

def number2ratio(x):
    return x * normalization / total_population * 100.0

def ratio2number(x):
    return x / normalization / 100 * total_population

fig = plt.figure(figsize=(figsize_x,figsize_y),tight_layout=True)
ax1 = fig.add_subplot(111)
ax1.fill_between(pd.to_datetime(date_array),np.add.accumulate(num1st_array)/normalization,
	step="mid",color='#F08080',alpha=0.5)
ax1.step(pd.to_datetime(date_array),np.add.accumulate(num1st_array)/normalization,
	'-',where='mid',markersize=0,label='合計(医療従事者+高齢者)',
	color='r')
#ax1.plot(df_korei['date'],np.add.accumulate(df_korei['#1st'])/normalization,'o-',
#	markersize=8,label='高齢者')
#plt.yscale('log')
ax1.set_xlim(date_start,date_end)
ax1.set_xlabel('日付 (2021年)')
ax1.set_ylabel('積算の接種人数 (万人)',labelpad=20)
myFmt = mdates.DateFormatter('%m/%d')
ax1.xaxis.set_major_formatter(myFmt)
ax1.legend(loc='upper left',borderaxespad=1,fontsize=20,ncol=2,
	title='１回目ワクチン接種者数\n (1日毎の積算値, %s時点)' % date.today())
fig.patch.set_alpha(0.0)
ax1.patch.set_alpha(0.0) 
plt.gca().set_ylim(bottom=0)
second_ay = ax1.secondary_yaxis('right',functions=(number2ratio, ratio2number))
second_ay.set_ylabel('人口に占める割合 (%)')
fig.savefig("fig/covid19_vaccine_accum_1st.pdf")
fig.savefig("fig/covid19_vaccine_accum_1st.jpg",dpi=300)


################
# Accumulation 1st, Olympic
################

def number2ratio_oku(x):
    return x * normalization**2 / total_population * 100.0

def ratio2number_oku(x):
    return x / normalization**2 / 100 * total_population


fig = plt.figure(figsize=(figsize_x,figsize_y),tight_layout=True)
ax1 = fig.add_subplot(111)
ax1.axvspan(pd.to_datetime('2021-07-23'),pd.to_datetime('2021-08-08'),
	color="#3498DB",alpha=0.2)
plt.text(pd.to_datetime('2021-08-01'), 
	0.5*total_population/normalization**2,
	'東京オリンピック',
	color='white',ha='center', va='center',rotation='vertical')
ax1.fill_between(pd.to_datetime(date_array),np.add.accumulate(num1st_array)/normalization**2,
	step="mid",color='#F08080',alpha=0.5)
ax1.step(pd.to_datetime(date_array),np.add.accumulate(num1st_array)/normalization**2,
	'-',where='mid',markersize=0,label='合計(医療従事者+高齢者)',
	color='r')
#ax1.plot(df_korei['date'],np.add.accumulate(df_korei['#1st'])/normalization,'o-',
#	markersize=8,label='高齢者')
#plt.yscale('log')
ax1.set_xlim(date_start,'2021-10-31')
ax1.set_ylim(0,total_population/normalization**2)
ax1.set_xlabel('日付 (2021年)')
ax1.set_ylabel('積算の接種人数 (億人)',labelpad=20)
myFmt = mdates.DateFormatter('%m/%d')
ax1.xaxis.set_major_formatter(myFmt)
ax1.legend(loc='upper left',borderaxespad=1,fontsize=20,ncol=2,
	title='１回目ワクチン接種者数\n (1日毎の積算値, %s時点)' % date.today())
fig.patch.set_alpha(0.0)
ax1.patch.set_alpha(0.0) 
#plt.gca().set_ylim(bottom=0)
plt.axvline(pd.to_datetime('2021-10-21'), color='k', linestyle='--')
plt.text(pd.to_datetime('2021-10-21'), 
	0.5*total_population/normalization**2,
	'衆議院議員選挙 (任期満了)', backgroundcolor='white',
	ha='center', va='center',rotation='vertical')
plt.axvline(pd.to_datetime('2021-07-22'), color='k', linestyle='--')
plt.text(pd.to_datetime('2021-07-22'), 
	0.5*total_population/normalization**2,
	'東京都議会議員選挙 (任期満了)', backgroundcolor='white',
	ha='center', va='center',rotation='vertical')
second_ay = ax1.secondary_yaxis('right',functions=(number2ratio_oku, ratio2number_oku))
second_ay.set_ylabel('人口に占める割合 (%)')
fig.savefig("fig/covid19_vaccine_accum_1st_prediction.pdf")
fig.savefig("fig/covid19_vaccine_accum_1st_prediction.jpg",dpi=300)
