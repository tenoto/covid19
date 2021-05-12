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
date_of_end_of_emergency = '2021-05-31'
normalization = 10000
normalization_en = 1000000
fit_width_days = 14
prediction_month = 8 

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

df_positive = pd.read_csv('data/pcr_positive_daily.csv',header=0,names=['date','positive'])
df_positive['date'] = pd.to_datetime(df_positive['date'])

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

latest = df_iryo['date'][0]
latest_date = '%s-%s-%s' % (latest.year, latest.month, latest.day)

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
ax1.set_xlabel('日付 (2021年)')
ax1.set_ylabel('接種人数 (万人)',labelpad=20)
myFmt = mdates.DateFormatter('%m/%d')
ax1.xaxis.set_major_formatter(myFmt)
ax1.legend(loc='upper left',borderaxespad=1,fontsize=20,ncol=2,
	title='１回目ワクチン接種者数\n (1日毎, %sまで)' % latest_date)
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
	step="mid",color='#85C1E9',alpha=0.5)
ax1.step(pd.to_datetime(date_array),num2nd_array/normalization,
	'-',where='mid',markersize=0,label='合計(医療従事者+高齢者)',
	color='b')
ax1.plot(df_korei['date'],df_korei['#2nd']/normalization,'o-',
	markersize=8,label='高齢者')
#plt.yscale('log')
ax1.set_xlim(date_start,date_end)
ax1.set_xlabel('日付 (2021年)')
ax1.set_ylabel('接種人数 (万人)',labelpad=20)
myFmt = mdates.DateFormatter('%m/%d')
ax1.xaxis.set_major_formatter(myFmt)
ax1.legend(loc='upper left',borderaxespad=1,fontsize=20,ncol=2,
	title='２回目ワクチン接種者数\n (1日毎, %sまで)' % latest_date)
fig.patch.set_alpha(0.0)
ax1.patch.set_alpha(0.0) 
plt.gca().set_ylim(bottom=0)
fig.savefig("fig/covid19_vaccine_daily_2nd.pdf")
fig.savefig("fig/covid19_vaccine_daily_2nd.jpg",dpi=300)


#fig = plt.figure(figsize=(figsize_x,2*figsize_y),tight_layout=True)
fig, axs = plt.subplots(2,1,sharex=True,figsize=(9,10),tight_layout=True)
fig.subplots_adjust(hspace=0.01)

axs[0].fill_between(pd.to_datetime(date_array),num1st_array/normalization,
	step="mid",color='#F08080',alpha=0.5)
axs[0].step(pd.to_datetime(date_array),num1st_array/normalization,
	'-',where='mid',markersize=0,label='１回目ワクチン接種者数',
	color='r')
axs[0].axhline(100.0,color='k', linestyle='--')
axs[0].set_xlim(date_start,date_end)
axs[0].set_ylim(0.0,130)
axs[0].set_ylabel('1日ごとの接種人数 (万人)',labelpad=20)
axs[0].legend(loc='upper left',borderaxespad=1,fontsize=20,ncol=2,
	title='首相官邸サイトの表から作成\n(1日毎, %sまで)' % latest_date)
fig.patch.set_alpha(0.0)
axs[0].patch.set_alpha(0.0) 

axs[1].fill_between(pd.to_datetime(date_array),num2nd_array/normalization,
	step="mid",color='#85C1E9',alpha=0.5)
axs[1].step(pd.to_datetime(date_array),num2nd_array/normalization,
	'-',where='mid',markersize=0,label='２回目ワクチン接種者数',
	color='b')
axs[1].axhline(100.0,color='k', linestyle='--')
axs[1].set_xlim(date_start,date_end)
axs[1].set_ylim(0.0,130)
axs[1].set_xlabel('日付 (2021年)')
axs[1].set_ylabel('1日ごとの接種人数 (万人)',labelpad=20)
myFmt = mdates.DateFormatter('%m/%d')
axs[1].xaxis.set_major_formatter(myFmt)
axs[1].set_ylim(0.0,130)
axs[1].legend(loc='upper left',borderaxespad=1,fontsize=20,ncol=2)	
fig.patch.set_alpha(0.0)
axs[1].patch.set_alpha(0.0) 
plt.gca().set_ylim(bottom=0)
fig.savefig("fig/covid19_vaccine_daily.pdf")
fig.savefig("fig/covid19_vaccine_daily.jpg",dpi=300)

################
# fit accumulation pace
################
elapsed_days = np.array((pd.to_datetime(date_array)-pd.to_datetime(date_array)[0]).total_seconds()/24/60/60)
accum_shots = np.add.accumulate(num1st_array)/normalization

pol1_fit_param = np.polyfit(
	elapsed_days[-fit_width_days:],
	accum_shots[-fit_width_days:],
	1)
func_pol1 = np.poly1d(pol1_fit_param)
print(pd.to_datetime(date_array)[0]+pd.DateOffset(1))
prediction_date = np.array([pd.to_datetime(date_array)[0]+pd.DateOffset(i) for i in range(30*prediction_month)])
prediction_shots = np.array([func_pol1(i) for i in range(30*prediction_month)])

################
# Accumulation 1st
################

def number2ratio(x):
    return x * normalization / total_population * 100.0

def ratio2number(x):
    return x / normalization / 100 * total_population

zorder = 0 

fig = plt.figure(figsize=(figsize_x,figsize_y),tight_layout=True)
ax1 = fig.add_subplot(111)
ax1.fill_between(pd.to_datetime(date_array),np.add.accumulate(num1st_array)/normalization,
	step="mid",color='#F08080',alpha=0.5,label='１回目の接種人数',zorder=zorder);zorder+=1
ax1.step(pd.to_datetime(date_array),np.add.accumulate(num1st_array)/normalization,
	'-',where='mid',markersize=0,color='r',zorder=zorder);zorder+=1
#ax1.plot(df_korei['date'],np.add.accumulate(df_korei['#1st'])/normalization,'o-',
#	markersize=8,label='高齢者')
#plt.yscale('log')
ax1.fill_between(pd.to_datetime(date_array),np.add.accumulate(num2nd_array)/normalization,
	step="mid",color='#85C1E9',alpha=0.5,label='２回目の接種人数',zorder=zorder);zorder+=1
ax1.step(pd.to_datetime(date_array),np.add.accumulate(num2nd_array)/normalization,
	'-',where='mid',markersize=0,color='b',zorder=zorder);zorder+=1
ax1.plot(prediction_date,
	prediction_shots,linestyle="--",color='k',zorder=zorder,
	label='直近 %d 日間の１次フィット (+%.1f 万人/日)' % (fit_width_days,pol1_fit_param[0]));
zorder+=1

# ====
plt.axvline(pd.to_datetime('2021-04-29'), 
	color='k', linestyle='--',zorder=zorder);zorder+=1
plt.axvline(pd.to_datetime('2021-05-09'), 
	color='k', linestyle='--',zorder=zorder);zorder+=1
plt.text(pd.to_datetime('2021-04-29')+pd.DateOffset(3),50,
	'ゴールデンウィーク', #backgroundcolor='white',
	ha='center', va='center',rotation='vertical',fontsize=15,
	zorder=zorder);zorder+=1

ax1.set_xlim(date_start,date_end)
ax1.set_ylim(0,1.2*max(np.add.accumulate(num1st_array)/normalization))
ax1.set_xlabel('日付 (2021年)')
ax1.set_ylabel('積算のワクチン接種人数 (万人)',labelpad=20)
myFmt = mdates.DateFormatter('%m/%d')
ax1.xaxis.set_major_formatter(myFmt)
ax1.legend(loc='upper left',borderaxespad=1,fontsize=20,ncol=1,
	title='首相官邸・厚生労働省のサイトから作成\n (1日毎の積算値, %sまで)' % latest_date)
fig.patch.set_alpha(0.0)
ax1.patch.set_alpha(0.0) 
#plt.gca().set_ylim(bottom=0)
second_ay = ax1.secondary_yaxis('right',functions=(number2ratio, ratio2number))
second_ay.set_ylabel('人口に占める割合 (%)')
fig.savefig("fig/covid19_vaccine_accum.pdf")
fig.savefig("fig/covid19_vaccine_accum.jpg",dpi=300)

# ======== English 


def number2ratio_en(x):
    return x * normalization_en / total_population * 100.0

def ratio2number_en(x):
    return x / normalization_en / 100 * total_population


zorder = 0 
fig = plt.figure(figsize=(figsize_x,figsize_y),tight_layout=True)
ax1 = fig.add_subplot(111)
ax1.fill_between(pd.to_datetime(date_array),np.add.accumulate(num1st_array)/normalization_en,
	step="mid",color='#F08080',alpha=0.5,label='Number of people vaccinated (1st)',zorder=zorder);zorder+=1
ax1.step(pd.to_datetime(date_array),np.add.accumulate(num1st_array)/normalization_en,
	'-',where='mid',markersize=0,color='r',zorder=zorder);zorder+=1
#ax1.plot(df_korei['date'],np.add.accumulate(df_korei['#1st'])/normalization_en,'o-',
#	markersize=8,label='高齢者')
#plt.yscale('log')
ax1.fill_between(pd.to_datetime(date_array),np.add.accumulate(num2nd_array)/normalization_en,
	step="mid",color='#85C1E9',alpha=0.5,label='Number of people vaccinated (2nd)',zorder=zorder);zorder+=1
ax1.step(pd.to_datetime(date_array),np.add.accumulate(num2nd_array)/normalization_en,
	'-',where='mid',markersize=0,color='b',zorder=zorder);zorder+=1
ax1.plot(prediction_date,
	prediction_shots/100,linestyle="--",color='k',zorder=zorder,
	label='Linear extrapolation from the last %d days (+%.3f million/day)' % (fit_width_days,pol1_fit_param[0]/100.0));
zorder+=1
# ====
plt.axvline(pd.to_datetime('2021-04-29'), 
	color='k', linestyle='--',zorder=zorder);zorder+=1
plt.axvline(pd.to_datetime('2021-05-09'), 
	color='k', linestyle='--',zorder=zorder);zorder+=1
plt.text(pd.to_datetime('2021-04-29')+pd.DateOffset(3),0.5,
	'Holidays', #backgroundcolor='white',
	ha='center', va='center',rotation='vertical',fontsize=15,
	zorder=zorder);zorder+=1

ax1.set_xlim(date_start,date_end)
ax1.set_ylim(0,1.2*max(np.add.accumulate(num1st_array)/normalization_en))
ax1.set_xlabel('Date (Year 2021)')
ax1.set_ylabel('Accumulated number vaccinated in Japan (million people)',labelpad=20)
myFmt = mdates.DateFormatter('%m/%d')
ax1.xaxis.set_major_formatter(myFmt)
ax1.legend(loc='upper left',borderaxespad=1,fontsize=20,ncol=1,
	title='Compiled from the Japanese government websites \n (Daily, as of %s)' % latest_date)
fig.patch.set_alpha(0.0)
ax1.patch.set_alpha(0.0) 
#plt.gca().set_ylim(bottom=0)
second_ay = ax1.secondary_yaxis('right',functions=(number2ratio_en, ratio2number_en))
second_ay.set_ylabel('Percentage of population (%)')
fig.savefig("fig/covid19_vaccine_accum_en.pdf")
fig.savefig("fig/covid19_vaccine_accum_en.jpg",dpi=300)



"""
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
"""


################
# Accumulation 1st, Olympic
################

fig = plt.figure(figsize=(figsize_x,figsize_y),tight_layout=True)
ax1 = fig.add_subplot(111)

zorder = 0
for i in range(1,12):
	plt.axvline(pd.to_datetime('2021-%02d-01' % i), 
		color='#E5E7E9',linestyle='-.',zorder=zorder)
	zorder += 1

ax1.axvspan(pd.to_datetime('2021-04-25'),pd.to_datetime(date_of_end_of_emergency),
	color="#FCF3CF",zorder=zorder,label='緊急事態宣言 (%s まで)' % date_of_end_of_emergency);zorder+=1
ax1.axvspan(pd.to_datetime('2021-01-08'),pd.to_datetime('2021-03-21'),
	color="#FCF3CF",zorder=zorder);zorder+=1
#plt.text(pd.to_datetime('2021-05-01'),50,
#	'緊急事態宣言',color='k',ha='center', 
#	va='center',rotation='vertical',zorder=zorder);zorder+=1
# ====
ax1.axvspan(pd.to_datetime('2021-07-23'),pd.to_datetime('2021-08-08'),
	color="#AED6F1",zorder=zorder);zorder+=1
plt.text(pd.to_datetime('2021-08-01'),60,
	'東京オリンピック',color='k',ha='center', 
	va='center',rotation='vertical',zorder=zorder);zorder+=1
# ====
ax1.axvspan(pd.to_datetime('2021-08-24'),pd.to_datetime('2021-09-05'),
	color="#AED6F1",zorder=zorder);zorder+=1
plt.text(pd.to_datetime('2021-08-30'),60,
	'東京パラリンピック',color='k',ha='center', 
	va='center',rotation='vertical',zorder=zorder);zorder+=1
# ====
plt.axvline(pd.to_datetime('2021-10-21'), 
	color='k', linestyle='--',zorder=zorder);zorder+=1
plt.text(pd.to_datetime('2021-10-21')-pd.DateOffset(4),50,
	'衆議院選挙 (任期満了)', backgroundcolor='white',fontsize=15,
	ha='center', va='center',
	rotation='vertical',zorder=zorder);zorder+=1
# ====
plt.axvline(pd.to_datetime('2021-07-04'), 
	color='k', linestyle='--',zorder=zorder);zorder+=1
plt.text(pd.to_datetime('2021-07-04')+pd.DateOffset(4),50,
	'都議会選挙 (選挙期日)', backgroundcolor='white',
	ha='center', va='center',rotation='vertical',fontsize=15,
	zorder=zorder);zorder+=1
# ====
plt.axvline(pd.to_datetime('2021-06-25'), 
	color='k', linestyle='--',zorder=zorder);zorder+=1
plt.text(pd.to_datetime('2021-06-25')-pd.DateOffset(4),50,
	'都議会選挙 (告示日)', backgroundcolor='white',
	ha='center', va='center',rotation='vertical',fontsize=15,
	zorder=zorder);zorder+=1
# ====
plt.axvline(pd.to_datetime('2021-04-29'), 
	color='k', linestyle='--',zorder=zorder);zorder+=1
plt.axvline(pd.to_datetime('2021-05-09'), 
	color='k', linestyle='--',zorder=zorder);zorder+=1
plt.text(pd.to_datetime('2021-04-29')+pd.DateOffset(5),20,
	'ゴールデンウィーク', #backgroundcolor='white',
	ha='center', va='center',rotation='vertical',fontsize=15,
	zorder=zorder);zorder+=1


ax1.fill_between(pd.to_datetime(date_array),np.add.accumulate(num1st_array)/total_population*100,
	step="mid",color='#F08080',alpha=0.5,label='１回目の接種人数',
	zorder=zorder);zorder+=1
ax1.step(pd.to_datetime(date_array),np.add.accumulate(num1st_array)/total_population*100,
	'-',where='mid',markersize=0,
	color='r',linewidth=2,zorder=zorder);zorder+=1
ax1.fill_between(pd.to_datetime(date_array),np.add.accumulate(num2nd_array)/total_population*100,
	step="mid",color='#85C1E9',alpha=0.5,label='２回目の接種人数',
	zorder=zorder);zorder+=1
ax1.step(pd.to_datetime(date_array),np.add.accumulate(num2nd_array)/total_population*100,
	'-',where='mid',markersize=0,
	color='b',linewidth=2,zorder=zorder);zorder+=1

expected_days = (total_population-np.add.accumulate(num1st_array)[-1])/(pol1_fit_param[0]*normalization)

ax1.plot(prediction_date,
	prediction_shots*normalization/total_population*100,
	linestyle="--",color='r',zorder=zorder,
	label='直近 %d 日間からの予測 (+%.1f 万人/日)\nこのペースでは全国民接種まで %d 日必要' % (fit_width_days,pol1_fit_param[0],expected_days))
zorder+=1
ax1.plot(
	[pd.to_datetime('2021-07-28'),pd.to_datetime('2021-08-03')],
	[3600*1e+4/total_population*100,3600*1e+4/total_population*100],
	'-',color='k',linewidth=3,zorder=zorder)
ax1.arrow(
	pd.to_datetime('2021-07-31'),
	3600*1e+4/total_population*100,
	0.0,
	400*1e+4/total_population*100,
	fc='#000000',ec='#000000',
	head_width=1.5,head_length=1.0,linewidth=3,
	zorder=zorder)
plt.text(pd.to_datetime('2021-07-28'),
	3200*1e+4/total_population*100,
	'7月末 3600万人(高齢者)\n接種完了目標',
	color='k',ha='center', fontsize=12,
	va='center',zorder=zorder);zorder+=1
ax1.plot(
	[pd.to_datetime('2021-05-10'),pd.to_datetime('2021-06-30')],
	[2800*1e+4/total_population*100,2800*1e+4/total_population*100],
	'-',color='k',linewidth=3,zorder=zorder)
plt.text(pd.to_datetime('2021-05-30'),
	3000*1e+4/total_population*100,
	'日本到着済?(2800万回分)',
	color='k',ha='center', fontsize=12,
	va='center',zorder=zorder);zorder+=1
zorder+=1

for key in [400,4000,8000]:
	ax1.plot(
		[pd.to_datetime(date_start),
		pd.to_datetime(date_start)+pd.DateOffset(10)],
		[key*1e+4/total_population*100,key*1e+4/total_population*100],
		'-',color='k',linewidth=3,zorder=zorder)
	plt.text(pd.to_datetime(date_start)+pd.DateOffset(19),
		key*1e+4/total_population*100,
		'%d万人' % key,
		color='k',ha='center', fontsize=12,
		va='center',zorder=zorder);zorder+=1

ax1.yaxis.label.set_color('red')
ax1.set_xlim(date_start,'2021-10-31')
ax1.set_ylim(0,100)
ax1.set_xlabel('日付 (2021年)')
ax1.set_ylabel('日本の総人口に対する積算のワクチン接種人数の割合 (%)',color='k',labelpad=20)
myFmt = mdates.DateFormatter('%m/%d')
ax1.xaxis.set_major_formatter(myFmt)
fig.patch.set_alpha(0.0)
ax1.patch.set_alpha(0.0) 

ax2 = ax1.twinx()  
ax2.set_ylabel('１日毎の全国の新型コロナウイルス陽性者数 (人)',color='#7D3C98',labelpad=20)  # we already handled the x-label with ax1
ax2.tick_params(axis='y',color='#7D3C98')
ax2.step(df_positive['date'], df_positive['positive'],'o-',where='mid',
	markersize=0,label='新型コロナウイルス陽性者(全国)',color='#7D3C98',
	linewidth=2)
ax2.set_ylim(0,1.3*max(df_positive['positive']))
#ax2.set_zorder(-1)


ax1.legend(loc='upper left',borderaxespad=1,fontsize=18,ncol=1,
	title='首相官邸サイトから作成 (%sまで)' % latest_date).set_zorder(zorder)

fig.savefig("fig/covid19_positive_vaccine_accum_long.pdf")
fig.savefig("fig/covid19_positive_vaccine_accum_long.jpg",dpi=300)

