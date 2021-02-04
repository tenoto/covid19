#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/pcr_positive_daily.csv',header=0,names=['date','positive'])
df['date'] = pd.to_datetime(df['date'])
print(df)

fig = plt.figure(figsize=(18,5))
ax1 = fig.add_subplot(111)
ax1.step(df['date'],df['positive'],'o-',where='mid',markersize=0,
	label='Daily positive cases of the COVID-19 PCR tests in Japan')
#plt.xscale('log')
#plt.yscale('log')
ax1.set_xlim('2020-01-01','2021-03-31')
#ax1.set_ylim(0.0,2.0)
ax1.set_xlabel('Date')
ax1.set_ylabel('Number of positive cases')
ax1.legend(loc='upper left',borderaxespad=1,fontsize=10,ncol=2)
plt.tight_layout()
fig.patch.set_alpha(0.0)
ax1.patch.set_alpha(0.0) 
fig.savefig("fig/covid19_pcr_positive_daily.pdf")

