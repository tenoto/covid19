#!/bin/sh -f

# download from https://www.mhlw.go.jp/stf/covid-19/open-data.html

for CSVFILE in pcr_positive_daily.csv pcr_tested_daily.csv cases_total.csv recovery_total.csv death_total.csv pcr_case_daily.csv
do 
	rm -f $CSVFILE
	wget https://www.mhlw.go.jp/content/$CSVFILE
	rm -f data/$CSVFILE	
	mv $CSVFILE data/
done 
