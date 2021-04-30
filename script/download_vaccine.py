#!/bin/sh -f

wget https://www.kantei.go.jp/jp/content/IRYO-vaccination_data.xlsx
mv IRYO-vaccination_data.xlsx data/

wget https://www.kantei.go.jp/jp/content/KOREI-vaccination_data.xlsx
mv KOREI-vaccination_data.xlsx data/