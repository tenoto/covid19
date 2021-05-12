#!/bin/sh -f

# https://www.kantei.go.jp/jp/headline/kansensho/vaccine.html

wget https://www.kantei.go.jp/jp/content/IRYO-vaccination_data.xlsx
mv IRYO-vaccination_data.xlsx data/

wget https://www.kantei.go.jp/jp/content/KOREI-vaccination_data.xlsx
mv KOREI-vaccination_data.xlsx data/