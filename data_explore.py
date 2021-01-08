#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:28:54 2020

@author: malika
"""

# This script is for long-term data analysis of Krakow data
# Analysis of wind, and entire dataset features, including model results

import numpy as np
import pandas as pd
from pandas import ExcelWriter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
import cmocean
import seaborn as sns
import os, shutil
import scipy as sc
from scipy import stats

import functions_KR as fun
import figures_KR as fg

plt.rcParams.update({'font.size': 15})

# data format:        filling time   	[CH4]dD-2 (ppb)	   1/MR	       dD-2 SMOW

dateformat='%d/%m/%Y %H:%M'
df_P='%d/%m/%y %H:%M:%S'

encoding='utf-8-sig'
d13C='\u03B4\u00B9\u00B3C'
dD='\u03B4D'
permil=' [\u2030]'
pm='\u00B1'
delta13C=d13C+' V-PDB'
deltaD=dD+' V-SMOW'

col_dt='filling time UTC'
col_dt_d13C='DateTime_d13C'
col_dt_dD='DateTime_dD'

col_MR_d13C='MR [ppb] -d13C'
col_errMR_d13C='SD MR -d13C'

col_d13C='d13C VPDB'
col_errd13C='SD d13C'

col_MR_dD='MR [ppb] -dD'
col_errMR_dD='SD MR -dD'

col_dD='dD SMOW'
col_errdD='SD dD'

col_MR_P='CH4'
col_P='CH4 [ppb]'
col_d13C_P='del13CH4_CORR'

col_MR='IRMS_'+col_P
col_errMR='IRMS_err'+col_P

col_wd='averageWindDirection'
col_wd_d13C='averageWindDirection_d13C'
col_wd_dD='averageWindDirection_dD'

col_ws='averageWindSpeed'
col_ws_d13C='averageWindSpeed_d13C'
col_ws_dD='averageWindSpeed_dD'
col_errwd='stdWindDirection'
col_errws='stdWindSpeed'

# For pretty strings

strMR_13C='MR_ppb - '+d13C
strMR_D='MR_ppb - '+dD
str_d13C=delta13C+permil
str_dD=deltaD+permil
str_MR='χ(CH\u2084) [ppb]'
str_EDGAR='EDGAR_v5.0'
str_TNO='CAMS-REG_v4.2'

vertclair='#009966'
web_blue='#333399'
web_green='#006633'
web_yellow='#FFFF00'
web_red='#FF3333'
web_magenta='#FF0099'
web_cyan='#0099FF'
web_orange='#FF6633'

plt.rcParams.update({'font.size': 12, 'font.sans-serif':'Helvetica'})

###############################################################################
# SAVE THE RESULT OF THE DAY IN A SEPERATE FOLDER
###############################################################################

f_path = '/Users/malika/surfdrive/Data/Krakow/Roof air/figures/'
t_path = '/Users/malika/surfdrive/Data/Krakow/Roof air/tables/'
today='2020-12-08 data_explore/'

#%%
###############################################################################
# DATA TABLES
###############################################################################

# IRMS data
df_IRMS=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Input files/KRK_merged_re-calibrated_corrected-IRMS.csv', sep=',', index_col=col_dt, parse_dates=True, dayfirst=True, keep_date_col=True)

# Modeled data
df_edgar=pd.read_table(t_path+'model/Krakow_CH4_EDGARv50_processed.csv', sep=',', index_col=0, parse_dates=True, dayfirst=True, keep_date_col=True)
df_tno=pd.read_table(t_path+'model/Krakow_CH4_TNO_CAMS_REGv221_processed.csv', sep=',', index_col=0, parse_dates=True, dayfirst=True, keep_date_col=True)

# Sample data
df_samples=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Krakow_city/Krakow_all-samples_signatures.csv', sep=',')

# Meteorological parameters
df_meteo=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Meteo data/meteo_data_all.csv', sep=',', index_col=0, parse_dates=True, dayfirst=True)

# Mace Head data
df_mh=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Mace Head/MHD-CH4-selected.csv', sep=',', index_col=0, parse_dates=True)
df_mh.where(df_mh['flag']==' ', inplace=True)

#%%
# Overview graph
f_IRMS = fg.overview(df_IRMS, data_MH=df_mh)
f_IRMS.savefig(f_path+today+'IRMS_overview_mace-head.png', dpi=600, transparent=True, bbox_inches = 'tight', pad_inches = 0)

f_chimere = fg.overview(df_IRMS, data_CHIM=[df_edgar, df_tno])
f_chimere.savefig(f_path+today+'overview_chimere.png', dpi=600, transparent=True, bbox_inches = 'tight', pad_inches = 0)

# Cut the 2 parts of the data
t0='2018-09-14 00:00:00'
t1='2018-11-15 00:00:00'
t2='2019-03-15 00:00:00'

#%%
# meteo analysis

meteo=list(df_meteo)
col_temp='averageAirTemp'
col_pm10='averagePm10'

for met in meteo:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5), sharex=True)
    df_IRMS.plot(y=col_MR, style='.', color='k', lw=0.7, label='x(CH4)', ax=ax)
    df_meteo.plot(y=met, style='.', ax=ax, secondary_y=True)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5), sharex=True)
df_IRMS.plot(y=col_MR, style='.', color='k', lw=0.7, label='x(CH4)', ax=ax)
df_meteo.plot(y=col_pm10, ax=ax, secondary_y=True)
fg.correlation([df_IRMS[t0:t1][col_MR]], [np.interp(df_IRMS[t0:t1].index, df_meteo[t0:t1].index, df_meteo[t0:t1][col_pm10])], 'x(CH4)', 'PM10', xy=True)

# d13C vs dD correlation
f=fg.correlation([df_IRMS['i_'+col_d13C]], [df_IRMS['i_'+col_dD]], 'd13C', 'dD', xy=True)
f[0].savefig(f_path+today+'corr_IRMS-d13C-dD_original.png', dpi=300, transparent=True)

#CH4 vs d13C and dD correlation
f=fg.correlation([df_IRMS[col_MR_d13C]], [df_IRMS[col_d13C]], str_MR, str_d13C, xy=True)
f[0].savefig(f_path+today+'corr_IRMS-x(CH4)-d13C_original.png', dpi=300, transparent=True)

f=fg.correlation([df_IRMS[col_MR_dD]], [df_IRMS[col_dD]], str_MR, str_dD, xy=True)
f[0].savefig(f_path+today+'corr_IRMS-x(CH4)-dD_original.png', dpi=300, transparent=True)

#%%
# day/night analysis
sunrise='06:00:00'
sunset='18:00:00'
t0='2018-09-14 00:00:00'
t1='2018-11-15 00:00:00'

data_day=df_IRMS.between_time(sunrise, sunset, False, False)
data_night=df_IRMS.between_time(sunset, sunrise)
hist_DayNight= plt.figure(figsize=(15,10))
ax_DayNight=hist_DayNight.add_subplot(111)
sns.kdeplot(data_day[col_MR], shade=True, ax=ax_DayNight, label='day', color=web_yellow)
sns.kdeplot(data_night[col_MR], shade=True, ax=ax_DayNight, label='night', color=web_blue)
bg=np.percentile(df_IRMS[col_MR], 10)
bg_day=len(data_day.loc[data_day<bg])/len(df_IRMS[col_MR].loc[df_IRMS[col_MR]<bg])
print(str(bg_day*100)+'% of the background values occurred during daytime.')

# day/night and wind speed
data_day_2018=data_day[t0:t1]
data_night_2018=data_night[t0:t1]
print('Average wind speed in the day, before Nov. 15th: '+str(data_day_2018[col_ws].mean())+' ± '+str(data_day_2018[col_ws].std())+' m/s')
print('Average wind speed in the night, before Nov. 15th: '+str(data_night_2018[col_ws].mean())+' ± '+str(data_night_2018[col_ws].std())+' m/s')

#%%
# Get the PKE data
df_pkdD = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE-dD_original_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=1, parse_dates=[ 1, 2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID',	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_errwd+'_dD', col_ws+'_dD', col_errws+'_dD'])
df_pkd13C = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE-d13C_original_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=1, parse_dates=[ 1, 2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID',	'start_'+col_dt_d13C,	'end_'+col_dt_d13C,	col_d13C,	 col_errd13C, 'slope_d13C',	'r2_d13C', 'res_d13C', 'n_d13C', col_wd+'_d13C', col_errwd+'_d13C', col_ws+'_d13C', col_errws+'_d13C'])
df_PKE = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE_original_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 13, 14, 15], dayfirst=True, keep_date_col=True, header=0,
                     names=[col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_errwd+'_dD', col_ws+'_dD', col_errws+'_dD', col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C,	col_d13C,	 col_errd13C, 'slope_d13C',	'r2_d13C', 'res_d13C', 'n_d13C', col_wd+'_d13C', col_errwd+'_d13C', col_ws+'_d13C', col_errws+'_d13C'])

df_pkdD_v2 = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE-dD_offset_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=1, parse_dates=[ 1, 2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID',	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_errwd+'_dD', col_ws+'_dD', col_errws+'_dD'])
df_pkd13C_v2 = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE-d13C_offset_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=1, parse_dates=[ 1, 2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID',	'start_'+col_dt_d13C,	'end_'+col_dt_d13C,	col_d13C,	 col_errd13C, 'slope_d13C',	'r2_d13C', 'res_d13C', 'n_d13C', col_wd+'_d13C', col_errwd+'_d13C', col_ws+'_d13C', col_errws+'_d13C'])
df_PKE_v2 = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE_offset_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 13, 14, 15], dayfirst=True, keep_date_col=True, header=0,
                     names=[col_dt_dD+'_v2',	'start_'+col_dt_dD+'_v2',	'end_'+col_dt_dD+'_v2', col_dD+'_v2',	col_errdD+'_v2',	'slope_dD'+'_v2',	'r2_dD'+'_v2', 'res_dD',	'n_dD'+'_v2', col_wd+'_dD'+'_v2', col_errwd+'_dD'+'_v2', col_ws+'_dD'+'_v2',	col_errws+'_dD'+'_v2', col_dt_d13C+'_v2',	'start_'+col_dt_d13C+'_v2',	'end_'+col_dt_d13C+'_v2',	col_d13C+'_v2',	 col_errd13C+'_v2', 'slope_d13C'+'_v2',	'r2_d13C'+'_v2', 'res_d13C'+'_v2', 'n_d13C'+'_v2', col_wd+'_d13C'+'_v2', col_errwd+'_d13C'+'_v2', col_ws+'_d13C'+'_v2', col_errws+'_d13C'+'_v2'])

df_pkdD_vo = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE-dD_cleared_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=1, parse_dates=[ 1, 2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID',	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_errwd+'_dD', col_ws+'_dD', col_errws+'_dD'])
df_pkd13C_vo = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE-d13C_cleared_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=1, parse_dates=[ 1, 2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID',	'start_'+col_dt_d13C,	'end_'+col_dt_d13C,	col_d13C,	 col_errd13C, 'slope_d13C',	'r2_d13C', 'res_d13C', 'n_d13C', col_wd+'_d13C', col_errwd+'_d13C', col_ws+'_d13C', col_errws+'_d13C'])
df_PKE_vo = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE_cleared_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 13, 14, 15], dayfirst=True, keep_date_col=True, header=0,
                     names=[col_dt_dD+'_vo',	'start_'+col_dt_dD+'_vo',	'end_'+col_dt_dD+'_vo', col_dD+'_vo',	col_errdD+'_vo',	'slope_dD'+'_vo',	'r2_dD'+'_vo', 'res_dD'+'_vo',	'n_dD'+'_vo', col_wd+'_dD'+'_vo', col_errwd+'_dD'+'_vo', col_ws+'_dD'+'_vo', col_errws+'_dD'+'_vo',	col_dt_d13C+'_vo',	'start_'+col_dt_d13C+'_vo',	'end_'+col_dt_d13C+'_vo',	col_d13C+'_vo',	 col_errd13C+'_vo', 'slope_d13C'+'_vo',	'r2_d13C'+'_vo', 'res_d13C'+'_vo', 'n_d13C'+'_vo', col_wd+'_d13C'+'_vo', col_errwd+'_d13C'+'_vo', col_ws+'_d13C'+'_vo', col_errws+'_d13C'+'_vo'])

# model peaks
df_pkdD_edgar5 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE-dD_edgar5_prom=100ppb_minH=1988.5753799999998ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -dD', col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_errwd+'_dD', col_ws+'_dD', col_errws+'_dD'])
df_pkd13C_edgar5 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE-d13C_edgar5_prom=100ppb_minH=1988.5753799999998ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -d13C', col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C, col_d13C,	col_errd13C,	'slope_d13C',	'r2_d13C', 'res_d13C',	'n_d13C', col_wd+'_d13C', col_errwd+'_d13C', col_ws+'_d13C', col_errws+'_d13C'])
df_PKE_edgar5 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE_edgar5_prom=100ppb_minH=1988.5753799999998ppb_n=3.csv',  sep=',', index_col=0, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True)

df_pkdD_tno5 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE-dD_tno5_prom=100ppb_minH=1996.73776ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -dD', col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_errwd+'_dD', col_ws+'_dD', col_errws+'_dD'])
df_pkd13C_tno5 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE-d13C_tno5_prom=100ppb_minH=1996.73776ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -d13C', col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C, col_d13C,	col_errd13C,	'slope_d13C',	'r2_d13C', 'res_d13C',	'n_d13C', col_wd+'_d13C', col_errwd+'_d13C', col_ws+'_d13C', col_errws+'_d13C'])
df_PKE_tno5 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE_tno5_prom=100ppb_minH=1996.73776ppb_n=3.csv',  sep=',', index_col=0, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True)

# 2D peak data
df_pk2D = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/2Dpeaks_6h_original_prom=100ppb_minH=1986ppb_n=3.csv', sep=',', index_col=0, parse_dates=[4,5,6, 14,15,16], dayfirst=True, keep_date_col=True)
df_pk2D_v2 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/2Dpeaks_6h_offset_prom=100ppb_minH=1986ppb_n=3.csv', sep=',', index_col=0, parse_dates=[4,5,6, 14,15,16], dayfirst=True, keep_date_col=True)
df_pk2D_vo = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/2Dpeaks_6h_cleared_prom=100ppb_minH=1986ppb_n=3.csv', sep=',', index_col=0, parse_dates=[4,5,6, 14,15,16], dayfirst=True, keep_date_col=True)

#%%

# Wind analysis 
peaksd13C_day=df_pkd13C.between_time(sunrise, sunset, False, False)
peaksd13C_night=df_pkd13C.between_time(sunset, sunrise)

peaksdD_day=df_pkdD.between_time(sunrise, sunset, False, False)
peaksdD_night=df_pkdD.between_time(sunset, sunrise)

print('Average wind speed of day peaks: '+str(np.mean([peaksd13C_day[col_ws_d13C].mean(), peaksdD_day[col_ws_dD].mean()]))+' ± '+str(np.mean([peaksd13C_day[col_ws_d13C].std(), peaksdD_day[col_ws_dD].std()]))+' m/s')
print('Average wind speed of night peaks: '+str(np.mean([peaksd13C_night[col_ws_d13C].mean(), peaksdD_night[col_ws_dD].mean()]))+' ± '+str(np.mean([peaksd13C_night[col_ws_d13C].std(), peaksdD_night[col_ws_dD].std()]))+' m/s')

peaksd13C_day_2018=peaksd13C_day[t0:t1]
peaksd13C_night_2018=peaksd13C_night[t0:t1]
peaksdD_day_2018=peaksdD_day[t0:t1]
peaksdD_night_2018=peaksdD_night[t0:t1]

print('Average wind speed of day peaks, before Nov. 15th: '+str(np.mean([peaksd13C_day_2018[col_ws_d13C].mean(), peaksdD_day_2018[col_ws_dD].mean()]))+' ± '+str(np.mean([peaksd13C_day_2018[col_ws_d13C].std(), peaksdD_day_2018[col_ws_dD].std()]))+' m/s')
print('Average wind speed of night peaks, before Nov. 15th: '+str(np.mean([peaksd13C_night_2018[col_ws_d13C].mean(), peaksdD_night_2018[col_ws_dD].mean()]))+' ± '+str(np.mean([peaksd13C_night_2018[col_ws_d13C].std(), peaksdD_night_2018[col_ws_dD].std()]))+' m/s')

#%% Top diagram with categories

# 2D graphs with wind direction colors

df_pk2D[col_wd]=np.mean([df_pk2D[col_wd_dD], df_pk2D[col_wd_d13C]], axis=0)
df_pk2D[col_errwd]=np.std([df_pk2D[col_wd_dD], df_pk2D[col_wd_d13C]], axis=0)
df_pk2D[col_ws]=np.mean([df_pk2D[col_ws_dD], df_pk2D[col_ws_d13C]], axis=0)
df_pk2D[col_errws]=np.std([df_pk2D[col_ws_dD], df_pk2D[col_ws_d13C]], axis=0)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
norm=plt.Normalize(0, 360)
wind_color=cmocean.cm.balance(norm(df_pk2D[col_wd].values))
scatter=ax.scatter(df_pk2D[col_d13C], df_pk2D[col_dD], marker='o', s=(df_pk2D[col_ws]*10)**2, color=wind_color)
ax.grid(axis='x')
ax.grid(axis='y')
ax.set_ylabel(str_dD)
ax.set_xlabel(str_d13C)
cax = fig.add_axes([0.9, 0.125, 0.015, 0.75])
sm = ScalarMappable(cmap=cmocean.cm.balance, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax, orientation='vertical', ticks=[0, 180, 360])
cbar.set_label('Wind dir. [º]', rotation=90,labelpad=5)
# legend of sizes
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, func=lambda s: np.sqrt(s)/10, num=5)
legend2 = ax.legend(handles, labels, loc="lower right", title="Wind speed\n[m/s]")

fig.savefig(f_path+today+'2D_PKE-per-wind_original.png', dpi=300, transparent=True, bbox_inches = 'tight', pad_inches = 0)

#%% Top diagram with error bars

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
norm=plt.Normalize(0, 360)
wind_color=cmocean.cm.balance(norm(df_pk2D[col_wd].values))
ax.errorbar(x=df_pk2D[col_d13C], y=df_pk2D[col_dD], xerr=df_pk2D[col_errd13C], yerr=df_pk2D[col_errdD], fmt='o', markersize=10, color=web_blue)
ax.grid(axis='x')
ax.grid(axis='y')
ax.set_ylabel(str_dD)
ax.set_xlabel(str_d13C)

fig.savefig(f_path+today+'2D_PKE-errors_original.png', dpi=300, transparent=True, bbox_inches = 'tight', pad_inches = 0)

#%%
# High d13C analysis
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
points=ax.scatter(x=df_PKE[col_wd_dD], y=df_PKE[col_dD],  c=df_PKE[col_ws_dD], cmap=cm.get_cmap('brg'))
plt.colorbar(points, label='Wind speed (m/s)', ax=ax)

#%%
# Non USCB signatures: wd<180 or ws<avg
df_PKE_nonUSCB = df_PKE.loc[((df_PKE[col_wd_d13C]<180) | (df_PKE[col_wd_dD]<180)) | ((df_PKE[col_ws_d13C]<np.mean(df_PKE[col_ws_d13C])) & (df_PKE[col_ws_dD]<np.mean(df_PKE[col_ws_dD])))]
df_PKE_nonUSCB[col_d13C].mean()
df_PKE_nonUSCB[col_dD].mean()
df_PKE_nonUSCB[col_d13C].std()
df_PKE_nonUSCB[col_dD].std()

# Potential USCB signatures: wd in [225,315] and ws>avg
df_PKE_USCB = df_PKE.loc[(((df_PKE[col_wd_d13C]>225) & (df_PKE[col_wd_d13C]<315)) | ((df_PKE[col_wd_dD]>225) & (df_PKE[col_wd_dD]<315))) & ((df_PKE[col_ws_d13C]>np.mean(df_PKE[col_ws_d13C])) & (df_PKE[col_ws_dD]>np.mean(df_PKE[col_ws_dD])))]

# All the rest is potentially from USCB then?
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
sns.kdeplot(df_PKE[col_d13C], shade=True, ax=ax, label='all', color=web_blue)
sns.kdeplot(df_PKE_nonUSCB[col_d13C], shade=False, ax=ax, label='non-USCB', color=web_cyan)
sns.kdeplot(df_PKE_USCB[col_d13C], shade=False, ax=ax, label='USCB', color=web_red)
ax.set_xlabel(str_d13C)
ax.legend()
#fig.savefig(f_path+'hist-dD_IRMS-USCB.png', dpi=300)


#%%
n_months=7
def monthly(data):
    
    # Cut the IRMS data per month

    t08='2018-09-01 00:00:00'
    t09='2018-10-01 00:00:00'
    t10='2018-11-01 00:00:00'
    t11='2018-12-01 00:00:00'
    t12='2019-01-01 00:00:00'
    t01='2019-02-01 00:00:00'
    t02='2019-03-01 00:00:00'
    t03='2019-04-01 00:00:00'
    t_months = [t08, t09, t10, t11, t12, t01, t02, t03]

    data_monthly=list(range(n_months))
    
    for m in range(1, n_months+1):
        data_monthly[m-1] = (data.loc[(data.index<t_months[m]) & (data.index>t_months[m-1])])
    
    return data_monthly

str_months = ['Sept. 18', 'Oct. 18', 'Nov. 18', 'Dec. 18', 'Jan. 19', 'Feb. 19', 'Mar. 19']
num_months = [9, 10, 11, 12, 1, 2, 3]
norm=mcolors.Normalize(vmin=0, vmax=len(str_months))
cmap=cm.get_cmap('jet')
c=cmap(norm(range(len(str_months))))
# Your own colors for each month
c=[web_magenta, vertclair, web_cyan, web_blue, 'k', web_green, web_red]

monthly_IRMS = monthly(df_IRMS)
monthly_edgar = monthly(df_edgar)
monthly_tno = monthly(df_tno)
monthly_met = monthly(df_meteo)

monthly_pkdD = monthly(df_pkdD)
monthly_pkd13C = monthly(df_pkd13C)

monthly_pkdD_v2 = monthly(df_pkdD_v2)
monthly_pkd13C_v2 = monthly(df_pkd13C_v2)

monthly_pkdD_vo = monthly(df_pkdD_vo)
monthly_pkd13C_vo = monthly(df_pkd13C_vo)

monthly_pkdD_edgar5 = monthly(df_pkdD_edgar5)
monthly_pkd13C_edgar5 = monthly(df_pkd13C_edgar5)
#monthly_pk_edgar5 = monthly(df_PKE_edgar5)

monthly_pkdD_tno5 = monthly(df_pkdD_tno5)
monthly_pkd13C_tno5 = monthly(df_pkd13C_tno5)
#monthly_pk_tno5 = monthly(df_PKE_tno5)

#%%
###############################################################################
# Wind analysis
###############################################################################

# Wind plots
windrose_ws = fg.wind_rose(df_IRMS[col_wd], df_IRMS[col_ws], col_ws, col_ws)
windrose_IRMS = fg.wind_rose(df_IRMS[col_wd], df_IRMS[col_MR], str_MR, '')
#windrose_IRMSd13C = fg.wind_rose(df_IRMS[col_wd], df_IRMS[col_d13C], str_d13C, str_d13C)
#windrose_IRMSdD = fg.wind_rose(df_IRMS[col_wd], df_IRMS[col_dD], str_dD, str_dD)
windrose_pkd13C = fg.wind_rose(df_PKE[col_wd_d13C], df_PKE[col_d13C], str_d13C, str_d13C)
windrose_pkdD = fg.wind_rose(df_PKE[col_wd_dD], df_PKE[col_dD], str_dD, str_dD)
windrose_pkd13C_v2 = fg.wind_rose(df_PKE_v2[col_wd_d13C], df_PKE_v2[col_d13C], str_d13C, str_d13C)
windrose_pkdD_v2 = fg.wind_rose(df_PKE_v2[col_wd_dD], df_PKE_v2[col_dD], str_dD, str_dD)
windrose_pkd13C_vo = fg.wind_rose(df_PKE_vo[col_wd_d13C], df_PKE_vo[col_d13C], str_d13C, str_d13C)
windrose_pkdD_vo = fg.wind_rose(df_PKE_vo[col_wd_dD], df_PKE_vo[col_dD], str_dD, str_dD)

windrose_IRMS.savefig(f_path+'windrose_wd-IRMS.png', bbox_inches='tight', dpi=600, transparent=True)
#windrose_IRMS.savefig(f_path+'windrose_x(CH4)-IRMS.png', bbox_inches='tight', dpi=300)
#windrose_IRMSd13C.savefig(f_path+'windrose_d13C-IRMS.png', dpi=300)
#windrose_IRMSdD.savefig(f_path+'windrose_dD-IRMS.png', dpi=300)
windrose_pkd13C.savefig(f_path+'windrose_d13C-peaks_original.png', dpi=300)
windrose_pkdD.savefig(f_path+'windrose_dD-peaks_original.png', dpi=300)
windrose_pkd13C_v2.savefig(f_path+'windrose_d13C-peaks_offset.png', dpi=300)
windrose_pkdD_v2.savefig(f_path+'windrose_dD-peaks_offset.png', dpi=300)
windrose_pkd13C_vo.savefig(f_path+'windrose_d13C-peaks_cleared.png', dpi=300)
windrose_pkdD_vo.savefig(f_path+'windrose_dD-peaks_cleared.png', dpi=300)

windrose_edgar = fg.wind_rose(df_edgar[col_wd], df_edgar['Total'], str_MR, 'EDGAR')
windrose_tno = fg.wind_rose(df_tno[col_wd], df_tno['Total'], str_MR, 'TNO')
windrose_edgar.savefig(f_path+'windrose_x(CH4)-EDGAR.png', bbox_inches='tight', dpi=300)
windrose_tno.savefig(f_path+'windrose_x(CH4)-TNO.png', bbox_inches='tight', dpi=300)

windrose_pkd13C = fg.wind_rose(df_pkd13C[col_wd_d13C], df_pkd13C[col_d13C], str_d13C, 'original data')
windrose_pkdD = fg.wind_rose(df_pkdD[col_wd_dD], df_pkdD[col_dD], str_dD, 'original data')
windrose_pkd13C.savefig(f_path+today+'windrose_d13C-peaks_original.png', dpi=300)
windrose_pkdD.savefig(f_path+today+'windrose_dD-peaks_original.png', dpi=300)

#%%

# Wind histograms 
fig_hist_W= plt.figure(figsize=(15,10))
ax_hist_W=fig_hist_W.add_subplot(111)
sns.kdeplot(df_IRMS[col_wd], shade=True, ax=ax_hist_W, label='IRMS')
#sns.kdeplot(df_edgar[col_wd], shade=True, ax=ax_hist_W, label='CHIMERE')
#sns.kdeplot(df_IRMS[col_ws], shade=True, ax=ax_hist_W, label='IRMS')
#sns.kdeplot(df_edgar[col_ws], shade=True, ax=ax_hist_W, label='CHIMERE')
ax_hist_W.set_xlabel('Wind dir. [º]')
#ax_hist_W.set_xlabel('Wind speed. [m/s]')
#fig_hist_W.savefig(f_path+'windhist-dir_IRMS&CHIMERE.png', dpi=300)


#%%
# Monthly histograms
for i in range(len(num_months)):
    df_pk2D['Month #'].replace({num_months[i]:str_months[i]}, inplace=True)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111)
for m in range(len(str_months)):
    
    month=str_months[m]
    month_n=num_months[m]
    
    # Wind
    #plt.hist(monthly_IRMS[m][col_wd])
    #monthly_IRMS[m].mean()[[col_wd, col_ws]]
    #monthly_IRMS[m].std()[[col_wd, col_ws]]
    #sns.kdeplot(monthly_IRMS[m][col_wd], shade=False, ax=ax, label=str_months[m], color=c[m])#, bw=10)
    sns.kdeplot(monthly_IRMS[m][col_ws], shade=False, ax=ax, label=str_months[m], color=c[m])
    #sns.kdeplot(monthly_edgar[m][col_wd], shade=True, ax=ax, label='CHIM month #: '+str(m), color=c[m], linestyle="")
    #sns.kdeplot(monthly_edgar[m][col_ws], shade=True, ax=ax, label='CHIM month #: '+str(m), color=c[m], linestyle="")
    
    # Signatures
    #sns.kdeplot(monthly_pkdD[m][col_dD], shade=False, ax=ax, label='month #: 'month, color=c[m])
    #sns.kdeplot(monthly_pkdD_edgar3[m][col_dD], shade=True, ax=ax, label='', color=c[m], linestyle="")
    #ax.set_xlabel(str_dD)
    #ax.hist(monthly_pkd13C[m][col_d13C],  label='month #: '+str(m), edgecolor=c[m], facecolor='None', lw=2, bins=bins_d13C)
    #sns.kdeplot(monthly_pkd13C[m][col_d13C], shade=False, ax=ax, label=month, color=c[m])
    #sns.kdeplot(monthly_pkd13C_edgar3[m][col_d13C], shade=True, ax=ax, color=c[m], label='', linestyle="")
    #ax.set_xlabel(str_d13C)
    
    #2D
    #ax.scatter(df_pk2D.loc[df_pk2D['Month #']==month][col_d13C], df_pk2D.loc[df_pk2D['Month #']==month][col_dD], marker='o', color=c[m], label=month)
    
#ax.set_xticks([0, 90, 180, 270, 360])# For wind plots
#ax.set_ylabel('Frequency')
#ax.set_ylabel(str_dD)
#ax.set_xlabel(str_d13C)
#ax.set_xlabel('Wind dir. [º]')
ax.set_xlabel('Wind speed. [m/s]')
ax.grid(axis='x')
ax.grid(axis='y')
ax.legend(loc='upper right',
               fancybox=False, framealpha=1, edgecolor='k')
fig.savefig(f_path+'hist_windspeed-monthly.png', bbox_inches='tight', dpi=600)

#%%
# Wind histograms+ correlation, IRMS-model & inventory-inventory comparison

winds=pd.DataFrame({'wd IRMS':df_IRMS[col_wd], 'wd CHIMERE':np.interp(df_IRMS.index, df_edgar.index, df_edgar[col_wd])}, index=df_IRMS.index)
#winds=pd.DataFrame({'wd IRMS':np.interp(df_edgar.index, df_IRMS.index, df_IRMS[col_wd]), 'wd CHIMERE':df_edgar[col_wd]}, index=df_edgar.index)

fig_corrwind=fg.correlation([winds['wd IRMS']], [winds['wd CHIMERE']], 'IRMS', 'CHIMERE')
#fig_corrwind[0].savefig(f_path+'corr_winddir_IRMS&CHIMERE.png', dpi=300)

# Wind to signature correlations
fig_corrwd_d13C=plt.plot(df_PKE[col_wd_d13C], df_PKE[col_d13C], '.')
fig_corrwd_dD=plt.plot(df_PKE[col_wd_dD], df_PKE[col_dD], '.')

fig_corrws_d13C=plt.plot(df_PKE[col_ws_d13C], df_PKE[col_d13C], '.')
fig_corrws_dD=plt.plot(df_PKE[col_ws_dD], df_PKE[col_dD], '.')

# d13C to dD correlations
corr_d13CdD=fg.correlation([df_pk2D[col_d13C]], [df_pk2D[col_dD]], 'd13C', 'dD')
corr_d13CdD[0].savefig(f_path+today+'correlation_d13C-dD_original.png', dpi=300)

# wd to ws correlations
fig_corwswd=plt.plot(df_IRMS[col_wd], df_IRMS[col_ws], '.')#, xlabel='Wind dir.', ylabel='Wind speed')

# CH4 EDGAR and CH4 TNO correlation
corr_CH4inv=fg.correlation([df_edgar['Total']], [df_tno['Total']], 'EDGAR v5.0', 'TNO-CAMS v4.2')
corr_CH4inv[0].savefig(f_path+today+'correlation_CH4_EDGAR-TNO.png', dpi=300, transparent=True)

#%%
# Interpolate the model datetimes to the measurement ones for histograms and correlation plots
df_all=df_IRMS.join(pd.concat([pd.DataFrame({'ch4_EDGAR':np.interp(df_IRMS.index, df_edgar.index, df_edgar['Total']), 'd13C_EDGAR': np.interp(df_IRMS.index, df_edgar.index, df_edgar['d13C calc5']), 'dD_EDGAR':np.interp(df_IRMS.index, df_edgar.index, df_edgar['dD calc5'])}, index=df_IRMS.index),
                               pd.DataFrame({'ch4_TNO':np.interp(df_IRMS.index, df_tno.index, df_tno['Total']), 'd13C_TNO': np.interp(df_IRMS.index, df_tno.index, df_tno['d13C calc5']), 'dD_TNO':np.interp(df_IRMS.index, df_tno.index, df_tno['dD calc5'])}, index=df_IRMS.index)], axis=1),
                    )

df_all_ch4=pd.DataFrame({col_MR:pd.concat([df_IRMS[col_MR_d13C], df_IRMS[col_MR_dD]]).dropna(), 'ch4_EDGAR':df_all['ch4_EDGAR'], 'ch4_TNO':df_all['ch4_TNO']})

# Root Mean Square Error (unit ppb)
np.sqrt(np.mean((df_all_ch4['ch4_TNO']-df_all_ch4[col_MR])**2))
np.sqrt(np.mean((df_all_ch4['ch4_EDGAR']-df_all_ch4[col_MR])**2))

# The other way around
# Interpolate the observation time series to the model times, and make hourly average, to account for the representation error
df_hourly=pd.DataFrame({col_MR+'_rep':np.interp(df_edgar.index, df_IRMS.index, df_IRMS.rolling('1h').mean()[col_MR]), col_MR+'_h':np.interp(df_edgar.index, df_IRMS.index, df_IRMS[col_MR])}, index=df_edgar.index)

#%%
# Correlation plots
fig_corr_ch4=fg.correlation([df_all_ch4[col_MR], df_all_ch4[col_MR]], [df_all_ch4['ch4_EDGAR'], df_all_ch4['ch4_TNO']], namex='observed x(CH4)', namey='modeled x(CH4)', leg=[str_EDGAR, str_TNO])
fig_corr_ch4_ssn=fg.correlation([df_all_ch4[col_MR], df_all_ch4[col_MR][t0:t1], df_all_ch4[col_MR][t1:t2]], [df_all_ch4['ch4_EDGAR'], df_all_ch4['ch4_EDGAR'][t0:t1], df_all_ch4['ch4_EDGAR'][t1:t2]], namex='observed x(CH4)', namey='modeled x(CH4)', leg=['all data', 'fall', 'winter'])
fig_corr_ch4_rep=fg.correlation([df_hourly[col_MR+'_h'], df_hourly[col_MR+'_h']], [df_edgar['Total'], df_tno['Total']], namex='observed-hourly x(CH4)', namey='modeled x(CH4)', leg=[str_EDGAR, str_TNO])

fig_corr_d13C=fg.correlation([df_all[col_d13C], df_all[col_d13C]], [df_all['d13C_EDGAR'], df_all['d13C_TNO']], namex='observed '+d13C, namey='modeled '+d13C, leg=[str_EDGAR, str_TNO])
fig_corr_dD=fg.correlation([df_all[col_dD], df_all[col_dD]], [df_all['dD_EDGAR'], df_all['dD_TNO']], namex='observed '+dD, namey='modeled '+dD, leg=[str_EDGAR, str_TNO])

fig_corr_ch4[0].savefig(f_path+today+'model-correlation_ch4.png', dpi=600, transparent=True, bbox_inches = 'tight', pad_inches = 0)
fig_corr_ch4_ssn[0].savefig(f_path+today+'model-correlation_ch4-seasonal.png', dpi=600, transparent=True, bbox_inches = 'tight', pad_inches = 0)
fig_corr_ch4_rep[0].savefig(f_path+today+'model-correlation_ch4-hourly.png', dpi=600, transparent=True, bbox_inches = 'tight', pad_inches = 0)

fig_corr_d13C[0].savefig(f_path+today+'model-correlation_d13C.png', dpi=300)
fig_corr_dD[0].savefig(f_path+today+'model-correlation_dD.png', dpi=300)

#%%

# Visualize peaks per month

for m in range(len(str_months)):
    y_lim1=np.min(monthly_IRMS[m][col_MR])
    y_lim2=np.max(monthly_IRMS[m][col_MR])
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    
    # obs x(CH4)
    #monthly_IRMS[m].plot(y=col_MR, ax=ax, markersize=2, style='o',  mec='black', mfc='black')#, label='observations')
    monthly_IRMS[m].plot(y=col_MR, ax=ax,  color='gray', lw=0.5)
    monthly_IRMS[m].plot(y=col_MR, ax=ax, markersize=1, style='o',  mec='black', mfc='black')#, label='observations')
    # model x(CH4)
    monthly_edgar[m].plot(y='Total', ax=ax,  color=web_magenta, label='edgar')
    monthly_tno[m].plot(y='Total', ax=ax,  color=web_cyan, label='tno')
    
    # Peaks from observation data
    ax.vlines(x=monthly_pkd13C[m].index.values, ymin=y_lim1, ymax=y_lim2, ls='--', color=web_green, label='obs. -d13C')
    ax.vlines(x=monthly_pkdD[m].index.values, ymin=y_lim1, ymax=y_lim2, ls='--', color=web_yellow, label='obs. -dD')
    ax.legend()
    
    # Peaks from model data
    ax.vlines(x=monthly_pkdD_edgar5[m].index.values, ymin=y_lim1, ymax=y_lim2, ls='--', color=web_magenta)
    ax.vlines(x=monthly_pkdD_edgar5[m].index.values, ymin=y_lim1, ymax=y_lim2, ls=':', color=web_cyan)
    
    # Meteorological data
    #monthly_met[m].plot(y=col_pm10, ax=ax, secondary_y=True)
    
    fig.savefig(f_path+today+'PKE_'+str(m)+'.png', dpi=300)
    #fig.savefig(f_path+today+'PM10_'+str(m)+'.png', dpi=300)

#%%

###############################################################################
# Model source analysis
###############################################################################

sources_edgar=['Wetland', 'Agriculture', 'Waste', 'FF_coal', 'FF_gas', 'FF_oil', 'ENB', 'Other']
sources_tno=['Wetland', 'Agriculture', 'Waste', 'FF', 'ENB', 'Other']
    
n_sources_edgar=len(sources_edgar)
norm_s=mcolors.Normalize(vmin=0, vmax=n_sources-1)
#c_s=cmap(norm_s(range(n_sources)))
c_s=['k', web_blue, web_red,  web_green, web_yellow]

# Participation from each source
#df_edgar.mean()[sources_added]
#df_tno.mean()[sources_added]

df_pp_edgar=pd.DataFrame(index=range(n_months+1), columns=[s+' added' for s in sources_edgar])
df_pp_tno=pd.DataFrame(index=range(n_months+1), columns=[s+' added' for s in sources_tno])
for m in range(n_months):
    for s in list(df_pp_edgar):
        df_pp_edgar[s][m]=monthly_edgar[m].mean()[s]
    for s in list(df_pp_tno):
        df_pp_tno[s][m]=monthly_tno[m].mean()[s]

df_pp_edgar.loc[n_months+1]=np.mean(df_pp_edgar)
df_pp_tno.loc[n_months+1]=np.mean(df_pp_tno)

# Save tables of general source proportions
df_pp_edgar.to_csv(t_path+today+'EDGAR-calc5_sources_pp.csv', encoding=encoding)
df_pp_tno.to_csv(t_path+today+'TNO-calc5_sources_pp.csv', encoding=encoding)

fig_ed = plt.figure(figsize=(15,10))
ax_ed = fig_ed.add_subplot(111)
df_pp_edgar.plot.bar(stacked=True, label=sources_edgar, ax=ax_ed, color=c_s[::-1], legend=False)
ax_ed.legend(sources_edgar, bbox_to_anchor=(1.01, 1.05))
ax_ed.set_xticklabels(str_months)
fig_ed.tight_layout()
#fig_ed.savefig(f_path+'sources-edgar_monthly.png', dpi=300)

fig_tno = plt.figure(figsize=(15,10))
ax_tno = fig_tno.add_subplot(111)
df_pp_tno.plot.bar(stacked=True, ax=ax_tno, color=c_s[::-1])
ax_tno.legend(sources_tno, bbox_to_anchor=(1.01, 1.05))
ax_tno.set_xticklabels(str_months)
fig_tno.tight_layout()
#fig_tno.savefig(f_path+'sources-tno_monthly.png', dpi=300)


#%%
###############################################################################
# Sample analysis
###############################################################################

avg_sources=df_samples.groupby('Source').mean()
avg_SNAP=df_samples.groupby('SNAP').mean()
std_sources=df_samples.groupby('Source').std()
std_SNAP=df_samples.groupby('SNAP').std()

avg_sources.to_csv(t_path+today+'samples_source-sig_average.csv')
avg_SNAP.to_csv(t_path+today+'samples_SNAP-sig_average.csv')
std_sources.to_csv(t_path+today+'samples_source-sig_stdev.csv')
std_SNAP.to_csv(t_path+today+'samples_SNAP-sig_stdev.csv')
