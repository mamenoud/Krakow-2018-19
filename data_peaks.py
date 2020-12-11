#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 16:36:27 2020

@author: malika
"""

# This script is for long-term data analysis of Krakow data
# Looking at individual peaks

import numpy as np
import pandas as pd
from pandas import ExcelWriter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import seaborn as sns
import os, shutil
import scipy as sc
from scipy import stats

import functions_KR as fun
import figures_KR as fg

plt.rcParams.update({'font.size': 22})

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

col_dt_obs='filling time UTC'
col_dt='DateTime'
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

col_MR_edgar='EDGARv4.3.2_CH4 [ppb]'
col_MR_tno='TNO-MACC_III_CH4 [ppb]'

col_wd='averageWindDirection'
col_ws='averageWindSpeed'
col_errwd='stdWindDirection'
col_errws='stdWindSpeed'

# For pretty strings

strMR_13C='MR_ppb - '+d13C
strMR_D='MR_ppb - '+dD
str_d13C=delta13C+permil
str_dD=deltaD+permil
str_MR=col_P

vertclair='#009966'
web_blue='#333399'
web_green='#006633'
web_yellow='#FFFF00'
web_red='#FF3333'
web_cyan='#0099FF'
web_magenta='#FF0099'

###############################################################################
# SAVE THE RESULT OF THE DAY IN A SEPERATE FOLDER
###############################################################################

f_path = '/Users/malika/surfdrive/Data/Krakow/Roof air/figures/'
t_path = '/Users/malika/surfdrive/Data/Krakow/Roof air/tables/'
today='2020-11-26 data_peaks/'

#%%
# Observation data
df_IRMS=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Input files/KRK_merged_re-calibrated_corrected-IRMS.csv', sep=',', index_col=col_dt_obs, parse_dates=True, dayfirst=True, keep_date_col=True)

# Modeled data
df_edgar_in=pd.read_table(t_path+'model/Krakow_CH4_EDGARv50_processed.csv', sep=',', index_col=0, parse_dates=True, dayfirst=True, keep_date_col=True)
df_tno_in=pd.read_table(t_path+'model/Krakow_CH4_TNO_CAMS_REGv221_processed.csv', sep=',', index_col=0, parse_dates=True, dayfirst=True, keep_date_col=True)
df_Wmod=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Modeled/wind_speed_direction_KRK_20180914_20190314_CHIMERE.csv', sep=';', index_col=0, parse_dates=True)
df_edgar=df_edgar_in.join(df_Wmod)
df_tno=df_tno_in.join(df_Wmod)

# Get the PKE data
df_PKE = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE_original_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 13, 14, 15], dayfirst=True, keep_date_col=True, header=0,
                     names=[col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_errwd+'_dD', col_ws+'_dD', col_errws+'_dD', col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C,	col_d13C,	 col_errd13C, 'slope_d13C',	'r2_d13C', 'res_d13C', 'n_d13C', col_wd+'_d13C', col_errwd+'_d13C', col_ws+'_d13C', col_errws+'_d13C'])
df_PKE_v2 = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE_offset_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 13, 14, 15], dayfirst=True, keep_date_col=True, header=0,
                     names=[col_dt_dD+'_v2',	'start_'+col_dt_dD+'_v2',	'end_'+col_dt_dD+'_v2', col_dD+'_v2',	col_errdD+'_v2',	'slope_dD'+'_v2',	'r2_dD'+'_v2', 'res_dD',	'n_dD'+'_v2', col_wd+'_dD'+'_v2', col_errwd+'_dD'+'_v2', col_ws+'_dD'+'_v2',	col_errws+'_dD'+'_v2', col_dt_d13C+'_v2',	'start_'+col_dt_d13C+'_v2',	'end_'+col_dt_d13C+'_v2',	col_d13C+'_v2',	 col_errd13C+'_v2', 'slope_d13C'+'_v2',	'r2_d13C'+'_v2', 'res_d13C'+'_v2', 'n_d13C'+'_v2', col_wd+'_d13C'+'_v2', col_errwd+'_d13C'+'_v2', col_ws+'_d13C'+'_v2', col_errws+'_d13C'+'_v2'])
df_PKE_vo = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE_cleared_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 13, 14, 15], dayfirst=True, keep_date_col=True, header=0,
                     names=[col_dt_dD+'_vo',	'start_'+col_dt_dD+'_vo',	'end_'+col_dt_dD+'_vo', col_dD+'_vo',	col_errdD+'_vo',	'slope_dD'+'_vo',	'r2_dD'+'_vo', 'res_dD'+'_vo',	'n_dD'+'_vo', col_wd+'_dD'+'_vo', col_errwd+'_dD'+'_vo', col_ws+'_dD'+'_vo', col_errws+'_dD'+'_vo',	col_dt_d13C+'_vo',	'start_'+col_dt_d13C+'_vo',	'end_'+col_dt_d13C+'_vo',	col_d13C+'_vo',	 col_errd13C+'_vo', 'slope_d13C'+'_vo',	'r2_d13C'+'_vo', 'res_d13C'+'_vo', 'n_d13C'+'_vo', col_wd+'_d13C'+'_vo', col_errwd+'_d13C'+'_vo', col_ws+'_d13C'+'_vo', col_errws+'_d13C'+'_vo'])

df_pkdD = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE-dD_original_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=1, parse_dates=[ 1, 2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID',	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_errwd+'_dD', col_ws+'_dD', col_errws+'_dD'])
df_pkd13C = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE-d13C_original_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=1, parse_dates=[ 1, 2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID',	'start_'+col_dt_d13C,	'end_'+col_dt_d13C,	col_d13C,	 col_errd13C, 'slope_d13C',	'r2_d13C', 'res_d13C', 'n_d13C', col_wd+'_d13C', col_errwd+'_d13C', col_ws+'_d13C', col_errws+'_d13C'])

# model peaks
df_pkdD_edgar5 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE-dD_edgar5_prom=100ppb_minH=1988.5753799999998ppb_n=3.csv',  sep=',', index_col=0, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -dD', col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_errwd+'_dD', col_ws+'_dD', col_errws+'_dD'])
df_pkd13C_edgar5 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE-d13C_edgar5_prom=100ppb_minH=1988.5753799999998ppb_n=3.csv',  sep=',', index_col=0, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -d13C', col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C, col_d13C,	col_errd13C,	'slope_d13C',	'r2_d13C', 'res_d13C',	'n_d13C', col_wd+'_d13C', col_errwd+'_d13C', col_ws+'_d13C', col_errws+'_d13C'])
df_PKE_edgar5 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE_edgar5_prom=100ppb_minH=1988.5753799999998ppb_n=3.csv',  sep=',', index_col=0, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True)

#df_pkdD_edgar4 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE-dD_edgar4_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2], dayfirst=True, keep_date_col=True, header=0,
#                     names=['peak ID -dD',	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_errwd+'_dD', col_ws+'_dD', col_errws+'_dD'])
#df_pkd13C_edgar4 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE-d13C_edgar4_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2], dayfirst=True, keep_date_col=True, header=0,
#                     names=['peak ID -d13C',	'start_'+col_dt_d13C,	'end_'+col_dt_d13C, col_d13C,	col_errd13C,	'slope_d13C',	'r2_d13C', 'res_d13C',	'n_d13C', col_wd+'_d13C', col_errwd+'_d13C', col_ws+'_d13C', col_errws+'_d13C'])
#df_PKE_edgar4 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE_edgar4_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=0, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True)

df_pkdD_tno5 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE-dD_tno5_prom=100ppb_minH=1996.73776ppb_n=3.csv',  sep=',', index_col=0, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -dD', col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_errwd+'_dD', col_ws+'_dD', col_errws+'_dD'])
df_pkd13C_tno5 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE-d13C_tno5_prom=100ppb_minH=1996.73776ppb_n=3.csv',  sep=',', index_col=0, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -d13C', col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C, col_d13C,	col_errd13C,	'slope_d13C',	'r2_d13C', 'res_d13C',	'n_d13C', col_wd+'_d13C', col_errwd+'_d13C', col_ws+'_d13C', col_errws+'_d13C'])
df_PKE_tno5 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE_tno5_prom=100ppb_minH=1996.73776ppb_n=3.csv',  sep=',', index_col=0, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True)

#df_pkdD_tno4 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE-dD_tno4_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2], dayfirst=True, keep_date_col=True, header=0,
#                     names=['peak ID',	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_errwd+'_dD', col_ws+'_dD', col_errws+'_dD'])
#df_pkd13C_tno4 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE-d13C_tno4_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2], dayfirst=True, keep_date_col=True, header=0,
#                     names=['peak ID -d13C',	'start_'+col_dt_d13C,	'end_'+col_dt_d13C, col_d13C,	col_errd13C,	'slope_d13C',	'r2_d13C', 'res_d13C',	'n_d13C', col_wd+'_d13C', col_errwd+'_d13C', col_ws+'_d13C', col_errws+'_d13C'])
#df_PKE_tno4 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE_tno4_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=0, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True)

#%%

def see_peak(t1,t2,sub):
    #f=fg.overview_pk(df_IRMS[t1:t2], [df_PKE], [df_PKE_edgar3, df_PKE_tno3], [df_edgar[t1:t2], df_tno[t1:t2]], names=['obs.', 'EDGAR calc3', 'TNO calc3'])
    f_edgar=fg.overview_pk(df_IRMS[t1:t2], [df_PKE], [df_PKE_edgar5], [df_edgar[t1:t2]], names=['obs.', 'EDGAR v5.0'])
    f_tno=fg.overview_pk(df_IRMS[t1:t2], [df_PKE], [df_PKE_tno5], [df_tno[t1:t2]], names=['obs.', 'TNO-CAMS v4.2'])
    
    return f_edgar, f_tno

def save_peak(t1,t2,sub,f_edgar, f_tno):
    df_IRMS[t1:t2].to_csv(t_path+today+'subset_'+sub+'_IRMS.csv', encoding=encoding)
    sub_PKE=pd.concat([df_pkdD.loc[t1:t2], df_pkd13C.loc[t1:t2]], axis=1, ignore_index=True)
    sub_PKE.to_csv(t_path+today+'subset_'+sub+'_peaks-original.csv', encoding=encoding)
    f_edgar[0].savefig(f_path+today+'overview_peak_'+sub+'_EDGAR-calc5.png', dpi=100, transparent=True, bbox_inches = 'tight', pad_inches = 0)
    f_tno[0].savefig(f_path+today+'overview_peak_'+sub+'_TNO-calc5.png', dpi=100, transparent=True, bbox_inches = 'tight', pad_inches = 0)
    return

#%%
############
# Special peaks

# d13C enriched
d1='Jan-08'
t11='2019-01-07 00:00:00'
t12='2019-01-10 00:00:00'
f1=see_peak(t11,t12,d1)
save_peak(t11,t12,d1,f1)

d2='Feb-09'
t21='2019-02-08 00:00:00'
t22='2019-02-10 00:00:00'
f2=see_peak(t21,t22,d2)

# dD enriched -birthday peak
da='Sep-25'
ta1='2018-09-24 00:00:00'
ta2='2018-09-26 00:00:00'
ta=see_peak(ta1,ta2,da)

# Most biogenic
d3a='Feb-10'
t31a='2019-02-10 00:00:00'
t32a='2019-02-12 00:00:00' 
f3a=see_peak(t31a,t32a,d3a)
save_peak(t31a,t32a,d3a,f3a)

# 2nd most biogenic (but bad data)
d3b='Mar-01'
t31b='2019-02-28 00:00:00'
t32b='2019-03-02 00:00:00'
f3b=see_peak(t31b,t32b,d3b)
save_peak(t31b,t32b,d3b,f3b)

# 3rd most biogenic
d3c='Oct-25'
t31c='2018-10-24 00:00:00'
t32c='2018-10-27 00:00:00'
f3c=see_peak(t31c,t32c,d3c)
save_peak(t31c,t32c,d3c,f3c)

# Mar-05: high wind speed, lowest d13C (but bad data)
d4='Mar-05'
t41='2019-03-05 00:00:00'
t42='2019-03-06 00:00:00'
f4=see_peak(t41,t42,d4)

############
# Special time periods

# Autumn daily peaks
s1='Sept-daily'
s1t1='2018-09-29 00:00:00'
s1t2='2018-10-04 00:00:00'
fs1=see_peak(s1t1,s1t2,s1)

# October highest event
s2='Oct-high'
s2t1='2018-10-15 00:00:00'
s2t2='2018-10-21 00:00:00'
fs2=see_peak(s2t1,s2t2,s2)

# October low dD
s12='Oct-lowdD'
s12t1='2018-10-09 00:00:00'
s12t2='2018-10-30 00:00:00'
fs12=see_peak(s12t1,s12t2,s12)

# November 2nd highest event
s3='Nov-high'
s3t1='2018-10-31 00:00:00'
s3t2='2018-11-04 00:00:00'
fs3=see_peak(s3t1, s3t2, s3)

# November peak serie ######################## SELECTED
s4='Nov-serie'
s4t1='2018-11-02 10:00:00'
s4t2='2018-11-10 16:00:00'
fs4_edgar, fs4_tno=see_peak(s4t1, s4t2, s4)
save_peak(s4t1,s4t2,s4,fs4_edgar,fs4_tno)

# November strange
s5='Nov-strange'
s5t1='2018-11-22 00:00:00'
s5t2='2018-11-27 00:00:00'
fs5=see_peak(s5t1, s5t2, s5)

# December shift
s6='Dec-shift'
s6t1='2018-12-19 00:00:00'
s6t2='2018-12-23 00:00:00'
fs6=see_peak(s6t1, s6t2, s6)
save_peak(s6t1, s6t2, s6, fs6)

# December high winds
s7='Dec-bigwind'
s7t1='2018-12-22 00:00:00'
s7t2='2019-01-01 00:00:00'
fs7=see_peak(s7t1, s7t2, s7)

# January peak serie -BAD
s8='Jan-serie'
s8t1='2019-01-07 00:00:00'
s8t2='2019-02-04 00:00:00'
fs8=see_peak(s8t1, s8t2, s8)

# January N wind -BAD
s9='Jan-Nwind'
s9t1='2019-01-18 00:00:00'
s9t2='2019-01-24 00:00:00'
fs9=see_peak(s9t1, s9t2, s9)

# February peak serie ######################## SELECTED
s10='Feb-serie'
s10t1='2019-02-15 00:00:00'
s10t2='2019-02-22 00:00:00'
fs10_edgar, fs10_tno=see_peak(s10t1, s10t2, s10)
save_peak(s10t1,s10t2,s10,fs10_edgar, fs10_tno)

# March peak serie
s11='March-serie'
s11t1='2019-03-01 00:00:00'
s11t2='2019-03-15 00:00:00'
fs11=see_peak(s11t1, s11t2, s11)

#%%
#fig = plt.figure(figsize=(10,10))
#ax1 = WindroseAxes.from_ax(fig=fig)
#data_angles=np.linspace(0,359,360)
#color_range = cmocean.cm.balance(norm(data_angles))
#ax1.bar(data_angles, np.array([1]*360), color=color_range)

#%%
############
# Selected time periods in tables
