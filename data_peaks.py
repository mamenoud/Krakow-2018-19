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
today='2020-09-22 data_peaks/'

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
df_PKE = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-11 data_signatures/PKE/PKE_original_prom=100ppb_minH=1986ppb_n=3.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 11, 12, 13], dayfirst=True, keep_date_col=True, header=0,
                     names=[col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_ws+'_dD',	col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C,	col_d13C,	 col_errd13C, 'slope_d13C',	'r2_d13C', 'res_d13C', 'n_d13C', col_wd+'_d13C', col_ws+'_d13C'])
df_PKE_v2 = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-11 data_signatures/PKE/PKE_offset_prom=100ppb_minH=1986ppb_n=3.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 11, 12, 13], dayfirst=True, keep_date_col=True, header=0,
                     names=[col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_ws+'_dD',	col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C,	col_d13C,	 col_errd13C, 'slope_d13C',	'r2_d13C', 'res_d13C', 'n_d13C', col_wd+'_d13C', col_ws+'_d13C'])
df_PKE_vo = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-11 data_signatures/PKE/PKE_cleared_prom=100ppb_minH=1986ppb_n=3.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 11, 12, 13], dayfirst=True, keep_date_col=True, header=0,
                     names=[col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_ws+'_dD',	col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C,	col_d13C,	 col_errd13C, 'slope_d13C',	'r2_d13C', 'res_d13C', 'n_d13C', col_wd+'_d13C', col_ws+'_d13C'])

# model peaks
df_pkdD_edgar3 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE-dD_edgar3_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID',	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_ws+'_dD'])
df_pkd13C_edgar3 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE-d13C_edgar3_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -d13C',	'start_'+col_dt_d13C,	'end_'+col_dt_d13C, col_d13C,	col_errd13C,	'slope_d13C',	'r2_d13C', 'res_d13C',	'n_d13C', col_wd+'_d13C', col_ws+'_d13C'])
df_PKE_edgar3 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE_edgar3_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=0, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True)

df_pkdD_edgar4 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE-dD_edgar4_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -dD',	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_ws+'_dD'])
df_pkd13C_edgar4 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE-d13C_edgar4_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -d13C',	'start_'+col_dt_d13C,	'end_'+col_dt_d13C, col_d13C,	col_errd13C,	'slope_d13C',	'r2_d13C', 'res_d13C',	'n_d13C', col_wd+'_d13C', col_ws+'_d13C'])
df_PKE_edgar4 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE_edgar4_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=0, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True)

df_pkdD_tno3 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE-dD_tno3_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID',	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_ws+'_dD'])
df_pkd13C_tno3 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE-d13C_tno3_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -d13C',	'start_'+col_dt_d13C,	'end_'+col_dt_d13C, col_d13C,	col_errd13C,	'slope_d13C',	'r2_d13C', 'res_d13C',	'n_d13C', col_wd+'_d13C', col_ws+'_d13C'])
df_PKE_tno3 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE_tno3_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=0, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True)

df_pkdD_tno4 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE-dD_tno4_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID',	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_ws+'_dD'])
df_pkd13C_tno4 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE-d13C_tno4_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -d13C',	'start_'+col_dt_d13C,	'end_'+col_dt_d13C, col_d13C,	col_errd13C,	'slope_d13C',	'r2_d13C', 'res_d13C',	'n_d13C', col_wd+'_d13C', col_ws+'_d13C'])
df_PKE_tno4 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE_tno4_prom=100ppb_minH=1984ppb_n=3.csv',  sep=',', index_col=0, parse_dates=[1,2,3], dayfirst=True, keep_date_col=True)

#%%

# Special peaks

# d13C enriched
d1='Jan-08'
t1='2019-01-07 00:00:00'
t2='2019-01-10 00:00:00'
f1=fg.overview_pk(df_IRMS[t1:t2], [df_PKE], [df_PKE_edgar3, df_PKE_tno3], names=['obs.', 'EDGAR calc3', 'TNO calc3'])
f1[0].savefig(f_path+today+'overview_peak_'+d1+'calc3.png', dpi=100)
# 21/01
# Check wind


