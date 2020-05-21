#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 08:29:38 2020

@author: malika
"""


# This script is for long-term data analysis of Krakow data

import numpy as np
import pandas as pd
from pandas import ExcelWriter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os, shutil
import scipy as sc
from scipy import stats

import functions_KR as fun
import figures_KR as fg

plt.rcParams.update({'font.size': 22})

# data format:        filling time   	[CH4]dD-2 (ppb)	   1/MR	       dD-2 SMOW

dateformat='%d/%m/%Y %H:%M'
df_P='%d/%m/%y %H:%M:%S'

encoding='utf-16le'
d13C='\u03B4\u00B9\u00B3C'
dD='\u03B4D'
permil=' [\u2030]'
pm='\u00B1'
delta13C=d13C+' V-PDB'
deltaD=dD+' V-SMOW'

col_dt='filling time UTC'

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

# For pretty strings

strMR_13C='MR_ppb - '+d13C
strMR_D='MR_ppb - '+dD
str_d13C=delta13C+permil
str_dD=deltaD+permil
str_MR=col_P

c1='xkcd:bright blue'
c2='xkcd:rust orange'
c3='xkcd:light purple'
c4='xkcd:iris'
cP='xkcd:grey'
vert='xkcd:green'
vertclair='xkcd:light green'

###############################################################################
# SAVE THE RESULT OF THE DAY IN A SEPERATE FOLDER
###############################################################################

today='2020-05-21 Definite data/'

#%%
############################################
# ALL DATA
############################################

# IRMS data
df_raw=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Input files/KRK_merged_re-calibrated_all-data.csv', sep=',', index_col=col_dt, parse_dates=True, dayfirst=True, keep_date_col=True)
df_raw.dropna(how='all', inplace=True) # drop empty rows

start_date = df_raw.index[0]
end_date = df_raw.index[-1]

# Make interpolated columns 
df_all=df_raw.copy()
df_all['i_'+col_MR_d13C] = np.interp(df_all.index, 
      df_all[col_MR_d13C].dropna().index, df_all[col_MR_d13C].dropna()) # MR -d13C
df_all['i_'+col_errMR_d13C] = np.interp(df_all.index, 
      df_all[col_errMR_d13C].dropna().index, df_all[col_errMR_d13C].dropna()) # err_MR -d13C

df_all['i_'+col_MR_dD] = np.interp(df_all.index, 
      df_all[col_MR_dD].dropna().index, df_all[col_MR_dD].dropna()) # MR -dD
df_all['i_'+col_errMR_dD] = np.interp(df_all.index, 
      df_all[col_errMR_dD].dropna().index, df_all[col_errMR_dD].dropna()) # err_MR -dD

df_all['i_'+col_d13C] = np.interp(df_all.index, 
      df_all[col_d13C].dropna().index, df_all[col_d13C].dropna()) # d13C
df_all['i_'+col_errd13C] = np.interp(df_all.index, 
      df_all[col_errd13C].dropna().index, df_all[col_errd13C].dropna()) # err_d13C

df_all['i_'+col_dD] = np.interp(df_all.index, 
      df_all[col_dD].dropna().index, df_all[col_dD].dropna()) # dD
df_all['i_'+col_errdD] = np.interp(df_all.index, 
      df_all[col_errdD].dropna().index, df_all[col_errdD].dropna()) # err_dD

# Get hourly values for comparison
df_hourly=df_all.resample('H').mean().add_prefix('1h_')

# picarro data
df_P_all=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Input files/Picarro_all_30m.csv', sep=',', index_col='date time', parse_dates=True, dayfirst=True)
df_P_all.dropna(inplace=True)
df_P=df_P_all[start_date:end_date]
df_P.insert(0, col_P, df_P[col_MR_P]*1000)           # Convert from ppm to ppb

P_hourly=df_P[col_P].resample('H').mean()

df_hourly_all=df_hourly.join(P_hourly)    # Add the hourly Picarro data to the hourly IRMS data for comparison
df_hourly_all.rename(columns={col_P:'1h_CH4_picarro'}, inplace=True)

#%%

############################################
# VISUALISE 
############################################

