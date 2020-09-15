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
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os, shutil
import scipy as sc
from scipy import stats
from scipy.optimize import curve_fit

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

col_wd='averageWindDirection'
col_ws='averageWindSpeed'

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

f_path = '/Users/malika/surfdrive/Data/Krakow/Roof air/figures/'
t_path = '/Users/malika/surfdrive/Data/Krakow/Roof air/tables/'
today='2020-09-02 data_init/'

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
      df_all[col_d13C].dropna().index, df_all[col_d13C].dropna()) # d13Clp
df_all['i_'+col_errd13C] = np.interp(df_all.index, 
      df_all[col_errd13C].dropna().index, df_all[col_errd13C].dropna()) # err_d13C

df_all['i_'+col_dD] = np.interp(df_all.index, 
      df_all[col_dD].dropna().index, df_all[col_dD].dropna()) # dD
df_all['i_'+col_errdD] = np.interp(df_all.index, 
      df_all[col_errdD].dropna().index, df_all[col_errdD].dropna()) # err_dD

# Make a column with all x(CH4) data
df_all[col_MR]=pd.concat([df_all[col_MR_d13C].dropna(), df_all[col_MR_dD].dropna()])
df_all[col_errMR]=pd.concat([df_all[col_errMR_d13C].dropna(), df_all[col_errMR_dD].dropna()])

# Get hourly values for comparison
df_hourly=df_all.resample('H').mean().add_prefix('1h_')

# Add wind data
df_wind_hourly=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Meteo data/data_all.csv', sep=',', index_col=0, parse_dates=True, dayfirst=True, keep_date_col=True)
df_all[col_wd]=np.interp(df_all.index, df_wind_hourly.index, df_wind_hourly[col_wd])
df_all[col_ws]=np.interp(df_all.index, df_wind_hourly.index, df_wind_hourly[col_ws])
df_hourly=df_hourly.join(df_wind_hourly)

# Split the data in 2
df_good=pd.concat([df_all['2018-09-01 00:00':'2019-01-12 14:30'], df_all['2019-02-05 00:00':'2019-02-27 13:00'], df_all['2019-03-08 14:00':'2019-03-14 05:00']], axis=0)
df_bad=pd.concat([df_all['2019-01-12 14:30':'2019-02-05 00:00'], df_all['2019-02-27 13:00':'2019-03-08 14:00']], axis=0)

#%%
############################################
# PICARRO data
############################################

# First part, already averaged by Jarek

end_date_Pic30m=pd.to_datetime('2019-02-01 00:00:00')
df_P_all=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Input files/Picarro_all_30m.csv', sep=',', index_col='date time', parse_dates=True, dayfirst=True)
df_P_all.dropna(inplace=True)
df_P_part1=df_P_all[start_date:end_date_Pic30m]
df_P_part1.insert(0, col_P, df_P_part1[col_MR_P]*1000)           # Convert from ppm to ppb

# Remove the jumps in Picarro data
jump1=['2018-11-09 11:30', '2018-11-09 13:30']
jump1bis=['2018-11-08 15:50', '2018-11-08 16:10']
jump2=['2018-11-21 10:30', '2018-11-21 11:30']
jump3=['2018-11-28 10:00', '2018-11-28 11:00']
jump4=['2018-12-19 11:00', '2018-12-19 13:00']
jump5=['2018-10-12 00:00', '2018-10-15 21:00']

def remove_jumps(data, jumps):
    for j in jumps:
        data=data.drop(data[j[0]:j[1]].index, axis=0)
    return data

df_P_part1=remove_jumps(df_P_part1, [jump1, jump1bis, jump2, jump3, jump4, jump5])

# Save the picarro data like this: cleaned and for the right time period
df_P_part1.to_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/Input files/picarro_cleaned_30m_sep-jan.csv', encoding=encoding)
#%%
# Second part, to average every 30 min

df_P_02=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Picarro/Krakow201902.csv', sep=',', index_col='date', parse_dates=True, dayfirst=True)
df_P_03=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Picarro/Krakow201903.csv', sep=',', index_col='date', parse_dates=True, dayfirst=True)

#def avg_30min(df_raw):
#    start=df_raw.index[0]
#    end=df_raw.index[-1]
#    n_30m=int((end-start)/datetime.timedelta(minutes=30))
#    target_index=[(start + datetime.timedelta(minutes=30*x)) for x in range(n_30m)]
#    df_avg=df_raw.rolling('30min').mean()
#    # get the middle of the time window
#   new_index=df_avg.index-datetime.timedelta(minutes=15)
#    df_avg.index=new_index
#    df_target=pd.DataFrame(index=target_index)
#   df_avg_centered=df_target.join(df_avg)
#   return df_avg_centered
# This whole function can be replaced by:
df_P_avg02=df_P_02.resample('30min').mean()
df_P_avg03=df_P_03.resample('30min').mean()

df_P_part2=pd.concat([df_P_avg02, df_P_avg03], axis=0)[end_date_Pic30m:end_date]
df_P_part2.insert(0, col_P, df_P_part2[col_MR_P]*1000)  

# remove jumps
jump1_=['2019-02-14 11:45', '2019-02-14 13:15']
jump2_=['2019-02-26 13:45', '2019-02-26 15:15']
jump3_=['2019-03-05 10:30', '2019-03-05 16:15']
jump4_=['2019-03-07 10:15', '2019-03-07 15:15']
jump5_=['2019-03-11 12:45', '2019-03-11 14:15']
jump6_=['2019-03-12 08:45', '2019-03-12 18:15']
jump7_=['2019-03-13 12:15', '2019-03-13 13:15']

df_P_part2=remove_jumps(df_P_part2, [jump1_, jump2_, jump3_, jump4_, jump5_, jump6_, jump7_])

df_P_part2.to_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/Input files/picarro_30m_feb-mar.csv', encoding=encoding)

#%%

####################### Final picarro file
df_P=pd.concat([df_P_part1[[col_P, col_MR_P]], df_P_part2[[col_P, col_MR_P]]], axis=0)
df_P.to_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/Input files/picarro_cleaned_30min_all.csv', encoding=encoding)

P_hourly=df_P[col_P].resample('H').mean()

df_hourly_all=df_hourly.join(P_hourly)    # Add the hourly Picarro data to the hourly IRMS data for comparison
df_hourly_all.rename(columns={col_P:'1h_CH4_picarro'}, inplace=True)

# Separate the 2 years for re-scaling x(CH4) data
df_hourly_good=pd.concat([df_hourly_all['2018-09-01 00:00':'2019-01-12 14:30'], df_hourly_all['2019-02-05 00:00':'2019-02-27 13:00'], df_hourly_all['2019-03-08 14:00':'2019-03-14 05:00']], axis=0)
df_hourly_bad=pd.concat([df_hourly_all['2019-01-12 14:30':'2019-02-05 00:00'], df_hourly_all['2019-02-27 13:00':'2019-03-08 14:00']], axis=0)

#%%
############################################
# CORRECTIONS on IRMS x(CH4) ------------------------> SKIP
############################################

# With a linear regression formula: v1 & v3

def rescale(x,y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    slope, intercept, r_value, p_value, std_err = sc.stats.linregress(x[mask], y[mask])
    y_new=(y-intercept)/slope
    fg.correlation([x, x], [y, y_new], namex='x(CH4)', namey='x(CH4)', leg=['v0', 'v1'])
    return y_new, slope, intercept

def rescale_0(x,y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    def func(x, a):
        return a*x
    slope, cov = curve_fit(func, x[mask], y[mask])
    #residuals = ydata- func(xdata, *slope)
    #ss_res = np.sum(residuals**2)
    #ss_tot = np.sum((y[mask]-np.mean(y[mask]))**2)
    #r_squared = 1 - (ss_res / ss_tot)
    y_new=y/slope
    fg.correlation([x, x], [y, y_new], namex='x(CH4)', namey='x(CH4)', leg=['v0', 'v3'])
    return y_new, slope

df_hourly_bad['1h_'+col_MR_d13C+'_v1'], alpha_d13C, beta_d13C = rescale(df_hourly_bad['1h_CH4_picarro'], df_hourly_bad['1h_'+col_MR_d13C])
df_hourly_bad['1h_'+col_MR_dD+'_v1'], alpha_dD, beta_dD = rescale(df_hourly_bad['1h_CH4_picarro'], df_hourly_bad['1h_'+col_MR_dD])

df_hourly_bad['1h_'+col_MR_d13C+'_v3'], slope_d13C = rescale_0(df_hourly_bad['1h_CH4_picarro'], df_hourly_bad['1h_'+col_MR_d13C])
df_hourly_bad['1h_'+col_MR_dD+'_v3'], slope_dD, = rescale_0(df_hourly_bad['1h_CH4_picarro'], df_hourly_bad['1h_'+col_MR_dD])


# apply the correction to the non-hourly averaged data of 2019
df_bad[col_MR_d13C+'_v1']=(df_bad[col_MR_d13C]-beta_d13C)/alpha_d13C
df_bad[col_MR_dD+'_v1']=(df_bad[col_MR_dD]-beta_dD)/alpha_dD
df_bad[col_MR+'_v1']=pd.concat([df_bad[col_MR_d13C+'_v1'].dropna(), df_bad[col_MR_dD+'_v1'].dropna()])

df_bad[col_MR_d13C+'_v3']=df_bad[col_MR_d13C]/slope_d13C
df_bad[col_MR_dD+'_v3']=df_bad[col_MR_dD]/slope_dD
df_bad[col_MR+'_v3']=pd.concat([df_bad[col_MR_d13C+'_v3'].dropna(), df_bad[col_MR_dD+'_v3'].dropna()])

#%%
# With an offset applied every 2 days: v2

# We cut the data in each subset of measurement setup
# The criteria for a new subset is when 2 measures were made within a time difference longer than t_lim 
# But a subset can't be longer than t_max
t_lim=300               # minutes, soit 5h
t_max=2                 # jours
df_bad['subset'], n_sub, dates_sub = fun.cut(df_bad, t_lim, t_max)
df_hourly_bad['subset'], n_sub, dates_sub = fun.cut(df_hourly_bad, t_lim, t_max)

limMR=4000 # limit MR value over which we don't take the difference into account in the mean
# Here it is set higher than the maximum value because we work with hourly averages, so we can compare them better

#['2018-09-14 00:00:00':'2018-10-14 00:00:00']
df_hourly_bad['1h_'+col_MR_d13C+'_v2'], stat_d13C = fun.diff2(df_hourly_bad['1h_CH4_picarro'], df_hourly_bad, '1h_'+col_MR_d13C, limMR, dates_sub)
df_hourly_bad['1h_'+col_MR_dD+'_v2'], stat_dD = fun.diff2(df_hourly_bad['1h_CH4_picarro'], df_hourly_bad, '1h_'+col_MR_dD, limMR, dates_sub)

# apply the corrections to the original x(CH4) values for the bad only
df_bad[col_MR_d13C+'_v2']=fun.corrMR(df_bad, col_MR_d13C, stat_d13C)
df_bad[col_MR_dD+'_v2']=fun.corrMR(df_bad, col_MR_dD, stat_dD)
df_bad[col_MR+'_v2']=pd.concat([df_bad[col_MR_d13C+'_v2'].dropna(), df_bad[col_MR_dD+'_v2'].dropna()]).sort_index()

#%%
# incorporate new series in df_all and df_hourly_all:

#df_good[col_MR_d13C+'_v1']=df_good[col_MR_d13C]     # First write the unchanged versions in the 2018 data table
#df_good[col_MR_dD+'_v1']=df_good[col_MR_dD]
#df_good[col_MR+'_v1']=df_good[col_MR]
#df_good[col_MR_d13C+'_v3']=df_good[col_MR_d13C]
#df_good[col_MR_dD+'_v3']=df_good[col_MR_dD]
#df_good[col_MR+'_v3']=df_good[col_MR]

df_good[col_MR_d13C+'_v2']=df_good[col_MR_d13C]
df_good[col_MR_dD+'_v2']=df_good[col_MR_dD]
df_good[col_MR+'_v2']=df_good[col_MR]
# Without the bad data: vo
df_good[col_MR_d13C+'_vo']=df_good[col_MR_d13C]
df_good[col_MR_dD+'_vo']=df_good[col_MR_dD]
df_good[col_MR+'_vo']=df_good[col_MR]

#%%
# re-update first table
df_all_corr=pd.concat([df_good, df_bad]).sort_index()
df_hourly_corr=df_all_corr.resample('H').mean().add_prefix('1h_')
df_hourly_corr=df_hourly_corr.join(P_hourly)    # Add the hourly Picarro data to the hourly IRMS data for comparison
df_hourly_corr.rename(columns={col_P:'1h_CH4_picarro'}, inplace=True)

#***************************************************************
# Save the overall table and the one with hourly averages
df_all_corr.to_csv(t_path+today+'/KRK_merged_re-calibrated_corrected-data.csv', encoding=encoding)
df_hourly_corr.to_csv(t_path+today+'/KRK_hourly-corrected-data.csv', encoding=encoding)
#***************************************************************

#%%
############################################
# VISUALISE 
############################################

# Correlation between Picarro and IRMS

f1 = fg.correlation([df_hourly_all['1h_CH4_picarro'], df_hourly_all['1h_CH4_picarro'], df_hourly_good['1h_CH4_picarro'], df_hourly_bad['1h_CH4_picarro']], [df_hourly_all['1h_'+col_MR_d13C], df_hourly_all['1h_'+col_MR_dD], df_hourly_good['1h_'+col_MR], df_hourly_bad['1h_'+col_MR]], namex='picarro x(CH4)', namey='IRMS x(CH4)', leg=['all-d13C', 'all-dD', 'only-good', 'only-bad'])
f2 = fg.correlation([df_hourly_all['1h_'+col_MR_d13C]], [df_hourly_all['1h_'+col_MR_dD]], namex='x(CH4) -d13C', namey='x(CH4) -dD')
#f1[0].savefig(f_path+today+'corr_pic-IRMS_xCH4_details.png', dpi=300)
#f2[0].savefig(f_path+today+'corr_d13C-dD_xCH4.png', dpi=300)

# Correlation with corrected values
f3 = fg.correlation([df_hourly_corr['1h_CH4_picarro'], df_hourly_corr['1h_CH4_picarro'], df_hourly_corr['1h_CH4_picarro']], [df_hourly_corr['1h_'+col_MR], df_hourly_corr['1h_'+col_MR+'_v2'], df_hourly_corr['1h_'+col_MR+'_vo']], namex='picarro x(CH4)', namey='IRMS x(CH4)', leg=['original', 'corr-offset', 'cleared'])
f3[0].savefig(f_path+today+'corr_pic-IRMS_xCH4_corrected_new.png', dpi=300)

#%%
# Overview of the data
t_01='2019-01-01 00:00:00'
t_02='2019-02-01 00:00:00'
t_03='2019-03-01 00:00:00'
t_04='2019-04-01 00:00:00'
f01=fg.ch4_zoom(df_all_corr, df_P[col_P], t_01, t_02)
f02=fg.ch4_zoom(df_all_corr, df_P[col_P], t_02, t_03)
f03=fg.ch4_zoom(df_all_corr, df_P[col_P], t_03, t_04)
f01.savefig(f_path+today+'xCH4_01-comparison_offset.png', dpi=300)
f02.savefig(f_path+today+'xCH4_02-comparison_offset.png', dpi=300)
f03.savefig(f_path+today+'xCH4_03-comparison_offset.png', dpi=300)

#%%
# Histograms
h1 = plt.subplots(1,1, figsize=(15,7.5))
plt.hist([df_P[col_P], df_all_corr[col_MR].dropna(), df_all_corr[col_MR+'_v2'].dropna(), df_all_corr[col_MR+'_vo'].dropna()], density=True, color=['black', 'blue', 'red', 'orange'], 
          bins=20)
plt.legend(['picarro', 'original', 'offset', 'cleared'])
h1[0].savefig(f_path+today+'xCH4_hist-comparison_offset.png', dpi=300)
