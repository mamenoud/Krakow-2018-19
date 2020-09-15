#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:29:19 2019

@author: malika
"""


# This script is for already corrected csv data from Picarro, which includes GPS data
#""""""""""""""

# Used for the Krakow campaigns (February and March 2019)


import datetime
now = datetime.datetime.now()
import shutil, os
import re
import csv
import math
import statistics
import sys
import openpyxl
import scipy as sc
from scipy.stats import shapiro
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib import colorbar
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import cartopy.io.shapereader as cshp

import mplleaflet

encoding='utf-8'

#%%

col_time = 'date'
col_CH4 = 'CH4_lgr'
col_lon = 'lon'
col_lat = 'lat'

# Importing the data
# If there is a problem: "find '.DS_Store' -delete"

path_in='/Users/malika/surfdrive/Data/Krakow/Krakow_city/Mobile_data/raw/'
path_out='/Users/malika/surfdrive/Data/Krakow/Krakow_city/Mobile_data/processed/'

dfs_list=[]
for f in os.listdir(path_in+'csv/'):
    df=pd.read_csv(path_in+'csv/'+f, index_col=0, parse_dates=True, dayfirst=True, infer_datetime_format=True)
    df.dropna(how='all', axis=1, inplace=True)
    dfs_list.append(df)
#%%
# AGH COMET 24/05
df_2405=pd.read_table(path_in+'20180524_AGH/AGH_20180524_CoMet_CH4.dat', skiprows=11, sep=';')
df_2405.index=pd.to_datetime(df_2405['Date_UTC'] + ' ' + df_2405['Time_UTC'], format='%Y-%m-%d %H:%M:%S')
df_2405['date']=df_2405.index
df_2405.rename(columns={'latitude':'lat', 'longitude':'lon'}, inplace=True)
df_2405[col_CH4]=df_2405['Value_cal_cor']/1000
df_2405.drop_duplicates(subset=[col_time], inplace=True)
df_2405.to_csv(path_in+'csv/20180524_AGH.csv')
#%%
# AGH last day
df_2905=pd.read_table(path_in+'20180529_AGH.dat', skiprows=11, sep=';')
df_2905.index=pd.to_datetime(df_2905['Date_UTC'] + ' ' + df_2905['Time_UTC'], format='%Y-%m-%d %H:%M:%S')
df_2905['date']=df_2905.index
df_2905.rename(columns={'latitude':'lat', 'longitude':'lon'}, inplace=True)
df_2905[col_CH4]=df_2905['Value_cal_cor']/1000
df_2905.drop_duplicates(subset=[col_time], inplace=True)
df_2905.to_csv(path_in+'csv/20180529_comet-AGH.csv')
#%%
# UU PICARRO 27 & 28 /05 ..... no GPS
dfs_UU=[]
for f in os.listdir(path_in+'20180527_UU/'):
    df=pd.read_table(path_in+'20180527_UU/'+f, delim_whitespace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df.index=pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
    df[col_time]=df.index
    df.rename(columns={'CH4_dry':col_CH4}, inplace=True)
    dfs_UU.append(df)

#%%
# AGH PICARRO 24/05 ..... no GPS
dfs_AGH=[]
for f in os.listdir(path_in+'20180524_AGH/Picarro/'):
    df=pd.read_table(path_in+'20180524_AGH/Picarro/'+f, delim_whitespace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df.index=pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
    df[col_time]=df.index
    df.rename(columns={'CH4_dry':col_CH4}, inplace=True)
    dfs_AGH.append(df)

#%%
# Check the data time series

fig, ax = plt.subplots(1,1, figsize=(20,10))

for df in dfs_list:
    ax.plot(df[col_CH4], '.', label=df.index[0].strftime('%Y-%m-%d'))

plt.yscale('log')
plt.legend()

#%%
# *************************************************************************
# PROCESSING OF THE FILES IN THE CSV FOLDER
# *************************************************************************
# Compute the x(CH4) excess
# Based on a rolling average made hourly to calculate the background
# Then substract the background to the measured x(CH4) 

fig, ax = plt.subplots(1,1, figsize=(20,10))
for df in dfs_list:
    df.sort_index(inplace=True)
    bg=df.rolling('1h').mean()
    df[col_CH4+' excess']=df[col_CH4]-bg[col_CH4]
    sns.kdeplot(df[col_CH4+' excess'], ax=ax, label=df.index[0].strftime('%Y-%m-%d'))
    plt.xscale('log')
    df[col_CH4+' excess-abs']=df[col_CH4+' excess']
    df[col_CH4+' excess-abs'][df[col_CH4+' excess']<0]= 0
    df.to_csv(path_out+df.index[0].strftime('%Y%m%d')+'_processed.csv')
    #df_mean=df.resample('1T', how='mean').dropna()      # Average per minute, because this is the precision we have
    dt=pd.date_range(df.index[0], df.index[-1], freq='10S')
    df_new=pd.DataFrame({'ch4_new':np.interp(dt, df.index, df[col_CH4+' excess-abs']), 
                         col_lat:np.interp(dt, df.index, df[col_lat]), 
                         col_lon:np.interp(dt, df.index, df[col_lon])}, index=dt)
    df_new.to_csv(path_out+df.index[0].strftime('%Y%m%d')+'_new.csv')

#%%
# Set the extend of the map window

def boundaries(table, col_dim):
    
    margin=0.1
    
    min_dim = min(table[col_dim])
    max_dim = max(table[col_dim])
    mid_dim = (max_dim - min_dim)/2
    
    lim_inf = min_dim - margin*mid_dim
    lim_sup = max_dim + margin*mid_dim
    
    return lim_inf, lim_sup

long_min, long_max = boundaries(krakow, col_lon)
lat_min, lat_max = boundaries(krakow, col_lat)

cmap = cm.get_cmap('Spectral_r')


#%%

# For the general plot (Leaflet HTML), we take average values every k measurements to have a lighter map
k=3

kr_round=pd.DataFrame({col_time:krakow[col_time], col_CH4:krakow[col_CH4], col_lon:round(krakow[col_lon], k), col_lat:round(krakow[col_lat], k)})
kr_light=kr_round.groupby([col_lat, col_lon], as_index=False).mean()

si_round=pd.DataFrame({col_time:silesia[col_time], col_CH4:silesia[col_CH4], col_lon:round(silesia[col_lon], k), col_lat:round(silesia[col_lat], k)})  
si_light=si_round.groupby([col_lat, col_lon], as_index=False).mean()

feb_round=pd.DataFrame({col_time:february[col_time], col_CH4:february[col_CH4], col_lon:round(february[col_lon], k), col_lat:round(february[col_lat], k)})
feb_light=feb_round.groupby([col_lat, col_lon], as_index=False).mean()

#%%
def html_map(x, y, z, c, style, title):

    figure, axes = plt.subplots()

    scat = axes.scatter(x, y, c=z, cmap=c, s=100, norm=colors.PowerNorm(1, max(min(z), 1.8), min(max(z), 10), clip=True))
    
    mplleaflet.show(fig=figure, tiles=style, title=title)
    
    mplleaflet.save_html(fig=figure, fileobj=title+'_map.html', tiles=style)
    
    return #figure

html_map(kr_light[col_lon], kr_light[col_lat], kr_light[col_CH4], cmap, 'esri_aerial', 'krakow-city_march_light')
html_map(silesia[col_lon], silesia[col_lat], silesia[col_CH4], cmap, 'esri_aerial', 'silesia_light')
html_map(feb_light[col_lon], feb_light[col_lat], feb_light.index, cmap, 'esri_aerial', 'krakow-city_february_light')

#%%

# Data per day

day1=['2019-02-05 10:00', '2019-02-05 18:00']
day2=['2019-02-06 13:00', '2019-02-06 18:00']
day3=['2019-02-07 13:00', '2019-02-07 15:00']               # I didn't use the picarro!

feb1 = february.loc[day1[0]:day1[1]]
feb2 = february.loc[day2[0]:day2[1]]
feb3 = february.loc[day3[0]:day3[1]]
feb_alldays = [feb1, feb2, feb3]

for d in range(len(feb_alldays)-1):
    html_map(feb_alldays[d][col_lon], feb_alldays[d][col_lat], feb_alldays[d][col_CH4], cmap, 'esri_aerial', 'krakow-city_february_day'+str(d+1))
