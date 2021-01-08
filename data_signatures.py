#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:20:25 2020

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

errdD=2
errd13C=0.1

###############################################################################
# SAVE THE RESULT OF THE DAY IN A SEPERATE FOLDER
###############################################################################

f_path = '/Users/malika/surfdrive/Data/Krakow/Roof air/figures/'
t_path = '/Users/malika/surfdrive/Data/Krakow/Roof air/tables/'
today='2021-01-08 data_signatures/'

#%%
############################################
# GET DATA
############################################

# IRMS data
df_all=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Input files/KRK_merged_re-calibrated_corrected-IRMS.csv', sep=',', index_col=col_dt, parse_dates=True, dayfirst=True, keep_date_col=True)

dfdD = pd.concat([df_all[col_MR_dD], df_all[col_dD], df_all[col_errMR_dD], df_all[col_errdD], df_all[col_wd], df_all[col_ws]], axis=1).dropna(subset=[col_MR_dD, col_dD])
dfdD_vo = pd.concat([df_all[col_MR_dD+'_vo'], df_all[col_dD], df_all[col_errMR_dD], df_all[col_errdD], df_all[col_wd], df_all[col_ws]], axis=1).dropna(subset=[col_MR_dD+'_vo', col_dD])
dfdD_v2 = pd.concat([df_all[col_MR_dD+'_v2'], df_all[col_dD], df_all[col_errMR_dD], df_all[col_errdD], df_all[col_wd], df_all[col_ws]], axis=1).dropna(subset=[col_MR_dD+'_v2', col_dD])

dfd13C = pd.concat([df_all[col_MR_d13C], df_all[col_d13C], df_all[col_errMR_d13C], df_all[col_errd13C], df_all[col_wd], df_all[col_ws]], axis=1).dropna(subset=[col_MR_d13C, col_d13C])
dfd13C_vo = pd.concat([df_all[col_MR_d13C+'_vo'], df_all[col_d13C], df_all[col_errMR_d13C], df_all[col_errd13C], df_all[col_wd], df_all[col_ws]], axis=1).dropna(subset=[col_MR_d13C+'_vo', col_d13C])
dfd13C_v2 = pd.concat([df_all[col_MR_d13C+'_v2'], df_all[col_d13C], df_all[col_errMR_d13C], df_all[col_errd13C], df_all[col_wd], df_all[col_ws]], axis=1).dropna(subset=[col_MR_d13C+'_v2', col_d13C])
#%%
# Model data
df_Wmod=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Modeled/wind_speed_direction_KRK_20180914_20190314_CHIMERE.csv', sep=';', index_col=0, parse_dates=True)

df_edgar=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/model/Krakow_CH4_EDGARv50_processed.csv', sep=',', index_col=0, parse_dates=True, dayfirst=True, keep_date_col=True)

dfdD_edgar5 = pd.concat([df_edgar['Total'], df_edgar['dD calc5'], df_edgar[col_wd], df_edgar[col_ws]], axis=1)
dfd13C_edgar5 = pd.concat([df_edgar['Total'], df_edgar['d13C calc5'], df_edgar[col_wd], df_edgar[col_ws]], axis=1)

dfdD_edgar6 = pd.concat([df_edgar['Total'], df_edgar['dD calc6'], df_edgar[col_wd], df_edgar[col_ws]], axis=1)
dfd13C_edgar6 = pd.concat([df_edgar['Total'], df_edgar['d13C calc6'], df_edgar[col_wd], df_edgar[col_ws]], axis=1)

dfdD_edgar7 = pd.concat([df_edgar['Total'], df_edgar['dD calc7'], df_edgar[col_wd], df_edgar[col_ws]], axis=1)
dfd13C_edgar7 = pd.concat([df_edgar['Total'], df_edgar['d13C calc7'], df_edgar[col_wd], df_edgar[col_ws]], axis=1)

dfdD_edgar8 = pd.concat([df_edgar['Total'], df_edgar['dD calc8'], df_edgar[col_wd], df_edgar[col_ws]], axis=1)
dfd13C_edgar8 = pd.concat([df_edgar['Total'], df_edgar['d13C calc8'], df_edgar[col_wd], df_edgar[col_ws]], axis=1)

df_tno=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/model/Krakow_CH4_TNO_CAMS_REGv221_processed.csv', sep=',', index_col=0, parse_dates=True, dayfirst=True, keep_date_col=True)

dfdD_tno5 = pd.concat([df_tno['Total'], df_tno['dD calc5'], df_tno[col_wd], df_tno[col_ws]], axis=1)
dfd13C_tno5 = pd.concat([df_tno['Total'], df_tno['d13C calc5'], df_tno[col_wd], df_tno[col_ws]], axis=1)

dfdD_tno6 = pd.concat([df_tno['Total'], df_tno['dD calc6'], df_tno[col_wd], df_tno[col_ws]], axis=1)
dfd13C_tno6 = pd.concat([df_tno['Total'], df_tno['d13C calc6'], df_tno[col_wd], df_tno[col_ws]], axis=1)

dfdD_tno7 = pd.concat([df_tno['Total'], df_tno['dD calc7'], df_tno[col_wd], df_tno[col_ws]], axis=1)
dfd13C_tno7 = pd.concat([df_tno['Total'], df_tno['d13C calc7'], df_tno[col_wd], df_tno[col_ws]], axis=1)

dfdD_tno8 = pd.concat([df_tno['Total'], df_tno['dD calc8'], df_tno[col_wd], df_tno[col_ws]], axis=1)
dfd13C_tno8 = pd.concat([df_tno['Total'], df_tno['d13C calc8'], df_tno[col_wd], df_tno[col_ws]], axis=1)

#%%
############################################
# AVERAGE SIGNATURES
############################################

# Keeling plots
reg_dD = fun.regression(dfdD, iso='D', graph=True, model='IRMS')
reg_dD_v2 = fun.regression(dfdD_v2, iso='D', graph=False)
reg_dD_vo = fun.regression(dfdD_vo, iso='D', graph=False)

reg_d13C = fun.regression(dfd13C, iso='13C', graph=True, model='IRMS')
reg_d13C_v2 = fun.regression(dfd13C_v2, iso='13C', graph=False)
reg_d13C_vo = fun.regression(dfd13C_vo, iso='13C', graph=False)

# Miller-Tans
MT_dD = fun.regression(dfdD, iso='D', graph=True, model='IRMS', MT=True)
MT_dD_v2 = fun.regression(dfdD_v2, iso='D', graph=True, model='IRMS', MT=True)
MT_dD_vo = fun.regression(dfdD_vo, iso='D', graph=True, model='IRMS', MT=True)

MT_d13C = fun.regression(dfd13C, iso='13C', graph=True, model='IRMS', MT=True)
MT_d13C_v2 = fun.regression(dfd13C_v2, iso='13C', graph=True, model='IRMS', MT=True)
MT_d13C_vo = fun.regression(dfd13C_vo, iso='13C', graph=True, model='IRMS', MT=True)

# Model regression - Keeling plots
reg_dD_edgar5 = fun.regression(dfdD_edgar5, iso='D', graph=True, model='edgar')
reg_d13C_edgar5 = fun.regression(dfd13C_edgar5, iso='13C', graph=True, model='edgar')
reg_dD_tno5 = fun.regression(dfdD_tno5, iso='D', graph=True, model='tno')
reg_d13C_tno5 = fun.regression(dfd13C_tno5, iso='13C', graph=True, model='tno')

reg_dD_edgar6 = fun.regression(dfdD_edgar6, iso='D', graph=True, model='edgar')
reg_d13C_edgar6 = fun.regression(dfd13C_edgar6, iso='13C', graph=True, model='edgar')
reg_dD_tno6 = fun.regression(dfdD_tno6, iso='D', graph=True, model='tno')
reg_d13C_tno6 = fun.regression(dfd13C_tno6, iso='13C', graph=True, model='tno')

reg_dD_edgar7 = fun.regression(dfdD_edgar7, iso='D', graph=True, model='edgar')
reg_d13C_edgar7 = fun.regression(dfd13C_edgar7, iso='13C', graph=True, model='edgar')
reg_dD_tno7 = fun.regression(dfdD_tno7, iso='D', graph=True, model='tno')
reg_d13C_tno7 = fun.regression(dfd13C_tno7, iso='13C', graph=True, model='tno')

reg_dD_edgar8 = fun.regression(dfdD_edgar8, iso='D', graph=True, model='edgar')
reg_d13C_edgar8 = fun.regression(dfd13C_edgar8, iso='13C', graph=True, model='edgar')
reg_dD_tno8 = fun.regression(dfdD_tno8, iso='D', graph=True, model='tno')
reg_d13C_tno8 = fun.regression(dfd13C_tno8, iso='13C', graph=True, model='tno')

# Model regression - Miller-Tans plots
MT_dD_edgar5 = fun.regression(dfdD_edgar5, iso='D', graph=True, model='edgar', MT=True)
MT_d13C_edgar5 = fun.regression(dfd13C_edgar5, iso='13C', graph=True, model='edgar', MT=True)
MT_dD_tno5 = fun.regression(dfdD_tno5, iso='D', graph=True, model='tno', MT=True)
MT_d13C_tno5 = fun.regression(dfd13C_tno5, iso='13C', graph=True, model='tno', MT=True)

MT_dD_edgar6 = fun.regression(dfdD_edgar6, iso='D', graph=True, model='edgar', MT=True)
MT_d13C_edgar6 = fun.regression(dfd13C_edgar6, iso='13C', graph=True, model='edgar', MT=True)
MT_dD_tno6 = fun.regression(dfdD_tno6, iso='D', graph=True, model='tno', MT=True)
MT_d13C_tno6 = fun.regression(dfd13C_tno6, iso='13C', graph=True, model='tno', MT=True)

MT_dD_edgar7 = fun.regression(dfdD_edgar7, iso='D', graph=True, model='edgar', MT=True)
MT_d13C_edgar7 = fun.regression(dfd13C_edgar7, iso='13C', graph=True, model='edgar', MT=True)
MT_dD_tno7 = fun.regression(dfdD_tno7, iso='D', graph=True, model='tno', MT=True)
MT_d13C_tno7 = fun.regression(dfd13C_tno7, iso='13C', graph=True, model='tno', MT=True)

MT_dD_edgar8 = fun.regression(dfdD_edgar8, iso='D', graph=True, model='edgar', MT=True)
MT_d13C_edgar8 = fun.regression(dfd13C_edgar8, iso='13C', graph=True, model='edgar', MT=True)
MT_dD_tno8 = fun.regression(dfdD_tno8, iso='D', graph=True, model='tno', MT=True)
MT_d13C_tno8 = fun.regression(dfd13C_tno8, iso='13C', graph=True, model='tno', MT=True)

# DataFrame to compare the different x(CH4) corrections
df_sig = pd.DataFrame({'KP sig '+col_d13C: [reg_d13C[1], reg_d13C_v2[1], reg_d13C_vo[1], reg_d13C_edgar5[1], reg_d13C_tno5[1], reg_d13C_edgar6[1], reg_d13C_tno6[1], reg_d13C_edgar7[1], reg_d13C_tno7[1], reg_d13C_edgar8[1], reg_d13C_tno8[1]], 'KP sig '+col_dD: [reg_dD[1], reg_dD_v2[1], reg_dD_vo[1], reg_dD_edgar5[1], reg_dD_tno5[1], reg_dD_edgar6[1], reg_dD_tno6[1], reg_dD_edgar7[1], reg_dD_tno7[1], reg_dD_edgar8[1], reg_dD_tno8[1]],
                       'KP sig '+col_errd13C: [reg_d13C[3], reg_d13C_v2[3], reg_d13C_vo[3], reg_d13C_edgar5[3], reg_d13C_tno5[3], reg_d13C_edgar6[3], reg_d13C_tno6[3], reg_d13C_edgar7[3], reg_d13C_tno7[3], reg_d13C_edgar8[3], reg_d13C_tno8[3]], 'KP sig '+col_errdD: [reg_dD[3], reg_dD_v2[3], reg_dD_vo[3], reg_dD_edgar5[3], reg_dD_tno5[3], reg_dD_edgar6[3], reg_dD_tno6[3], reg_dD_edgar7[3], reg_dD_tno7[3], reg_dD_edgar8[3], reg_dD_tno8[3]], 
                       'KP slope '+col_d13C: [reg_d13C[0], reg_d13C_v2[0], reg_d13C_vo[0], reg_d13C_edgar5[0], reg_d13C_tno5[0], reg_d13C_edgar6[0], reg_d13C_tno6[0], reg_d13C_edgar7[0], reg_d13C_tno7[0], reg_d13C_edgar8[0], reg_d13C_tno8[0]], 'KP slope '+col_dD: [reg_dD[0], reg_dD_v2[0], reg_dD_vo[0], reg_dD_edgar5[0], reg_dD_tno5[0], reg_dD_edgar6[0], reg_dD_tno6[0], reg_dD_edgar7[0], reg_dD_tno7[0], reg_dD_edgar8[0], reg_dD_tno8[0]],
                       'MT sig '+col_d13C: [MT_d13C[0], MT_d13C_v2[0], MT_d13C_vo[0], MT_d13C_edgar5[0], MT_d13C_tno5[0], MT_d13C_edgar6[0], MT_d13C_tno6[0], MT_d13C_edgar7[0], MT_d13C_tno7[0], MT_d13C_edgar8[0], MT_d13C_tno8[0]], 'MT sig '+col_dD: [MT_dD[0], MT_dD_v2[0], MT_dD_vo[0], MT_dD_edgar5[0], MT_dD_tno5[0], MT_dD_edgar6[0], MT_dD_tno6[0], MT_dD_edgar7[0], MT_dD_tno7[0], MT_dD_edgar8[0], MT_dD_tno8[0]],
                       'MT sig '+col_errd13C: [MT_d13C[2], MT_d13C_v2[2], MT_d13C_vo[2], MT_d13C_edgar5[2], MT_d13C_tno5[2], MT_d13C_edgar6[2], MT_d13C_tno6[2], MT_d13C_edgar7[2], MT_d13C_tno7[2], MT_d13C_edgar8[2], MT_d13C_tno8[2]], 'MT sig '+col_errdD: [MT_dD[2], MT_dD_v2[2], MT_dD_vo[2], MT_dD_edgar5[2], MT_dD_tno5[2], MT_dD_edgar6[2], MT_dD_tno6[2], MT_dD_edgar7[2], MT_dD_tno7[2], MT_dD_edgar8[2], MT_dD_tno8[2]],
                       'MT intercept '+col_d13C: [MT_d13C[1], MT_d13C_v2[1], MT_d13C_vo[1], MT_d13C_edgar5[1], MT_d13C_tno5[1], MT_d13C_edgar6[1], MT_d13C_tno6[1], MT_d13C_edgar7[1], MT_d13C_tno7[1], MT_d13C_edgar8[1], MT_d13C_tno8[1]], 'MT intercept '+col_dD: [MT_dD[1], MT_dD_v2[1], MT_dD_vo[1], MT_dD_edgar5[1], MT_dD_tno5[1], MT_dD_edgar6[1], MT_dD_tno6[1], MT_dD_edgar7[1], MT_dD_tno7[1], MT_dD_edgar8[1], MT_dD_tno8[1]]},
                       index=['original', 'offset', 'cleared', 'EDGAR-calc5', 'TNO-calc5', 'EDGAR-calc6', 'TNO-calc6', 'EDGAR-calc7', 'TNO-calc7', 'EDGAR-calc8', 'TNO-calc8'])
df_sig.to_csv(t_path+today+'avg_signatures_corrections_BCES.csv', encoding=encoding)

#%%
############################################
# AUTOMATIZED PEAK EXTRACTION
############################################

H_obs=np.percentile(df_all[col_MR], 10)
H_edgar=np.percentile(df_edgar_in['Total'], 10)
H_tno=np.percentile(df_tno_in['Total'], 10)
min_prom = 100
min_height=H_obs

min_width= 3
h_relative=0.6
strPKE_obs='prom='+str(min_prom)+'ppb_minH='+str(H_obs)+'ppb_n='+str(min_width)
strPKE_edgar='prom='+str(min_prom)+'ppb_minH='+str(H_edgar)+'ppb_n='+str(min_width)
strPKE_tno='prom='+str(min_prom)+'ppb_minH='+str(H_tno)+'ppb_n='+str(min_width)

#dfdD_PKE, fdD_PKE, paramsdD_PKE = fun.extract_peaks(dfdD, 'D',  min_prom, H_obs, min_width, h_relative, ind_figs=True)
#dfd13C_PKE, fd13C_PKE, paramsd13C_PKE = fun.extract_peaks(dfd13C, '13C',  min_prom, H_obs, min_width, h_relative, ind_figs=True)

#dfdD_PKE_v2, fdD_PKE_v2, paramsdD_PKE_v2 = fun.extract_peaks(dfdD_v2, 'D',  min_prom, H_obs, min_width, h_relative, ind_figs=False, suf='_v2')
#dfd13C_PKE_v2, fd13C_PKE_v2, paramsd13C_PKE_v2 = fun.extract_peaks(dfd13C_v2, '13C',  min_prom, H_obs, min_width, h_relative,ind_figs=False, suf='_v2')

#dfdD_PKE_vo, fdD_PKE_vo, paramsdD_PKE_vo = fun.extract_peaks(dfdD_vo, 'D',  min_prom, H_obs, min_width, h_relative, ind_figs=False, suf='_vo')
#dfd13C_PKE_vo, fd13C_PKE_vo, paramsd13C_PKE_vo = fun.extract_peaks(dfd13C_vo, '13C',  min_prom, H_obs, min_width, h_relative, ind_figs=False, suf='_vo')

dfdD_PKE_edgar5, fdD_PKE_edgar5, paramsdD_PKE_edgar5 = fun.extract_peaks(dfdD_edgar5, 'D', min_prom, H_edgar, min_width, h_relative, ind_figs=False, model=1)
dfd13C_PKE_edgar5, fd13C_PKE_edgar5, paramsd13C_PKE_edgar5 = fun.extract_peaks(dfd13C_edgar5, '13C', min_prom, H_edgar, min_width, h_relative, ind_figs=False, model=1)

dfdD_PKE_tno5, fdD_PKE_tno5, paramsdD_PKE_tno5 = fun.extract_peaks(dfdD_tno5, 'D', min_prom, H_tno, min_width, h_relative, ind_figs=False, model=1)
dfd13C_PKE_tno5, fd13C_PKE_tno5, paramsd13C_PKE_tno5 = fun.extract_peaks(dfd13C_tno5, '13C', min_prom, H_tno, min_width, h_relative, ind_figs=False, model=1)

dfdD_PKE_edgar6, fdD_PKE_edgar6, paramsdD_PKE_edgar6 = fun.extract_peaks(dfdD_edgar6, 'D', min_prom, H_edgar, min_width, h_relative, ind_figs=False, model=1)
dfd13C_PKE_edgar6, fd13C_PKE_edgar6, paramsd13C_PKE_edgar6 = fun.extract_peaks(dfd13C_edgar6, '13C', min_prom, H_edgar, min_width, h_relative, ind_figs=False, model=1)

dfdD_PKE_tno6, fdD_PKE_tno6, paramsdD_PKE_tno6 = fun.extract_peaks(dfdD_tno6, 'D', min_prom, H_tno, min_width, h_relative, ind_figs=False, model=1)
dfd13C_PKE_tno6, fd13C_PKE_tno6, paramsd13C_PKE_tno6 = fun.extract_peaks(dfd13C_tno6, '13C', min_prom, H_tno, min_width, h_relative, ind_figs=False, model=1)

dfdD_PKE_edgar7, fdD_PKE_edgar7, paramsdD_PKE_edgar7 = fun.extract_peaks(dfdD_edgar7, 'D', min_prom, H_edgar, min_width, h_relative, ind_figs=False, model=1)
dfd13C_PKE_edgar7, fd13C_PKE_edgar7, paramsd13C_PKE_edgar7 = fun.extract_peaks(dfd13C_edgar7, '13C', min_prom, H_edgar, min_width, h_relative, ind_figs=False, model=1)

dfdD_PKE_tno7, fdD_PKE_tno7, paramsdD_PKE_tno7 = fun.extract_peaks(dfdD_tno7, 'D', min_prom, H_tno, min_width, h_relative, ind_figs=False, model=1)
dfd13C_PKE_tno7, fd13C_PKE_tno7, paramsd13C_PKE_tno7 = fun.extract_peaks(dfd13C_tno7, '13C', min_prom, H_tno, min_width, h_relative, ind_figs=False, model=1)

dfdD_PKE_edgar8, fdD_PKE_edgar8, paramsdD_PKE_edgar8 = fun.extract_peaks(dfdD_edgar8, 'D', min_prom, H_edgar, min_width, h_relative, ind_figs=False, model=1)
dfd13C_PKE_edgar8, fd13C_PKE_edgar8, paramsd13C_PKE_edgar8 = fun.extract_peaks(dfd13C_edgar8, '13C', min_prom, H_edgar, min_width, h_relative, ind_figs=False, model=1)

dfdD_PKE_tno8, fdD_PKE_tno8, paramsdD_PKE_tno8 = fun.extract_peaks(dfdD_tno8, 'D', min_prom, H_tno, min_width, h_relative, ind_figs=False, model=1)
dfd13C_PKE_tno8, fd13C_PKE_tno8, paramsd13C_PKE_tno8 = fun.extract_peaks(dfd13C_tno8, '13C', min_prom, H_tno, min_width, h_relative, ind_figs=False, model=1)

#%%

# Clear the PKE peaks from crazy data
crazy_d13C = [-100, 0]
crazy_dD = [-600, 0]
crazy_errd13C = 30
crazy_errdD = 200

#dfd13C_PKE, dfdD_PKE = fun.clear_crazy(dfd13C_PKE, crazy_d13C, crazy_errd13C, dfdD_PKE, crazy_dD, crazy_errdD)
#dfd13C_PKE_v2, dfdD_PKE_v2 = fun.clear_crazy(dfd13C_PKE_v2, crazy_d13C, crazy_errd13C, dfdD_PKE_v2, crazy_dD, crazy_errdD, suf='_v2')
#dfd13C_PKE_vo, dfdD_PKE_vo = fun.clear_crazy(dfd13C_PKE_vo, crazy_d13C, crazy_errd13C, dfdD_PKE_vo, crazy_dD, crazy_errdD, suf='_vo')

dfd13C_PKE_edgar5, dfdD_PKE_edgar5 = fun.clear_crazy(dfd13C_PKE_edgar5, crazy_d13C, crazy_errd13C, dfdD_PKE_edgar5, crazy_dD, crazy_errdD)
dfd13C_PKE_tno5, dfdD_PKE_tno5 = fun.clear_crazy(dfd13C_PKE_tno5, crazy_d13C, crazy_errd13C, dfdD_PKE_tno5, crazy_dD, crazy_errdD)

dfd13C_PKE_edgar6, dfdD_PKE_edgar6 = fun.clear_crazy(dfd13C_PKE_edgar6, crazy_d13C, crazy_errd13C, dfdD_PKE_edgar6, crazy_dD, crazy_errdD)
dfd13C_PKE_tno6, dfdD_PKE_tno6 = fun.clear_crazy(dfd13C_PKE_tno6, crazy_d13C, crazy_errd13C, dfdD_PKE_tno6, crazy_dD, crazy_errdD)

dfd13C_PKE_edgar7, dfdD_PKE_edgar7 = fun.clear_crazy(dfd13C_PKE_edgar7, crazy_d13C, crazy_errd13C, dfdD_PKE_edgar7, crazy_dD, crazy_errdD)
dfd13C_PKE_tno7, dfdD_PKE_tno7 = fun.clear_crazy(dfd13C_PKE_tno7, crazy_d13C, crazy_errd13C, dfdD_PKE_tno7, crazy_dD, crazy_errdD)

dfd13C_PKE_edgar8, dfdD_PKE_edgar8 = fun.clear_crazy(dfd13C_PKE_edgar8, crazy_d13C, crazy_errd13C, dfdD_PKE_edgar8, crazy_dD, crazy_errdD)
dfd13C_PKE_tno8, dfdD_PKE_tno8 = fun.clear_crazy(dfd13C_PKE_tno8, crazy_d13C, crazy_errd13C, dfdD_PKE_tno8, crazy_dD, crazy_errdD)

#%%
# Save PKE data ...

# Obs
#dfdD_PKE.to_csv(t_path+today+'PKE-dD_original_'+strPKE_obs+'.csv', encoding=encoding)
#dfd13C_PKE.to_csv(t_path+today+'PKE-d13C_original_'+strPKE_obs+'.csv', encoding=encoding)

#dfdD_PKE_v2.to_csv(t_path+today+'PKE-dD_offset_'+strPKE_obs+'.csv', encoding=encoding)
#dfd13C_PKE_v2.to_csv(t_path+today+'PKE-d13C_offset_'+strPKE_obs+'.csv', encoding=encoding)

#dfdD_PKE_vo.to_csv(t_path+today+'PKE-dD_cleared_'+strPKE_obs+'.csv', encoding=encoding)
#dfd13C_PKE_vo.to_csv(t_path+today+'PKE-d13C_cleared_'+strPKE_obs+'.csv', encoding=encoding)

#pd.concat([dfdD_PKE, dfd13C_PKE], axis=1).to_csv(t_path+today+'PKE_original_'+strPKE_obs+'.csv', encoding=encoding)
#pd.concat([dfdD_PKE_v2, dfd13C_PKE_v2], axis=1).to_csv(t_path+today+'PKE_offset_'+strPKE_obs+'.csv', encoding=encoding)
#pd.concat([dfdD_PKE_vo, dfd13C_PKE_vo], axis=1).to_csv(t_path+today+'PKE_cleared_'+strPKE_obs+'.csv', encoding=encoding)

# Model
dfdD_PKE_edgar5.to_csv(t_path+today+'PKE-dD_edgar5_'+strPKE_edgar+'.csv', encoding=encoding)
dfd13C_PKE_edgar5.to_csv(t_path+today+'PKE-d13C_edgar5_'+strPKE_edgar+'.csv', encoding=encoding)

dfdD_PKE_tno5.to_csv(t_path+today+'PKE-dD_tno5_'+strPKE_tno+'.csv', encoding=encoding)
dfd13C_PKE_tno5.to_csv(t_path+today+'PKE-d13C_tno5_'+strPKE_tno+'.csv', encoding=encoding)

dfdD_PKE_edgar6.to_csv(t_path+today+'PKE-dD_edgar6_'+strPKE_edgar+'.csv', encoding=encoding)
dfd13C_PKE_edgar6.to_csv(t_path+today+'PKE-d13C_edgar6_'+strPKE_edgar+'.csv', encoding=encoding)

dfdD_PKE_tno6.to_csv(t_path+today+'PKE-dD_tno6_'+strPKE_tno+'.csv', encoding=encoding)
dfd13C_PKE_tno6.to_csv(t_path+today+'PKE-d13C_tno6_'+strPKE_tno+'.csv', encoding=encoding)

dfdD_PKE_edgar7.to_csv(t_path+today+'PKE-dD_edgar7_'+strPKE_edgar+'.csv', encoding=encoding)
dfd13C_PKE_edgar7.to_csv(t_path+today+'PKE-d13C_edgar7_'+strPKE_edgar+'.csv', encoding=encoding)

dfdD_PKE_tno7.to_csv(t_path+today+'PKE-dD_tno7_'+strPKE_tno+'.csv', encoding=encoding)
dfd13C_PKE_tno7.to_csv(t_path+today+'PKE-d13C_tno7_'+strPKE_tno+'.csv', encoding=encoding)

dfdD_PKE_edgar8.to_csv(t_path+today+'PKE-dD_edgar8_'+strPKE_edgar+'.csv', encoding=encoding)
dfd13C_PKE_edgar8.to_csv(t_path+today+'PKE-d13C_edgar8_'+strPKE_edgar+'.csv', encoding=encoding)

dfdD_PKE_tno8.to_csv(t_path+today+'PKE-dD_tno8_'+strPKE_tno+'.csv', encoding=encoding)
dfd13C_PKE_tno8.to_csv(t_path+today+'PKE-d13C_tno8_'+strPKE_tno+'.csv', encoding=encoding)

# The datatimes are the exact sames for the model! Because x(ch4) is the same for both isotoopes, and there is no gap in time series.
# Use another function than concat to have only 1 copy of the time limits of the peaks
pd.merge(dfdD_PKE_edgar5, dfd13C_PKE_edgar5, on=['DateTime', 'start_DateTime', 'end_DateTime'], suffixes=['_dD', '_d13C']).to_csv(t_path+today+'PKE_edgar5_'+strPKE_edgar+'.csv', encoding=encoding)
pd.merge(dfdD_PKE_tno5, dfd13C_PKE_tno5, on=['DateTime', 'start_DateTime', 'end_DateTime'], suffixes=['_dD', '_d13C']).to_csv(t_path+today+'PKE_tno5_'+strPKE_tno+'.csv', encoding=encoding)

pd.merge(dfdD_PKE_edgar6, dfd13C_PKE_edgar6, on=['DateTime', 'start_DateTime', 'end_DateTime'], suffixes=['_dD', '_d13C']).to_csv(t_path+today+'PKE_edgar6_'+strPKE_edgar+'.csv', encoding=encoding)
pd.merge(dfdD_PKE_tno6, dfd13C_PKE_tno6, on=['DateTime', 'start_DateTime', 'end_DateTime'], suffixes=['_dD', '_d13C']).to_csv(t_path+today+'PKE_tno6_'+strPKE_tno+'.csv', encoding=encoding)

pd.merge(dfdD_PKE_edgar7, dfd13C_PKE_edgar7, on=['DateTime', 'start_DateTime', 'end_DateTime'], suffixes=['_dD', '_d13C']).to_csv(t_path+today+'PKE_edgar7_'+strPKE_edgar+'.csv', encoding=encoding)
pd.merge(dfdD_PKE_tno7, dfd13C_PKE_tno7, on=['DateTime', 'start_DateTime', 'end_DateTime'], suffixes=['_dD', '_d13C']).to_csv(t_path+today+'PKE_tno7_'+strPKE_tno+'.csv', encoding=encoding)

pd.merge(dfdD_PKE_edgar8, dfd13C_PKE_edgar8, on=['DateTime', 'start_DateTime', 'end_DateTime'], suffixes=['_dD', '_d13C']).to_csv(t_path+today+'PKE_edgar8_'+strPKE_edgar+'.csv', encoding=encoding)
pd.merge(dfdD_PKE_tno8, dfd13C_PKE_tno8, on=['DateTime', 'start_DateTime', 'end_DateTime'], suffixes=['_dD', '_d13C']).to_csv(t_path+today+'PKE_tno8_'+strPKE_tno+'.csv', encoding=encoding)

#%%
# ... & figures
#fdD_PKE.savefig(f_path+today+'PKE/dD_PKE_original_'+strPKE+'.png', dpi=300)
#fd13C_PKE.savefig(f_path+today+'PKE/d13C_PKE_original_'+strPKE+'.png', dpi=300)

#fdD_PKE_v2.savefig(f_path+today+'PKE/dD_PKE_offset_'+strPKE+'.png', dpi=300)
#fd13C_PKE_v2.savefig(f_path+today+'PKE/d13C_PKE_offset_'+strPKE+'.png', dpi=300)

#fdD_PKE_vo.savefig(f_path+today+'PKE/dD_PKE_cleared_'+strPKE+'.png', dpi=300)
#fd13C_PKE_vo.savefig(f_path+today+'PKE/d13C_PKE_cleared_'+strPKE+'.png', dpi=300)

fdD_PKE_edgar5.savefig(f_path+today+'dD_PKE_edgar_'+strPKE_edgar+'.png', dpi=300)
fd13C_PKE_edgar5.savefig(f_path+today+'d13C_PKE_edgar_'+strPKE_edgar+'.png', dpi=300)

fdD_PKE_tno5.savefig(f_path+today+'dD_PKE_tno_'+strPKE_tno+'.png', dpi=300)
fd13C_PKE_tno5.savefig(f_path+today+'d13C_PKE_tno_'+strPKE_tno+'.png', dpi=300)