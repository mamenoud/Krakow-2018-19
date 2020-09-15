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
today='2020-09-14 data_signatures/'

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
df_edgar_in=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/model/Krakow_CH4_EDGARv50_processed.csv', sep=',', index_col=0, parse_dates=True, dayfirst=True, keep_date_col=True)
df_Wmod=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Modeled/wind_speed_direction_KRK_20180914_20190314_CHIMERE.csv', sep=';', index_col=0, parse_dates=True)
df_edgar=df_edgar_in.join(df_Wmod)

df_dD_edgar = pd.concat([df_edgar['Total'], df_edgar['dD'], df_edgar[col_wd], df_edgar[col_ws]], axis=1)
df_d13C_edgar = pd.concat([df_edgar['Total'], df_edgar['d13C'], df_edgar[col_wd], df_edgar[col_ws]], axis=1)
dfdD_edgar3 = pd.concat([df_edgar['Total'], df_edgar['dD calc3'], df_edgar[col_wd], df_edgar[col_ws]], axis=1)
dfd13C_edgar3 = pd.concat([df_edgar['Total'], df_edgar['d13C calc3'], df_edgar[col_wd], df_edgar[col_ws]], axis=1)
dfdD_edgar4 = pd.concat([df_edgar['Total'], df_edgar['dD calc4'], df_edgar[col_wd], df_edgar[col_ws]], axis=1)
dfd13C_edgar4 = pd.concat([df_edgar['Total'], df_edgar['d13C calc4'], df_edgar[col_wd], df_edgar[col_ws]], axis=1)

#%%
############################################
# AVERAGE SIGNATURES
############################################

reg_dD = fun.regression(dfdD, iso='D', graph=True)
reg_dD_v2 = fun.regression(dfdD_v2, iso='D', graph=True)
reg_dD_vo = fun.regression(dfdD_vo, iso='D', graph=True)

reg_d13C = fun.regression(dfd13C, iso='13C', graph=True)
reg_d13C_v2 = fun.regression(dfd13C_v2, iso='13C', graph=True)
reg_d13C_vo = fun.regression(dfd13C_vo, iso='13C', graph=True)

# Model regression
reg_dD_edgar = fun.regression(df_dD_edgar, iso='D', graph=True)
reg_d13C_edgar = fun.regression(df_d13C_edgar, iso='13C', graph=True)
reg_dD_edgar3 = fun.regression(dfdD_edgar3, iso='D', graph=True)
reg_d13C_edgar3 = fun.regression(dfd13C_edgar3, iso='13C', graph=True)
reg_dD_edgar4 = fun.regression(dfdD_edgar4, iso='D', graph=True)
reg_d13C_edgar4 = fun.regression(dfd13C_edgar4, iso='13C', graph=True)

# DataFrame to compare the different x(CH4) corrections
df_sig = pd.DataFrame({col_d13C: [reg_d13C[1], reg_d13C_v2[1], reg_d13C_vo[1], reg_d13C_edgar[1], reg_d13C_edgar3[1], reg_d13C_edgar4[1]], col_dD: [reg_dD[1], reg_dD_v2[1], reg_dD_vo[1], reg_dD_edgar[1], reg_dD_edgar3[1], reg_dD_edgar4[1]],
                       col_errd13C: [reg_d13C[3], reg_d13C_v2[3], reg_d13C_vo[3], reg_d13C_edgar[3], reg_d13C_edgar3[3], reg_d13C_edgar4[3]], col_errdD: [reg_dD[3], reg_dD_v2[3], reg_dD_vo[3], reg_dD_edgar[3], reg_dD_edgar3[3], reg_dD_edgar4[3]]},
                       index=['original', 'offset', 'cleared', 'EDGAR-original', 'EDGAR-calc3', 'EDGAR-calc4'])
df_sig.to_csv(t_path+today+'avg_signatures_corrections_BCES.csv', encoding=encoding)

#%%
############################################
# MOVING WINDOW KEELING PLOT
############################################

t1='2018-09-01 00:00'
t2='2019-01-12 14:30'
t3='2019-04-01 00:00'
df_temp=pd.concat([df_all[col_MR_dD], df_all[col_dD], df_all[col_errMR_dD], df_all[col_errdD]], axis=1).dropna(subset=[col_MR_dD, col_dD])[t1:t2]

# Parameters
window=12       # time window in hours
n_min=5         # minimum number of datapoint per Keeling plot
fit=0.1
fitd13C=errd13C*fit
fitdD=errdD*fit         # limit value to accept the regression (s=sqrt(res_var))
window_CH4=200  # minimun x(CH4) range for a Keeling plot to be made
bg_CH4=10     # percentile for maximum x(CH4) value to consider as background
strMWKP='dt='+str(window)+'h_n='+str(n_min)+'_fit='+str(fit)+'sq-sc_x-range='+str(window_CH4)+'ppb_x-bg='+str(bg_CH4)+'per'

#dfdD_MW, fdD_MW, errsdD_MW = fun.moving_window(dfdD, 'D', window, n_min, fit, window_CH4, bg_CH4, ind_figs=False)
dfdD_MW_vo, fdD_MW_vo, errsdD_MW_vo = fun.moving_window(dfdD_vo, 'D', window, n_min, fit, window_CH4, bg_CH4)#, wd=None, ind_figs=False)
dfdD_MW_v2, fdD_MW_v2, errsdD_MW_v2 = fun.moving_window(dfdD_v2, 'D', window, n_min, fit, window_CH4, bg_CH4)#, wd=None, ind_figs=False)

#dfd13C_MW, fd13C_MW, errsd13C_MW = fun.moving_window(dfd13C, '13C', window, n_min, fit, window_CH4, bg_CH4, ind_figs=False)#, wd=None, ind_figs=False)
dfd13C_MW_vo, fd13C_MW_vo, errsd13C_MW_vo = fun.moving_window(dfd13C_vo, '13C', window, n_min, fit, window_CH4, bg_CH4)#, wd=None, ind_figs=False)
dfd13C_MW_v2, fd13C_MW_v2, errsd13C_MW_v2 = fun.moving_window(dfd13C_v2, '13C', window, n_min, fit, window_CH4, bg_CH4)#, wd=None, ind_figs=False)

# Model MWKP
#dfdD_MW_edgar, fdD_MW_edgar, errsdD_MW_edgar = fun.moving_window(df_dD_edgar, 'D', window, n_min, fit, window_CH4, bg_CH4, wd=None, ind_figs=False)
#dfd13C_MW_edgar, fd13C_MW_edgar, errsd13C_MW_edgar = fun.moving_window(df_d13C_edgar, '13C', window, n_min, fit, window_CH4, bg_CH4, wd=None, ind_figs=False)
#dfdD_MW_edgar3, fdD_MW_edgar3, errsdD_MW_edgar3 = fun.moving_window(dfdD_edgar3, 'D', window, n_min, fit, window_CH4, bg_CH4, wd=None, ind_figs=False)
#dfd13C_MW_edgar3, fd13C_MW_edgar3, errsd13C_MW_edgar3 = fun.moving_window(dfd13C_edgar3, '13C', window, n_min, fit, window_CH4, bg_CH4, wd=None, ind_figs=False)

#%%

# When concanenating the d13C and dD MWKP signature tables, 
# you get the following column list:
# (int) DateTime,	start_DateTime,	end_DateTime,	dD SMOW,	SD dD,	slope,	r2,	n,	 col_wd, col_ws, DateTime,	start_DateTime,	end_DateTime,	d13C VPDB,	SD d13C,	slope,	r2,	n, col_wd, col_ws
# Re-import the data after saving to rename the columns and distinguish d13C and dD
# ... in next script: 'process_signatures.py'

# Save the data!

# All the MWKP data
pd.concat([dfdD_MW, dfd13C_MW], axis=1).to_csv(t_path+today+'MWKP/MWKP_original_'+strMWKP+'.csv', encoding=encoding)
fdD_MW.savefig(f_path+today+'MWKP/dD_MWKP_original_'+strMWKP+'.png', dpi=300)
fd13C_MW.savefig(f_path+today+'MWKP/d13C_MWKP_original_'+strMWKP+'.png', dpi=300)

pd.concat([dfdD_MW_vo, dfd13C_MW_vo], axis=1).to_csv(t_path+today+'MWKP/MWKP_cleared_'+strMWKP+'.csv', encoding=encoding)
fdD_MW_vo.savefig(f_path+today+'MWKP/dD_MWKP_linear_'+strMWKP+'.png', dpi=300)
fd13C_MW_vo.savefig(f_path+today+'MWKP/d13C_MWKP_linear_'+strMWKP+'.png', dpi=300)

pd.concat([dfdD_MW_v2, dfd13C_MW_v2], axis=1).to_csv(t_path+today+'MWKP/MWKP_offset_'+strMWKP+'.csv', encoding=encoding)
fdD_MW_v2.savefig(f_path+today+'MWKP/dD_MWKP_offset_'+strMWKP+'.png', dpi=300)
fd13C_MW_v2.savefig(f_path+today+'MWKP/d13C_MWKP_offset_'+strMWKP+'.png', dpi=300)

#Model data
pd.concat([dfdD_MW_edgar, dfd13C_MW_edgar], axis=1).to_csv(t_path+today+'MWKP/MWKP_edgar-original_'+strMWKP+'.csv', encoding=encoding)
pd.concat([dfdD_MW_edgar3, dfd13C_MW_edgar3], axis=1).to_csv(t_path+today+'MWKP/MWKP_edgar-calc3_'+strMWKP+'.csv', encoding=encoding)


#%%
############################################
# AUTOMATIZED PEAK EXTRACTION
############################################
min_prom = 100
#min_height= 1986       # for observations
min_height = 1984       # for model with EDGAR
min_width= 3
strPKE='prom='+str(min_prom)+'ppb_minH='+str(min_height)+'ppb_n='+str(min_width)

dfdD_PKE, fdD_PKE, paramsdD_PKE = fun.extract_peaks(dfdD, 'D',  min_prom, min_height, min_width, ind_figs=True)
dfd13C_PKE, fd13C_PKE, paramsd13C_PKE = fun.extract_peaks(dfd13C, '13C',  min_prom, min_height, min_width, ind_figs=True)

dfdD_PKE_v2, fdD_PKE_v2, paramsdD_PKE_v2 = fun.extract_peaks(dfdD_v2, 'D',  min_prom, min_height, min_width,ind_figs=False)
dfd13C_PKE_v2, fd13C_PKE_v2, paramsd13C_PKE_v2 = fun.extract_peaks(dfd13C_v2, '13C',  min_prom, min_height, min_width,ind_figs=False)

dfdD_PKE_vo, fdD_PKE_vo, paramsdD_PKE_vo = fun.extract_peaks(dfdD_vo, 'D',  min_prom, min_height, min_width,ind_figs=False)
dfd13C_PKE_vo, fd13C_PKE_vo, paramsd13C_PKE_vo = fun.extract_peaks(dfd13C_vo, '13C',  min_prom, min_height, min_width,ind_figs=False)

dfdD_PKE_edgar3, fdD_PKE_edgar3, paramsdD_PKE_edgar3 = fun.extract_peaks(dfdD_edgar3, 'D', min_prom, min_height, min_width, ind_figs=False, model=1)
dfd13C_PKE_edgar3, fd13C_PKE_edgar3, paramsd13C_PKE_edgar3 = fun.extract_peaks(dfd13C_edgar3, '13C', min_prom, min_height, min_width, ind_figs=False, model=1)

dfdD_PKE_edgar4, fdD_PKE_edgar4, paramsdD_PKE_edgar4 = fun.extract_peaks(dfdD_edgar4, 'D', min_prom, min_height, min_width, ind_figs=False, model=1)
dfd13C_PKE_edgar4, fd13C_PKE_edgar4, paramsd13C_PKE_edgar4 = fun.extract_peaks(dfd13C_edgar4, '13C', min_prom, min_height, min_width, ind_figs=False, model=1)

#%%

# Clear the PKE peaks from crazy data
crazy_d13C = [-100, 0]
crazy_dD = [-600, 0]
crazy_errd13C = 30
crazy_errdD = 200
dfd13C_PKE, dfdD_PKE = fun.clear_crazy(dfd13C_PKE, crazy_d13C, crazy_errd13C, dfdD_PKE, crazy_dD, crazy_errdD)
dfd13C_PKE_v2, dfdD_PKE_v2 = fun.clear_crazy(dfd13C_PKE_v2, crazy_d13C, crazy_errd13C, dfdD_PKE_v2, crazy_dD, crazy_errdD)
dfd13C_PKE_vo, dfdD_PKE_vo = fun.clear_crazy(dfd13C_PKE_vo, crazy_d13C, crazy_errd13C, dfdD_PKE_vo, crazy_dD, crazy_errdD)

dfd13C_PKE_edgar3, dfdD_PKE_edgar3 = fun.clear_crazy(dfd13C_PKE_edgar3, crazy_d13C, crazy_errd13C, dfdD_PKE_edgar3, crazy_dD, crazy_errdD)
dfd13C_PKE_edgar4, dfdD_PKE_edgar4 = fun.clear_crazy(dfd13C_PKE_edgar4, crazy_d13C, crazy_errd13C, dfdD_PKE_edgar4, crazy_dD, crazy_errdD)

#%%
# Save PKE data ...
dfdD_PKE.to_csv(t_path+today+'PKE/PKE-dD_original_'+strPKE+'.csv', encoding=encoding)
dfd13C_PKE.to_csv(t_path+today+'PKE/PKE-d13C_original_'+strPKE+'.csv', encoding=encoding)

dfdD_PKE_v2.to_csv(t_path+today+'PKE/PKE-dD_offset_'+strPKE+'.csv', encoding=encoding)
dfd13C_PKE_v2.to_csv(t_path+today+'PKE/PKE-d13C_offset_'+strPKE+'.csv', encoding=encoding)

dfdD_PKE_vo.to_csv(t_path+today+'PKE/PKE-dD_cleared_'+strPKE+'.csv', encoding=encoding)
dfd13C_PKE_vo.to_csv(t_path+today+'PKE/PKE-d13C_cleared_'+strPKE+'.csv', encoding=encoding)

dfdD_PKE_edgar3.to_csv(t_path+today+'PKE-dD_edgar3_'+strPKE+'.csv', encoding=encoding)
dfd13C_PKE_edgar3.to_csv(t_path+today+'PKE-d13C_edgar3_'+strPKE+'.csv', encoding=encoding)

dfdD_PKE_edgar4.to_csv(t_path+today+'PKE-dD_edgar4_'+strPKE+'.csv', encoding=encoding)
dfd13C_PKE_edgar4.to_csv(t_path+today+'PKE-d13C_edgar4_'+strPKE+'.csv', encoding=encoding)

pd.concat([dfdD_PKE, dfd13C_PKE], axis=1).to_csv(t_path+today+'PKE/PKE_original_'+strPKE+'.csv', encoding=encoding)
pd.concat([dfdD_PKE_v2, dfd13C_PKE_v2], axis=1).to_csv(t_path+today+'PKE/PKE_offset_'+strPKE+'.csv', encoding=encoding)
pd.concat([dfdD_PKE_vo, dfd13C_PKE_vo], axis=1).to_csv(t_path+today+'PKE/PKE_cleared_'+strPKE+'.csv', encoding=encoding)

pd.concat([dfdD_PKE_edgar3, dfd13C_PKE_edgar3], axis=1).to_csv(t_path+today+'PKE_edgar3_'+strPKE+'.csv', encoding=encoding)
pd.concat([dfdD_PKE_edgar4, dfd13C_PKE_edgar4], axis=1).to_csv(t_path+today+'PKE_edgar4_'+strPKE+'.csv', encoding=encoding)

# ... & figures
fdD_PKE.savefig(f_path+today+'PKE/dD_PKE_original_'+strPKE+'.png', dpi=300)
fd13C_PKE.savefig(f_path+today+'PKE/d13C_PKE_original_'+strPKE+'.png', dpi=300)

fdD_PKE_v2.savefig(f_path+today+'PKE/dD_PKE_offset_'+strPKE+'.png', dpi=300)
fd13C_PKE_v2.savefig(f_path+today+'PKE/d13C_PKE_offset_'+strPKE+'.png', dpi=300)

fdD_PKE_vo.savefig(f_path+today+'PKE/dD_PKE_cleared_'+strPKE+'.png', dpi=300)
fd13C_PKE_vo.savefig(f_path+today+'PKE/d13C_PKE_cleared_'+strPKE+'.png', dpi=300)

fdD_PKE_edgar3.savefig(f_path+today+'dD_PKE_edgar3_'+strPKE+'.png', dpi=300)
fd13C_PKE_edgar3.savefig(f_path+today+'d13C_PKE_edgar3_'+strPKE+'.png', dpi=300)

fdD_PKE_edgar4.savefig(f_path+today+'dD_PKE_edgar4_'+strPKE+'.png', dpi=300)
fd13C_PKE_edgar4.savefig(f_path+today+'d13C_PKE_edgar4_'+strPKE+'.png', dpi=300)
#%%
# Manual removal of outliers
df_MWKP_2019=df_MWKP_2019.loc[df_MWKP_2019[col_d13C]>-100]

############################################
# Make figures
############################################

# Plot the signatures
h1 = fg.hist(df_MWKP_2018[col_d13C], df_MWKP_2018[col_dD], 'Raw signatures', 
             data13C_list=[df_MWKP_2019.loc[df_MWKP_2019[col_d13C]>-100][col_d13C], df_MWKP_2019_v1.loc[df_MWKP_2019_v1[col_d13C]>-100][col_d13C], df_MWKP_2019_v2.loc[df_MWKP_2019_v2[col_d13C]>-100][col_d13C]], 
             dataD_list=[df_MWKP_2019[col_dD], df_MWKP_2019_v1[col_dD], df_MWKP_2019_v2[col_dD]], 
             str_leg=['2018', '2019-original', '2019-linear', '2019-offset'])

# Plot the errors
hr2=fg.hist(dfd13C_MW['r2'], dfdD_MW['r2'], title='all residual variances', data13C_list=[dfd13C_MW_v1['r2'], dfd13C_MW_v2['r2']], dataD_list=[dfdD_MW_v1['r2'], dfdD_MW_v2['r2']], str_leg=['original', 'linear', 'offset'])

# top diagramme
f2D1 = fg.top_diagrams(df_1=df_MWKP_2018[[col_dD, col_errdD, col_d13C, col_errd13C]].reindex(), df_2=df_MWKP_2019[[col_dD, col_errdD, col_d13C, col_errd13C]].reindex(),
                df_3=df_MWKP_2019_v1[[col_dD, col_errdD, col_d13C, col_errd13C]].reindex(), df_4=df_MWKP_2019_v2[[col_dD, col_errdD, col_d13C, col_errd13C]].reindex())

h1.savefig(f_path+today+'hist_MW_comparison_'+strMWKP+'.png', dpi=300)
