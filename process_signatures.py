#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 14:00:43 2020

@author: malika
"""


# This script is for long-term data analysis of Krakow data

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
web_purple='#663399'

###############################################################################
# SAVE THE RESULT OF THE DAY IN A SEPERATE FOLDER
###############################################################################

f_path = '/Users/malika/surfdrive/Data/Krakow/Roof air/figures/'
t_path = '/Users/malika/surfdrive/Data/Krakow/Roof air/tables/'
today='2020-11-26 data_signatures/'

#%%
# Observation data
df_IRMS=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Input files/KRK_merged_re-calibrated_corrected-IRMS.csv', sep=',', index_col=col_dt_obs, parse_dates=True, dayfirst=True, keep_date_col=True)

# Get the PKE data
df_PKE = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE_original_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 13, 14, 15], dayfirst=True, keep_date_col=True, header=0,
                     names=[col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_errwd+'_dD', col_ws+'_dD', col_errws+'_dD', col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C,	col_d13C,	 col_errd13C, 'slope_d13C',	'r2_d13C', 'res_d13C', 'n_d13C', col_wd+'_d13C', col_errwd+'_d13C', col_ws+'_d13C', col_errws+'_d13C'])
df_PKE_v2 = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE_offset_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 13, 14, 15], dayfirst=True, keep_date_col=True, header=0,
                     names=[col_dt_dD+'_v2',	'start_'+col_dt_dD+'_v2',	'end_'+col_dt_dD+'_v2', col_dD+'_v2',	col_errdD+'_v2',	'slope_dD'+'_v2',	'r2_dD'+'_v2', 'res_dD',	'n_dD'+'_v2', col_wd+'_dD'+'_v2', col_errwd+'_dD'+'_v2', col_ws+'_dD'+'_v2',	col_errws+'_dD'+'_v2', col_dt_d13C+'_v2',	'start_'+col_dt_d13C+'_v2',	'end_'+col_dt_d13C+'_v2',	col_d13C+'_v2',	 col_errd13C+'_v2', 'slope_d13C'+'_v2',	'r2_d13C'+'_v2', 'res_d13C'+'_v2', 'n_d13C'+'_v2', col_wd+'_d13C'+'_v2', col_errwd+'_d13C'+'_v2', col_ws+'_d13C'+'_v2', col_errws+'_d13C'+'_v2'])
df_PKE_vo = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-11-26 data_signatures/PKE_cleared_prom=100ppb_minH=1986.0ppb_n=3.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 13, 14, 15], dayfirst=True, keep_date_col=True, header=0,
                     names=[col_dt_dD+'_vo',	'start_'+col_dt_dD+'_vo',	'end_'+col_dt_dD+'_vo', col_dD+'_vo',	col_errdD+'_vo',	'slope_dD'+'_vo',	'r2_dD'+'_vo', 'res_dD'+'_vo',	'n_dD'+'_vo', col_wd+'_dD'+'_vo', col_errwd+'_dD'+'_vo', col_ws+'_dD'+'_vo', col_errws+'_dD'+'_vo',	col_dt_d13C+'_vo',	'start_'+col_dt_d13C+'_vo',	'end_'+col_dt_d13C+'_vo',	col_d13C+'_vo',	 col_errd13C+'_vo', 'slope_d13C'+'_vo',	'r2_d13C'+'_vo', 'res_d13C'+'_vo', 'n_d13C'+'_vo', col_wd+'_d13C'+'_vo', col_errwd+'_d13C'+'_vo', col_ws+'_d13C'+'_vo', col_errws+'_d13C'+'_vo'])

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

# Histogram bins for isotopic signatures
d13C_bins=np.linspace(-61, -35, 14)
dD_bins=np.linspace(-350, -90, 14)

# COMPARISON IRMS vs CHIMERE peaks
hist_pk_d13C= plt.figure(figsize=(15,10))
ax_pk_d13C=hist_pk_d13C.add_subplot(111)
sns.histplot(df_PKE[col_d13C],  stat='density', bins=d13C_bins, ax=ax_pk_d13C, label='IRMS obs., n='+str(len(df_PKE[col_d13C].dropna())), color='gray')
sns.kdeplot(df_pkd13C_edgar5[col_d13C], shade=True, ax=ax_pk_d13C, label='EDGAR v5.0, n='+str(len(df_pkd13C_edgar5[col_d13C].dropna())), color=web_red)
sns.kdeplot(df_pkd13C_tno5[col_d13C], shade=True, ax=ax_pk_d13C, label='TNO-CAMS v4.2, n='+str(len(df_pkd13C_tno5[col_d13C].dropna())), color=vertclair)
ax_pk_d13C.set_xlabel(str_d13C)
#ax_pk_d13C.legend(loc='upper left')
hist_pk_d13C.savefig(f_path+today+'hist_peaks-d13C_original-calc5.png', dpi=300, transparent=True)

hist_pk_dD= plt.figure(figsize=(15,10))
ax_pk_dD=hist_pk_dD.add_subplot(111)
sns.histplot(df_PKE[col_dD], stat='density', bins=dD_bins, ax=ax_pk_dD, label='IRMS obs., n='+str(len(df_PKE[col_dD].dropna())), color='gray')
sns.kdeplot(df_pkdD_edgar5[col_dD], shade=True, ax=ax_pk_dD, label='EDGAR v5.0, n='+str(len(df_pkdD_edgar5[col_dD].dropna())), color=web_red)
sns.kdeplot(df_pkdD_tno5[col_dD], shade=True, ax=ax_pk_dD, label='TNO-CAMS v4.2, n='+str(len(df_pkdD_tno5[col_dD].dropna())), color=vertclair)
ax_pk_dD.set_xlabel(str_dD)
ax_pk_dD.legend(loc='upper left')
hist_pk_dD.savefig(f_path+today+'hist_peaks-dD_original-calc5.png', dpi=300, transparent=True)

#%%

# Visualize the peak signatures 

# PKE derived peaks
f1, ax1 = fg.overview_pk(df_IRMS, [df_PKE], [df_PKE_edgar5, df_PKE_tno5], names=['obs -original', 'EDGAR -calc5', 'TNO -calc5'])
f1.savefig(f_path+today+'/overview_PKEpeaks_original&calc3_prom=100ppb_minH=10perc_n=3.png', dpi=300)

#%%
# Merge the peak lists, so that the peaks get 2 isotopes
h=6

# PKE peaks
dfPKE_2D = fun.two_isotopes(df_PKE[[col_dt_dD, 'start_'+col_dt_dD, 'end_'+col_dt_dD, col_dD, col_errdD, col_wd+'_dD', col_errwd+'_dD', col_ws+'_dD', col_errws+'_dD']].dropna(), df_PKE[[col_dt_d13C, 'start_'+col_dt_d13C, 'end_'+col_dt_d13C, col_d13C, col_errd13C, col_wd+'_d13C', col_errwd+'_d13C', col_ws+'_d13C', col_errws+'_d13C']].dropna(), h)

dfPKE_2D_v2 = fun.two_isotopes(df_PKE_v2[[col_dt_dD+'_v2', 'start_'+col_dt_dD+'_v2', 'end_'+col_dt_dD+'_v2', col_dD+'_v2', col_errdD+'_v2', col_wd+'_dD_v2', col_errwd+'_dD_v2', col_ws+'_dD_v2', col_errws+'_dD_v2']].dropna(), df_PKE_v2[[col_dt_d13C+'_v2', 'start_'+col_dt_d13C+'_v2', 'end_'+col_dt_d13C+'_v2', col_d13C+'_v2', col_errd13C+'_v2', col_wd+'_d13C_v2', col_errwd+'_d13C_v2', col_ws+'_d13C_v2', col_errws+'_d13C_v2']].dropna(), h, suf='_v2')

dfPKE_2D_vo = fun.two_isotopes(df_PKE_vo[[col_dt_dD+'_vo', 'start_'+col_dt_dD+'_vo', 'end_'+col_dt_dD+'_vo', col_dD+'_vo', col_errdD+'_vo', col_wd+'_dD_vo', col_errwd+'_dD_vo', col_ws+'_dD_vo', col_errws+'_dD_vo']].dropna(), df_PKE_vo[[col_dt_d13C+'_vo', 'start_'+col_dt_d13C+'_vo', 'end_'+col_dt_d13C+'_vo', col_d13C+'_vo', col_errd13C+'_vo', col_wd+'_d13C_vo', col_errwd+'_d13C_vo', col_ws+'_d13C_vo', col_errws+'_d13C_vo']].dropna(), h, suf='_vo')
#%%

# 2D diagrams
ambient=pd.DataFrame({'avg_d13C':-47.8, 'avg_dD':-90.0}, index=[0])

# Compare the original, offset and cleared
f3=fg.top_diagrams([dfPKE_2D[1], dfPKE_2D_v2[1], dfPKE_2D_vo[1]], ['original PKE', 'offset PKE', 'cleared PKE'], ['', '_v2', '_vo'], ambient=ambient)
f3.savefig(f_path+today+'/top-diagram_PKE-bg_original&offset&cleared.png', dpi=300)

# Compare obs and model
f4=fg.top_diagrams([dfPKE_2D[1], df_PKE_edgar5, df_PKE_tno5], ['obs -original', 'EDGAR -calc5', 'TNO -calc5'], ['', '', ''], ambient=ambient)
f4.savefig(f_path+today+'/top-diagram_PKE-bg_original&calc5.png', dpi=300)

#%%
# Add column for month #
df_pk2D = dfPKE_2D[1]
df_pk2D['Month #'] = [df_pk2D[col_dt_dD][i].month for i in df_pk2D.index]

df_pk2D_v2 = dfPKE_2D_v2[1]
df_pk2D_v2['Month #'] = [df_pk2D_v2[col_dt_dD+'_v2'][i].month for i in df_pk2D_v2.index]

df_pk2D_vo = dfPKE_2D_vo[1]
df_pk2D_vo['Month #'] = [df_pk2D_vo[col_dt_dD+'_vo'][i].month for i in df_pk2D_vo.index]

fig= plt.figure(figsize=(15,15))
ax=fig.add_subplot(111)

sns.jointplot(x=df_pk2D[col_d13C], y=df_pk2D[col_dD], hue=df_pk2D['Month #'])

# Add column for wind dir and speed


#%%
# save the 2D tables
df_pk2D.to_csv(t_path+today+'/2Dpeaks_'+str(h)+'h_original_prom=100ppb_minH=1986ppb_n=3.csv', encoding=encoding)
df_pk2D_v2.to_csv(t_path+today+'/2Dpeaks_'+str(h)+'h_offset_prom=100ppb_minH=1986ppb_n=3.csv', encoding=encoding)
df_pk2D_vo.to_csv(t_path+today+'/2Dpeaks_'+str(h)+'h_cleared_prom=100ppb_minH=1986ppb_n=3.csv', encoding=encoding)

#%%

# Merge the obs. and model peaks, to compare in a correlation plot
dfPKE_dD_all, dfPKE_dD_edgar5i = fun.two_isotopes(df_PKE[[col_dt_dD, 'start_'+col_dt_dD, 'end_'+col_dt_dD, col_dD, col_errdD]].dropna(), df_pkdD_edgar5[[col_dt_dD, 'start_'+col_dt_dD, 'end_'+col_dt_dD, col_dD, col_errdD]].dropna(), h, iso='D')
dfPKE_d13C_all, dfPKE_d13C_edgar5i = fun.two_isotopes(df_PKE[[col_dt_d13C, 'start_'+col_dt_d13C, 'end_'+col_dt_d13C, col_d13C, col_errd13C]].dropna(), df_pkd13C_edgar5[[col_dt_d13C, 'start_'+col_dt_d13C, 'end_'+col_dt_d13C, col_d13C, col_errd13C]].dropna(), h, iso='13C')
dfPKE_dD_all, dfPKE_dD_tno5i = fun.two_isotopes(df_PKE[[col_dt_dD, 'start_'+col_dt_dD, 'end_'+col_dt_dD, col_dD, col_errdD]].dropna(), df_pkdD_tno5[[col_dt_dD, 'start_'+col_dt_dD, 'end_'+col_dt_dD, col_dD, col_errdD]].dropna(), h, iso='D')
dfPKE_d13C_all, dfPKE_d13C_tno5i = fun.two_isotopes(df_PKE[[col_dt_d13C, 'start_'+col_dt_d13C, 'end_'+col_dt_d13C, col_d13C, col_errd13C]].dropna(), df_pkd13C_tno5[[col_dt_d13C, 'start_'+col_dt_d13C, 'end_'+col_dt_d13C, col_d13C, col_errd13C]].dropna(), h, iso='13C')

f_corr_pkd13C = fg.correlation([dfPKE_d13C_edgar5i[col_d13C+'_obs'], dfPKE_d13C_tno5i[col_d13C+'_obs']], [dfPKE_d13C_edgar5i[col_d13C+'_model'], dfPKE_d13C_tno5i[col_d13C+'_model']], 'obs. '+str_d13C, 'modeled '+str_d13C)#, leg=['EDGAR calc5', 'TNO calc5'])
f_corr_pkd13C[0].savefig(f_path+today+'/corr_PKE-d13C_IRMS&calc5.png', dpi=300, transparent=True)

f_corr_pkdD = fg.correlation([dfPKE_dD_edgar5i[col_dD+'_obs'], dfPKE_dD_tno5i[col_dD+'_obs']], [dfPKE_dD_edgar5i[col_dD+'_model'], dfPKE_dD_tno5i[col_dD+'_model']], 'obs. '+str_dD, 'modeled '+str_dD)#, leg=['EDGAR calc5', 'TNO calc5'])
f_corr_pkdD[0].savefig(f_path+today+'/corr_PKE-dD_IRMS&calc5.png', dpi=300, transparent=True)

# Put the 2D model peaks in the same dataframe to overview them against obs. peaks
dfPKE_edgar5i=pd.concat([dfPKE_d13C_edgar5i, dfPKE_dD_edgar5i], axis=1)
dfPKE_tno5i=pd.concat([dfPKE_d13C_tno5i, dfPKE_dD_tno5i], axis=1)

#f1, ax1 = fg.overview_pk(df_IRMS, [df_PKE, df_PKE_edgar5, df_PKE_tno5, dfPKE_edgar5i, dfPKE_tno5i])
#f1.savefig(f_path+today+'overview_PKEpeaks_original&edgar&edgar-i.png', dpi=300)

