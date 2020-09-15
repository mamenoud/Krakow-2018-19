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

vertclair='#009966'
web_blue='#333399'
web_green='#006633'
web_yellow='#FFFF00'
web_red='#FF3333'

###############################################################################
# SAVE THE RESULT OF THE DAY IN A SEPERATE FOLDER
###############################################################################

f_path = '/Users/malika/surfdrive/Data/Krakow/Roof air/figures/'
t_path = '/Users/malika/surfdrive/Data/Krakow/Roof air/tables/'
today='2020-09-11 data_signatures/'

#%%
# Observation data
df_IRMS=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Input files/KRK_merged_re-calibrated_corrected-IRMS.csv', sep=',', index_col=col_dt, parse_dates=True, dayfirst=True, keep_date_col=True)

# Get the MWKP data
#(int) DateTime,	start_DateTime,	end_DateTime,	dD SMOW,	SD dD,	slope,	r2,	n,	DateTime,	start_DateTime,	end_DateTime,	d13C VPDB,	SD d13C,	slope,	r2,	n
df_MW = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-03 data_signatures/MWKP/MWKP_original_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 10, 11, 12], dayfirst=True, keep_date_col=True, header=0,
                    names=[col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD',	'n_dD', col_wd+'_dD', col_ws+'_dD',	col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C,	col_d13C,	 col_errd13C, 'slope_d13C',	'r2_d13C', 'n_d13C', col_wd+'_d13C', col_ws+'_d13C'])
df_MW_v2 = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-03 data_signatures/MWKP/MWKP_offset_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 10, 11, 12], dayfirst=True, keep_date_col=True, header=0,
                    names=[col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD',	'n_dD', col_wd+'_dD', col_ws+'_dD',	col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C,	col_d13C,	 col_errd13C, 'slope_d13C',	'r2_d13C', 'n_d13C', col_wd+'_d13C', col_ws+'_d13C'])
df_MW_vo = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-03 data_signatures/MWKP/MWKP_cleared_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 10, 11, 12], dayfirst=True, keep_date_col=True, header=0,
                    names=[col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD',	'n_dD', col_wd+'_dD', col_ws+'_dD',	col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C,	col_d13C,	 col_errd13C, 'slope_d13C',	'r2_d13C', 'n_d13C', col_wd+'_d13C', col_ws+'_d13C'])

df_MW_edgar3 = pd.read_csv(t_path+'2020-09-03 data_signatures/MWKP/MWKP_edgar-calc3_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 8, 9, 10], dayfirst=True, keep_date_col=True, header=0,
                           names=[col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD',	'n_dD',	col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C, col_d13C, col_errd13C, 'slope_d13C', 'r2_d13C', 'n_d13C'])

# Get the PKE data
df_PKE = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-11 data_signatures/PKE/PKE_original_prom=100ppb_minH=1986ppb_n=3.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 11, 12, 13], dayfirst=True, keep_date_col=True, header=0,
                     names=[col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_ws+'_dD',	col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C,	col_d13C,	 col_errd13C, 'slope_d13C',	'r2_d13C', 'res_d13C', 'n_d13C', col_wd+'_d13C', col_ws+'_d13C'])
df_PKE_v2 = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-11 data_signatures/PKE/PKE_offset_prom=100ppb_minH=1986ppb_n=3.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 11, 12, 13], dayfirst=True, keep_date_col=True, header=0,
                     names=[col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_ws+'_dD',	col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C,	col_d13C,	 col_errd13C, 'slope_d13C',	'r2_d13C', 'res_d13C', 'n_d13C', col_wd+'_d13C', col_ws+'_d13C'])
df_PKE_vo = pd.read_csv('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-11 data_signatures/PKE/PKE_cleared_prom=100ppb_minH=1986ppb_n=3.csv', sep=',', index_col=0, parse_dates=[0, 1, 2, 11, 12, 13], dayfirst=True, keep_date_col=True, header=0,
                     names=[col_dt_dD,	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_ws+'_dD',	col_dt_d13C,	'start_'+col_dt_d13C,	'end_'+col_dt_d13C,	col_d13C,	 col_errd13C, 'slope_d13C',	'r2_d13C', 'res_d13C', 'n_d13C', col_wd+'_d13C', col_ws+'_d13C'])

# model peaks
df_pkdD_edgar3 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE-dD_edgar3_prom=100ppb_minH=1986ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -dD',	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_ws+'_dD'])
df_pkd13C_edgar3 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE-d13C_edgar3_prom=100ppb_minH=1986ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -d13C',	'start_'+col_dt_d13C,	'end_'+col_dt_d13C, col_d13C,	col_errd13C,	'slope_d13C',	'r2_d13C', 'res_d13C',	'n_d13C', col_wd+'_d13C', col_ws+'_d13C'])

df_pkdD_edgar4 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE-dD_edgar4_prom=100ppb_minH=1986ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -dD',	'start_'+col_dt_dD,	'end_'+col_dt_dD, col_dD,	col_errdD,	'slope_dD',	'r2_dD', 'res_dD',	'n_dD', col_wd+'_dD', col_ws+'_dD'])
df_pkd13C_edgar4 = pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/tables/2020-09-14 data_signatures/PKE-d13C_edgar4_prom=100ppb_minH=1986ppb_n=3.csv',  sep=',', index_col=1, parse_dates=[1,2], dayfirst=True, keep_date_col=True, header=0,
                     names=['peak ID -d13C',	'start_'+col_dt_d13C,	'end_'+col_dt_d13C, col_d13C,	col_errd13C,	'slope_d13C',	'r2_d13C', 'res_d13C',	'n_d13C', col_wd+'_d13C', col_ws+'_d13C'])

#%%
# Visualise the raw MWKP signatures
f1, ax1 = fg.overview_sig(df_IRMS, df_MW)
f1.savefig(f_path+today+'/overwiew_MW_original_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.png', dpi=300)
f3, ax3 = fg.overview_sig(df_IRMS, df_MW_edgar3)
f3.savefig(f_path+today+'/overwiew_MW_edgar-calc3_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.png', dpi=300)

#%%

# COMPARISON IRMS vs CHIMERE signatures
hist_MW_d13C= plt.figure(figsize=(15,10))
ax_MW_d13C=hist_MW_d13C.add_subplot(111)
sns.kdeplot(df_MW[col_d13C], shade=True, ax=ax_MW_d13C, label='IRMS obs., n='+str(len(df_MW_3[col_d13C].dropna())), color=web_blue)
sns.kdeplot(df_MW_edgar3[col_d13C], shade=True, ax=ax_MW_d13C, label='CHIMERE-EDGAR, n='+str(len(df_MW_edgar3[col_d13C].dropna())), color=web_red)
ax_MW_d13C.set_xlabel(str_d13C)
hist_MW_d13C.savefig(f_path+today+'hist_MW-d13C_obs-edgar3.png', dpi=300)

hist_MW_dD= plt.figure(figsize=(15,10))
ax_MW_dD=hist_MW_dD.add_subplot(111)
sns.kdeplot(df_MW_3[col_dD], shade=True, ax=ax_MW_dD, label='IRMS obs., n='+str(len(df_MW_3[col_dD].dropna())), color=web_blue)
sns.kdeplot(df_MW_edgar3[col_dD], shade=True, ax=ax_MW_dD, label='CHIMERE-EDGAR, n='+str(len(df_MW_edgar3[col_dD].dropna())), color=web_red)
ax_MW_dD.set_xlabel(str_dD)
hist_MW_dD.savefig(f_path+today+'hist_MW-dD_obs-edgar3.png', dpi=300)

#%%
# Average signatures that are sharing p % of time 
p_overlap=70
diffdD=16
diffd13C=4

# It leads to a new table with the peak signatures

df_pk13C = fun.MW_peaks(df_MW, p_overlap, '13C', delta_thr=diffd13C)
df_pk13C_v2 = fun.MW_peaks(df_MW_v2, p_overlap, '13C', delta_thr=diffd13C)
df_pk13C_vo = fun.MW_peaks(df_MW_vo, p_overlap, '13C', delta_thr=diffd13C)

df_pkD = fun.MW_peaks(df_MW, p_overlap, 'D', delta_thr=diffdD)
df_pkD_v2 = fun.MW_peaks(df_MW_v2, p_overlap, 'D', delta_thr=diffdD)
df_pkD_vo = fun.MW_peaks(df_MW_vo, p_overlap, 'D', delta_thr=diffdD)

df_MWpk = pd.concat([df_pkD, df_pk13C], axis=1)
df_MWpk_v2 = pd.concat([df_pkD_v2, df_pk13C_v2], axis=1)
df_MWpk_vo = pd.concat([df_pkD_vo, df_pk13C_vo], axis=1)

df_MWpk.to_csv(t_path+today+'/MWKP-peaks_overlap'+str(p_overlap)+'_original_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.csv', encoding=encoding)
df_MWpk_v2.to_csv(t_path+today+'/MWKP-peaks_overlap'+str(p_overlap)+'_offset_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.csv', encoding=encoding)
df_MWpk_vo.to_csv(t_path+today+'/MWKP-peaks_overlap'+str(p_overlap)+'_cleared_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.csv', encoding=encoding)

# CHIMERE-edgar peaks
df_pk13C_edgar3 = fun.MW_peaks(df_MW_edgar3, p_overlap, '13C', delta_thr=diffd13C)
df_pkD_edgar3 = fun.MW_peaks(df_MW_edgar3, p_overlap, 'D', delta_thr=diffdD)
df_MW_pk_edgar3 = pd.concat([df_pkD_edgar3, df_pk13C_edgar3], axis=1)

df_MW_pk_edgar3.to_csv(t_path+today+'/MWKP-peaks_overlap'+str(p_overlap)+'_edgar-calc3_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.png', encoding=encoding)

#%%
# COMPARISON IRMS vs CHIMERE peaks
hist_pk_d13C= plt.figure(figsize=(15,10))
ax_pk_d13C=hist_pk_d13C.add_subplot(111)
sns.kdeplot(df_pk[col_d13C], shade=True, ax=ax_pk_d13C, label='IRMS obs., n='+str(len(df_pk[col_d13C].dropna())), color=web_blue)
sns.kdeplot(df_pk_edgar3[col_d13C], shade=True, ax=ax_pk_d13C, label='CHIMERE-EDGAR, n='+str(len(df_pk_edgar3[col_d13C].dropna())), color=web_red)
ax_pk_d13C.set_xlabel(str_d13C)
hist_pk_d13C.savefig(f_path+today+'hist_pk'+str(p_overlap)+'-d13C_obs-edgar3.png', dpi=300)

hist_pk_dD= plt.figure(figsize=(15,10))
ax_pk_dD=hist_pk_dD.add_subplot(111)
sns.kdeplot(df_pk[col_dD], shade=True, ax=ax_pk_dD, label='IRMS obs., n='+str(len(df_pk[col_dD].dropna())), color=web_blue)
sns.kdeplot(df_MW_edgar3[col_dD], shade=True, ax=ax_pk_dD, label='CHIMERE-EDGAR, n='+str(len(df_pk_edgar3[col_dD].dropna())), color=web_red)
ax_pk_dD.set_xlabel(str_dD)
hist_pk_dD.savefig(f_path+today+'hist_pk'+str(p_overlap)+'-dD_obs-edgar3.png', dpi=300)

#%%

# Visualize the peak signatures 

# MWKP derived peaks
f1, ax1 = fg.overview_pk(df_IRMS, df_MWpk)
f2, ax2 = fg.overview_pk(df_IRMS, df_MWpk_v2)
f1.savefig(f_path+today+'/overwiew_MWpeaks_overlap'+str(p_overlap)+'_original_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.png', dpi=300)
f2.savefig(f_path+today+'/overwiew_MWpeaks_overlap'+str(p_overlap)+'_offset_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.png', dpi=300)

# PKE derived peaks
f1, ax1 = fg.overview_pk(df_IRMS, [df_PKE])
f1.savefig(f_path+today+'/overview_PKEpeaks_cleaned_original_prom=100ppb_minH=1900ppb_n=3.png', dpi=300)

#f3, ax3 = fg.overview_pk(df_IRMS, df_pk_edgar3)
#f3.savefig(f_path+today+'/overview_MWKPpeaks_overlap'+str(p_overlap)+'_edgar-calc3_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.png', dpi=300)

#%%
# Merge the peak lists, so that the peaks get 2 isotopes
h=6

# MWKP peaks
dfMW_2D = fun.two_isotopes(df_pkD.dropna(), df_pk13C.dropna(), h)
dfMW_2D[1].to_csv(t_path+today+'/2Dpeaks-'+str(h)+'h_overlap'+str(p_overlap)+'_original_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.csv', encoding=encoding)

dfMW_2D_v2 = fun.two_isotopes(df_pkD_v2.dropna(), df_pk13C_v2.dropna(), h)
dfMW_2D_v2[1].to_csv(t_path+today+'/2Dpeaks-'+str(h)+'h_overlap'+str(p_overlap)+'_offset_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.csv', encoding=encoding)

dfMW_2D_vo = fun.two_isotopes(df_pkD_vo.dropna(), df_pk13C_vo.dropna(), h)
dfMW_2D_vo[1].to_csv(t_path+today+'/2Dpeaks-'+str(h)+'h_overlap'+str(p_overlap)+'_cleared_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.csv', encoding=encoding)

dfMW_2D_edgar3 = fun.two_isotopes(df_pkD_edgar3.dropna(), df_pk13C_edgar3.dropna(), 6)
dfMW_2D_edgar3[1].to_csv(t_path+today+'/2Dpeaks_overlap'+str(p_overlap)+'_edgar-calc3_dt=12h_n=5_fit=0.1sq-sc_x-range=200ppb_x-bg=10per.csv', encoding=encoding)

# PKE peaks
dfPKE_2D = fun.two_isotopes(df_PKE[[col_dt_dD, 'start_'+col_dt_dD, 'end_'+col_dt_dD, col_dD, col_errdD]].dropna(), df_PKE[[col_dt_d13C, 'start_'+col_dt_d13C, 'end_'+col_dt_d13C, col_d13C, col_errd13C]].dropna(), h)
dfPKE_2D[1].to_csv(t_path+today+'/2Dpeaks_'+str(h)+'h_original_prom=100ppb_minH=1986ppb_n=3.csv', encoding=encoding)

dfPKE_2D_v2 = fun.two_isotopes(df_PKE_v2[[col_dt_dD, 'start_'+col_dt_dD, 'end_'+col_dt_dD, col_dD, col_errdD]].dropna(), df_PKE_v2[[col_dt_d13C, 'start_'+col_dt_d13C, 'end_'+col_dt_d13C, col_d13C, col_errd13C]].dropna(), h)
dfPKE_2D_v2[1].to_csv(t_path+today+'/2Dpeaks_'+str(h)+'h_offset_prom=100ppb_minH=1986ppb_n=3.csv', encoding=encoding)

dfPKE_2D_vo = fun.two_isotopes(df_PKE_vo[[col_dt_dD, 'start_'+col_dt_dD, 'end_'+col_dt_dD, col_dD, col_errdD]].dropna(), df_PKE_vo[[col_dt_d13C, 'start_'+col_dt_d13C, 'end_'+col_dt_d13C, col_d13C, col_errd13C]].dropna(), h)
dfPKE_2D_vo[1].to_csv(t_path+today+'/2Dpeaks_'+str(h)+'h_cleared_prom=100ppb_minH=1986ppb_n=3.csv', encoding=encoding)
#%%

# 2D diagrams
ambient=pd.DataFrame({'avg_d13C':-47.8, 'avg_dD':-90.0}, index=[0])

# Compare MW and PKE
f1 = fg.top_diagrams([dfPKE_2D[1], dfMW_2D[1]], ['original PKE', 'original MWKP'], ambient=ambient)
f1.savefig(f_path+today+'/top-diagram_MW&PKE-bg_original.png', dpi=300)

f2 = fg.top_diagrams([dfPKE_2D_v2[1], dfMW_2D_v2[1]], ['offset PKE', 'offset MWKP'], ambient=ambient)
f2.savefig(f_path+today+'/top-diagram_MW&PKE-bg_offset.png', dpi=300)

# Compare the original, offset and cleared
f3=fg.top_diagrams([dfPKE_2D[1], dfPKE_2D_v2[1], dfPKE_2D_vo[1]], ['original PKE', 'offset PKE', 'cleared PKE'], ambient=ambient)
f3.savefig(f_path+today+'/top-diagram_PKE-bg_original&offset&cleared.png', dpi=300)

#%%


