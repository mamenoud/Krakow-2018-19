#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 08:48:52 2020

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
str_EDGAR='EDGARv50'
str_TNO='TNO_CAMS_REGv221'

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
today='2020-09-14 model_init/'

#%%
###############################################################################
# DATA TABLES
###############################################################################

edgar_ch4=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Modeled/Krakow_201809_201903/Krakow_CH4_EDGARv50_sectors_and_total_201809_201903.csv', sep=';', index_col=0, parse_dates=True, keep_date_col=True)
tno_ch4=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Modeled/Krakow_201809_201903/Krakow_CH4_TNO_CAMS_REGv221_sectors_and_total_201809_201903.csv', sep=';', index_col=0, parse_dates=True, keep_date_col=True)
# Get the list of sources before you add new columns to that table
sources=edgar_ch4.sum().index.to_list()

df_d13C=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Modeled/Krakow_201809_201903/krk_d13_EDGAR_TNO_201809_201903.csv', sep=';', index_col=0, parse_dates=True, keep_date_col=True)
df_dD=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Modeled/Krakow_201809_201903/krk_dD_EDGAR_TNO_201809_201903.csv', sep=';', index_col=0, parse_dates=True, keep_date_col=True)

# IRMS data
df_IRMS=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Input files/KRK_merged_re-calibrated_corrected-IRMS.csv', sep=',', index_col=col_dt, parse_dates=True, dayfirst=True, keep_date_col=True)

#%%
# Make one df for each inventory
df_edgar=edgar_ch4.join(pd.concat([df_d13C['EDGARv50'].rename('d13C'), df_dD['EDGARv50'].rename('dD')], axis=1))

df_tno=tno_ch4.join(pd.concat([df_d13C['TNO_CAMS_REGv221'].rename('d13C'), df_dD['TNO_CAMS_REGv221'].rename('dD')], axis=1))

#%%
# Start from previous calc configuration
df_edgar=pd.read_table(t_path+'/model/Krakow_CH4_EDGARv50_processed.csv', sep=',', index_col=0, parse_dates=True, keep_date_col=True)
df_tno=pd.read_table(t_path+'/model/Krakow_CH4_TNO_CAMS_REGv221_processed.csv',sep=',', index_col=0, parse_dates=True, keep_date_col=True)

#%%
#########################
# One function to evaluate one signature configuration
#########################

def eval_model(df_model, inventory, conf):
    
    df_model_bis=df_model[['Agriculture', 'Waste', 'FF', 'Other', 'Wetland', 'BC', 'Total', 'd13C '+conf, 'dD '+conf]]
    df_model_bis.rename(columns={'d13C '+conf:'d13C', 'dD '+conf:'dD'}, inplace=True)
    
    fig_overview, ax_overview = fg.overview_model(df_IRMS, df_model_bis, inventory)
    fig_overview.savefig(f_path+today+'overview_'+inventory+'_'+conf+'.png', dpi=300)
    # For overview with both, just copy this line
    #fig_overview_chimere, ax_overview_chimere = fg.overview_model(df_IRMS, df_edgar, 'both', df_tno)
    #
    
    df_merged=df_IRMS.join(pd.DataFrame({'ch4_'+inventory:np.interp(df_IRMS.index, df_model_bis.index, df_model_bis['Total']), 'd13C_'+inventory: np.interp(df_IRMS.index, df_model_bis.index, df_model_bis['d13C']), 'dD_'+inventory:np.interp(df_IRMS.index, df_model_bis.index, df_model_bis['dD'])}, index=df_IRMS.index)
                    )
    df_merged_ch4=pd.DataFrame({col_MR:pd.concat([df_IRMS[col_MR_d13C], df_IRMS[col_MR_dD]]).dropna(), 'ch4_'+inventory:df_merged['ch4_'+inventory]})
    
    fig_corr_ch4=fg.correlation([df_merged_ch4[col_MR]], [df_merged_ch4['ch4_'+inventory]], namex='observed x(CH4)', namey='modeled x(CH4)', leg=[inventory+'_'+conf])
    fig_corr_d13C=fg.correlation([df_merged[col_d13C]], [df_merged['d13C_'+inventory]], namex='observed '+d13C, namey='modeled '+d13C, leg=[inventory+'_'+conf])
    fig_corr_dD=fg.correlation([df_merged[col_dD]], [df_merged['dD_'+inventory]], namex='observed '+dD, namey='modeled '+dD, leg=[inventory+'_'+conf])
    
    fig_corr_ch4[0].savefig(f_path+today+'model-correlation_ch4_'+inventory+'_'+conf+'.png', dpi=300)
    fig_corr_d13C[0].savefig(f_path+today+'model-correlation_d13C_'+inventory+'_'+conf+'.png', dpi=300)
    fig_corr_dD[0].savefig(f_path+today+'model-correlation_dD_'+inventory+'_'+conf+'.png', dpi=300)
    
    return
    
eval_model(df_edgar, 'EDGAR', 'calc3')
#%%
# Plot model data with original datetimes
fig_overview_edgar, ax_overview_edgar = fg.overview_model(df_IRMS, df_edgar, 'EDGAR')
fig_overview_tno, ax_overview_tno = fg.overview_model(df_IRMS, df_tno, 'TNO')
fig_overview_chimere, ax_overview_chimere = fg.overview_model(df_IRMS, df_edgar, 'both', df_tno)

fig_overview_edgar.savefig(f_path+today+'overview_edgar.png', dpi=300)
fig_overview_tno.savefig(f_path+today+'overview_tno.png', dpi=300)
fig_overview_chimere.savefig(f_path+today+'overview_chimere.png', dpi=300)

#%%
# Interpolate the model datetimes to the measurement ones for histograms and correlation plots
df_all=df_IRMS.join(pd.concat([pd.DataFrame({'ch4_EDGAR':np.interp(df_IRMS.index, df_edgar.index, df_edgar['Total']), 'd13C_EDGAR': np.interp(df_IRMS.index, df_edgar.index, df_edgar['d13C']), 'dD_EDGAR':np.interp(df_IRMS.index, df_edgar.index, df_edgar['dD'])}, index=df_IRMS.index),
                               pd.DataFrame({'ch4_TNO':np.interp(df_IRMS.index, df_tno.index, df_tno['Total']), 'd13C_TNO': np.interp(df_IRMS.index, df_tno.index, df_tno['d13C']), 'dD_TNO':np.interp(df_IRMS.index, df_tno.index, df_tno['dD'])}, index=df_IRMS.index)], axis=1),
                    )

df_all_ch4=pd.DataFrame({col_MR:pd.concat([df_IRMS[col_MR_d13C], df_IRMS[col_MR_dD]]).dropna(), 'ch4_EDGAR':df_all['ch4_EDGAR'], 'ch4_TNO':df_all['ch4_TNO']})

#%%
# Correlation plots
fig_corr_ch4=fg.correlation([df_all_ch4[col_MR], df_all_ch4[col_MR]], [df_all_ch4['ch4_EDGAR'], df_all_ch4['ch4_TNO']], namex='observed x(CH4)', namey='modeled x(CH4)', leg=[str_EDGAR, str_TNO])
fig_corr_d13C=fg.correlation([df_all[col_d13C], df_all[col_d13C]], [df_all['d13C_EDGAR'], df_all['d13C_TNO']], namex='observed '+d13C, namey='modeled '+d13C, leg=[str_EDGAR, str_TNO])
fig_corr_dD=fg.correlation([df_all[col_dD], df_all[col_dD]], [df_all['dD_EDGAR'], df_all['dD_TNO']], namex='observed '+dD, namey='modeled '+dD, leg=[str_EDGAR, str_TNO])

fig_corr_ch4[0].savefig(f_path+today+'model-correlation_ch4.png', dpi=300)
fig_corr_d13C[0].savefig(f_path+today+'model-correlation_d13C.png', dpi=300)
fig_corr_dD[0].savefig(f_path+today+'model-correlation_dD.png', dpi=300)


#%%
# Contributions of each source

for s in sources:
    if s=='BC' or s=='Total':
        continue
    else:
        df_edgar[s+' added'] = edgar_ch4[s]/(edgar_ch4['Total']-edgar_ch4['BC'])
        df_tno[s+' added'] = tno_ch4[s]/(tno_ch4['Total']-tno_ch4['BC'])

#%%
# Model isotopic signatures calculations
[d13C_agr, d13C_wst, d13C_ff, d13C_oth, d13C_wet, d13C_bg] = [-63.9, -56, -46.7, -26.2, -61.5, -47.8]
[dD_agr, dD_wst, dD_ff, dD_oth, dD_wet, dD_bg] = [-319, -298, -185, -211, -322, -90]

df_edgar['d13C calc4']=(d13C_agr*df_edgar['Agriculture']+d13C_wst*df_edgar['Waste']+d13C_ff*df_edgar['FF']+d13C_oth*df_edgar['Other']+d13C_wet*df_edgar['Wetland']+d13C_bg*df_edgar['BC'])/df_edgar['Total']
df_tno['d13C calc4']=(d13C_agr*df_tno['Agriculture']+d13C_wst*df_tno['Waste']+d13C_ff*df_tno['FF']+d13C_oth*df_tno['Other']+d13C_wet*df_tno['Wetland']+d13C_bg*df_tno['BC'])/df_tno['Total']

df_edgar['dD calc4']=(dD_agr*df_edgar['Agriculture']+dD_wst*df_edgar['Waste']+dD_ff*df_edgar['FF']+dD_oth*df_edgar['Other']+dD_wet*df_edgar['Wetland']+dD_bg*df_edgar['BC'])/df_edgar['Total']
df_tno['dD calc4']=(dD_agr*df_tno['Agriculture']+dD_wst*df_tno['Waste']+dD_ff*df_tno['FF']+dD_oth*df_tno['Other']+dD_wet*df_tno['Wetland']+dD_bg*df_tno['BC'])/df_tno['Total']

# Save the merged model tables
df_edgar.to_csv(t_path+'model/Krakow_CH4_EDGARv50_processed.csv')
df_tno.to_csv(t_path+'model/Krakow_CH4_TNO_CAMS_REGv221_processed.csv')
