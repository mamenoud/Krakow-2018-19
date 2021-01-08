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
today='2021-01-08 model_init/'

#%%
###############################################################################
# DATA TABLES
###############################################################################

edgar_ch4=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Modeled/Krakow_Updated categories/Krakow_CH4_EDGARv50_sectors_and_total_201809_201903.csv', sep=',', index_col=0, parse_dates=True, keep_date_col=True, dayfirst=True)
tno_ch4=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Modeled/Krakow_Updated categories/Krakow_CH4_TNO_CAMS_REGv42_sectors_and_total_201809_201903.csv', sep=',', index_col=0, parse_dates=True, keep_date_col=True, dayfirst=True)
# Get the list of sources before you add new columns to that table
tno_sources=tno_ch4.sum().index.to_list()
edgar_sources=edgar_ch4.sum().index.to_list()

#df_d13C=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Modeled/Krakow_201809_201903/krk_d13_EDGAR_TNO_201809_201903.csv', sep=';', index_col=0, parse_dates=True, keep_date_col=True)
#df_dD=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Modeled/Krakow_201809_201903/krk_dD_EDGAR_TNO_201809_201903.csv', sep=';', index_col=0, parse_dates=True, keep_date_col=True)

# IRMS data
df_IRMS=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Input files/KRK_merged_re-calibrated_corrected-IRMS.csv', sep=',', index_col=col_dt, parse_dates=True, dayfirst=True, keep_date_col=True)

#%%
# Model isotopic signatures calculations

def model_isotopes(sigs_d13C, sigs_dD, name):
    
    [d13C_agr, d13C_wst, d13C_ff, d13C_ffc, d13C_ffg, d13C_ffo, d13C_enb, d13C_oth, d13C_wet, d13C_bg]=sigs_d13C
    [dD_agr, dD_wst, dD_ff, dD_ffc, dD_ffg, dD_ffo, dD_enb, dD_oth, dD_wet, dD_bg]=sigs_dD
    
    d13C_edgar=(d13C_agr*edgar_ch4['Agriculture']+d13C_wst*edgar_ch4['Waste']+d13C_ffc*edgar_ch4['FF_coal']+d13C_ffg*edgar_ch4['FF_gas']+d13C_ffo*edgar_ch4['FF_oil']+d13C_enb*edgar_ch4['ENB']+d13C_oth*edgar_ch4['Other']+d13C_wet*edgar_ch4['Wetland']+d13C_bg*edgar_ch4['BC'])/edgar_ch4['Total']
    d13C_tno=(d13C_agr*tno_ch4['Agriculture']+d13C_wst*tno_ch4['Waste']+d13C_ff*tno_ch4['FF']+d13C_enb*tno_ch4['ENB']+d13C_oth*tno_ch4['Other']+d13C_wet*tno_ch4['Wetland']+d13C_bg*tno_ch4['BC'])/tno_ch4['Total']

    dD_edgar=(dD_agr*edgar_ch4['Agriculture']+dD_wst*edgar_ch4['Waste']+dD_ffc*edgar_ch4['FF_coal']+dD_ffg*edgar_ch4['FF_gas']+dD_ffo*edgar_ch4['FF_oil']+dD_enb*edgar_ch4['ENB']+dD_oth*edgar_ch4['Other']+dD_wet*edgar_ch4['Wetland']+dD_bg*edgar_ch4['BC'])/edgar_ch4['Total']
    dD_tno=(dD_agr*tno_ch4['Agriculture']+dD_wst*tno_ch4['Waste']+dD_ff*tno_ch4['FF']+dD_enb*tno_ch4['ENB']+dD_oth*tno_ch4['Other']+dD_wet*tno_ch4['Wetland']+dD_bg*tno_ch4['BC'])/tno_ch4['Total']

    return d13C_edgar.rename('d13C '+name), d13C_tno.rename('d13C '+name), dD_edgar.rename('dD '+name), dD_tno.rename('dD '+name)


#%%
sigs_d13C_calc5 = [-63, -51.6, -48.5, -50.6, -47.6, -48.5, -48.5, -32.1, -73.2, -47.8]
sigs_dD_calc5 = [-359, -299, -193, -192, -194, -193, -193, -185, -323, -90]
d13C_edgar_calc5, d13C_tno_calc5, dD_edgar_calc5, dD_tno_calc5 = model_isotopes(sigs_d13C_calc5, sigs_dD_calc5, 'calc5')

sigs_d13C_calc6 = [-63, -51.6, -48.5, -50.6, -47.6, -48.5, -32.1, -48.5, -73.2, -47.8]
sigs_dD_calc6 = [-359, -299, -193, -192, -194, -193, -185, -193, -323, -90]
d13C_edgar_calc6, d13C_tno_calc6, dD_edgar_calc6, dD_tno_calc6 = model_isotopes(sigs_d13C_calc6, sigs_dD_calc6, 'calc6')

sigs_d13C_calc7 = [-63, -51.6, -48.5, -50.6, -47.6, -48.5, -32.1, -32.1, -73.2, -47.8]
sigs_dD_calc7 = [-359, -299, -193, -192, -194, -193, -185, -185, -323, -90]
d13C_edgar_calc7, d13C_tno_calc7, dD_edgar_calc7, dD_tno_calc7 = model_isotopes(sigs_d13C_calc7, sigs_dD_calc7, 'calc7')

sigs_d13C_calc8 = [-63, -51.6, -48.5, -50.6, -47.6, -48.5, -48.5, -48.5, -73.2, -47.8]
sigs_dD_calc8 = [-359, -299, -193, -192, -194, -193, -193, -193, -323, -90]
d13C_edgar_calc8, d13C_tno_calc8, dD_edgar_calc8, dD_tno_calc8 = model_isotopes(sigs_d13C_calc8, sigs_dD_calc8, 'calc8')

# Start from previous calc configuration
#df_edgar=pd.read_table(t_path+'model/old/Krakow_CH4_EDGARv50_processed_dec2020.csv', sep=',', index_col=0, parse_dates=True, keep_date_col=True)
#df_tno=pd.read_table(t_path+'model/old/Krakow_CH4_TNO_CAMS_REGv221_processed_dec2020.csv',sep=',', index_col=0, parse_dates=True, keep_date_col=True)

# And add the new calc configurations
df_edgar=edgar_ch4.join(pd.concat([d13C_edgar_calc5, dD_edgar_calc5, d13C_edgar_calc6, dD_edgar_calc6, d13C_edgar_calc7, dD_edgar_calc7, d13C_edgar_calc8, dD_edgar_calc8], axis=1))
df_tno=tno_ch4.join(pd.concat([d13C_tno_calc5, dD_tno_calc5, d13C_tno_calc6, dD_tno_calc6, d13C_tno_calc7, dD_tno_calc7, d13C_tno_calc8, dD_tno_calc8], axis=1))

#%%
#########################
# One function to evaluate one signature configuration
#########################

def eval_model(df_model, inventory, scenari):
    
    for scenario in scenari:
        
        if inventory=='EDGAR':
            df_model_bis=df_model[['Agriculture', 'Waste', 'FF_coal', 'FF_gas', 'FF_oil', 'ENB', 'Other', 'Wetland', 'BC', 'Total', 'd13C '+scenario, 'dD '+scenario]]
        elif inventory=='TNO':
            df_model_bis=df_model[['Agriculture', 'Waste', 'FF', 'ENB', 'Other', 'Wetland', 'BC', 'Total', 'd13C '+scenario, 'dD '+scenario]]
        
        df_model_bis.rename(columns={'d13C '+scenario:'d13C', 'dD '+scenario:'dD'}, inplace=True)
    
        fig_overview, ax_overview = fg.overview_model(df_IRMS, df_model_bis, inventory)
        fig_overview.savefig(f_path+today+'overview_'+inventory+'_'+scenario+'.png', dpi=300)
        # For overview with both, just copy this line
        #fig_overview_chimere, ax_overview_chimere = fg.overview_model(df_IRMS, df_edgar, 'both', df_tno)
        #
    
        df_merged=df_IRMS.join(pd.DataFrame({'ch4_'+inventory:np.interp(df_IRMS.index, df_model_bis.index, df_model_bis['Total']), 'd13C_'+inventory: np.interp(df_IRMS.index, df_model_bis.index, df_model_bis['d13C']), 'dD_'+inventory:np.interp(df_IRMS.index, df_model_bis.index, df_model_bis['dD'])}, index=df_IRMS.index)
                               )
        df_merged_ch4=pd.DataFrame({col_MR:pd.concat([df_IRMS[col_MR_d13C], df_IRMS[col_MR_dD]]).dropna(), 'ch4_'+inventory:df_merged['ch4_'+inventory]})
    
        fig_corr_ch4=fg.correlation([df_merged_ch4[col_MR]], [df_merged_ch4['ch4_'+inventory]], namex='observed x(CH4)', namey='modeled x(CH4)', leg=[inventory+'_'+scenario])
        fig_corr_d13C=fg.correlation([df_merged[col_d13C]], [df_merged['d13C_'+inventory]], namex='observed '+d13C, namey='modeled '+d13C, leg=[inventory+'_'+scenario])
        fig_corr_dD=fg.correlation([df_merged[col_dD]], [df_merged['dD_'+inventory]], namex='observed '+dD, namey='modeled '+dD, leg=[inventory+'_'+scenario])
        
        fig_corr_ch4[0].savefig(f_path+today+'model-correlation_ch4_'+inventory+'_'+scenario+'.png', dpi=300)
        fig_corr_d13C[0].savefig(f_path+today+'model-correlation_d13C_'+inventory+'_'+scenario+'.png', dpi=300)
        fig_corr_dD[0].savefig(f_path+today+'model-correlation_dD_'+inventory+'_'+scenario+'.png', dpi=300)
    
    return
    
scenari = ['calc5', 'calc6', 'calc7', 'calc8']
eval_model(df_edgar, 'EDGAR', scenari)
eval_model(df_tno, 'TNO', scenari)

#%%
# Plot model data with original datetimes, for both inventories 

for scenario in scenari:
    fig=fg.overview(df_IRMS, data_CHIM=[df_edgar, df_tno], suf=scenario)
    fig.savefig(f_path+today+'overview_chimere_'+scenario+'.png', dpi=600, transparent=True, bbox_inches = 'tight', pad_inches = 0)

#%%
# Contributions of each source

sources_edgar=['Wetland', 'Agriculture', 'Waste', 'FF_coal', 'FF_gas', 'FF_oil', 'ENB', 'Other']
sources_tno=['Wetland', 'Agriculture', 'Waste', 'FF', 'ENB', 'Other']
for s in sources_edgar:
    # proportion of each source
    df_edgar[s+' added'] = edgar_ch4[s]/(edgar_ch4['Total']-edgar_ch4['BC'])
for s in sources_tno:    
    df_tno[s+' added'] = tno_ch4[s]/(tno_ch4['Total']-tno_ch4['BC'])

#%%
# Add model wind and ave the merged model tables
df_Wmod=pd.read_table('/Users/malika/surfdrive/Data/Krakow/Roof air/Modeled/wind_speed_direction_KRK_20180914_20190314_CHIMERE.csv', sep=';', index_col=0, parse_dates=True)

df_edgar.join(df_Wmod).to_csv(t_path+'model/Krakow_CH4_EDGARv50_processed.csv')
df_tno.join(df_Wmod).to_csv(t_path+'model/Krakow_CH4_TNO_CAMS_REGv221_processed.csv')