#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 08:07:38 2018

@author: malika
"""

import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import os

cdir=os.getcwd()

import figures as fg

d13C='\u03B4\u00B9\u00B3C'
dD='\u03B4D'

permil=' [\u2030]'

deltaD=dD+' SMOW'+permil
delta13C=d13C+' VPDB'+permil

formatdate='%d/%m/%Y %H:%M'

c1='xkcd:bright blue'
c2='xkcd:rust orange'
c3='xkcd:light purple'
c4='xkcd:iris'
cP='xkcd:grey'
vert='xkcd:green'
vertclair='xkcd:light green'

d13C='\u03B4\u00B9\u00B3C'
dD='\u03B4D'
permil=' [\u2030]'
pm=' \u00B1 '

deltaD=dD+' SMOW'+permil
delta13C=d13C+' VPDB'+permil

MRx='1/MR'
eMRx='err_1/MR'

#col_dt='DateTime'
col_dt='filling time UTC'

#col_MR_d13C='MR [ppb] -d13C'
col_MR_d13C='[CH4 13C] (ppb)'

#col_errMR_d13C='err_refMR [ppb] -d13C'
col_errMR_d13C='SD [CH4] C'

col_d13C='d13C VPDB'

#col_errd13C='err_ref_d13C [‰]'
col_errd13C='SD d13C'

#col_MR_dD='MR [ppb] -dD'
col_MR_dD='[CH4 dD] (ppb)'

#col_errMR_dD='err_refMR [ppb] -dD'
col_errMR_dD='SD [CH4] D'

col_dD='dD SMOW'

#col_errdD='err_ref_dD [‰]'
col_errdD='SD dD'

#col_MR_P='picarro CH4 [ppb]'
col_MR_P='CH4 Picaro ppb'
strMR_13C='MR_ppb - '+d13C
strMR_D='MR_ppb - '+dD


def cut(raw_data, deltat):
    
    s=deltat*60
    #strt=list(raw_data)[0]
    indexchange=[]
    n=0
    subsets=[n]
    
    for i in range(1,len(raw_data)):
        t=raw_data.index[i]-raw_data.index[i-1]
        ts=t.days*24*3600+t.seconds
        if ts>s:
            indexchange.append(i)
            n=n+1
        subsets.append(n)
    
    n_subsets=len(indexchange)
    subsets=np.asarray(subsets)
    new_data=raw_data.copy()
    new_data.insert(loc=1, column='subset', value=subsets, allow_duplicates=True)
    new_data.index=np.asarray(raw_data.index)
    
    startdate=pd.Series(new_data.index[0])
    startdates=startdate.append(pd.Series(new_data.index[indexchange]))

    dates=pd.DataFrame(data={'Date':startdates})
    dates.index=np.asarray(list(range(n_subsets+1)))

    return new_data, n_subsets, dates


def merge(picarro, IRMS, t_vec):
    
    strt='DateTime'
    strdata=list(IRMS)[1]
    
    # First select only the mixing ratios 
    
    data=IRMS.loc[:,(list(IRMS)[0], strdata)]
    data.rename(columns={list(IRMS)[0]: strt}, inplace=True)
    
    dataout=data 
    
    # Put normal numbers as index
    data.index=pd.Series(range(len(data)))  

    # Merge with PICARRO data on common filling time 
    merged=data.merge(picarro, on=[strt], how='inner', sort=True)

    # Put back the time in index
    merged.index=merged[strt]
    #del merged[strt]

    merged.insert(loc=len(merged.columns) , column='diff', value=merged[strdata]-merged['picarro'], allow_duplicates=True)
    
    return dataout, merged

def diff2(picarro, IRMS, strt, strdata, t_max, limMR, dates):
    
    newt=[]
    difft=[]
    difft_int=[]
    
    strP='picarro'
    
    # First select only the mixing ratios 
        
    IRMS_t = pd.Series(IRMS.index)
    data=pd.DataFrame({strt: IRMS_t, strdata: IRMS[strdata].values}, columns=[strt, strdata])
    
    IRMS = IRMS.set_index(data.index) # Update the initial data table with the new indexes (ints)
    
    picarro_t = pd.Series(picarro.index)
    pic=pd.DataFrame({strt: picarro_t, strP: picarro.values}, columns=[strt, strP])

    for t in data[strt]:
        diff_t=(pic[strt]-t)
        match=pic.ix[(diff_t).abs().argsort()[:1]]
        newt.append(match)
        difft.append(match[strt].iloc[0]-t)
        difft_int.append((match[strt].iloc[0]-t).seconds)

    newt_df=pd.concat(newt)

    difft_s=np.asarray(difft)
    difft_int_s=np.asarray(difft_int)
    
    newt_df.index=data.index
    
    newt_df.insert(loc=1 , column='subset', value=IRMS['subset'], allow_duplicates=True)
    newt_df.insert(loc=len(newt_df.columns) , column=strdata, value=data[strdata], allow_duplicates=True)    
    newt_df.insert(loc=len(newt_df.columns) , column='diff', value=newt_df[strP]-newt_df[strdata], allow_duplicates=True)
    newt_df.insert(loc=len(newt_df.columns) , column='abs_diff', value=abs(newt_df['diff']), allow_duplicates=True)
    newt_df.insert(loc=len(newt_df.columns), column='t_diff', value=difft_s, allow_duplicates=True)
    newt_df.insert(loc=len(newt_df.columns), column='t_diff_s', value=difft_int_s, allow_duplicates=True)
    
    max_diff=t_max*60

    clear_df=newt_df.loc[abs(newt_df['t_diff_s'])<max_diff]
    
    clear_df.insert(loc=len(clear_df.columns), column='t_diff_m', value=clear_df['t_diff_s']/60, allow_duplicates=True)

    #clear_df['t_diff_m']=clear_df['t_diff_s']/60
    
    #stat=pd.DataFrame({'avg diffMR':np.mean(clear_df['diff']), 'std diffMR':np.std(clear_df['diff']), 'n':len(clear_df['diff'])}, 
    #                  columns=['avg diffMR', 'std diffMR', 'n'], index = [1])
    stat=pd.DataFrame(dates)
        
    sub_df=clear_df[clear_df[strdata]<limMR]
    
    #stat['n'] = sub_df['diff'].size()
    #stat['avg diffMR'] = sub_df['diff'].mean()
    #stat['sem diffMR'] = sub_df['diff'].sem()
    #offsets=abs(stat['avg diffMR'])-stat['std diffMR']
    stat.insert(loc=1, column='n_group', value=newt_df.groupby('subset')['diff'].size(), allow_duplicates=True)
    stat.insert(loc=len(stat.columns), value=sub_df.groupby('subset')['diff'].mean(), column='avg diffMR', allow_duplicates=True)
    stat.insert(loc=len(stat.columns), column='sem diffMR', value=sub_df.groupby('subset')['diff'].sem(), allow_duplicates=True)
    offsets=abs(stat['avg diffMR'])-stat['sem diffMR']
    # Don't apply when Picarro did bulshit (12/12/2018 and 13/12/2018), by cutting the differences higher than 500 ppb
    offsets[(offsets < 0) | (offsets>100)] = 0
    offsets[offsets > 0] = 1
    stat.insert(loc=len(stat.columns), column='offset', value=offsets)
    stat.replace(to_replace=np.nan, value=0, inplace=True)

    return clear_df, stat


def corrMR(data, stat, col):
    
    new_col=col+'_corr'
    
    df=data.join(stat, on='subset')

    df[new_col]=df[col]+df['avg diffMR']*df['offset']
    
    return df


def inverseMR(df, col, col_err):
    
    df['1/MR']=1/df[col]
    df['err_1/MR']=1/(df[col]**2)*df[col_err]
    
    return df


def join_data(raw1, raw2, iso):
    
    d13C='\u03B4\u00B9\u00B3C'
    dD='\u03B4D'
    
    new1=raw1.loc[:,(list(raw1))]
    new2=raw2.loc[:,(list(raw2))]
    strt1=list(raw1)[0]
    strt2=list(raw2)[0]
    
    if iso=='13C':
        str1=d13C+'-1'
        str2=d13C+'-2'
        new1.rename(columns={strt1:'filling time', str1:'MR '+d13C, str1+' VPDB':d13C+' VPDB'}, inplace=True)
        new2.rename(columns={strt2:'filling time', str2:'MR '+d13C, str2+' VPDB':d13C+' VPDB'}, inplace=True)
    if iso=='D':
        str1=dD+'-1'
        str2=dD+'-2'
        new1.rename(columns={strt1:'filling time', str1:'MR '+dD, str1+' SMOW':dD+' SMOW'}, inplace=True)
        new2.rename(columns={strt2:'filling time', str2:'MR '+dD, str2+' SMOW':dD+' SMOW'}, inplace=True)
        
    #frames=[new1, new2]
    all_data=pd.merge(new1, new2, how='outer', sort=True, indicator='switch')
    all_data['switch'].replace('left_only', 0, inplace=True)
    all_data['switch'].replace('right_only', 1, inplace=True)
    all_data.index=all_data[list(all_data)[0]]
    
    return all_data


def select(start, end, dformat, data13C, dataD, pica):
    
    strt13C=list(data13C)[0]
    strtD=list(dataD)[0]
    strtP=list(pica)[0]
    
    startdate=pd.to_datetime(start, format=dformat)
    enddate=pd.to_datetime(end, format=dformat)

    sub_dataD=dataD.loc[(dataD[strtD]>startdate) & (dataD[strtD]<enddate)]
    sub_dataD.index=sub_dataD[strtD]
    
    sub_data13C=data13C.loc[(data13C[strt13C]>startdate) & (data13C[strt13C]<enddate)]
    sub_data13C.index=sub_data13C[strt13C]

    sub_P=pica.loc[(pica[strtP]>startdate) & (pica[strtP]<enddate)]
    sub_P.index=sub_P[strtP]

    return sub_data13C, sub_dataD, sub_P

def take_subset(index, start_date, end_date, date_format, data_13C, data_D, picarro):
    
    sub_dir = os.path.join(cdir, 'ex'+str(index)+'/')
    if not os.path.isdir(sub_dir):
        os.makedirs(sub_dir)
    
    df13C, dfD, dfP = select(start=start_date, end=end_date, dformat=date_format, data13C=data_13C, dataD=data_D, pica=picarro)
    
    fig_overview = fg.overview_sub(df13C, dfD, dfP)
    
    sig13C, err13C, R13C, fig13C = fg.keeling(data=df13C, iso='13C', strID=str(index))
    sigD, errD, RD, figD = fg.keeling(data=dfD, iso='D', strID=str(index))
    
    cols=['intercept',  'err_intercept',  'r2']
    df_keeling=pd.DataFrame({cols[0]: [sig13C, sigD], cols[1]: [err13C, errD], cols[2]: [R13C, RD]}, columns=cols, index=[delta13C, deltaD])
    
    # Moving Keeling plot
    t_window=24            # in hours, time of the window
    n=8                    # minimum number of point to make one linear fit
    select_R=0.8           # minimum correlation coefficient of the linear fit
    rangeMR_13C=150        # minimum spread of the mixing ratios for d13C data
    rangeMR_D=150          # minimum spread of the mixing ratios for dD data
    limMR=1800             # minimum mixing ratio to select this set of points
    bgMR=2200              # max bg value in each dataset
    bgMR_MT=2500           #
    
    df13C_MKP, fig13C_MKP = fg.moving_window(data=df13C, time=t_window, n=n, lim_r=select_R, iso='13C', MRmin=limMR, MRbg=bgMR, MRwindow=rangeMR_13C, miller=False)
    dfD_MKP, figD_MKP = fg.moving_window(data=dfD, time=t_window, n=n, lim_r=select_R, iso='D', MRmin=limMR, MRbg=bgMR, MRwindow=rangeMR_D, miller=False)
    
    # Plotting the results of moving window Keeling plot with the MR time series
    title='dt = '+str(t_window)+' h, spread > '+str(rangeMR_13C)+' ('+d13C+'), '+str(rangeMR_D)+' ('+dD+') [ppb], n>'+str(n)+', r2>'+str(select_R)
    fig_sigs = fg.signatures(moving_data13C=df13C_MKP, moving_dataD=dfD_MKP, data13C=df13C, dataD=dfD, title=title)
    
    # Moving Miller-Tans plot
    df13C_MMT, fig13C_MMT = fg.moving_window(data=df13C, time=t_window, n=n, lim_r=select_R, iso='13C', MRmin=limMR, MRbg=bgMR_MT, MRwindow=rangeMR_13C, miller=True)
    dfD_MMT, figD_MMT = fg.moving_window(data=dfD, time=t_window, n=n, lim_r=select_R, iso='D', MRmin=limMR, MRbg=bgMR_MT, MRwindow=rangeMR_D, miller=True)
    
    # Plotting the results of moving window Miller-Tans plot with the MR time series
    title_MT='dt = '+str(t_window)+' h, spread > '+str(rangeMR_13C)+' ('+d13C+'), '+str(rangeMR_D)+' ('+dD+') [ppb], n>'+str(n)+', r2>'+str(select_R)
    fig_sigs_MT = fg.signatures(moving_data13C=df13C_MMT, moving_dataD=dfD_MMT, data13C=df13C, dataD=dfD, title=title_MT)
    
    fig_overview.savefig('ex'+str(index)+'/ex'+str(index)+'_overview.png', dpi=300)
    fig13C.savefig('ex'+str(index)+'/keeling13C_ex'+str(index)+'.png', dpi=300)
    figD.savefig('ex'+str(index)+'/keelingD_ex'+str(index)+'.png', dpi=300)
    fig13C_MKP.savefig('ex'+str(index)+'/MKP_13C_ex'+str(index)+'_t-'+str(t_window)+'h.png', dpi=300, bbox_inches='tight')
    figD_MKP.savefig('ex'+str(index)+'/MKP_D_ex'+str(index)+'_t-'+str(t_window)+'h.png', dpi=300, bbox_inches='tight')
    fig_sigs.savefig('ex'+str(index)+'/sigs_ex'+str(index)+'_t-'+str(t_window)+'h.png', dpi=300)
    fig13C_MMT.savefig('ex'+str(index)+'/MMT_13C_ex'+str(index)+'_t-'+str(t_window)+'h.png', dpi=300, bbox_inches='tight')
    figD_MMT.savefig('ex'+str(index)+'/MMT_D_ex'+str(index)+'_t-'+str(t_window)+'h.png', dpi=300, bbox_inches='tight')
    fig_sigs_MT.savefig('ex'+str(index)+'/MT_sigs_ex'+str(index)+'_t-'+str(t_window)+'h.png', dpi=300)
    
    return df13C, dfD, dfP, fig_overview, fig13C, figD, fig13C_MKP, figD_MKP, fig_sigs, fig13C_MMT, figD_MMT, fig_sigs_MT, df_keeling, df13C_MKP, dfD_MKP, df13C_MMT, dfD_MMT

def ind_peak(data13C, dataD, start, end, ID_ex, ID_peak):
    
    startdate = pd.to_datetime(start, format=formatdate)
    enddate = pd.to_datetime(end, format=formatdate)
    
    peak_data_13C = data13C.loc[(data13C.index>startdate) & (data13C.index<enddate)]
    peak_data_D = dataD.loc[(dataD.index>startdate) & (dataD.index<enddate)]
    
    n_13C = len(peak_data_13C)
    n_D = len(peak_data_D)
    
    sig_13C, std_intercept_13C, r_value_13C, fig_13C = fg.keeling(peak_data_13C, '13C', str(ID_peak))
    sig_D, std_intercept_D, r_value_D, fig_D = fg.keeling(peak_data_D, 'D', str(ID_peak))
    
    values = pd.DataFrame({'isotope': [d13C, dD], 
                           'intercept'+permil: [sig_13C, sig_D], 
                           'err_intercept'+permil: [std_intercept_13C, std_intercept_D],
                           'r2': [r_value_13C, r_value_D], 
                           'n': [n_13C, n_D], 
                           'start date': [startdate, startdate], 
                           'end date': [enddate, enddate]},
    columns=['isotope', 'intercept'+permil, 'err_intercept'+permil, 'r2', 'n', 'start date', 'end date'])
    
    #fig_13C.savefig('ex'+str(ID_ex)+'/p'+str(ID_peak)+'_keeling13C.png', dpi=300)
    #fig_D.savefig('ex'+str(ID_ex)+'/p'+str(ID_peak)+'_keelingD.png', dpi=300)
    
    return values, fig_13C, fig_D



def select_sigs(moving_data, err_lim, time_lim, iso, title):
    
    if iso=='d13C':
        delta=col_d13C
        err_delta='err '+col_d13C
    if iso=='dD':
        delta=col_dD
        err_delta='err '+col_dD
    
    df_sigs = moving_data.loc[moving_data[err_delta]<err_lim]     # select only the signatures with a small error
    
    df_sigs.index=pd.Series(df_sigs['DateTime'])                       # re-index the table to fill wholes in the line numbering

    df_sigs, n_peaks, dates_peaks = cut(df_sigs, time_lim)          # cut the peaks that are separated of more than time_lim 
    s_sigs = df_sigs[delta]
    roll_out_mu = s_sigs.rolling('6h').mean()
    roll_out_sd = s_sigs.rolling('6h').std()
    
    # Compute the averages and the std for each peak 
    avg_sigs = df_sigs.groupby('subset').mean()[delta]
    std_sigs = df_sigs.groupby('subset').std()[delta]
    
    df_rolling=pd.DataFrame({'DateTime':df_sigs.index, delta:roll_out_mu, err_delta:roll_out_sd})
    
    # Compute the starting date and end date of each peak, and its duration
    dates = pd.Series(index=range(n_peaks))
    start_dates = pd.Series(index=range(n_peaks))
    end_dates = pd.Series(index=range(n_peaks))
    lengths = pd.Series(index=range(n_peaks))
    print(df_sigs['subset'].value_counts(sort=False))

    for i in range(n_peaks):
        sigs = df_sigs.loc[df_sigs['subset']==i]
        if len(sigs)==1:
            unique_sig_std = sigs[err_delta]
            std_sigs[i]=unique_sig_std
            
        dates[i] = df_sigs.loc[df_sigs['subset']==i]['DateTime'].iloc[0]
        lengths[i] = df_sigs.loc[df_sigs['subset']==i]['DateTime'].iloc[-1] - dates[i]
        start_dates[i] = df_sigs.loc[df_sigs['subset']==i]['start_DateTime'].iloc[0]
        end_dates[i] = df_sigs.loc[df_sigs['subset']==i]['end_DateTime'].iloc[0]
        
    df_peaks = pd.DataFrame({'DateTime': dates, 'start_DateTime': start_dates, 'end_DateTime': end_dates, 'Peak length':lengths, 'Avg peak '+delta+permil: avg_sigs, 'Std peak '+delta+permil: std_sigs}, 
                             columns = ['DateTime', 'start_DateTime', 'end_DateTime', 'Peak length', 'Avg peak '+delta+permil, 'Std peak '+delta+permil])
    
    df_peaks['DateTime'][i+1]=df_sigs.loc[df_sigs['subset']==(i+1)]['DateTime'].iloc[0]
    df_peaks['start_DateTime'][i+1]=df_sigs.loc[df_sigs['subset']==(i+1)]['start_DateTime'].iloc[0]
    df_peaks['end_DateTime'][i+1]=df_sigs.loc[df_sigs['subset']==(i+1)]['end_DateTime'].iloc[0]
    df_peaks['Peak length'][i+1]=df_sigs.loc[df_sigs['subset']==i+1]['DateTime'].iloc[-1] - df_peaks['DateTime'][i+1]

    moving_data.to_csv('Signatures/'+iso+'_'+title+'.csv')
    df_peaks.to_csv('Signatures/'+iso+'_'+title+'_peaks.csv')
    df_rolling.to_csv('Signatures/'+iso+'_'+title+'_rolling.csv')
    
    return df_sigs, df_peaks, df_rolling



def two_isotopes(peaks_1, peaks_2, t_lim):

    if len(peaks_1)<len(peaks_2):
        small=peaks_1
        large=peaks_2
    elif len(peaks_1)>=len(peaks_2):
        small=peaks_2
        large=peaks_1
    
    large.index=pd.Series(range(len(large)))
    small.index=pd.Series(range(len(small)))
        
    id_large=[]
    new_large=pd.DataFrame(index=small.index)
    
    #dt_large=pd.Series(large.index)
    #dt_small=pd.Series(small.index)

    for t in range(len(small)):
        id_large.append((np.abs(large['DateTime'] - small['DateTime'][t])).idxmin())
        # For each peak on the small vector, get the closest peak in the large vector

    new_large=large.iloc[id_large] # Save the large vector peaks that were matched

    new = pd.merge(small, new_large, on=small.index)

    diffs=[]

    for t in range(len(new)):
        diff = new['DateTime_x'][t]-new['DateTime_y'][t]
        diffs.append(np.abs(diff.days*24*3600+diff.seconds))
        #diffs.append(str(np.abs(diff)))
        # Calculate the time difference between each d13C and dD peak
    
    new.insert(0, 'time diff', diffs)

    t_lim = t_lim*60*60

    new_skeemed = new.loc[new['time diff']<t_lim]  # remove the peaks that were matched but are more than t_lim appart

    return new, new_skeemed


