#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 08:07:38 2018

@author: malika
"""

import pandas as pd
import numpy as np
import scipy as sc
from scipy import stats
import matplotlib.pyplot as plt
import os
import scipy.odr.odrpack as odrpack
import bces.bces
import nmmn.stats
import datetime
from datetime import datetime, timedelta
import scipy.signal as scgl

import matplotlib.gridspec as gridspec

cdir=os.getcwd()

import figures_KR as fg
import lnr4

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

col_dt_obs='filling time UTC'
col_dt='DateTime'
col_dt_d13C='DateTime_d13C'
col_dt_dD='DateTime_dD'

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

col_wd='averageWindDirection'
col_ws='averageWindSpeed'
col_errwd='stdWindDirection'
col_errws='stdWindSpeed'

# For pretty strings

strMR_13C='MR_ppb - '+d13C
strMR_D='MR_ppb - '+dD
str_d13C=delta13C+permil
str_dD=deltaD+permil
str_MR='x(CH4) [ppb]'
str_EDGAR='EDGARv50'
str_TNO='TNO_CAMS_REGv221'

#col_MR_P='picarro CH4 [ppb]'
col_MR_P='CH4 Picaro ppb'
strMR_13C='MR_ppb - '+d13C
strMR_D='MR_ppb - '+dD


def cut(raw_data, deltat, maxt):
    
    s=deltat*60
    s_max=maxt*24*60*60

    indexchange=[]
    n=0
    j=0
    subsets=[n]
    
    for i in range(1,len(raw_data)):
        t=raw_data.index[i]-raw_data.index[i-1]
        ts=t.days*24*3600+t.seconds
        if ts>s:
            indexchange.append(i)
            j=i
            n=n+1
        t_sub=raw_data.index[i]-raw_data.index[j] # difference since last change
        ts_sub=t_sub.days*24*3600+t_sub.seconds 
        if ts_sub>s_max:
            indexchange.append(i)
            j=i
            n=n+1
        subsets.append(n)
    
    n_subsets=len(indexchange)
    subsets=np.asarray(subsets)
    
    startdate=pd.Series(raw_data.index[0])
    startdates=startdate.append(pd.Series(raw_data.index[indexchange]))

    dates=pd.Series(startdates, name='Date')
    dates.index=np.asarray(list(range(n_subsets+1)))

    return subsets, n_subsets, dates


def diff2(ch4_picarro, df_IRMS, col_ch4, limMR, dates_cut):
    
    df_temp = df_IRMS[[col_ch4, 'subset']]
    
    # Compute the difference between picarro and IRMS hourly average
    df_temp['MR diff']=ch4_picarro-df_IRMS[col_ch4]
    # only select MR lower than a certain threshold, they are more representative
    df_low=df_temp.loc[df_temp[col_ch4]<limMR]
    print('Number of measurements without Picarro verification: '+str(len(df_IRMS)-len(df_low)))

    # Table with the statistics per subset
    stat=pd.DataFrame(dates_cut) # the index is the subset number
    # calculate number of points per subset
    stat['n_group']=df_IRMS.groupby('subset').size()
    stat['n_calc']=df_low.groupby('subset').size()
    # caluculate the average ch4 difference in the subset, on the low values dataframe
    stat['avg diffMR']=df_low.groupby('subset')['MR diff'].mean()
    # caluculate the stdev ch4 difference in the subset, on the low values dataframe
    stat['sem diffMR']=df_low.groupby('subset')['MR diff'].sem()
    # see if the offset is significant (i.e. within its uncertainty)
    offsets=abs(stat['avg diffMR'])-stat['sem diffMR']
    # Make it binary: yes or no and add it to the stat table
    for i in range(len(offsets)):
        if (offsets[i]>0): #& (stat['n_calc'][i]>10):
            offsets[i]=1
        else:
            offsets[i]=0
            
    stat['offset']=offsets
    # When there is no possible offset calculcation, the offset is 0
    stat.replace(to_replace=np.nan, value=0, inplace=True)
    
    # insert the offsets in the data table to make a temporary DataFrame where all the information is
    df_temp=df_temp.join(stat, on='subset')
    ch4_corr=df_temp[col_ch4]+df_temp['avg diffMR']*df_temp['offset']
   
    return ch4_corr, stat


def corrMR(df, var, stat):
    
    df_temp=df.join(stat, on='subset')

    ch4_corr=df_temp[var]+df_temp['avg diffMR']*df_temp['offset']
    
    return ch4_corr


def inverseMR(MR, err_MR):
    
    MRx=1/MR
    err_MRx=1/(MR**2)*err_MR
    
    return MRx, err_MRx


def f(B, x):
    return B[0]*x + B[1]


def regression(df_in, iso, graph=False, model=None, MT=False):
    
    df_in.replace(0.0, 1e-8, inplace=True)
    print(list(df_in))
    data_MR = df_in[list(df_in)[0]]
    data_y = df_in[list(df_in)[1]]
    if (model==None) or (model=='IRMS'):
        err_MR = df_in[list(df_in)[2]]
        err_y = df_in[list(df_in)[3]]
        reg='yx'
    else:
        err_MR = pd.Series([np.nan]*len(df_in), index=df_in.index)
        err_y = pd.Series([np.nan]*len(df_in), index=df_in.index)
        reg='yx'
    
    st_err_MR=10
    if iso=='D':
        st_err_delta=2.0
    if iso=='13C':
        st_err_delta=0.1
    
    err_MR.replace(np.nan, st_err_MR, inplace=True)
    err_y.replace(np.nan, st_err_delta, inplace=True)
    data_x, err_x = inverseMR(data_MR, err_MR)
    if MT is True:
        data_x=data_MR
        err_x=err_MR
        err_y=data_MR*data_y *(err_y/data_y+err_MR/data_MR)
        data_y=data_MR*data_y
    
    # Print some variables
    print('\n '+str(len(df_in))+' data points.')
    # First estimation with a linear regression
    slope_0, intercept_0, r_value_0, p_value_0, std_err_0 = sc.stats.linregress(data_x, data_y)
    print('\n Linear regression\nSlope: '+str(slope_0)+'\nIntercept: '+str(intercept_0))
    
    # Set up linear model
    linear = odrpack.Model(f)
    
    # run an ordinary distance regression model
    mydata = odrpack.RealData(data_x, data_y, sx=err_x, sy=err_y)
    myodr = odrpack.ODR(mydata, linear, beta0=(slope_0, intercept_0))
    odr_out = myodr.run()
        
    # save the relevant output values 
    n = len(data_x)
    slope = odr_out.beta[0]
    intercept = odr_out.beta[1]
    cov_mat = odr_out.cov_beta
    std_slope = np.sqrt(cov_mat[0][0])
    std_intercept = np.sqrt(cov_mat[1][1]) # get the standard deviation of the b parameter from the covariance matrix
    res_var = odr_out.res_var
    xrange = 1/min(data_x) - 1/max(data_x)
    #err_xrange = err_x*1/(max(data_x)**2) + err_x*1/(max(data_x)**2)
    info=odr_out.stopreason
    #odr_out.pprint()  
    
    print('\n Orthogonal distance regression')
    print('Slope: '+str(slope)+'\nIntercept: '+str(intercept))
    print(str(info))
    
    print('\n Bivariate correlated errors and intrinsic scatter')
    bces_out=lnr4.bces(data_x,data_y,err_x,err_y, nsim=5000, model=reg, bootstrap=False, verbose='normal', full_output=True)
    # Treat y as the dependant variable
    slope=bces_out[1][0]
    std_slope=bces_out[1][1]
    intercept=bces_out[0][0]
    std_intercept=bces_out[0][1]
    #cov2=bces_out[2][1][1]
    
    bces_r,bces_rho,bces_rchisq,bces_scat,bces_iscat,bces_r2 = nmmn.stats.fitstats(data_x, data_y, err_x, err_y, slope, intercept)
    res_var = nmmn.stats.scatterfit(data_x, data_y, slope, intercept)**2
    sc_err=np.sqrt(res_var/xrange)
    #https://www.aanda.org/articles/aa/pdf/2009/17/aa10994-08.pdf
    alpha=0.05
    dof = n-3
    rchisq = sc.stats.chi2.ppf(q=1-alpha, df=dof)/dof
    print('rchisq: '+str(bces_rchisq))
    print('crit rchisq: '+str(rchisq))
    #if odr_rchisq > rchisq, we are 100*(1-alpha)% confident in rejecting our model
    #http://physics.ucsc.edu/~drip/133/ch4.pdf

    print('\nBCES r2: '+str(bces_r2))

    if (np.abs(bces_r2)<0.5) or (np.abs(bces_r2)>1):# | (bces_r2<0.5):
        print('Linear fit is not good enough!')
    if (bces_rchisq>rchisq):
        print('Regression model is rejected for alpha='+str(alpha)+'.')    
    else:
        print('\n Orthogonal distance regression')
        print('Slope: '+str(slope)+'\nIntercept: '+str(intercept))
        print(str(info))
        
    reg_out = [slope, intercept, std_slope, std_intercept, [bces_r2,sc_err,res_var]]#[odr_r,odr_rho,odr_rchisq,odr_scat,odr_iscat,odr_r2,res_var]]
    if (graph is True) & (MT is False):
        figure = fg.keeling_plot([data_x, err_x], [data_y, err_y], reg_out, iso)
        #figure.savefig('/Users/malika/surfdrive/Data/Krakow/Roof air/figures/Keeling plots/'+iso+'_'+str(df_in.index[0])+'.png', dpi=100)
        figure.savefig('/Users/malika/surfdrive/Data/Krakow/Roof air/figures/2020-11-26 data_signatures/KP_all-data_d'+iso+'_'+str(model)+'.png', dpi=100)
        reg_out.append(figure)
    elif (graph is True) & (MT is True):
        figure = fg.keeling_plot([data_x, err_x], [data_y, err_y], reg_out, iso)
        figure.savefig('/Users/malika/surfdrive/Data/Krakow/Roof air/figures/2020-11-26 data_signatures/MT_all-data_d'+iso+'_'+str(model)+'.png', dpi=100)
        reg_out.append(figure)
    return reg_out


def extract_peaks(data, iso, size, min_h, wid, rel_h, ind_figs=False, model=None, suf=''):
    
    st_err_MR=10
    if iso=='D':
        nameMR=col_MR_dD
        delta=col_dD
        edelta=col_errdD
        str_delta=col_dD+permil
        st_err_delta=2.0
    if iso=='13C':
        nameMR=col_MR_d13C
        delta=col_d13C
        edelta=col_errd13C
        str_delta=col_d13C+permil
        st_err_delta=0.1
    
    data.rename(columns={list(data)[0]: nameMR}, inplace=True)
    data.rename(columns={list(data)[1]: delta}, inplace=True)
    if model==None:
        data.rename(columns={list(data)[2]: 'err_'+nameMR}, inplace=True)
        data.rename(columns={list(data)[3]: edelta}, inplace=True)
    else:
        data['err_'+nameMR]=st_err_MR
        data[edelta]=st_err_delta
    
    if data.empty==True:
        empty = pd.DataFrame(np.nan, index=['0'], columns=['start_DateTime', 'end_DateTime', delta+permil, 'err '+delta+permil, 'r2', 'n'])
        empty_fig = plt.figure()
        return empty, empty_fig
        
    #Overview figure
    big_f=plt.figure(figsize=(15,10))
    axm=big_f.add_subplot(111)
    # cut data first
    
    if model==None:
        sub, n_sub, new_sub = cut(data, 12*60, 300)      # Cut the data if no record for more than 12h, but no max time (300 days)
    
        list_peaks=[]
        list_params=[]
        list_dates=[]
        list_startdates=[]
        list_enddates=[]
        for s in range(len(new_sub)):
            
            d1=new_sub.iloc[s]
            # For the last subset, get the last date
            if s==len(new_sub)-1:
                d2=data.index[-1]
            else:
                d2=new_sub.iloc[s+1]
            
            data_sub = data[d1:d2]
            
            # Get the peaks
            peak, p_peak = scgl.find_peaks(data_sub[nameMR], height=min_h, prominence=size, width=wid, rel_height=rel_h)
            #print(peak)
            
            startdates=data_sub.iloc[p_peak["left_ips"]].index
            enddates=data_sub.iloc[p_peak["right_ips"]].index
            diffs=enddates-startdates
            dates=startdates+diffs/2
    
            # Figure for overview
            prominences = scgl.peak_prominences(data_sub[nameMR], peak)[0]
            axm.plot(data_sub[nameMR], '.', markersize=2,  mec=cP, mfc=cP)
            axm.plot(dates, p_peak['peak_heights'], 'x', c='r')
            axm.fill([startdates, enddates, enddates, startdates], [data_sub.iloc[peak][nameMR]-prominences, data_sub.iloc[peak][nameMR]-prominences, data_sub.iloc[peak][nameMR]+prominences, data_sub.iloc[peak][nameMR]+prominences], c='r', alpha=0.3)
            axm.hlines(y=p_peak["width_heights"], xmin=startdates, xmax=enddates, color = 'r')
            #axm.set_xlabel()
            axm.set_ylabel(str_MR)
            
            # Save the peaks start, peaks end and values
            list_startdates.append(pd.Series(startdates))
            list_enddates.append(pd.Series(enddates))
            list_dates.append(pd.Series(dates))
            list_peaks.append(pd.Series(peak))
            list_params.append(pd.DataFrame(p_peak))
        
        peaks=pd.concat(list_peaks).reset_index(drop=True)
        p_peaks=pd.concat(list_params).reset_index(drop=True)
        p_startdates=pd.concat(list_startdates).reset_index(drop=True)
        p_enddates=pd.concat(list_enddates).reset_index(drop=True)
        p_dates=pd.concat(list_dates).reset_index(drop=True)
        
    else:
        print(data)
        peaks, p_peaks = scgl.find_peaks(data[nameMR], height=min_h, prominence=size, width=wid, rel_height=rel_h)
        p_startdates=data.iloc[p_peaks["left_ips"]].index
        p_enddates=data.iloc[p_peaks["right_ips"]].index
        p_dates=p_startdates+(p_enddates-p_startdates)/2
        # Figure for overview
        prominences = scgl.peak_prominences(data[nameMR], peaks)[0]
        axm.plot(data[nameMR], '.', markersize=2,  mec=cP, mfc=cP)
        axm.plot(p_dates, p_peaks['peak_heights'], 'x', c='r')
        axm.fill([p_startdates, p_enddates, p_enddates, p_startdates], [data.iloc[peaks][nameMR]-prominences, data.iloc[peaks][nameMR]-prominences, data.iloc[peaks][nameMR]+prominences, data.iloc[peaks][nameMR]+prominences], c='r', alpha=0.3)
        axm.hlines(y=p_peaks["width_heights"], xmin=p_startdates, xmax=p_enddates, color = 'r')
        #axm.set_xlabel()
        axm.set_ylabel(str_MR)
    
    #Get the peaks and apply Keeling plot on them
    sigs=[]
    std_intercepts=[]
    slopes=[]
    r2s=[]
    res_vars=[]
    n_points=[]
    wds=[]
    err_wds=[]
    wss=[]
    err_wss=[]
    
    for i in range(len(peaks)):
        # Get the data points within that peak
        peak_data=data[p_startdates[i]:p_enddates[i]]
        ext_data=data[p_startdates[i]-timedelta(hours=24):p_enddates[i]+timedelta(hours=24)]
        bg_lim=np.percentile(ext_data[nameMR].dropna(), 10)
        bg_data=ext_data.loc[ext_data[nameMR]<bg_lim]
        # Add some background
        dataset=pd.concat([peak_data, bg_data], axis=0).drop_duplicates().sort_index()
        #dataset=data[p_startdates[i]:p_enddates[i]]
        #dataset.dropna(subset=[nameMR], inplace=True)
            
        myoutput = regression(dataset, iso, graph=ind_figs, model=model)

        # save the relevant output values
        slope=myoutput[0]
        intercept=myoutput[1]
        std_intercept=myoutput[3]
        
        slopes.append(slope)
        sigs.append(intercept)
        std_intercepts.append(std_intercept)
        r2s.append(myoutput[4][0])
        res_vars.append(myoutput[4][2])
        n_points.append(len(dataset))
        
        # get wind information only during the peak event, not with BG!!!
        wds.append(np.mean(peak_data[col_wd]))
        err_wds.append(np.std(peak_data[col_wd]))
        wss.append(np.mean(peak_data[col_ws]))
        err_wss.append(np.std(peak_data[col_ws]))

        if ind_figs==True:
            f=myoutput[5]
            f.savefig('/Users/malika/surfdrive/Data/Krakow/Roof air/figures/Keeling plots/PKE/'+iso+'_'+str(p_dates[i])+'.png', dpi=100)
    
    df=pd.DataFrame({'DateTime'+suf: p_dates, 'start_DateTime'+suf: p_startdates,'end_DateTime'+suf: p_enddates, delta+suf: sigs, edelta+suf: std_intercepts, 'slope'+suf:slopes, 'r2'+suf:r2s, 'res'+suf:res_vars, 'n'+suf:n_points, col_wd+suf:wds, col_errwd+suf:err_wds, col_ws+suf:wss, col_errws+suf:err_wss},
                    columns=['DateTime'+suf, 'start_DateTime'+suf, 'end_DateTime'+suf, delta+suf, edelta+suf, 'slope'+suf, 'r2'+suf, 'res'+suf, 'n'+suf, col_wd+suf, col_errwd+suf, col_ws+suf, col_errws+suf], index=range(len(peaks)))
    
    return df, big_f, p_peaks


def clear_crazy(df_d13C_in, range_d13C, lim_ed13C, df_dD_in, range_dD, lim_edD, suf=''):
    
    df_d13C_clear = df_d13C_in.loc[(df_d13C_in[col_d13C+suf]<range_d13C[1]) & (df_d13C_in[col_d13C+suf]>range_d13C[0]) & (df_d13C_in[col_errd13C+suf]<lim_ed13C)]# | (df_in[col_d13C].isnull() & df_in[col_errd13C].isnull())]
    df_dD_clear = df_dD_in.loc[(df_dD_in[col_dD+suf]<range_dD[1]) & (df_dD_in[col_dD+suf]>range_dD[0])  & (df_dD_in[col_errdD+suf]<lim_edD)]# | (df_in[col_dD].isnull() & df_in[col_errdD].isnull()) ]
    
    return df_d13C_clear, df_dD_clear


def two_isotopes(peaks_D, peaks_13C, t_lim, iso=None, suf=''):
    
    if len(peaks_D)<len(peaks_13C):
        small=peaks_D
        large=peaks_13C
        col_t_l=col_dt_d13C+suf
        col_t_s=col_dt_dD+suf
        str_l='_d13C'
        str_s='_dD'
        delta_s=col_dD+suf
        delta_l=col_d13C+suf
        errdelta_s=col_errdD+suf
        errdelta_l=col_errd13C+suf
        winddir_l=col_wd+'_d13C'+suf
        errwinddir_l=col_errwd+'_d13C'+suf
        windspeed_l=col_ws+'_d13C'+suf
        errwindspeed_l=col_errws+'_d13C'+suf
        winddir_s=col_wd+'_dD'+suf
        errwinddir_s=col_errwd+'_dD'+suf
        windspeed_s=col_ws+'_dD'+suf
        errwindspeed_s=col_errws+'_dD'+suf
        if iso=='D': # it means we have only one isotope & model is the largest
            col_t_l=col_dt_dD+suf
            delta_l=col_dD+suf
            errdelta_l=col_errdD+suf
            winddir_l=col_wd+'_dD'+suf
            errwinddir_l=col_errwd+'_dD'+suf
            windspeed_l=col_ws+'_dD'+suf
            errwindspeed_l=col_errws+'_dD'+suf
        elif iso=='13C': # it means we have only one isotope & model is the largest
            col_t_s=col_dt_d13C+suf
            delta_s=col_d13C+suf
            errdelta_s=col_errd13C+suf
            winddir_s=col_wd+'_d13C'+suf
            errwinddir_s=col_errwd+'_d13C'+suf
            windspeed_s=col_ws+'_d13C'+suf
            errwindspeed_s=col_errws+'_d13C'+suf
        if iso is not None:
            str_l='_model'
            str_s='_obs'
            
    elif len(peaks_D)>=len(peaks_13C):
        small=peaks_13C
        large=peaks_D
        col_t_l=col_dt_dD+suf
        col_t_s=col_dt_d13C+suf
        str_s='_d13C'
        str_l='_dD'
        delta_l=col_dD+suf
        delta_s=col_d13C+suf
        errdelta_l=col_errdD+suf
        errdelta_s=col_errd13C+suf
        winddir_l=col_wd+'_dD'+suf
        errwinddir_l=col_errwd+'_dD'+suf
        windspeed_l=col_ws+'_dD'+suf
        errwindspeed_l=col_errws+'_dD'+suf
        winddir_s=col_wd+'_d13C'+suf
        errwinddir_s=col_errwd+'_d13C'+suf
        windspeed_s=col_ws+'_d13C'+suf
        errwindspeed_s=col_errws+'_d13C'+suf
        if iso=='D': # it means we have only one isotope & model is the shortest
            col_t_s=col_dt_dD+suf
            delta_s=col_dD+suf
            errdelta_s=col_errdD+suf
            winddir_s=col_wd+'_dD'+suf
            errwinddir_s=col_errwd+'_dD'+suf
            windspeed_s=col_ws+'_dD'+suf
            errwindspeed_s=col_errws+'_dD'+suf
        elif iso=='13C': # it means we have only one isotope & model is the shortest
            col_t_l=col_dt_d13C+suf
            delta_l=col_d13C+suf
            errdelta_l=col_errd13C+suf
            winddir_l=col_wd+'_d13C'+suf
            errwinddir_l=col_errwd+'_d13C'+suf
            windspeed_l=col_ws+'_d13C'+suf
            errwindspeed_l=col_errws+'_d13C'+suf
        if iso is not None:
            str_l='_obs'
            str_s='_model'
        
    small.reset_index(inplace=True)
    large.reset_index(inplace=True)
    id_large=[]
    new_large=pd.DataFrame(index=small.index)

    for t in range(len(small)):
        id_large.append((np.abs(large[col_t_l] - small.iloc[t][col_t_s])).idxmin())
        # For each peak on the small vector, get the closest peak in the large vector

    print(id_large)
    new_large=large.iloc[id_large] # Save the large vector peaks that were matched

    new = pd.merge(small, new_large, on=small.index, suffixes=[str_s, str_l])
    
    diffs=[]

    for t in range(len(new)):
        if iso is not None:
            diff = new[col_t_s+str_s][t]-new[col_t_l+str_l][t]
        else:
            diff = new[col_t_s][t]-new[col_t_l][t]
        diffs.append(np.abs(diff.days*24*3600+diff.seconds))
        #diffs.append(str(np.abs(diff)))
        # Calculate the time difference between each d13C and dD peak
    
    new.insert(0, 'time diff', diffs)

    t_lim = t_lim*60*60

    new_skeemed = new.loc[new['time diff']<t_lim]  # remove the peaks that were matched but are more than t_lim appart

    if iso is not None:
        df_l = pd.DataFrame({'DateTime_d'+iso+suf: new_skeemed[col_t_l+str_l], 'start_DateTime_d'+iso+suf: new_skeemed['start_DateTime_d'+iso+suf+str_l],'end_DateTime_d'+iso+suf: new_skeemed['end_DateTime_d'+iso+suf+str_l], delta_l: new_skeemed[delta_l+str_l], errdelta_l: new_skeemed[errdelta_l+str_l], winddir_l: new_skeemed[winddir_l+suf+str_l], errwinddir_l: new_skeemed[errwinddir_l+suf+str_l], windspeed_l: new_skeemed[windspeed_l+suf+str_l], errwindspeed_l: new_skeemed[windspeed_l+suf+str_l]})
        df_s = pd.DataFrame({'DateTime_d'+iso+suf: new_skeemed[col_t_s+str_s], 'start_DateTime_d'+iso+suf: new_skeemed['start_DateTime_d'+iso+suf+str_s],'end_DateTime_d'+iso+suf: new_skeemed['end_DateTime_d'+iso+suf+str_s], delta_s: new_skeemed[delta_s+str_s], errdelta_s: new_skeemed[errdelta_s+str_s], winddir_s: new_skeemed[winddir_s+suf+str_s], errwinddir_s: new_skeemed[errwinddir_s+suf+str_s], windspeed_s: new_skeemed[windspeed_s+suf+str_s], errwindspeed_s: new_skeemed[windspeed_s+suf+str_s]})
        if str_l=='_obs':
            #df_out = [df_l, df_s]
            df_out = pd.concat([df_l, df_s], axis=1)
            df_out.columns=['DateTime_d'+iso+str_l, 'start_DateTime_d'+iso+str_l, 'end_DateTime_d'+iso+str_l, delta_l+str_l, errdelta_l+str_l, winddir_l, errwinddir_l, windspeed_l, errwindspeed_l,
                            'DateTime_d'+iso+str_s, 'start_DateTime_d'+iso+str_s, 'end_DateTime_d'+iso+str_s, delta_s+str_s, errdelta_s+str_s, winddir_s, errwinddir_s, windspeed_s, errwindspeed_s]
        elif str_l=='_model':
            #df_out = [df_s, df_l]
            df_out = pd.concat([df_s, df_l], axis=1)
            df_out.columns=['DateTime_d'+iso+str_s, 'start_DateTime_d'+iso+str_s, 'end_DateTime_d'+iso+str_s, delta_s+str_s, errdelta_s+str_s, winddir_s, errwinddir_s, windspeed_s, errwindspeed_s,
                            'DateTime_d'+iso+str_l, 'start_DateTime_d'+iso+str_l, 'end_DateTime_d'+iso+str_l, delta_l+str_l, errdelta_l+str_l, winddir_l, errwinddir_l, windspeed_l, errwindspeed_l]
    else:
        df_out = new_skeemed
        
    return new, df_out


def dt(serie, strt, t1, t2):
    
    selected_serie = serie.loc[(serie[strt]>t1) & (serie[strt]<t2)]
    
    return selected_serie


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




