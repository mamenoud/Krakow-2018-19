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

#col_dt='DateTime'
col_dt='filling time UTC'
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


def regression(df_in, iso, graph=False, model=None):
    
    df_in.replace(0.0, 1e-8, inplace=True)
    print(list(df_in))
    data_MR = df_in[list(df_in)[0]]
    data_y = df_in[list(df_in)[1]]
    if model==None:
        err_MR = df_in[list(df_in)[2]]
        err_y = df_in[list(df_in)[3]]
    else:
        err_MR = pd.Series([np.nan]*len(df_in), index=df_in.index)
        err_y = pd.Series([np.nan]*len(df_in), index=df_in.index)
    
    st_err_MR=10
    if iso=='D':
        st_err_delta=2.0
    if iso=='13C':
        st_err_delta=0.1
    
    err_MR.replace(np.nan, st_err_MR, inplace=True)
    err_y.replace(np.nan, st_err_delta, inplace=True)
    data_x, err_x = inverseMR(data_MR, err_MR)
    
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
    bces_out=lnr4.bces(data_x,data_y,err_x,err_y, nsim=5000, model='yx', bootstrap=False, verbose='normal', full_output=True)
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
    if graph is True:
        figure = fg.keeling_plot([data_x, err_x], [data_y, err_y], reg_out, iso)
        #figure.savefig('/Users/malika/surfdrive/Data/Krakow/Roof air/figures/Keeling plots/'+iso+'_'+str(df_in.index[0])+'.png', dpi=100)
        reg_out.append(figure)
    
    return reg_out


def moving_window(data, iso, time, n, lim_err, MRwindow, bg, ind_figs=False):
    
    st_err_MR=1
    if iso=='D':
        #c=c4
        ymar=0.05
        nameMR=col_MR_dD
        delta=col_dD
        edelta=col_errdD
        sy0=0.1
        sx0=1e-8
        str_delta=col_dD+permil
        st_err_delta=2.0
    if iso=='13C':
        #c=c2
        ymar=0.01
        nameMR=col_MR_d13C
        delta=col_d13C
        edelta=col_errd13C
        sy0=0.005
        sx0=1e-8
        str_delta=col_d13C+permil
        st_err_delta=0.1
    
    data.rename(columns={list(data)[0]: nameMR}, inplace=True)
    data.rename(columns={list(data)[1]: delta}, inplace=True)
    if len(list(data))>2:
        data.rename(columns={list(data)[2]: 'err_'+nameMR}, inplace=True)
        data.rename(columns={list(data)[3]: edelta}, inplace=True)
    else:
        data['err_'+nameMR]=st_err_MR
        data[edelta]=st_err_delta
    
    if data.empty==True:
        empty = pd.DataFrame(np.nan, index=['0'], columns=['start_DateTime', 'end_DateTime', delta+permil, 'err '+delta+permil, 'r2', 'n'])
        empty_fig = plt.figure()
        return empty, empty_fig
    
    time_s=timedelta(hours=time)
    
    xmar=0.01 
    
    xmin=1
    xmax=0
    
    ymin=0
    ymax=-500
    
    sigmin=ymin
    sigmax=ymax

    # create a table that can contain the background values (during the day from 10 to 18h)
    #data_bg=data.between_time('10:00:00', '18:00:00')
    data_bg=data
    MRmin=np.percentile(data[nameMR],10)
    
    #fig, (ax1, ax2, ax3a, ax3b)= plt.subplots(2,2, gridspec_kw = {'width_ratios':[1, 5]}, figsize=(15,10))
    #fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    
    gs=gridspec.GridSpec(2,6)
    fig=plt.figure(figsize=(15,10), tight_layout=True)
    ax1=plt.subplot(gs[0,0])
    ax2=plt.subplot(gs[0,1:])
    ax3=plt.subplot(gs[1,:])
    
    #########################
    # Add the colored MR data, before you select parts! we want it all
    #########################
    data.plot(y=nameMR, ax=ax3, markersize=2, style='o',  mec=cP, mfc=cP, legend=False, clip_box=[0,1,1,0])
    data.dropna(inplace=True, subset=[delta, nameMR])
    
    sig=[]
    std_intercepts=[]
    mu=[]
    startdates=[]
    enddates=[]
    dates=[]
    n_points=[]
    n_tot=len(data)
    slopes=[]
    errs=[]
    r2s=[]
    wds=[]
    err_wds=[]
    wss=[]
    err_wss=[]
    
    date0 = data.index[0]
    n_hours = np.round((data.index[-1]-date0).days*24 + (data.index[-1]-date0).seconds/3600)
    hours = date0 + pd.to_timedelta(time/2, 'h') + pd.to_timedelta(np.arange(n_hours), 'h')
    times = pd.Series(data.index)
    
    cm=plt.get_cmap('jet', n_hours)
    
    i=0
    j=0
    
    for h in hours:
        
        print('h: '+str(h))
        
        n_trials=1
        
        while True:

            # Time-based window 
            t_mid=h
            t1=t_mid-time_s/2+(timedelta(hours=n_trials-1))        # start time, with a potentially reduced window 
            t2=t_mid+time_s/2-(timedelta(hours=n_trials-1))         # end time
            
            print(str(t1))
            print(str(type(t1)))
            print(str(t2))
            print(str(type(t2)))
            # select data from the time range we want
            set_window=data[t1:t2]
            #print(set_window)
            
            # one-shot conditions
            ###################################################
            # is the dataset consisting in at least n points?
            if len(set_window)<n:
                print('Not enough data points!')
                break
            
            # add some background from the surrounding 48h (±24)
            deltat=pd.to_timedelta(24, unit='h')
            set_bg=data_bg.loc[data_bg[nameMR]<MRmin]
            
            # gather the set of points
            dataset=pd.concat([set_window, set_bg[(t1-deltat):(t2+deltat)]])
            dataset.drop_duplicates(inplace=True)
            #print(dataset)
            
            # get the CH4 mixing ratios
            MRs=dataset[nameMR]
            
            # is the window of MR wide enough?
            MRwidth=max(MRs)-min(MRs)
            if MRwidth<MRwindow:
                print('Range of [CH4] is too small!')
                # refuse the selected set of MR
                break
            
            c=cm(j)
        
            # loop conditions
            ###################################################
            
            y=dataset[delta]
            sx=dataset['err_'+nameMR]
            sy=dataset[edelta]
            sy = sy.replace(0.0, sy0)
            sx = sx.replace(0.0, sx0)
            
            myoutput = regression(dataset, iso, graph=ind_figs)

            # save the relevant output values 
            slope = myoutput[0]
            intercept = myoutput[1]
            std_err = myoutput[2]
            std_intercept = myoutput[3]
            r2=myoutput[4][0]
            error=myoutput[4][1]
            #error=std_intercept/np.sqrt(sc)
            #error_1=myoutput[4][0]
            #error_2=myoutput[4][1]
            #error_3=myoutput[4][2]
            #error_4=myoutput[4][3]
            #error_5=myoutput[4][4]
            #error_rv=myoutput[4][6]
            
            #if error>10:
            #    regression(dataset, iso, graph=True)
            
            if np.isnan(slope)==True:
                print('No linear regression made!\n')
                continue
        
            if (np.abs(error)>lim_err):
                print('Linear fit is not good enough!')
                n_trials=n_trials+1
                continue
            else:
                break
        
        #print(info)
        j=j+1
        
        if len(set_window)<n:
            print('Not enough data points!')
            continue

        if MRwidth<MRwindow:
            print('Range of [CH4] is too small!')
            # refuse the selected set of MR
            continue
            
        #errs_all.append(error)#[error_1, error_2, error_3, error_4, error_5, error, error_rv])
        r2s.append(r2)
        errs.append(error)
        print('Keeling plot found: '+str(intercept)+permil)
        #cm=plt.get_cmap('jet', j)
        
        i=i+1
        # If you want individual figures, uncomment the following lines
        #figs, (ax1, ax2)= plt.subplots(1,2, gridspec_kw = {'width_ratios':[1, 5]}, figsize=(20,10))
        #figs.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
    
        # Get the peak start and end dates, before adding the background
        datestart=set_window.index[0]
        startdates.append(datestart)
        dateend=set_window.index[len(set_window)-1]
        enddates.append(dateend)
        dates.append(datestart+(dateend-datestart)/2)
        # Based on the fact that std_err is the error of the calculated slope, we calculate the error of the intercept (see formula in Wikipedia - Linear regression)
        n_points.append(len(dataset))
        slopes.append(slope)               # save the statistical significiance parameter

        wds.append(np.mean(dataset[col_wd]))
        err_wds.append(np.std(dataset[col_wd]))
        wss.append(np.mean(dataset[col_ws]))
        err_wss.append(np.std(dataset[col_ws]))

        sig.append(intercept)
        std_intercepts.append(std_intercept)
        
        # Make the regression line
        r_x, r_y = zip(*((j, j*slope + intercept) for j in range(len(dataset))))
        lm = pd.DataFrame({'x': r_x, 'y': r_y})
        lm.index=lm['x']
            
        # Draw the datapoints that are lineary fitted
        dataset[MRx]=1/dataset[nameMR]
        x=dataset[MRx]
        dataset.plot(style='.', marker='d', x=MRx, y=delta, ax = ax2, legend=False, color=c)

        # Draw the regression line
        lm.plot(kind='line', x='x', y='y', ax=ax2, style='--', legend=False, color=c)
        lm.plot(kind='line', x='x', y='y', ax=ax1, style='--', legend=False, color=c)

        ax1.set_ylabel(str_delta)
        ax2.set_ylabel('')
        ax1.set_xlabel('')
        ax2.set_xlabel('1/x(CH4) [1/ppb]')
        #ax2.set_xlabel(str_MRx)
            
        # Color the points in the bottom graph of [CH4] data
        dataset.plot(y=nameMR, ax=ax3, markersize=5, style='o',  mec=c, mfc=c, legend=False, clip_box=[0,1,1,0]) # overwrite the mixing ratios with the colored ones        
        
        # For saving individual figures
        #ax1.legend(bbox_to_anchor=(7, 1), loc=2)
        ax2.legend([str(t_mid), 'Intercept: '+str(round(intercept, 3))+' '+pm+' '+str(round(std_intercept, 3))+permil],
                    fontsize='xx-small')
        
        if min(x)<xmin:
            xmin=min(x)
        if max(x)>xmax:
            xmax=max(x)
            
        if min(y)<ymin:
            ymin=min(y)
        if max(y)>ymax:
            ymax=max(y)

        if min(sig)<sigmin:
            sigmin=min(sig)
        if max(sig)>sigmax:
            sigmax=max(sig)
            
        ##
        xsmall=round((xmax-xmin)/4, 5)
        xmid=(xmin+xmax)/2
        xmin1=(xmid+xmin)/2
        xmax1=(xmid+xmax)/2
        ymid=(ymin+ymax)/2
        sigmid=(sigmin+sigmax)/2
        ax1.set_xlim([int(0.0), xsmall])
        ax1.set_ylim([sigmin+ymar*sigmid, sigmax-ymar*sigmid])
            
        print('xmin: '+str(xmin-xmar*xmid))
        print('xmax: '+str(xmax+xmar*xmid))
        ax2.set_xlim([xmin-xmar*xmid, xmax+xmar*xmid])
         
        ax2.set_ylim([ymin+ymar*ymid, ymax-ymar*ymid])
        #ax2.set_ylabel()
        
        ax1.set_xticks([0])
        #ax2.set_xticks([xmin, xmin1, xmid, xmax1, xmax])
        ax2.set_xticks([xmin, xmax])
        
        # Save each figure for gif making
        #figs.tight_layout()
        #figs.savefig('2020-04-02 New/MWKP/individual plots/december/chimere_EDGAR_12h-d'+iso+'_keeling-plot_'+str(i)+'.png', dpi=100)
        ##
    
    #plt.suptitle(title, y=1.05)
    
    ax3.set_ylabel('x(CH4) [ppb]')
    
    df=pd.DataFrame({'DateTime': dates, 'start_DateTime': startdates,'end_DateTime': enddates, delta: sig, edelta: std_intercepts, 'slope':slopes, 'r2':r2s, 'n':n_points, col_wd:wds, col_ws:wss},
                    columns=['DateTime', 'start_DateTime', 'end_DateTime', delta, edelta, 'slope', 'r2', 'n', col_wd, col_ws])
    
    if len(df)==0:
        df=pd.DataFrame({'DateTime':'nan', 'start_DateTime': 'nan','end_DateTime': 'nan', delta: 'nan', edelta: 'nan', 'ss':'nan', 'n':'nan', col_wd:'nan', col_ws:'nan'}, columns=['DateTime', 'start_DateTime', 'end_DateTime', delta, 'err '+delta, 'slope', 'res_var', 'n', col_wd, col_ws], index=[0])
    else:
        df.drop_duplicates(inplace=True)
    
    fig.tight_layout()
    
    return df, fig, errs


def MW_peaks(df_MW, p_overlap, iso, delta_thr):
    
    p=p_overlap/100
    
    if iso=='13C':
        col_st='start_'+col_dt_d13C
        col_et='end_'+col_dt_d13C
        col_t=col_dt_d13C
        col_delta=col_d13C
        col_edelta=col_errd13C
        col_errdelta='err '+col_d13C
    elif iso=='D':
        col_st='start_'+col_dt_dD
        col_et='end_'+col_dt_dD
        col_t=col_dt_dD
        col_delta=col_dD
        col_edelta=col_errdD
        col_errdelta='err '+col_dD
    
    winddir=col_wd+'_d'+iso
    windsp=col_ws+'_d'+iso
    
    delta_t=df_MW[col_et]-df_MW[col_st]

    new_peak=[]
    peaks_id=[]
    peak_mean=[]
    peak_sd=[]
    peak_err=[]
    peak_ds=[]
    peak_de=[]
    peak_n=[]
    peak_wm=[]
    peak_wsd=[]
    j=0
    for i in range(1, len(df_MW)):
        if ((delta_t[i-1]*(1-p))>(df_MW.iloc[i][col_st]-df_MW.iloc[i-1][col_st]+df_MW.iloc[i][col_et]-df_MW.iloc[i-1][col_et])) and ((df_MW.iloc[i][col_delta]-df_MW.iloc[i-1][col_delta])<delta_thr):
            new_peak.append(False)
        else:
            new_peak.append(True)
            peaks_id.append(j)
            id1=j
            id2=i
            peak_mean.append(np.mean(df_MW.iloc[id1:id2][col_delta]))
            peak_sd.append(np.std(df_MW.iloc[id1:id2][col_delta]))
            peak_err.append(np.mean(df_MW.iloc[id1:id2][col_edelta]))
            peak_ds.append(df_MW.iloc[id1][col_st])
            peak_de.append(df_MW.iloc[id2-1][col_et])
            peak_n.append(len(df_MW.iloc[id1:id2]))
            peak_wm.append(np.mean(df_MW.iloc[id1:id2][winddir]))
            peak_wsd.append(np.std(df_MW.iloc[id1:id2][winddir]))
            j=i

    # The last peak
    peaks_id.append(j)
    peak_mean.append(np.mean(df_MW.iloc[j:i+1][col_delta]))
    peak_sd.append(np.std(df_MW.iloc[j:i+1][col_delta]))
    peak_err.append(np.mean(df_MW.iloc[j:i+1][col_edelta]))
    peak_ds.append(df_MW.iloc[j][col_st])
    peak_de.append(df_MW.iloc[i][col_et])
    peak_d=np.array(peak_ds) + (np.array(peak_de) - np.array(peak_ds)) / 2
    peak_n.append(len(df_MW.iloc[j:i+1]))
    peak_wm.append(np.mean(df_MW.iloc[j:i+1][winddir]))
    peak_wsd.append(np.std(df_MW.iloc[j:i+1][winddir]))
    
    plt.plot(peak_wsd, peak_sd, '.')
    # Fill-in the table
    df_peaks=pd.DataFrame({col_t: peak_d, col_st:peak_ds, col_et:peak_de, col_delta: peak_mean, col_errdelta:peak_err, col_edelta:peak_sd, 'n_sig':peak_n, winddir:peak_wm, 'SD '+winddir: peak_wsd}, 
                          index=range(len(peaks_id)))
    
    # +++++++++++++ faire un critere sur la variance au sein d'un peak
    
    return df_peaks


def extract_peaks(data, iso, size, min_h, wid, ind_figs=False, model=None):
    
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
            peak, p_peak = scgl.find_peaks(data_sub[nameMR], height=min_h, prominence=size, width=wid, rel_height=0.5)
            
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
        peaks, p_peaks = scgl.find_peaks(data[nameMR], height=min_h, prominence=size, width=wid, rel_height=0.5)
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
        wds.append(np.mean(dataset[col_wd]))
        err_wds.append(np.std(dataset[col_wd]))
        wss.append(np.mean(dataset[col_ws]))
        err_wss.append(np.std(dataset[col_ws]))

        if ind_figs==True:
            f=myoutput[5]
            #f.savefig('/Users/malika/surfdrive/Data/Krakow/Roof air/figures/Keeling plots/PKE/'+iso+'_'+str(p_dates[i])+'.png', dpi=100)
    
    df=pd.DataFrame({'DateTime': p_dates, 'start_DateTime': p_startdates,'end_DateTime': p_enddates, delta: sigs, edelta: std_intercepts, 'slope':slopes, 'r2':r2s, 'res':res_vars, 'n':n_points, col_wd:wds, col_ws:wss},
                    columns=['DateTime', 'start_DateTime', 'end_DateTime', delta, edelta, 'slope', 'r2', 'res', 'n', col_wd, col_ws], index=range(len(peaks)))
    
    return df, big_f, p_peaks


def clear_crazy(df_d13C_in, range_d13C, lim_ed13C, df_dD_in, range_dD, lim_edD):
    
    df_d13C_clear = df_d13C_in.loc[(df_d13C_in[col_d13C]<range_d13C[1]) & (df_d13C_in[col_d13C]>range_d13C[0]) & (df_d13C_in[col_errd13C]<lim_ed13C)]# | (df_in[col_d13C].isnull() & df_in[col_errd13C].isnull())]
    df_dD_clear = df_dD_in.loc[(df_dD_in[col_dD]<range_dD[1]) & (df_dD_in[col_dD]>range_dD[0])  & (df_dD_in[col_errdD]<lim_edD)]# | (df_in[col_dD].isnull() & df_in[col_errdD].isnull()) ]
    
    return df_d13C_clear, df_dD_clear


def two_isotopes(peaks_D, peaks_13C, t_lim, iso=None):
    
    if len(peaks_D)<len(peaks_13C):
        small=peaks_D
        large=peaks_13C
        col_t_l=col_dt_d13C
        col_t_s=col_dt_dD
        str_l='_d13C'
        str_s='_dD'
        delta_s=col_dD
        delta_l=col_d13C
        errdelta_s=col_errdD
        errdelta_l=col_errd13C
        if iso=='D':
            col_t_l=col_dt_dD
            delta_l=col_dD
            errdelta_l=col_errdD
        elif iso=='13C':
            col_t_s=col_dt_d13C
            delta_s=col_d13C
            errdelta_s=col_errd13C
        if iso is not None:
            str_l='_model'
            str_s='_obs'
            
    elif len(peaks_D)>=len(peaks_13C):
        small=peaks_13C
        large=peaks_D
        col_t_l=col_dt_dD
        col_t_s=col_dt_d13C
        str_s='_d13C'
        str_l='_dD'
        delta_l=col_dD
        delta_s=col_d13C
        errdelta_l=col_errdD
        errdelta_s=col_errd13C
        if iso=='D':
            col_t_s=col_dt_dD
            delta_s=col_dD
            errdelta_s=col_errdD
        elif iso=='13C':
            col_t_l=col_dt_d13C
            delta_l=col_d13C
            errdelta_l=col_errd13C
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
        df_l = pd.DataFrame({'DateTime_d'+iso: new_skeemed[col_t_l+str_l], 'start_DateTime_d'+iso: new_skeemed['start_DateTime_d'+iso+str_l],'end_DateTime_d'+iso: new_skeemed['end_DateTime_d'+iso+str_l], delta_l: new_skeemed[delta_l+str_l], errdelta_l: new_skeemed[errdelta_l+str_l]})
        df_s = pd.DataFrame({'DateTime_d'+iso: new_skeemed[col_t_s+str_s], 'start_DateTime_d'+iso: new_skeemed['start_DateTime_d'+iso+str_s],'end_DateTime_d'+iso: new_skeemed['end_DateTime_d'+iso+str_s], delta_s: new_skeemed[delta_s+str_s], errdelta_s: new_skeemed[errdelta_s+str_s]})
        if str_l=='_obs':
            df_out = [df_l, df_s]
        elif str_l=='_model':
            df_out = [df_s, df_l]
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




