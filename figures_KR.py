#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:32:39 2018

@author: malika
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import os, shutil
import scipy as sc
import scipy.odr.odrpack as odrpack
from scipy import stats
from scipy.stats import t as studentt
from pandas.tools.plotting import table
import datetime

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

# possibilities are ‘xx-small’, ‘x-small’, ‘small’, ‘medium’, ‘large’, ‘x-large’, ‘xx-large’

c1='xkcd:bright blue'
c2='xkcd:brick orange'
c3='xkcd:light purple'
c4='xkcd:wine'
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

def overview(raw1, raw2, rawP):
    
    date_start = min(min(raw1.index), min(raw2.index))
    date_end = max(max(raw1.index), max(raw2.index))
    
    rawP = rawP.loc[(rawP.index>date_start) & (rawP.index<date_end)]
    
    plt.rcParams.update({'font.size': 22})

    # Check if the grey data should be on the same figure (Picarro for comparison) or another below (wind data)
    
    figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 10), sharex=True)
    ax1=axes[0]
        #ax0=ax1.twinx()
    ax2=axes[1]
    ax3=axes[2]
    
    # Graph of the mixing ratios (top)
    
    rawP.plot(y=col_MR_P, ax=ax1, color=cP, lw=0.7)
    raw1.plot(y=col_MR_d13C, ax=ax1, markersize=1, style='o',  mec=c2, mfc=c2)
    raw2.plot(y=col_MR_dD, ax=ax1, markersize=1, style='o', mec=c4, mfc=c4)
    #raw3.plot(y=MR3, style='o', ax=ax1, markersize=1,  mec=c3, mfc=c3)
    #raw4.plot(y=MR4, style='o', ax=ax1, markersize=1,  mec=c4, mfc=c4)
    
    ax1.set_ylabel('x(CH4) [ppb]')
    ax1.legend(fontsize=15)

    # Graph of the C isotopic signatures
    
    raw1.plot(y=col_d13C, markersize=1, style='o', legend=False, ax=ax2, mec=c2, mfc=c2)

    ax2.set_ylabel(d13C+' [\u2030]')
    ax2.legend(fontsize=15)

    # Graph of the H isotopic signatures

    raw2.plot(y=col_dD, markersize=1, style='o', ax=ax3,   mec=c4, mfc=c4)

    ax3.set_ylabel(dD+' [\u2030]')
    ax3.legend(fontsize=15)
    ax3.xaxis_date()
    #ax3.format_xdata = mdates.DateFormatter('%Y-%b')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
    ax3.xaxis.set_major_locator(plt.MaxNLocator(10))
    figure.autofmt_xdate()
    
    ax1.yaxis.grid()
    ax2.yaxis.grid()
    ax3.yaxis.grid()
    
    plt.suptitle('Overview of the data-set')

    return figure, axes


def overview_diff(table1, table2, table3, table4, steps1, steps2, steps3, steps4):
        
    #figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 10), sharex=True)
    #ax1=axes[0]
    
    def dateindex(table):
        table.index=table[list(table)[0]]
        return table
    
    table1=dateindex(table1)
    table2=dateindex(table2)
    table3=dateindex(table3)
    table4=dateindex(table4)

    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(20, 10), sharex=True)

    ax2=axes[0]
    ax3=axes[1]

    ax2bis=ax2.twinx()
    ax3bis=ax3.twinx()
    
    # Plot the difference

    #picarro.plot(y='picarro', x='DateTime', ax=ax1, color=cP, lw=0.7)
    #raw_13C_1.plot(y=str1, ax=ax1, markersize=1, style='o',  mec=c1, mfc=c1)
    #raw_13C_2.plot(y=str2, ax=ax1, markersize=1, style='o', mec=c2, mfc=c2)
    #raw_D_1.plot(y=str3, style='o', ax=ax1, markersize=1,  mec=c3, mfc=c3)
    #raw_D_2.plot(y=str4, style='o', ax=ax1, markersize=1,  mec=c4, mfc=c4)
    #ax1.set_ylabel('x(CH4) [ppb]')

    table1.plot(y='diff', ax=ax2, markersize=1, style='o',  mec=c1, mfc=c1, legend=False)
    table2.plot(y='diff', ax=ax2, markersize=1, style='o',  mec=c2, mfc=c2, legend=False)

    table3.plot(y='diff', ax=ax3, markersize=1, style='o',  mec=c3, mfc=c3, legend=False)
    table4.plot(y='diff', ax=ax3, markersize=1, style='o',  mec=c4, mfc=c4, legend=False)
    ax2.set_ylabel('x(CH4)_diff(picarro-IRMS) [ppb]')
    ax3.set_ylabel('x(CH4)_diff(picarro-IRMS) [ppb]')

    table1.plot(y='t_diff_m', ax=ax2bis, markersize=1, style='o',  mec=vertclair, mfc=vertclair, legend=False)
    table2.plot(y='t_diff_m', ax=ax2bis, markersize=1, style='o',  mec=vert, mfc=vert, legend=False)
    
    table3.plot(y='t_diff_m', ax=ax3bis, markersize=1, style='o',  mec=vertclair, mfc=vertclair, legend=False)
    table4.plot(y='t_diff_m', ax=ax3bis, markersize=1, style='o',  mec=vert, mfc=vert, legend=False)
    ax2bis.set_ylabel('comparison time diff [min]', color=vert)
    ax3bis.set_ylabel('comparison time diff [min]', color=vert)
    
    ax2.axhline(y=0, xmin=0, xmax=1, color=cP)
    ax3.axhline(y=0, xmin=0, xmax=1, color=cP)
    
    # put a vertical line at the begining of each subset
    ycoords2 = [min(min(table1['diff']), min(table2['diff'])), 
               max(max(table1['diff']), max(table2['diff']))]
    for i1 in range(len(steps1)):
        xcoords2 = [steps1['Date'][i1], steps1['Date'][i1]]
        ax2.plot(x=xcoords2, y=ycoords2, style='-')
    
    
    return figure


def overview_sub(sub_13C, sub_D):
    
    plt.rcParams.update({'font.size': 22})
    
    c13C=c2
    cD=c4
    
    figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,10), sharex=True)
    ax1=axes[0]
    ax2=axes[1]
    ax3=axes[2]
    
    #sub_P.plot(y='picarro', ax=ax1, color=cP, lw=0.7)
    if sub_13C.empty==False:
        sub_13C.plot(y=col_MR_d13C, ax=ax1, markersize=1, style='o',  mec=c13C, mfc=c13C)
        sub_13C.plot(y=col_d13C, markersize=1.5, style='o', legend=False, ax=ax2, mec=c13C, mfc=c13C)
    if sub_D.empty==False:
        sub_D.plot(y=col_MR_dD, ax=ax1, markersize=1, style='o',  mec=cD, mfc=cD)
        sub_D.plot(y=col_dD, markersize=1.5, style='o', legend=False, ax=ax3, mec=cD, mfc=cD)
    
    ax1.set_ylabel('x(CH4) [ppb]')
    ax1.legend(['[CH4] IRMS '+d13C, '[CH4] IRMS '+dD], markerscale=5, loc=1, fontsize='x-small')
    
    ax2.set_ylabel(d13C+permil)
    ax2.legend(['IRMS '+d13C], markerscale=5, loc=3, fontsize='x-small')
    
    ax3.set_ylabel(dD+permil)
    ax3.set_xlabel('Months')
    ax3.legend(['IRMS '+dD], markerscale=5, loc=3, fontsize='x-small')
    
    return figure


def keeling(data, iso, strID):
    
    plt.rcParams.update({'font.size': 10})
    
    if iso=='D':
        
        color=c4
        strd=dD
        margin=40
        ymar=0.1
        
        delta=col_dD
        edelta=col_errdD
        
        str_delta=col_dD+permil
        
    if iso=='13C':
        
        color=c2
        strd=d13C
        margin=5
        ymar=0.05
        
        delta=col_d13C
        edelta=col_errd13C
        
        str_delta=col_d13C+permil

    if data.empty==True:
        empty_fig = plt.figure()
        return np.nan, np.nan, np.nan, empty_fig
    
    xmar=0.1
    
    fig, (ax1, ax2)= plt.subplots(1,2, gridspec_kw = {'width_ratios':[1, 5]})
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    
    sig=[]
    mu=[]

    x=data[MRx]
    sx=data[eMRx]
    y=data[delta]
    sy=data[edelta]
    
    slope, intercept, r_value, p_value, std_err = sc.stats.linregress(x, y)
    print('Satandard linear regression')
    print('Slope: '+str(slope)+'\nIntercept: '+str(intercept))
    
    # Set up linear model
    linear = odrpack.Model(f)
    
    # run an ordinary distance regression model
    mydata = odrpack.RealData(x, y, sx=sx, sy=sy)
    myodr = odrpack.ODR(mydata, linear, beta0=(slope, intercept))
    myoutput = myodr.run()
        
    # save the relevant output values 
    #slope = myoutput.beta[0]
    #intercept = myoutput.beta[1]
    #std_err = myoutput.sd_beta[0]
    #std_intercept = myoutput.sd_beta[1]
    #sum_square=myoutput.sum_square
    #info=myoutput.stopreason
        
    sx2 = (x**2).sum()
    sy2 = (y**2).sum()
    std_intercept = std_err * np.sqrt(sx2/len(x)) 
    
    if std_intercept>(2*std_err):
        print('Linear fit is not good enough!')
    
    xmid=round((max(x)+min(x))/2,5)
    xmin=round(min(x)-xmar*xmid,4)
    xmax=round(max(x)+xmar*xmid,4)
    xmin1=round((xmid+xmin)/2,5)
    xmax1=round((xmid+xmax)/2,5)
    xsmall=round((xmax-xmin)/10, 5)
    
    ymid=(min(y)+max(y))/2
    ymin=min(y)-ymar*abs(ymid)
    ymax=max(y)+ymar*abs(ymid)
    
    # Based on the fact that std_err is the error of the calculated slope, we calculate the error of the intercept (see formula in Wikipedia - Linear regression)
    sig=intercept
    mu=np.mean(x)
    
    #print(info)
    
    # Making the plot
    
    y_lm='linregress, n='+str(len(y))+'\n'+strd+'= '+str(round(sig,2))+pm+str(round(std_intercept,2))+permil
    r_x, r_y = zip(*((j, j*slope + intercept) for j in range(len(data))))
    lm = pd.DataFrame({MRx : r_x, y_lm: r_y})
    lm.index=lm[MRx]
    
    # Intercept allow to calculate y_lim1 and y_lim2 for the left small graph
    y_lim1=intercept-0.5*margin
    y_lim2=intercept+margin
    
    # Plot of the points and line on the main graph (right)
    data.plot(kind='scatter', x=MRx, y=delta, marker='o', color=color, ax = ax2, xlim=[xmin, xmax], ylim=[ymin, ymax], xticks=[xmin, xmin1, xmid, xmax1, xmax])
    lm.plot(kind='line', y=y_lm, color=color, ax=ax2, style='--', legend=True)
    
    # Plot of the line on the left small graph
    lm.plot(kind='line', y=y_lm, color=color, ax=ax1, style='--', legend=False, xlim=[0, xsmall], ylim=[y_lim1, y_lim2], xticks=[round(0,0), xsmall])
    
    ax1.set_ylabel(str_delta)
    ax2.set_ylabel('')
    ax1.set_xlabel('')
    plt.suptitle('Keeling plot, '+strID, y=1.05)
    fig.tight_layout()
    
        
    return sig, std_intercept, fig


def miller(data, iso):
    
    plt.rcParams.update({'font.size': 15})
    
    if iso=='13C':
        
        xname=strMR_13C
        yname='MR * '+d13C
        strd=d13C
        color=c2
        ymar=0.05
        
        delta=delta13C
        
    if iso=='D':
        
        xname=strMR_D
        yname='MR * '+dD
        strd=dD
        color=c4
        ymar=0.1

        delta=deltaD
        
    xmar=0.1
    
    fig, ax1 = plt.subplots(1,1)
    
    x=data[xname]
    y=data[xname]*data[delta]
    data.insert(loc=len(data.columns), value=y, column=yname, allow_duplicates=True)
    mask= ~np.isnan(x) & ~np.isnan(y)
    
    xmid=(max(x[mask])+min(x[mask]))/2
    xmin=min(x[mask])-xmid*xmar
    xmax=max(x[mask])+xmid*xmar
    xmin1=(xmid+xmin)/2
    xmax1=(xmid+xmax)/2
    
    ymid=(min(y[mask])+max(y[mask]))/2
    ymin=min(y[mask])-ymar*abs(ymid)
    ymax=max(y[mask])+ymar*abs(ymid)
    
    slope, intercept, r_value, p_value, std_err = sc.stats.linregress(x[mask], y[mask])
        
    sig=slope
    std_slope=std_err
    
    y_lm='r2='+str(round(r_value, 3))+'\n'+strd+'= '+str(round(sig,2))+pm+str(round(std_slope,2))+permil
    r_x, r_y = zip(*((j, j*slope + intercept) for j in range(len(data))))
    lm = pd.DataFrame({xname : r_x, y_lm: r_y})
    lm.index=lm[xname]
    
    # Plot of the points and line on the graph
    data.plot(kind='scatter', x=xname, y=yname, marker='o', color=color, ax = ax1, xlim=[xmin, xmax], ylim=[ymin, ymax], xticks=[xmin, xmin1, xmid, xmax1, xmax])
    lm.plot(kind='line', x=xname, y=y_lm, color=color, ax=ax1, style='--', legend=True)

    ax1.set_ylabel(yname)
    plt.suptitle('Miller-Tans plot, all data', y=1.05)

    return sig, std_slope, r_value, fig

def correlation(x, y, namex, namey, text):
    
    mar=0.1
    
    midx=(max(x)-min(x))/2
    xmin=min(x)-mar*midx
    xmax=max(x)+mar*midx
    
    midy=(max(y)-min(y))/2
    ymin=min(y)-mar*miny
    ymax=max(y)+mar*midy
    
    slope, intercept, r_value, p_value, std_err = sc.stats.linregress(x, y)
    
    y_lm='r= '+str(round(r_value, 3))
    r_x, r_y = zip(*((j, j*slope + intercept) for j in range(3000)))
    lm = pd.DataFrame({namex: r_x, namey: r_y})
    #lm.index=lm[nm13C_2[1]]
    #figdiff = plt.figure()
    #rect = 1,1,1,1
    #ax1=figdiff.add_axes()
    #plt.plot(kind='scatter', y=data13C[nm13C_2[1]], x=dataP['picarro'])
    #plt.plot(kind='scatter', y=dataD[nmD_2[1]], x=dataP['picarro'])
    something=plt.scatter(y=y, x=x, color='g')
    fig=something.get_figure()
    
    fig.text(0.4,0.9, text, size='large')
    rect=0,0,1,1
    ax1=fig.add_axes(rect)
    
    lm.plot(kind='line', y=y_lm, color='g', ax=ax1, style='--', legend=True, xlim=[xmin, xmax], ylim=[ymin, ymax])
    plt.scatter(y=y, x=x, color='g')
    
    ax1.set_xlabel(namex)
    ax1.set_ylabel(namey)
    
    return fig

def moving_window(data, time, n, lim_err, iso, MRmin, MRbg, MRwindow, miller):
    
    if miller==False:
        title='Moving window Keeling plot'+', n='+str(n)+', min MR window = '+str(MRwindow)+' ppb, min MR value = '+str(MRmin)+' ppb'
    elif miller==True:
        title='Moving window Miller-Tans plot'+', n='+str(n)+', min MR window = '+str(MRwindow)+' ppb, min MR value = '+str(MRmin)+' ppb'
    
    if iso=='D':
        #c=c4
        ymar=0.05
        nameMR=col_MR_dD
        delta=col_dD
        edelta=col_errdD
        yname='MR * '+dD
        sy0=0.1
        sx0=1e-8
        str_delta=col_dD+permil
    if iso=='13C':
        #c=c2
        ymar=0.01
        nameMR=col_MR_d13C
        delta=col_d13C
        edelta=col_errd13C
        yname='MR * '+d13C
        sy0=0.005
        sx0=1e-8
        str_delta=col_d13C+permil
    
    if data.empty==True:
        empty = pd.DataFrame(np.nan, index=['0'], columns=['start_DateTime', 'end_DateTime', delta+permil, 'err '+delta+permil, 'r2', 'n'])
        empty_fig = plt.figure()
        return empty, empty_fig
    
    time_s=datetime.timedelta(hours=time)
    
    xmar=0.01 
    
    xmin=1
    xmax=0
    
    ymin=0
    ymax=-500
    
    sigmin=ymin
    sigmax=ymax

    # Set up linear model
    linear = odrpack.Model(f)
    
    #fig, (ax1, ax2, ax3a, ax3b)= plt.subplots(2,2, gridspec_kw = {'width_ratios':[1, 5]}, figsize=(15,10))
    #fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    
    #########################
    # Add the colored MR data
    #########################
    gs=gridspec.GridSpec(2,6)
    fig=plt.figure(figsize=(15,10), tight_layout=True)
    if miller==False:
        ax1=plt.subplot(gs[0,0])
        ax2=plt.subplot(gs[0,1:])
        ax3=plt.subplot(gs[1,:])
    elif miller==True:
        ax1=plt.subplot(gs[0,:])
        ax2=ax1
        ax3=plt.subplot(gs[1,:])
    #ax3.set_position([0,-1,1,1])
    
    #gs.update()
    
    #########################
    # Add the colored MR data, before you select parts! we want it all
    #########################
    #sub_P.plot(y='picarro', ax=ax3, color=cP, lw=0.7)
    data.plot(y=nameMR, ax=ax3, markersize=2, style='o',  mec=cP, mfc=cP, legend=False, clip_box=[0,1,1,0])
    
    n_sets=int(len(data)/n)
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
    
    cm=plt.get_cmap('jet', n_tot)
    
    j=0
    
    for i in range(n_tot):
        
        t_mid=data.index[i]         # time of the middle of the window
        hypt1=t_mid-time_s/2        # start time hypothetique, according to the window size
        hypt2=t_mid+time_s/2
        t1=min(data.index, key=lambda t: abs(t-hypt1))           # start time real, corresponding to the closest data point of the time hyptothetique
        t2=min(data.index, key=lambda t: abs(t-hypt2))
        
        # select data from the time range we want
        dataset=data[t1:t2]
        
        MRs=dataset[nameMR]
        
        c=cm(i)
        
        # one-shot conditions
        ###################################################
        # is the dataset consisting in at least n points?
        if len(MRs)<n:
            print('Not enough data points!')
            continue     
        # is the window of MR wide enough?
        MRwidth=max(MRs)-min(MRs)
        if MRwidth<MRwindow:
            print('Range of [CH4] is too small!')
            # refuse the selected set of MR
            continue
        
        # loop conditions
        ###################################################
        cancel=0
        cancel2=0
        
        # is there at least one MR higher than the min MR required?
        for MR in MRs:
            if MR>MRmin:
                cancel=0
                break
            if MR<MRmin:
                cancel=1
                pass 
            
        if (cancel==1):
            print('No [CH4] value high enough!')
            continue
            
        # is there at least one MR at a bg value?
        for MR in MRs:
            if MR<MRbg:
                cancel2=0
                break 
            else:
                cancel2=1
                pass
        
        if (cancel2==1):
            print('No value for background!')
            continue
        
        #MRx='1/$x(CH_4)$ [$ppb^{-1}$]'
        if miller==False:
            x=dataset[MRx]
            y=dataset[delta]
            sx=dataset['err_'+MRx]
            sy=dataset[edelta]
            sy = sy.replace(0.0, sy0)
            sx = sx.replace(0.0, sx0)
        elif miller==True:
            x=dataset[nameMR]
            y=dataset[yname]
         
        slope, intercept, r_value, p_value, std_err = sc.stats.linregress(x, y)
        
        # run an ordinary distance regression model
        mydata = odrpack.RealData(x, y, sx=sx, sy=sy)
        myodr = odrpack.ODR(mydata, linear, beta0=(slope, intercept))
        myoutput = myodr.run()
        
        # save the relevant output values 
        #slope = myoutput.beta[0]
        #intercept = myoutput.beta[1]
        #std_err = myoutput.sd_beta[0]
        #std_intercept = myoutput.sd_beta[1]
        #sum_square=myoutput.sum_square
        #info=myoutput.stopreason
        # don't stop the function if it doesn't work
        if np.isnan(slope)==True:
            print('No linear regression made!\nReason: '+info)
            continue
        
        sx2 = (x**2).sum()
        sy2 = (y**2).sum()
        std_intercept = std_err * np.sqrt(sx2/len(x))
        
        #error = myoutput.res_var
        error = p_value
        
        if error>lim_err:
            print(str(error))
            print('Linear fit is not good enough!')
            continue
        
        #print(info)

        errs.append(error)        
        
        #cm=plt.get_cmap('jet', j)
        
        # If you want individual figures, uncomment the following lines
        #fig, (ax1, ax2)= plt.subplots(1,2, gridspec_kw = {'width_ratios':[1, 5]})
        #fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)
    
        # Based on the fact that std_err is the error of the calculated slope, we calculate the error of the intercept (see formula in Wikipedia - Linear regression)
        datestart=dataset.index[0]
        startdates.append(datestart)
        dateend=dataset.index[len(dataset)-1]
        enddates.append(dateend)
        dates.append(datestart+(dateend-datestart)/2)
        n_points.append(len(dataset))
        slopes.append(slope)               # save the statistical significiance parameter
        mu.append(np.mean(x))
        #p_values.append(p_value)
        

        if miller==True:
            signature=round(slope,3)
            err_sig=round(std_err, 3)
        elif miller==False:
            signature=round(intercept, 3)
            err_sig=round(std_intercept, 3)

        sig.append(signature)
        std_intercepts.append(err_sig)
        
        # Make the regression line
        #y_lm=str(datestart)+' conf='+str(round(r_value, 3))+'\n'+strd+'= '+str(signature)+' \u00B1 '+str(err_sig)+' \u2030'
        r_x, r_y = zip(*((j, j*slope + intercept) for j in range(len(dataset))))
        lm = pd.DataFrame({'x': r_x, 'y': r_y})
        lm.index=lm['x']
            
        # Draw the datapoints that are lineary fitted
        if miller==False:
            dataset.plot(style='.', marker='d', x=MRx, y=delta, ax = ax2, legend=False, color=c)
        elif miller==True:
            dataset.plot(style='.', marker='d', x=nameMR, y=yname, ax = ax2, legend=False, color=c)
   
        # Draw the regression line
        lm.plot(kind='line', x='x', y='y', ax=ax2, style='--', legend=False, color=c)
        lm.plot(kind='line', x='x', y='y', ax=ax1, style='--', legend=False, color=c)

        if miller==False:
            ax1.set_ylabel(str_delta)
            ax2.set_ylabel('1/[CH4] (ppb-1)')
            ax1.set_xlabel('')
        elif miller==True:
            ax1.set_ylabel(delta+' * [CH4]')
            ax2.set_ylabel('')
            ax1.set_xlabel('[CH4] (ppb)')

        # Color the points in the bottom graph of [CH4] data
        dataset.plot(y=nameMR, ax=ax3, markersize=5, style='o',  mec=c, mfc=c, legend=False, clip_box=[0,1,1,0]) # overwrite the mixing ratios with the colored ones
        
        fig.savefig('figures/MWKP/d'+iso+'_'+str(j)+'.png')
        #ax1.legend(bbox_to_anchor=(7, 1), loc=2)
        
        if min(x)<xmin:
            xmin=min(x)
        if max(x)>xmax:
            xmax=max(x)
            
        if min(y)<ymin:
            ymin=min(y)
        if max(y)>ymax:
            ymax=max(y)
            
        if miller==False:
            if min(sig)<sigmin:
                sigmin=min(sig)
            if max(sig)>sigmax:
                sigmax=max(sig)
        if miller==True:
            sigmin=min(y)
            sigmax=max(y)
            
            ##
        xsmall=round((xmax-xmin)/4, 5)
        xmid=(xmin+xmax)/2
        xmin1=(xmid+xmin)/2
        xmax1=(xmid+xmax)/2
        ymid=(ymin+ymax)/2
        sigmid=(sigmin+sigmax)/2
        ax1.set_xlim([int(0.0), xsmall])
        ax1.set_ylim([sigmin+ymar*sigmid, sigmax-ymar*sigmid])
            
        ax2.set_xlim([xmin-xmar*xmid, xmax+xmar*xmid])
         
        ax2.set_ylim([ymin+ymar*ymid, ymax-ymar*ymid])
    
        ax1.set_xticks([0])
        ax2.set_xticks([xmin, xmin1, xmid, xmax1, xmax])
        ##
            
        j=j+1
    
    if sigmax>0:
        sigmax=0
    xsmall=round((xmax-xmin)/4, 5)
    xmid=(xmin+xmax)/2
    xmin1=(xmid+xmin)/2
    xmax1=(xmid+xmax)/2
    ymid=(ymin+ymax)/2
    sigmid=(sigmin+sigmax)/2
    ax1.set_xlim([int(0.0), xsmall])
    ax1.set_ylim([sigmin+ymar*sigmid, sigmax-ymar*sigmid])
        
    ax2.set_xlim([xmin-xmar*xmid, xmax+xmar*xmid])
         
    ax2.set_ylim([ymin+ymar*ymid, ymax-ymar*ymid])
    
    ax1.set_xticks([0])
    ax2.set_xticks([xmin, xmin1, xmid, xmax1, xmax])
    plt.suptitle(title, y=1.05)
    
    ax3.set_ylabel('x(CH4) [ppb]')
    #plt.subplots_adjust(top=0.8)
    
    df=pd.DataFrame({'DateTime': dates, 'start_DateTime': startdates,'end_DateTime': enddates, delta: sig, 'err '+delta: std_intercepts, 'slope':slopes, 'res_var':errs, 'n':n_points}, 
                    columns=['DateTime', 'start_DateTime', 'end_DateTime', delta, 'err '+delta, 'slope', 'res_var', 'n'])
    if len(df)==0:
        df=pd.DataFrame({'DateTime':'nan', 'start_DateTime': 'nan','end_DateTime': 'nan', delta: 'nan', 'err '+delta: 'nan', 'ss':'nan', 'n':'nan'}, columns=['DateTime', 'start_DateTime', 'end_DateTime', delta, 'err '+delta, 'slope', 'res_var', 'n'], index=[0])
    
    fig.tight_layout()
    
    return df, fig, errs


def make_gif(df_points, df_lin, iso):
    
    if iso=='D':
        #c=c4
        ymar=0.05
        nameMR=col_MR_dD
        delta=col_dD
        edelta=col_errdD
        yname='MR * '+dD
        sy0=0.1
        sx0=1e-8
        str_delta=col_dD+permil
    if iso=='13C':
        #c=c2
        ymar=0.01
        nameMR=col_MR_d13C
        delta=col_d13C
        edelta=col_errd13C
        yname='MR * '+d13C
        sy0=0.005
        sx0=1e-8
        str_delta=col_d13C+permil
        
    xmar=0.01

    n_tot=len(df_points)
    
    gs=gridspec.GridSpec(2,6)
    fig=plt.figure(figsize=(15,10), tight_layout=True)
    
    ax1=plt.subplot(gs[0,0])
    ax2=plt.subplot(gs[0,1:])
    ax3=plt.subplot(gs[1,:])
    
    # get min and max values to set the axis limits
    xmin=np.min(df_points[MRx])
    xmax=np.max(df_points[MRx])
    xmid=(xmax+xmin)/2
    ymin=np.min(df_points[delta])
    ymax=np.max(df_points[delta])
    ymid=(ymin+ymax)/2
    sigmin=np.min(df_lin[delta])
    sigmax=np.max(df_lin[delta])
    sigmid=(sigmax+sigmin)/2
    
    # set the axis limits of the intercepts
    xsmall=round((xmax-xmin)/4, 5)
    ax1.set_xlim([int(0.0), xsmall])
    ax1.set_ylim([sigmin+ymar*sigmid, sigmax-ymar*sigmid])
    ax1.set_xticks([0])
    
    # set axis limit of the keeling plot panel
    ax2.set_xlim([xmin-xmar*xmid, xmax+xmar*xmid])
    ax2.set_ylim([ymin+ymar*ymid, ymax-ymar*ymid])

    # Plot of the grey [CH4] data, that will persist
    df_points.plot(y=nameMR, ax=ax3, markersize=2, style='o',  mec=cP, mfc=cP, legend=False, clip_box=[0,1,1,0])
    
    def update(i):
    
        # Get the data points of this step
        start=df_lin.loc[i, 'start_DateTime']
        end=df_lin.loc[i, 'end_DateTime']
        
        dataset=df_points.loc[(df_points.index>start) & (df_points.index<end)]
        
        # Make the regression line 
        slope = df_lin.loc[i, 'slope']
        intercept = df_lin.loc[i, delta]
        r_x, r_y = zip(*((j, j*slope + intercept) for j in range(len(dataset))))
        lm = pd.DataFrame({'x': r_x, 'y': r_y})
        lm.index=lm['x']
        
        # Get the color of this step
        cm=plt.get_cmap('jet', n_tot)
        c=cm(i)
        
        # Draw the datapoints
        dataset.plot(style='.', marker='d', x=MRx, y=delta, ax = ax2, legend=False, color=c)

        # Draw the regression line
        lm.plot(kind='line', x='x', y='y', ax=ax2, style='--', legend=False, color=c)
        lm.plot(kind='line', x='x', y='y', ax=ax1, style='--', legend=False, color=c)
        
        ax1.set_ylabel(str_delta)
        ax2.set_ylabel('1/[CH4] (ppb-1)')
        ax1.set_xlabel('')

        # Color the points in the bottom graph of [CH4] data
        dataset.plot(y=nameMR, ax=ax3, markersize=5, style='o',  mec=c, mfc=c, legend=False, clip_box=[0,1,1,0]) # overwrite the mixing ratios with the colored ones
        
        return ax1, ax2, ax3
        
    anim = FuncAnimation(fig, update, frames=np.arange(0,n_tot), interval=200)
    plt.show()
    
    return anim
        

def signatures(moving_data13C, moving_dataD, data13C, dataD, title):
    
    strt='start_DateTime'
    str_d13C=delta13C+permil
    str_dD=deltaD+permil
        
    edelta13C='err '+delta13C+permil
    edeltaD='err '+deltaD+permil
    
    s13C='source '+delta13C
    sD='source '+deltaD
    err_s13C='err '+d13C
    err_sD='err '+dD
    
    #n_13C=len(moving_data13C)
    #n_D=len(moving_dataD)
    
    if moving_data13C.isnull().all().all()==False:
        merged_13C=pd.DataFrame(
                {'filling time': moving_data13C[strt],
                 s13C: moving_data13C[delta13C+permil],
                 err_s13C: moving_data13C[edelta13C]
                 })
        merged_13C.index=merged_13C['filling time']
    
    if moving_dataD.isnull().all().all()==False:
        merged_D=pd.DataFrame(
                {'filling time': moving_dataD[strt],
                 sD: moving_dataD[deltaD+permil],
                 err_sD: moving_dataD[edeltaD]
                 })
        merged_D.index=merged_D['filling time']
    
    # Creating the plot

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 10), sharex=True)
    ax1=axes[0]
    ax2=axes[1]
    ax3=axes[2]

    # Graph of the mixing ratios (top)

    if data13C.empty==False:
        data13C.plot(y=strMR_13C, ax=ax1, markersize=1, style='o',  mec=c2, mfc=c2)
        ax2.errorbar(x=merged_13C.index, y=merged_13C[s13C], yerr=merged_13C[err_s13C], markersize=3, marker='o', ls='None', mec=c2, mfc=c2, c=c1)
        data13C.plot(y=str_d13C, ax=ax2, markersize=1, style='o', mec=cP, mfc=cP)
        
    if dataD.empty==False:
        dataD.plot(y=strMR_D, style='o', ax=ax1, markersize=1,  mec=c4, mfc=c4)
        ax3.errorbar(x=merged_D.index, y=merged_D[sD], yerr=merged_D[err_sD], markersize=3, marker='o', ls='None',  mec=c4, mfc=c4, c=c3)
        dataD.plot(y=str_dD, ax=ax3, markersize=1, style='o', mec=cP, mfc=cP)
    
    ax1.set_ylabel('x(CH4) [ppb]')

    ax1.legend(fontsize=15)

    # Graph of the C isotopic signatures
    
    ax2.set_ylabel(str_d13C)
    
    #ax2.legend(fontsize=15)
    
    # Graph of the H isotopic signatures

    ax3.set_ylabel(str_dD)
    
    #ax3.legend(fontsize=15)

    plt.suptitle('Source CH4 isotopic signatures\n'+title)

    ax3.xaxis_date()
    #ax3.format_xdata = mdates.DateFormatter('%Y-%b')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
    ax3.xaxis.set_major_locator(plt.MaxNLocator(10))
    fig.autofmt_xdate()

    return fig


def f(B, x):
    return B[0]*x + B[1]



def top_diagrams(df_peaks=None, ambient=None, patches=True, title=None):
        
    # Axis limits
    lim_dD=[-420, -70]
    lim_d13C=[-80, -10]
    
    # Define the boundaries of the patches, and their color
        
    x_m=[-71, -56, -56, -71]
    y_m=[-260, -260, -380, -380]
    c_m='xkcd:neon green'
        
    x_w=[-61.5, -46, -46, -61.5]
    y_w=[-252, -252, -330, -330]
    c_w='xkcd:neon pink'
        
    x_t=[-60, -31.5, -31.5, -60]
    y_t=[-155, -155, -195, -195]
    c_t='xkcd:neon blue'
        
    x_p=[-30, -21.5, -21.5, -30]
    y_p=[-180, -180, -290, -290]
    c_p='xkcd:neon yellow'
    
    x_air=[ambient['avg_d13C']-2*ambient['std_d13C'], ambient['avg_d13C'], ambient['avg_d13C']+2*ambient['std_d13C'], ambient['avg_d13C']]
    y_air=[ambient['avg_dD'], ambient['avg_dD']+2*ambient['std_dD'], ambient['avg_dD'], ambient['avg_dD']-2*ambient['std_dD']]
    c_amb='gray' 
    
    # Make the figure
    top_fig_top= plt.figure(figsize=(15,10))
    top_top=top_fig_top.add_subplot(111)

    # Add the areas
    top_top.fill(x_m, y_m, alpha=0.5, c=c_m)
    top_top.fill(x_w, y_w, alpha=0.5, c=c_w)
    top_top.fill(x_t, y_t, alpha=0.5, c=c_t)
    top_top.fill(x_p, y_p, alpha=0.5, c=c_p)
    # ... and corresponding patches
    bio_patch = mpatches.Patch(color=c_m, label='Biogenic', alpha=0.5)
    waste_patch = mpatches.Patch(color=c_w, label='Waste', alpha=0.5)
    thermo_patch = mpatches.Patch(color=c_t, label='Thermogenic', alpha=0.5)
    pyro_patch = mpatches.Patch(color=c_p, label='Pyrogenic', alpha=0.5)
        
    # Add the ambient air point
    top_top.scatter(x=ambient['avg_d13C'], y=ambient['avg_dD'], c=c_amb,  s=150)
    # ... and corresponding patches
    air_patch = mlines.Line2D([], [], color=c_amb, marker='o',
                      markersize=15, label='Ambient air', linestyle='')

    # Add the peak signatures if necessary
    if df_peaks is not None:
        df_peaks.plot(kind='scatter', ax=top_top, x=col_d13C, y=col_dD, xerr=col_errd13C, yerr=col_errdD, marker='o', color='black', zorder=1)
        data_patch = mlines.Line2D([], [], color='black', marker='o',
                           label='Krakow 2018-19', linestyle='')
        # Put the legend
        top_fig_top.legend(handles=[bio_patch, waste_patch, thermo_patch, pyro_patch, air_patch, data_patch], bbox_to_anchor=(0.63, 0.38), loc=2, borderaxespad=0.5)
    else:
        # Put the legend
        top_fig_top.legend(handles=[bio_patch, waste_patch, thermo_patch, pyro_patch, air_patch], bbox_to_anchor=(0.63, 0.35), loc=2, borderaxespad=0.5)
  
    # Arrange the axis limits
    plt.xlim(lim_d13C)
    plt.ylim(lim_dD)
    plt.xlabel(d13C+permil)
    plt.ylabel(dD+permil)

    top_fig_top.savefig('figures/top_diagram_'+title+'.png', dpi=300, bbox_inches='tight')
    
    return top_fig_top