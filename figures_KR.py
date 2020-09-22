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
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import cmocean
import os, shutil
import scipy as sc
import scipy.odr.odrpack as odrpack
import nmmn.stats
from scipy import stats
#from pandas.tools.plotting import table
import datetime

from windrose import WindroseAxes

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}

# possibilities are ‘xx-small’, ‘x-small’, ‘small’, ‘medium’, ‘large’, ‘x-large’, ‘xx-large’

c1='xkcd:blue'
c2='xkcd:green'
c3='xkcd:red'
c4='xkcd:yellow'
cP='xkcd:grey'
vertclair='#009966'
web_blue='#333399'
web_green='#006633'
web_yellow='#FFFF00'
web_red='#FF3333'

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

strMR_13C='MR_ppb - '+d13C
strMR_D='MR_ppb - '+dD
strMR='x(CH4) [ppb]'
strMRx='1/x(CH4) [1/ppb]'

str_EDGAR='EDGARv50'
str_TNO='TNO_CAMS_REGv221'

col_d13C_mod='d13C'
col_dD_mod='dD'

# x and y must be arrays of n columns
def correlation(x, y, namex, namey, leg=None):
    
    mar=0.1

    #midy=(np.nanmax(y)-np.nanmin(y))/2
    #ymin=np.nanmin(y)-mar*midy
    #ymax=np.nanmax(y)+mar*midy
    
    n=len(x)
    colors=plt.get_cmap('brg', n+1)
    fig = plt.subplots(1,1, figsize=(10,10))
    
    for i in range(n):
        
        midx=(np.nanmax(x[i])-np.nanmin(x[i]))/2
        xmin=np.nanmin(x[i])-mar*midx
        xmax=np.nanmax(x[i])+mar*midx
    
        #linear regression
        mask = ~np.isnan(x[i]) & ~np.isnan(y[i])
        slope, intercept, r_value, p_value, std_err = sc.stats.linregress(x[i][mask], y[i][mask])
        x_lin=np.linspace(xmin, xmax, 1000)
        lin = pd.Series(slope*x_lin+intercept, index=x_lin)
        lin11 = pd.Series(x_lin, index=x_lin)
        
        #report r2 in legend
        if leg is not None:
            str_lin = leg[i]+', '+'r= '+str(round(r_value, 3))
        else:
            str_lin = 'r= '+str(round(r_value, 3))
    
        #plot the 1:1 line
        plt.plot(lin11, color='grey', linestyle=':', linewidth=0.7)
        #plot data points
        plt.scatter(y=y[i], x=x[i], color=colors(i), marker='x')
        #plot the regression line
        plt.plot(lin, color=colors(i), label=str_lin)
        #
    
    fig[1].set_xlabel(namex)
    fig[1].set_ylabel(namey)
    plt.legend()
    
    return fig


def ch4_zoom(data_IRMS, MR_P, t1, t2):
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5), sharex=True)
    
    # Graph of the mixing ratios (top)
    #if t1!='2019-03-01 00:00:00':
    plt.plot(MR_P[t1:t2], color='k', lw=0.7, label='picarro')
    
    data_IRMS[t1:t2].plot(y=col_MR, markersize=1, style='x',  mec='b', mfc='b', label='original', ax=ax)
    #data_IRMS[t1:t2].plot(y=col_MR+'_v1', markersize=1, style='x',  mec='orange', mfc='orange', label='corr-linear', ax=ax)
    data_IRMS[t1:t2].plot(y=col_MR+'_v2', markersize=1, style='x',  mec='red', mfc='red', label='corr-offset', ax=ax)
    #data_IRMS[t1:t2].plot(y=col_MR+'_v3', markersize=1, style='x',  mec='purple', mfc='purple', label='corr-linear0', ax=ax)
    plt.legend()
    
    return fig

  
def keeling_plot(serie_x, serie_y, params, iso):
    
    plt.rcParams.update({'font.size': 10})
    
    if iso=='D':
        color='xkcd:grey'
        strd='\u03B4D'
        title=strd+' SMOW'+permil
        margin=40
        ymar=0.1
        
    if iso=='13C':
        color='xkcd:grey'
        strd='\u03B4\u00B9\u00B3C'
        title=strd+' VPDB'+permil
        margin=5
        ymar=0.05
    
    color2='xkcd:red'
    xmar=0.1
    
    MRx = serie_x[0]
    delta = serie_y[0]
    err_MRx = serie_x[1]
    err_delta = serie_y[1]

    data = pd.concat([MRx, delta, err_MRx, err_delta], axis=1).dropna()
    
    print(str(data))
    #return data
    
    slope=params[0]
    sig=params[1]
    std_beta=params[3]
    r2=params[4][0]
    res_var=params[4][1]
    
    #c.fillna(0, inplace=True)

    print(str(len(MRx))+' mole fraction values.')
    print(str(len(delta))+' isotopes values of d'+iso+'.')
    print('slope: '+str(slope)+'\nintercept: '+str(sig))
    
    # Prepare the figure layout
    figure, (ax1, ax2)= plt.subplots(1,2, gridspec_kw = {'width_ratios':[1, 5]})
    figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=None)

    xmid=round((np.nanmax(MRx)+np.nanmin(MRx))/2,5)
    xmin=round(np.nanmin(MRx)-xmar*xmid,4)
    xmax=round(np.nanmax(MRx)+xmar*xmid,4)
    xmin1=round((xmid+xmin)/2,5)
    xmax1=round((xmid+xmax)/2,5)
    xsmall=round((xmax-xmin)/10, 5)
    xrange=round(1/np.nanmin(MRx)-1/np.nanmax(MRx))

    ymid=(np.nanmin(delta)+np.nanmax(delta))/2
    ymin=min(delta)-ymar*abs(ymid)
    ymax=max(delta)+ymar*abs(ymid)
    
    lcb,ucb,xcb=nmmn.stats.confband(MRx, delta, slope, sig, conf=0.95)#, x=np.linspace(0,xmax,len(MRx)).all())
    #print([lcb,ucb,xcb])
    
    # Making the plot
    
    y_lm='BCES, n='+str(len(delta))+', r2='+str(round(r2,3))+', rv='+str(round(res_var,3))+', sc='+str(round(np.sqrt(res_var/xrange),5))+'\n'+strd+'= '+str(round(sig,2))+pm+str(round(std_beta,2))+permil
    r_x, r_y = zip(*((j, j*slope + sig) for j in range(len(MRx))))
    lm = pd.DataFrame({strMRx : r_x, y_lm: r_y})
    lm.index=lm[strMRx]
        
    # Intercept allow to calculate y_lim1 and y_lim2 for the left small graph
    y_lim1=sig-0.5*margin
    y_lim2=sig+margin
        
    # Plot of the points, line and area on the main graph (right)
    ax2.fill_between(xcb, lcb, ucb, alpha=0.3, facecolor='orange')
    plt.scatter(x=MRx, y=delta, c='k', s=10)
    lm.plot(kind='line', y=y_lm, color=color, style='--', legend=True, ax=ax2)
    
    # Plot of the line on the left small graph
    #print(lm)
    if lm[y_lm].isna().values.all()==False:
        lm.plot(kind='line', y=y_lm, color=color, ax=ax1, style='--', legend=False, xlim=[0, xsmall], ylim=[y_lim1, y_lim2], xticks=[round(0,0), xsmall])
        ax1.fill_between(xcb, lcb, ucb, alpha=0.3, facecolor='orange')
        
    ax2.set_xlim([xmin, xmax])
    ax2.set_ylim([ymin, ymax])
    ax2.set_xticks([xmin, xmin1, xmid, xmax1, xmax])
        
    ax1.set_ylabel(title)
    ax2.set_ylabel('')
    ax1.set_xlabel('')
    ax2.set_xlabel(strMRx)
    plt.suptitle('Keeling plot, '+strd, y=1.05)
    figure.tight_layout()
    
    return figure


def hist(data13C, dataD, title, data13C_list=None, dataD_list=None, str_leg=None):
    
    plt.rcParams.update({'font.size': 22})
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,20), sharex=False)
    ax1=axes[0]
    #ax1b=ax1.twinx()
    ax2=axes[1]
    #ax2b=ax2.twinx()
    
    bins_d13C=np.linspace(np.nanmin(data13C), np.nanmax(data13C), 10)
    #bins_d13C=10
    c_d13C=plt.get_cmap('brg', len(data13C_list)+1)
    data13C.plot.hist(ax=ax1, bins=bins_d13C,  lw=3, edgecolor='k', facecolor='None', label=str_leg[0])
    
    for i in range(len(data13C_list)):
        data=data13C_list[i]
        bins_d13C=np.linspace(np.nanmin(data), np.nanmax(data), 10)
        data.plot.hist(ax=ax1, alpha=0.5, bins=bins_d13C, color=c_d13C(i), label=str_leg[i+1])
        ax1.set_ylabel('')
    
    #bins_dD=[-350, -340, -330, -320, -310, -300, -290, -280, -270, -260, -250, -240]
    #bins_dD=np.arange(-355, -175, 5)
    bins_dD=10
    c_dD=plt.get_cmap('brg', len(dataD_list)+1)
    dataD.plot.hist(ax=ax2, bins=bins_dD,  lw=3, edgecolor='k', facecolor='None', label=str_leg[0])
    for i in range(len(dataD_list)):
        data=dataD_list[i]
        data.plot.hist(ax=ax2, bins=bins_dD, alpha=0.5, color=c_dD(i), label=str_leg[i+1])
        ax2.set_ylabel('')
        
    plt.legend(loc=2)
    ax1.set_xlabel(delta13C)
    ax2.set_xlabel(deltaD)

    fig.suptitle(t=title, y=0.9)

    return fig

  
def overview_model(data_obs, data_mod, inventory, data2_mod=None):
    
    plt.rcParams.update({'font.size': 22})
    
    if (inventory=='EDGAR') | (inventory=='both'):
        cM=web_red
        str_inv=str_EDGAR
        cM2='g'
        str_inv2=str_TNO
        cP=web_blue
    elif inventory=='TNO':
        cM='g'
        str_inv=str_TNO

    figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 10), sharex=True)
    ax1=axes[0]
    ax2=axes[1]
    ax3=axes[2]
    
    # Graph of the mixing ratios (top)
    data_obs.plot(y=col_MR, ax=ax1, markersize=1, style='o',  mec=cP, mfc=cP, label='observations')    
    data_mod.plot(y='Total', ax=ax1, color=cM, lw=0.7, label=str_inv)
    if inventory=='both':
        data2_mod.plot(y='Total', ax=ax1, color=cM2, lw=0.7, label=str_inv2)
    ax1.set_ylabel(strMR)
    ax1.legend(fontsize=15)

    # Graph of the C isotopic signatures
    data_obs.plot(y=col_d13C, markersize=1, style='o', legend=False, ax=ax2, mec=cP, mfc=cP)
    data_mod.plot(y=col_d13C_mod, ax=ax2, color=cM, lw=0.7, label=str_inv, legend=False)
    if inventory=='both':
        data2_mod.plot(y=col_d13C_mod, ax=ax2, color=cM2, lw=0.7, label=str_inv2, legend=False)
    ax2.set_ylabel(d13C+' [\u2030]')
    #ax2.legend(fontsize=15)

    # Graph of the H isotopic signatures

    data_obs.plot(y=col_dD, markersize=1, style='o', legend=False, ax=ax3,  mec=cP, mfc=cP)
    data_mod.plot(y=col_dD_mod, ax=ax3, color=cM, lw=0.7, label=str_inv, legend=False)
    if inventory=='both':
        data2_mod.plot(y=col_dD_mod, ax=ax3, color=cM2, lw=0.7, label=str_inv2, legend=False)
    ax3.set_ylabel(dD+' [\u2030]')
    #ax3.legend(fontsize=15)
    
    ax3.xaxis_date()
    ax3.format_xdata = mdates.DateFormatter('%Y-%b')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
    ax3.xaxis.set_major_locator(plt.MaxNLocator(10))
    figure.autofmt_xdate()
    
    ax1.grid(which= 'major', axis='both', linestyle='-')
    ax2.grid(which= 'major', axis='both', linestyle='-')
    ax3.grid(which= 'major', axis='both', linestyle='-')
    
    plt.suptitle('Overview of the observations with modeled time series using '+str_inv)

    return figure, axes


def wind_rose(direction, var, name, title):
    
    #plt.rcParams.update({'font.size': 14})
    colors=[web_blue, web_green, vertclair, web_yellow, web_red]
    cm = mcolors.LinearSegmentedColormap.from_list('web', colors, N=6)
    #bins=[min(var), 2050, 2200, 2350, 2900]
    bins=5
    
    fig = plt.figure(figsize=(10,10))
    ax1 = WindroseAxes.from_ax(fig=fig)
    
    ax1.bar(direction, var, nsector=32, opening=0.8, cmap=cm, bins=bins, edgecolor='white')
    
    ax1.legend(title=name, loc=4, bbox_to_anchor=(1.2, -0.15), fontsize='medium')
    ax1.set_title(title, pad=10)
    #ax1.set_rlim([0, 1000])
    #ax1.yaxis.set_ticks([0, 200, 400, 600, 800, 1000])
    #ax1.yaxis.set_ticklabels([0, 200, 400, 600, 800, 1000])
    #ax1.set_rlabel_position([0, 200, 400, 600, 800, 1000])
    
    return fig


def overview_pk(data_obs, data_obs_pk, data_model_pk, data_model=None, names=None):
    
    colors_obs=['red', 'green', 'orange', 'black']
    colors_model=['blue', 'orange', 'green']
    
    plt.rcParams.update({'font.size': 22})
    figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 10), sharex=True)
    ax1=axes[0]
    ax2=axes[1]
    ax3=axes[2]
    
    # Graph of the mixing ratios (top)
    data_obs.plot(y=col_MR, ax=ax1, markersize=2, style='o',  mec='k', mfc='k', label='observations')    
    ax1.set_ylabel(strMR)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.legend(fontsize=15)
    # Graph of the C isotopic signatures
    data_obs.plot(y=col_d13C, markersize=2, style='o', legend=False, ax=ax2, mec='k', mfc='k')
    # Graph of the H isotopic signatures
    data_obs.plot(y=col_dD, markersize=2, style='o', legend=False, ax=ax3,  mec='k', mfc='k')
    
    obs_patches=[]
    i=0
    for df in data_obs_pk:
    
        start_d13C=df['start_'+col_dt_d13C].dropna()
        start_dD=df['start_'+col_dt_dD].dropna()
        end_d13C=df['end_'+col_dt_d13C].dropna()
        end_dD=df['end_'+col_dt_dD].dropna()
    
        top_d13C=df[col_d13C].dropna()+df[col_errd13C].dropna()
        top_dD=df[col_dD].dropna()+df[col_errdD].dropna()
        bot_d13C=df[col_d13C].dropna()-df[col_errd13C].dropna()
        bot_dD=df[col_dD].dropna()-df[col_errdD].dropna()
    
        # Plot the C peak signatures
        ax2.fill([start_d13C, end_d13C, end_d13C, start_d13C], 
                 [top_d13C, top_d13C, bot_d13C, bot_d13C],
                 alpha=0.5, c=colors_obs[i], label=names[i])
        
        # Plot the D peak signatures
        ax3.fill([start_dD, end_dD, end_dD, start_dD], 
                 [top_dD, top_dD, bot_dD, bot_dD],
                 alpha=0.5, c=colors_obs[i])
        
        obs_patches.append(mpatches.Patch(facecolor=colors_obs[i], edgecolor=colors_obs[i], 
                         label=names[i]))
        
        i=i+1
    
    j=0
    if data_model is not None:
        for df in data_model:
            df.plot(y='Total', ax=ax1, markersize=1, style='-',  color=colors_model[j], legend=False)
            j=j+1
            
    model_patches=[]
    j=0
    for df in data_model_pk:
    
        start=df['start_'+col_dt].dropna()
        end=df['end_'+col_dt].dropna()
    
        top_d13C=df[col_d13C].dropna()+df[col_errd13C].dropna()
        top_dD=df[col_dD].dropna()+df[col_errdD].dropna()
        bot_d13C=df[col_d13C].dropna()-df[col_errd13C].dropna()
        bot_dD=df[col_dD].dropna()-df[col_errdD].dropna()
    
        # Plot the C peak signatures
        ax2.fill([start, end, end, start], 
                 [top_d13C, top_d13C, bot_d13C, bot_d13C],
                 alpha=0.5, c=colors_model[j], label=names[j+i])
        
        # Plot the D peak signatures
        ax3.fill([start, end, end, start], 
                 [top_dD, top_dD, bot_dD, bot_dD],
                 alpha=0.5, c=colors_model[j])
        
        model_patches.append(mpatches.Patch(facecolor=colors_model[j], edgecolor=colors_model[j],
                         label=names[j+i]))
        
        j=j+1
    
    ax2.set_ylabel(d13C+' [\u2030]')
    ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax2.legend(handles=obs_patches+model_patches, fontsize=15, loc='upper right')
    
    ax3.set_ylabel(dD+' [\u2030]')
    ax3.yaxis.set_major_locator(plt.MaxNLocator(5))
    #ax3.legend(fontsize=15)
        
    ax3.xaxis_date()
    ax3.format_xdata = mdates.DateFormatter('%Y-%b')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b-%d'))
    ax3.xaxis.set_major_locator(plt.MaxNLocator(10))
    figure.autofmt_xdate()
    
    ax1.grid(which= 'major', axis='both', linestyle='-')
    ax2.grid(which= 'major', axis='both', linestyle='-')
    ax3.grid(which= 'major', axis='both', linestyle='-')
    
    plt.suptitle('Overview of the observations with peak signatures.')

    return figure, axes


def overview_ch4(data_obs, data_mod, inventory, data2_mod=None):
    
    colors=['red', 'blue', 'green', 'black', 'black']
    cM='g'
    cM2='b'
    
    plt.rcParams.update({'font.size': 22})
    figure, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(30, 10), sharex=True)
    
    # Graph of the mixing ratios (top)
    data_obs.plot(y=col_MR, ax=ax1, markersize=1, style='o',  mec=cP, mfc=cP, label='observations')
    data_mod.plot(y='Total', ax=ax1, color=cM, lw=0.7, label=inventory[0])
    if inventory=='both':
        data2_mod.plot(y='Total', ax=ax1, color=cM2, lw=0.7, label=inventory[1])
    ax1.legend(fontsize=15)
    ax1.set_ylabel(strMR)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    
    return figure


def overview(data_obs, data_P=None):
    
    plt.rcParams.update({'font.size': 22})
    
    c13C='#333399'
    cD='#333399'
    
    figure, axes = plt.subplots(nrows=3, ncols=1, figsize=(15,10), sharex=True)
    ax1=axes[0]
    ax2=axes[1]
    ax3=axes[2]
    
    if data_P is not None:
        data_P.plot(y=col_P, ax=ax1, color=cP, lw=0.7)
        
    data_obs.plot(y=col_MR_d13C, ax=ax1, markersize=1.5, style='.',  mec=c13C, mfc=c13C, legend=False)
    data_obs.plot(y=col_d13C, markersize=1.5, style='.', legend=False, ax=ax2, mec=c13C, mfc=c13C)
    
    data_obs.plot(y=col_MR_dD, ax=ax1, markersize=1.5, style='.',  mec=cD, mfc=cD, legend=False)
    data_obs.plot(y=col_dD, markersize=1.5, style='.', legend=False, ax=ax3, mec=cD, mfc=cD)
    
    ax1.set_ylabel('x(CH4) [ppb]')
    #ax1.legend(['[CH4] IRMS '+d13C, '[CH4] IRMS '+dD], markerscale=5, loc=1, fontsize='x-small')
    
    ax2.set_ylabel(d13C+permil)
    #ax2.legend(['IRMS '+d13C], markerscale=5, loc=3, fontsize='x-small')
    
    ax3.set_ylabel(dD+permil)
    ax3.set_xlabel('Months')
    #ax3.legend(['IRMS '+dD], markerscale=5, loc=3, fontsize='x-small')
    
    return figure


def wind_contour(direction, var, name, fill, title, fig=None):
    
    if name==str_d13C:
        real_bins=np.flip(np.arange(-52, -47, 1),0)
        bins=np.arange(47, 52, 1)
        color_list = np.flip(plt.cm.Spectral(np.linspace(0, 1, len(bins))),0)
    elif name==str_dD:
        real_bins=np.flip(np.arange(-145, -70, 15),0)
        bins=np.arange(70, 145, 15)
        color_list = np.flip(plt.cm.Spectral(np.linspace(0, 1, len(bins))),0)
    elif name==strMR_D or name==strMR_13C:
        bins=np.arange(1800, 2900, 200)
        real_bins=np.arange(1800, 2900, 200)
        color_list = np.flip(plt.cm.Spectral(np.linspace(0, 1, len(bins))),0)
    else:
        bins=6
        color_list = np.flip(plt.cm.Spectral(np.linspace(0, 1, len(bins))),0)
    
    if fig is None:
        fig = plt.figure(figsize=(10,10))
    
    ax1 = WindroseAxes.from_ax(fig=fig)
    #ax2 = ax1.twinx()
    labels = [u'[{} : {})'.format(real_bins[i], real_bins[i+1]) for i in range(len(real_bins)-1)]
    labels.append(u'[{} : inf)'.format(real_bins[-1]))
    
    if fill==False:
        ax1.contour(direction, abs(var), bins=bins, nsector=32, colors=color_list)
    elif fill==True:
        ax1.contourf(direction, abs(var), bins=bins, nsector=32, colors=color_list)
    
    handles = ['']*len(bins)
    for i in range(len(bins)-1):
        handles[i] = mpatches.Patch(color=color_list[i], label='['+str(real_bins[i])+': '+str(real_bins[i+1])+')')
    handles[len(bins)-1] = mpatches.Patch(color=color_list[len(bins)-1], label='['+str(real_bins[len(bins)-1])+': -inf)')
    print(str(labels))
    #ax1.legend(handles=handles, title=name, loc=4, bbox_to_anchor=(1.2, -0.15), fontsize='medium')
    #ax1.legend(labels=labels)
    ax1.set_title(title, pad=10)
    #ax2.legend(handles=handles, title=name, loc=4, bbox_to_anchor=(1.2, -0.15), fontsize='medium', framealpha=1)
    #plt.show()
        
    return fig


def top_diagrams(list_df, list_labels, ambient=None, patches=True, title=None):
        
    # Axis limits
    lim_dD=[-420, -50]
    lim_d13C=[-80, -10]
    
    # Define the boundaries of the patches, and their color
        
    x_mf=[-90, -50, -50, -90]
    y_mf=[-275, -275, -450, -450]
    c_m='xkcd:neon green'
    
    x_mc=[-90, -60, -60, -90]
    y_mc=[-125, -125, -350, -350]
    #c_mc='xkcd:neon green'
        
    x_w=[-61.5, -46, -46, -61.5]
    y_w=[-252, -252, -330, -330]
    c_w='xkcd:neon pink'
        
    x_t=[-40, -20, -40, -75]
    y_t=[-100, -140, -300, -350]
    c_t='xkcd:neon blue'
        
    x_p=[-30, -15, -15, -30]
    y_p=[-100, -100, -250, -250]
    c_p='xkcd:neon yellow'
    
    #x_air=[ambient['avg_d13C']-2*ambient['std_d13C'], ambient['avg_d13C'], ambient['avg_d13C']+2*ambient['std_d13C'], ambient['avg_d13C']]
    #y_air=[ambient['avg_dD'], ambient['avg_dD']+2*ambient['std_dD'], ambient['avg_dD'], ambient['avg_dD']-2*ambient['std_dD']]
    c_amb='gray' 
    c=['black', web_red, web_blue, web_green]
    
    # Make the figure
    top_fig_top= plt.figure(figsize=(15,10))
    top_top=top_fig_top.add_subplot(111)

    # Add the areas
    top_top.fill(x_mf, y_mf, alpha=0.5, c=c_m)
    top_top.fill(x_mc, y_mc, alpha=0.1, c=c_m)
    #top_top.fill(x_w, y_w, alpha=0.5, c=c_w)
    top_top.fill(x_t, y_t, alpha=0.5, c=c_t)
    top_top.fill(x_p, y_p, alpha=0.5, c=c_p)
    # ... and corresponding patches
    biof_patch = mpatches.Patch(color=c_m, label='Microbial -fermentation', alpha=0.5)
    bioc_patch = mpatches.Patch(color=c_m, label='Microbial -C-reduction', alpha=0.1)
    #waste_patch = mpatches.Patch(color=c_w, label='Waste', alpha=0.5)
    thermo_patch = mpatches.Patch(color=c_t, label='Thermogenic', alpha=0.5)
    pyro_patch = mpatches.Patch(color=c_p, label='Pyrogenic', alpha=0.5)
        
    # Add the ambient air point
    if ambient is not None:
        top_top.scatter(x=ambient['avg_d13C'], y=ambient['avg_dD'], c=c_amb,  s=150)
    # ... and corresponding patches
        air_patch = mlines.Line2D([], [], color=c_amb, marker='o',
                      markersize=15, label='Ambient air', linestyle='')

    data_patches=[]
    i=0
    # Add the peak signatures if necessary
    for df in list_df:
        df.plot(kind='scatter', ax=top_top, x=col_d13C, y=col_dD, xerr=col_errd13C, yerr=col_errdD, marker='o', color=c[i], zorder=1)
        data_patches.append(mlines.Line2D([], [], color=c[i], marker='o',
                           label=list_labels[i], linestyle=''))
        i=i+1
        
    # add legend
    all_patches=[biof_patch, bioc_patch, thermo_patch, pyro_patch, air_patch]+data_patches
    top_fig_top.legend(handles=all_patches, bbox_to_anchor=(0.55, 0.38), loc=2, borderaxespad=0.5)

    # Arrange the axis limits
    plt.xlim(lim_d13C)
    plt.ylim(lim_dD)
    plt.xlabel(d13C+permil)
    plt.ylabel(dD+permil)

    #top_fig_top.savefig('figures/top_diagram_'+title+'.png', dpi=300, bbox_inches='tight')
    
    return top_fig_top


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

