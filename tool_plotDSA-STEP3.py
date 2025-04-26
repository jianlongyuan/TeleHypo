#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 01 00:00:00 2020

@author: Jianlong Yuan (yuan_jianlong@126.com)
    
    Supervisors: Honn Kao & Jiashun Yu


Algorithm name: 
        Depth-Scanning Algorithm (DSA)


Framework (this version only uses Z and T components):
 1. Automatic generation of synthetic waveforms for all possible depth phases.    
 2. Match-filtering of all possible depth phases.
 3. Determination of the focal depth.

    

Input:
  1. Stations data: three-component waveforms and inventory responses.
      Notice: Waveform  should be MSEED (e.g., 1A.CORRE.00.BHE.mseed )
              Inventory should be xml (e.g., 1A.PIDGE.xml )
  3. Velocity model.
      Notice: TauP Toolkit format (see Section 5 in
              https://www.seis.sc.edu/downloads/TauP/taup.pdf )

Output:
  Focal depth (median) 
  

Any questions or advices? Please contact at:
    yuan_jianlong@126.com
    1334631943@qq.com
    j.yu@cdut.edu.cn
     
"""

from obspy.taup import TauPyModel, taup_create
import matplotlib.pyplot as plt
import matplotlib.pyplot as pltDebug
from obspy.geodetics.base import kilometer2degrees
from obspy.core import UTCDateTime
import math
from pathlib import Path
import numpy as np
from scipy.signal import hilbert, find_peaks
from obspy import read, read_inventory
from obspy.geodetics import gps2dist_azimuth
import pandas as pd
from scipy.stats import kurtosis as kurt
from scipy import signal
import os, fnmatch, sys
import timeit
import shutil
import csv
import numba as nb
import pickle
plt.rcParams["font.family"] = "Times New Roman"



#%% 主程序
outfilePath = './catalog_GCMT_2010-03-04_2010-05-13_Mw6.0-8.0_50-300km/2010-03-04-22-39-29/DSA_results'

#%%######################################################
# Plot matching results of Step 3                     #
#########################################################
#-- 读取DSA的震源深度扫描结果文件
try:
    data = pd.read_csv( outfilePath+'/0_PreliminarySolution.csv' )
    scanDepthMembers           = data[ 'ScanDep(km)' ]
    sumGlobal3090              = data[ 'NumMatPha' ]
    sumAvgArrTimeDiffResGlobal = data[ 'TimeRes(s)' ]
    prelimiSolution3090        = data[ 'Solution(km)' ]
    prelimiSolution3090 = np.median( prelimiSolution3090 )
except:
    print( 'Can not find: '+outfilePath+'/0_PreliminarySolution.csv' )    

# set figure layout
fig = plt.figure( constrained_layout=True, figsize=(3.5,1.8))
fig.subplots_adjust(hspace=0.0)
fig.subplots_adjust(wspace=0.0)
gs0 = fig.add_gridspec(1, 1 )
gs00 = gs0[0].subgridspec(1,1)
ax0 = fig.add_subplot(gs00[0, 0])

plotXlim = [ min(scanDepthMembers), max(scanDepthMembers) ] 
thresholdMaxNumb = max(sumGlobal3090) * 0.9
scanDepthFrom = np.min(scanDepthMembers)
scanDepthTo   = np.max(scanDepthMembers)

t = scanDepthMembers
ax0.plot(t, sumGlobal3090, color="black", linewidth=0.9, alpha=1) 
ax0.set_xlim( plotXlim )
ax0.set_ylim( 0, max(sumGlobal3090)*1.3 )
  
# set labels
ax0.set_ylabel('Number of matches', fontsize=10)
ax0.set_xlabel('Depth (km)', fontsize=10)
ax0.grid(True, linestyle='--', linewidth=0.25)
ax0.axvline( prelimiSolution3090, lw=1, color='black', ls='--') 
# ax0.text( prelimiSolution3090+0.25, max(sumGlobal3090)*1.1, 'DSA',
#           color='black', fontsize=14 )
ax0.tick_params(axis='both', which='major', labelsize=9)

#-- set axis
ax0.margins(x=0)
tmp = plotXlim[1] - plotXlim[0]
if tmp <= 50:
    depthStep = 5
elif tmp <= 200:
    depthStep = 20
else:
    depthStep = 40
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
ax0.xaxis.set_major_locator( MultipleLocator(depthStep))
ax0.yaxis.set_major_locator( MultipleLocator(20))


#-- save    
plt.tight_layout()
figNamePng = outfilePath+'/0_DSA_STEP3.png'
plt.savefig( figNamePng, dpi=300 )
figNameSVG = outfilePath+'/0_DSA_STEP3.svg'
plt.savefig( figNameSVG, dpi=300 )
plt.show()

#-- 将图件复制到当前程序所在目录用于汇总其他图件
shutil.copy(figNameSVG, './')
shutil.copy(figNamePng, './')


