#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Function: Function: plot step 3 of DSA (precise depth location)
     
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd, os
plt.rcParams["font.family"] = "Times New Roman"



#%% 主程序
filePath = './catalog_GCMT_2010-03-04_2010-05-13_Mw6.0-8.0_50-300km/2010-03-04-22-39-29/DSA_results'

#%%######################################################
# Plot matching results of Step 3                     #
#########################################################
#-- 读取DSA的震源深度扫描结果文件
try:
    data = pd.read_csv( filePath+'/0_PreliminarySolution.csv' )
    scanDepthMembers           = data[ 'ScanDep(km)' ]
    sumGlobal3090              = data[ 'NumMatPha' ]
    sumAvgArrTimeDiffResGlobal = data[ 'TimeRes(s)' ]
    prelimiSolution3090        = data[ 'Solution(km)' ]
    prelimiSolution3090 = np.median( prelimiSolution3090 )
except:
    print( 'Can not find: '+filePath+'/0_PreliminarySolution.csv' )    

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

outfilePath = str('./outputFigures/')
if not os.path.exists(outfilePath):
    os.mkdir(outfilePath)
else:
    print( '\n Warning: outfilePath already exists!\n')
#-- save    
plt.tight_layout()
figNamePng = outfilePath+'DSA_depth_solution.png'
plt.savefig( figNamePng, dpi=300 )
figNameSVG = outfilePath+'DSA_depth_solution.svg'
plt.savefig( figNameSVG, dpi=300 )
plt.show()



