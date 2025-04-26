#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 01:33:57 2019

    Function:
        Plot brightness in 2d and 3d slices

    Input(4D dataset):
        200MAXIvaluePAPER.p (x, y, z, t)

    Output:
        2d slice (XOY) of each event
        3d slice (YOZ, XOY, XOZ) of each event

@author: jianlongyuan

Any questions or advices? Please contact at:
    yuan_jianlong@126.com
    1334631943@qq.com
    j.yu@cdut.edu.cn
    
"""

import pickle
import csv
import sys,fnmatch,os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
plt.rcParams['font.family'] = 'Times New Roman'


dataPath = './catalog_GCMT_2010-03-04_2010-05-13_Mw6.0-8.0_50-300km/2010-03-04-22-39-29/ssnapresults/'

data = pd.read_csv( str(dataPath)+'0_brmax.csv' )   

xBr   = data['time(s)']
brmax = data['brmax']

print(data)



fig = plt.figure(figsize=(3.5,1.8))

fig.subplots_adjust(hspace=0.01)
fig.subplots_adjust(wspace=0.01)
gs0 = fig.add_gridspec(1 )
ax0 = fig.add_subplot(gs0[0])
ax0.plot( xBr, brmax, color='black', linestyle='-', lw=0.9)
ax0.margins(x=0)
ax0.set_ylim(0.5,1.6)
ax0.axhline(y=1, color='red', lw=0.9)
ax0.tick_params(axis='both', which='major', labelsize=9)
ax0.set_xlabel('Scanning time (s)', fontsize=10)
ax0.set_ylabel('Maximum brightness', fontsize=10)
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
ax0.xaxis.set_major_locator( MultipleLocator(300))
ax0.yaxis.set_major_locator( MultipleLocator(0.5))
plt.tight_layout()
plt.savefig(dataPath+'brmax_ForPaper.png', dpi=300)
plt.savefig(dataPath+'brmax_ForPaper.svg', dpi=300)
plt.show()

#-- 将图件复制到当前程序所在目录用于汇总其他图件
shutil.copy(dataPath+'brmax_ForPaper.svg', './')
shutil.copy(dataPath+'brmax_ForPaper.png', './')