#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Function: plot brightness function
    
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


outfilePath = str('./outputFigures/')
if not os.path.exists(outfilePath):
    os.mkdir(outfilePath)
else:
    print( '\n Warning: outfilePath already exists!\n')

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
plt.savefig(outfilePath+'brmax_ForPaper.png', dpi=300)
plt.savefig(outfilePath+'brmax_ForPaper.svg', dpi=300)
plt.show()
