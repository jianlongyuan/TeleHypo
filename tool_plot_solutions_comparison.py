#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Function: plot solutions of TeleHypo, CGMT and ISC
    
"""

import pickle
import csv
import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
from obspy.geodetics.base import degrees2kilometers
from obspy.geodetics.base import kilometer2degrees
import sys,fnmatch,os
import shutil
plt.rcParams['font.family'] = 'Times New Roman'


#- http://www.isc.ac.uk/iscbulletin/search/catalogue/
#- http://www.isc.ac.uk/isc-ehb/search/catalogue/
ISC_LAT,  ISC_LON, ISC_DEPTH          = -22.3151, -68.4615, 105.0
ISCEHB_LAT,  ISCEHB_LON, ISCEHB_DEPTH = -22.2610, -68.4000, 103.4

def load_settings():
    
    try:
        SETTINGS = pd.read_csv('./SETTINGS.txt', delim_whitespace=True, index_col='PARAMETER')
        
   
        par7 = float( SETTINGS.VALUE.loc['studydepthbottom'] )
        par8 = float( SETTINGS.VALUE.loc['studydepthtop']  )
        par9 = float( SETTINGS.VALUE.loc['depgrid']  )
        par10= float( SETTINGS.VALUE.loc['latgrid']  )
        par11 = float( SETTINGS.VALUE.loc['longrid']  )     
        par16 =  int( SETTINGS.VALUE.loc['wantedSnapShotID_id']  )
        par17 =  int( SETTINGS.VALUE.loc['deplimRangebottom']  )
        par18 =  int( SETTINGS.VALUE.loc['deplimRangetop']  )       

        return  par7, par8, par9, \
                par10, par11, par16, par17, par18

    except:
        sys.exit("Errors in 'SETTINGS.txt' !\n")

def plotMAXI(dataPath):
    studydepth_for_locatebottom, \
    studydepth_for_locatetop, depgrid, latgrid, longrid, \
    wantedSnapShotID_id, deplimRangebottom,deplimRangetop = load_settings()
    
    outputdir = dataPath + '/ssnapresults/'
    stationsFile = dataPath+'0_StationWithHighSNRforDSA_OrgTimeGCMT_DepthGCMT.csv' 
    data = pd.read_csv( stationsFile, header=0 )
    evLatList = data['epLat']
    evLonList = data['epLon']
    evDepList = data['evDp']
    eventLat = evLatList[0]
    eventLon = evLonList[0]
    eventDep = evDepList[0]
        
    # dsa file
    dsafilePath = dataPath + '/DSA_results/0_PreliminarySolution.csv'
    dsadata = pd.read_csv(dsafilePath, header=0)
    Solution = dsadata['Solution(km)']
    dsaSrcDep = Solution[0]
        
    eventfile = fnmatch.filter(os.listdir(outputdir), '*cataloghigh*')
    print("11111=", eventfile)
    eventfile = pd.read_csv(outputdir+eventfile[0], delim_whitespace=True)
    eventfile = list(eventfile)
    snapSrcLat = float(eventfile[1])
    snapSrcLon = float(eventfile[2])
    snapSrcDep = float(eventfile[3])
    wantedDep_id = int((snapSrcDep-studydepth_for_locatebottom)//depgrid)
    snapMag   = float(eventfile[7])
    # dsaSrcDep = float(eventfile[8])
    wantedDep = [ wantedDep_id ]
    wantedSnapShot = [wantedSnapShotID_id] #poltmaxi事件切片
    deplimRange = [80, 130] # km
    studydepth_for_locate = [studydepth_for_locatebottom, studydepth_for_locatetop] # km   
    
    
    lonFrom = eventLon-1
    lonTo = eventLon+1
    latFrom = eventLat-1
    latTo =  eventLat+1
    studyarea=[ latFrom, lonFrom, latTo-latFrom, lonTo-lonFrom] # degree
    xlimRange   = [ lonFrom, lonTo]
    ylimRange   = [ latFrom, latTo ]
    print( 'xlimRangeInKm = ',
            degrees2kilometers( 0 ),
            degrees2kilometers(xlimRange[1] - xlimRange[0] ) )
    print( 'ylimRangeInKm = ',
            degrees2kilometers( 0 ),
            degrees2kilometers(ylimRange[1] - ylimRange[0] ) )
    lats = np.arange(studyarea[0], studyarea[0] + studyarea[2], latgrid)
    lons = np.arange(studyarea[1], studyarea[1] + studyarea[3], longrid)
    deps = np.arange(studydepth_for_locate[0], studydepth_for_locate[1], depgrid)
    
    latnum = len(lats)
    lonnum = len(lons)
    depnum = len(deps)
    print('latnum, lonnum, depnum =', latnum, lonnum, depnum, latnum * lonnum * depnum)
    print("latnum=", lats)
    # -- stations
    df = pd.read_csv( '{0}/0_StationWithHighSNRforDSA_OrgTimeGCMT_DepthGCMT.csv'.format( dataPath ) )

    stLat = df['lat']
    stLon = df['lon']

    
    minLat = np.min(lats)
    maxLat = np.max(lats)
    minLon = np.min(lons)
    maxLon = np.max(lons)
    print("minLat, maxLat=", minLat, maxLat)
    print("minLon, maxLon=", minLon, maxLon)
    
    srcXdegIdx = int(np.floor((eventLat - minLat) / latgrid))
    srcYdegIdx = int(np.floor((eventLon - minLon) / longrid))
    print('srcXdegIdx = ', srcXdegIdx)
    print('srcYdegIdx = ', srcYdegIdx)
    
    with open(outputdir + 'studygrids.p', 'rb') as f:
        comein = pickle.load(f)
    studygrid = comein[0]
    print("len(studygrid)=", len(studygrid))
    
    # -- 4D dataset: x and y are in degree unit, z is in km unit, t is steps
    MAXIvaluePAPER = fnmatch.filter(os.listdir(outputdir), '*MAXIvaluePAPER*')
    with open(outputdir + MAXIvaluePAPER[0], 'rb') as f:
        comein = pickle.load(f)
    MAXI = np.array(comein[0])
    shape = np.shape(MAXI)
    print("shape =", shape, 'len(MAXI)', len(MAXI))
    
    for idx, iSnapShot in enumerate(wantedSnapShot):
        print('iSnapShot =', iSnapShot, '/', len(wantedSnapShot))
    
        snapShot3d = np.zeros([latnum, lonnum, depnum])
        for i in range(latnum):
            for j in range(lonnum):
                for k in range(depnum):
                    index = i * lonnum * depnum + j * depnum + k
                    snapShot3d[i, j, k] = MAXI[iSnapShot, index]
    
        # %% set 3d-slice figure layout
        # -- get data and normalize it
        maxVal = np.max(snapShot3d)
        print('maxVal =', maxVal)
        snapShot2dYOZ = snapShot3d[:, srcYdegIdx, :] / maxVal
        snapShot2dXOY = snapShot3d[:, :, wantedDep[idx]] / maxVal
        snapShot2dXOZ = snapShot3d[srcXdegIdx, :, :] / maxVal
    
        shape_yoz = np.shape(snapShot2dYOZ)
        shape_xoy = np.shape(snapShot2dXOY)
        shape_xoz = np.shape(snapShot2dXOZ)
        print('shape_yoz =', shape_yoz)
        print('shape_xoy =', shape_xoy)
        print('shape_xoz =', shape_xoz)
    
        fig = plt.figure( figsize=(3.5, 3.5))
        fig.subplots_adjust(hspace=0.0)
        fig.subplots_adjust(wspace=0.2)
        gs0 = fig.add_gridspec(9, 9)
        ax1 = fig.add_subplot(gs0[0:5, 0:3])
        ax2 = fig.add_subplot(gs0[0:5, 4:9])
        ax3 = fig.add_subplot(gs0[6:9, 4:9])
    
        # -- yoz slice
        extent = [studydepth_for_locate[0], studydepth_for_locate[1], minLat, maxLat]
        imageYOZ = ax1.imshow(snapShot2dYOZ, interpolation='bilinear',
                              cmap='rainbow', extent=extent, origin='lower')
        ax1.tick_params(axis='both', which='major', labelsize=10)
        ax1.set_xlabel('Depth (km)', fontsize=10)
        ax1.set_ylabel('Latitude ($^\circ$)', fontsize=10)
        ax1.set_xticks(np.arange(studydepth_for_locate[0], studydepth_for_locate[1], step=10))
        ax1.set_yticks(np.arange(round(minLat,0)-0.5, round(maxLat,0)+0.5, step=0.5))
        ax1.set_xlim(deplimRange)
        # ax1.set_ylim(ylimRange)
        ax1.set_xlim(ax1.get_xlim()[::-1])
        ax1.set_aspect('auto')
        # ax1.scatter(snapSrcDep, snapSrcLat, marker="o",
        #             facecolors='none', edgecolors='white', linewidths=2,
        #             s=200, color='white', zorder=111, label='TeleHypo')
        ax1.scatter(dsaSrcDep, snapSrcLat, marker="o",
                    facecolors='none', edgecolors='white', linewidths=2,
                    s=50, color='gold', zorder=111, label='TeleHypo')
        ax1.scatter(eventDep, eventLat, marker="o",
                    facecolors='none', edgecolors='black', linewidths=2,
                    s=50, color='black', zorder=111, label='GCMT')
        # ax1.scatter(ISC_DEPTH, ISC_LAT, marker="o",
        #             facecolors='none', edgecolors='lime', linewidths=2,
        #             s=200, color='lime', zorder=111, label='ISC')
        ax1.scatter(ISCEHB_DEPTH, ISCEHB_LAT, marker="o",
                    facecolors='none', edgecolors='lime', linewidths=2,
                    s=50, color='green', zorder=111, label='ISCEHB')
    
        # -- xoy slice
        extent = [minLon, maxLon, minLat, maxLat]
        xlimRange = [minLon, maxLon]
        ylimRange = [minLat, maxLat]
        imageXOY = ax2.imshow(snapShot2dXOY, interpolation='bilinear',
                              cmap='rainbow', extent=extent, origin='lower')
        #ax2.scatter(stLon, stLat, marker="^", s=50, color='white', zorder=101)
        # ax2.set_xticks([])
        # ax2.set_yticks([])
        #ax2.set_xlim(xlimRange)
        #ax2.set_ylim(ylimRange)
        ax2.tick_params(axis='both', which='major', labelsize=0, labelcolor='white')
        ax2.scatter(snapSrcLon, snapSrcLat, marker="o",
                    facecolors='none', edgecolors='white', linewidths=2,
                    s=50, color='white', zorder=111, label='TeleHypo')
        ax2.scatter(eventLon, eventLat, marker="o",
                    facecolors='none', edgecolors='black', linewidths=2,
                    s=50, color='black', zorder=111, label='GCMT')
        # ax2.scatter(ISC_LON, ISC_LAT, marker="o",
        #             facecolors='none', edgecolors='lime', linewidths=2,
        #             s=200, color='black', zorder=111, label='ISC')
        ax2.scatter(ISCEHB_LON, ISCEHB_LAT, marker="o",
                    facecolors='none', edgecolors='lime', linewidths=2,
                    s=50, color='green', zorder=111, label='ISCEHB')
        ax2.set_aspect('auto')
    
        # -- xoz slice
        extent = [minLon, maxLon, studydepth_for_locate[0], studydepth_for_locate[1]]
        imageXOZ = ax3.imshow(snapShot2dXOZ.T, interpolation='bilinear',
                              cmap='rainbow', extent=extent, origin='lower')
        ax3.tick_params(axis='both', which='major', labelsize=10)
        ax3.set_xlabel('Longitude ($^\circ$)', fontsize=10)
        ax3.set_ylabel('Depth (km)', fontsize=10)
        ax3.set_xticks(np.arange(round(minLon,0)-0.5, round(maxLon,0)+0.5, step=0.5))
        ax3.set_yticks(np.arange(studydepth_for_locate[0], studydepth_for_locate[1], step=10))
        ax3.set_ylim(deplimRange)
        ax3.set_ylim(ax3.get_ylim()[::-1])
        ax3.set_aspect('auto')
        ax3.scatter(snapSrcLon, dsaSrcDep, marker="o",
                    facecolors='none', edgecolors='white', linewidths=2,
                    s=50, color='white', zorder=111, label='TeleHypo')
        ax3.scatter(eventLon, eventDep, marker="o",
                    facecolors='none', edgecolors='black', linewidths=2,
                    s=50, color='black', zorder=111, label='GCMT')
        ax3.scatter(ISCEHB_LON, ISCEHB_DEPTH, marker="o",
                    facecolors='none', edgecolors='lime', linewidths=2,
                    s=50, color='green', zorder=111, label='ISCEHB')
        from matplotlib.ticker import AutoMinorLocator, MultipleLocator
        ax1.xaxis.set_major_locator( MultipleLocator(20))
        ax1.yaxis.set_major_locator( MultipleLocator(0.5))
        ax2.xaxis.set_major_locator( MultipleLocator(0.5))
        ax2.yaxis.set_major_locator( MultipleLocator(0.5))
        ax3.xaxis.set_major_locator( MultipleLocator(0.5))
        ax3.yaxis.set_major_locator( MultipleLocator(20))
    
        outfilePath = str('./outputFigures/')
        if not os.path.exists(outfilePath):
            os.mkdir(outfilePath)
        else:
            print( '\n Warning: outfilePath already exists!\n')
        #plt.tight_layout()
        plt.savefig("{0}/solutionsComparison{1}.png".format(outfilePath,iSnapShot), bbox_inches='tight', dpi=300)
        plt.savefig("{0}/solutionsComparison{1}.svg".format(outfilePath,iSnapShot), bbox_inches='tight', dpi=300)
        plt.show()
#%% 
if __name__=="__main__":        
    dataPath = './catalog_GCMT_2010-03-04_2010-05-13_Mw6.0-8.0_50-300km/2010-03-04-22-39-29/'
    plotMAXI(dataPath)
