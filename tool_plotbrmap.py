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
"""

import pickle
import csv
import sys,fnmatch,os
import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
from obspy.geodetics.base import degrees2kilometers
from obspy.geodetics.base import kilometer2degrees
from obspy.core import UTCDateTime
plt.rcParams['font.family'] = 'Times New Roman'



def load_settings():
    
    try:
        SETTINGS = pd.read_csv('./SETTINGS.txt', delim_whitespace=True, index_col='PARAMETER')
        
        
        par7 = float( SETTINGS.VALUE.loc['studydepthbottom'] )
        par8 = float( SETTINGS.VALUE.loc['studydepthtop']  )
        par9 = float( SETTINGS.VALUE.loc['depgrid']  )
        par10= float( SETTINGS.VALUE.loc['latgrid']  )
        par11 = float( SETTINGS.VALUE.loc['longrid']  )     
        par17 =  int( SETTINGS.VALUE.loc['deplimRangebottom']  )
        par18 =  int( SETTINGS.VALUE.loc['deplimRangetop']  )       
      

        
        
        
        return  par7, par8, par9, \
                par10, par11, par17, par18
               
    
    except:
        sys.exit("Errors in 'SETTINGS.txt' !\n")
      

      
def plotbrmap(dataPath,peaktimes):
    studydepthbottom, \
    studydepthtop, depgrid, latgrid, longrid, \
    deplimRangebottom,deplimRangetop = load_settings()
    
    outputdir = dataPath + '/ssnapresults/'
    stationsFile = dataPath+'0_StationWithHighSNRforDSA_OrgTimeGCMT_DepthGCMT.csv' 
    
    data = pd.read_csv( stationsFile, header=0 )
    evLatList = data['epLat']
    evLonList = data['epLon']
    evDepList = data['evDp']
    epTime = data['epTime']
    eventLat = evLatList[0]
    eventLon = evLonList[0]
    eventDep = evDepList[0]
    starttime = epTime[0]
    
    lonFrom = eventLon-1
    lonTo = eventLon+1
    latFrom = eventLat-1
    latTo =  eventLat+1
    #获取起震时间

 
        
    eventfile = fnmatch.filter(os.listdir(outputdir), '*cataloghigh*')
    eventfile = pd.read_csv(outputdir+eventfile[0], delim_whitespace=True)
    eventfile = list(eventfile)
    ssnaptime = eventfile[0]
    snapSrcLat = float(eventfile[1])
    snapSrcLon = float(eventfile[2])
    snapSrcDep  = float(eventfile[3])
    wantedDep_id = int((snapSrcDep-studydepthbottom)//depgrid)
    epMag = float(eventfile[6])
    refertime = int(UTCDateTime(ssnaptime)-UTCDateTime(starttime))
    
    
    wantedDep = []
    wantedSnapShot = peaktimes #poltbrmap事件切片
    for i in np.arange(len(wantedSnapShot)):
        wantedDep.append(wantedDep_id)

    
    deplimRange = [deplimRangebottom, deplimRangetop] # km
    studydepth = [studydepthbottom, studydepthtop] # km 
    studyarea=[ latFrom, lonFrom, latTo-latFrom, lonTo-lonFrom] # degree
    xlimRange   = [ lonFrom, lonTo]
    ylimRange   = [ latFrom, latTo ]
    print( 'xlimRangeInKm = ',
            degrees2kilometers( 0 ),
            degrees2kilometers(xlimRange[1] - xlimRange[0] ) )
    print( 'ylimRangeInKm = ',
            degrees2kilometers( 0 ),
            degrees2kilometers(ylimRange[1] - ylimRange[0] ) )
    
    lats = np.arange( studyarea[0],  studyarea[0]+studyarea[2], latgrid)
    lons = np.arange( studyarea[1],  studyarea[1]+studyarea[3], longrid)
    deps = np.arange( studydepth[0], studydepth[1], depgrid)
    
    
    latnum = len( lats )
    lonnum = len( lons )
    depnum = len( deps )
    print('latnum, lonnum, depnum =', latnum, lonnum, depnum, latnum*lonnum*depnum)
    print("latnum=",lats)
    #-- stations
    
    df = pd.read_csv( '{0}/0_StationWithHighSNRforDSA_OrgTimeGCMT_DepthGCMT.csv'.format( dataPath ) )

    stLat = df['lat']
    stLon = df['lon']

    
    minLat = np.min(lats)
    maxLat = np.max(lats)
    minLon = np.min(lons)
    maxLon = np.max(lons)
    print( "minLat, maxLat=", minLat, maxLat )
    print( "minLon, maxLon=", minLon, maxLon )
    
    
    srcXdegIdx = int( np.floor( (eventLat-minLat)/latgrid) )
    srcYdegIdx = int( np.floor( (eventLon-minLon)/longrid) )
    print('srcXdegIdx = ', srcXdegIdx)
    print('srcYdegIdx = ', srcYdegIdx)
    
    
    with open(outputdir+'studygrids.p' , 'rb') as f:
        comein = pickle.load(f)
    studygrid=comein[0]
    print("len(studygrid)=", len(studygrid))
    
    #-- 4D dataset: x and y are in degree unit, z is in km unit, t is steps
    ssabrmap_S = fnmatch.filter(os.listdir(outputdir), '*ssabrmap_S*')
    with open(outputdir + ssabrmap_S[0] , 'rb') as f:
        comein=pickle.load(f)
    brmaps  = np.array(comein[0])
    shape_s = np.shape(brmaps)
    print("shape_s =", shape_s, 'len(brmaps)', len(brmaps[0]) )
    
    #-- 4D dataset: x and y are in degree unit, z is in km unit, t is steps
    ssabrmap_P = fnmatch.filter(os.listdir(outputdir), '*ssabrmap_P*')
    with open(outputdir + ssabrmap_P[0] , 'rb') as f:
        comein=pickle.load(f)
    brmapp = np.array(comein[0])
    shape_p = np.shape(brmapp)
    print("shape_p =", shape_p )
    
    
    brmapp = brmapp[:,0:shape_s[1]]
    
    brmap = np.multiply(brmaps,brmapp)
    shape_ps = np.shape(brmap)
    print("shape_ps =", shape_ps )
    
    nSnapShots = np.shape(brmap)[1]
    print("nSnapShots=", nSnapShots )
    
    for idx, iSnapShot in enumerate(wantedSnapShot):
        print('iSnapShot =', iSnapShot, '/', len(wantedSnapShot) )
        snapShot3dP  = np.zeros([ latnum, lonnum, depnum ])
        snapShot3dS  = np.zeros([ latnum, lonnum, depnum ])
        snapShot3dPS = np.zeros([ latnum, lonnum, depnum ])
        for i in range( latnum ):
            for j in range( lonnum ):
                for k in range( depnum ):
                    index = i*lonnum*depnum + j*depnum + k
                    snapShot3dP[i, j, k] = brmapp[index, iSnapShot]
                    snapShot3dS[i, j, k] = brmaps[index, iSnapShot]
                    snapShot3dPS[i, j, k] = brmap[index, iSnapShot]
    
        #%% set 3d-slice figure layout
        #-- get data and normalize it
        snapShot2dP  = snapShot3dP[:,:,wantedDep[idx]]
        snapShot2dS  = snapShot3dS[:,:,wantedDep[idx]]
        snapShot2dPS = snapShot3dPS[:,:,wantedDep[idx]]
    
        tmpMax = [np.max(snapShot2dP), np.max(snapShot2dS), np.max(snapShot2dPS)]
        print(tmpMax)
    
        #最大值的坐标
        tmpargMax = [np.argmax(snapShot2dP), np.argmax(snapShot2dS), np.argmax(snapShot2dPS)]
        latppid = int((tmpargMax[0]+1)/lonnum)
        if latppid == latnum:
            latppid = latppid - 1
        latpp = lats[latppid]
        latssid = int((tmpargMax[1]+1)/lonnum)
        if latssid == latnum:
            latssid = latssid - 1        
        latss = lats[latssid]
        latpsid = int((tmpargMax[2]+1)/lonnum)
        if latpsid == latnum:
            latpsid = latpsid - 1
        latps = lats[latpsid]
    
    
        lonppid = tmpargMax[0]-latppid*lonnum
        lonpp = lons[lonppid]
        lonssid = tmpargMax[1]-latssid*lonnum
        lonss = lons[lonssid]
        lonpsid = tmpargMax[2]-latpsid*lonnum
        lonps = lons[lonpsid]
    
        dep_alllonlat = np.zeros(depnum)
        depmax = 0
        for k in range(depnum):
            dep_alllonlat[k] = snapShot3dPS[latpsid, lonpsid, k]
            if depmax < dep_alllonlat[k]:
                depargMaxid = k
                depmax = dep_alllonlat[k]
    
        depps = deps[depargMaxid]
    
        print(tmpargMax)
    
        maxGlobal = np.max(tmpMax)
        print(maxGlobal)
        snapShot2dP = snapShot2dP / maxGlobal
        snapShot2dS = snapShot2dS / maxGlobal
        snapShot2dPS = snapShot2dPS / maxGlobal
        print(np.max(snapShot2dP), np.max(snapShot2dS), np.max(snapShot2dPS))
    
        fig = plt.figure(tight_layout=True, figsize=(24, 5))
        fig.subplots_adjust(hspace=0.0)
        fig.subplots_adjust(wspace=0.2)
        gs0 = fig.add_gridspec(1, 1)
        gs00 = gs0[0].subgridspec(1, 3)
        ax1 = fig.add_subplot(gs00[0, 0])
        ax2 = fig.add_subplot(gs00[0, 1])
        ax3 = fig.add_subplot(gs00[0, 2])
    
    
        extent=[ minLon, maxLon, minLat, maxLat]
        xlimRange = [minLon,maxLon]
        ylimRange = [minLat, maxLat]
        imageP = ax1.imshow( snapShot2dP, interpolation='none',
                             cmap='rainbow', extent=extent, origin='lower' )
        ax1.scatter( stLon, stLat, marker="^", s=50, color='white', zorder=101 )
        ax1.set_xlabel('Longitude ($^\circ$)', fontsize=12)
        ax1.set_ylabel('Latitude ($^\circ$)', fontsize=12)
        ax1.set_xlim( xlimRange )
        ax1.set_ylim( ylimRange )
        ax1.scatter(snapSrcLon, snapSrcLat, marker="o",
                    facecolors='none', edgecolors='white', linewidths=2,
                    s=200, color='white', zorder=111)
        ax1.scatter(lonpp, latpp, marker="o",
                    facecolors='none', edgecolors='grey', linewidths=2,
                    s=200, color='grey', zorder=111)
        ax1.scatter( eventLon, eventLat, marker="o",
                     facecolors='none', edgecolors='black', linewidths=2,
                     s=200, color='black', zorder=111, label='SSANP' )
        ax1.set_aspect('auto')
    
    
        extent = [minLon, maxLon, minLat, maxLat]
        imageS = ax2.imshow( snapShot2dS, interpolation='none',
                             cmap='rainbow', extent=extent, origin='lower' )
        ax2.scatter( stLon, stLat, marker="^", s=50, color='white', zorder=101 )
        ax2.set_xlabel('Longitude ($^\circ$)', fontsize=12)
        ax2.set_ylabel('Latitude ($^\circ$)', fontsize=12)
        ax2.set_xlim( xlimRange )
        ax2.set_ylim( ylimRange )
        ax2.scatter(snapSrcLon, snapSrcLat, marker="o",
                    facecolors='none', edgecolors='white', linewidths=2,
                    s=200, color='white', zorder=111)
        ax2.scatter(lonss, latss, marker="o",
                    facecolors='none', edgecolors='grey', linewidths=2,
                    s=200, color='grey', zorder=111)
        ax2.scatter( eventLon, eventLat, marker="o",
                     facecolors='none', edgecolors='black', linewidths=2,
                     s=200, color='black', zorder=111, label='SSANP' )
        ax2.set_aspect('auto')
    
        extent = [minLon, maxLon, minLat, maxLat]
        imagePS = ax3.imshow(snapShot2dPS, interpolation='none',
                            cmap='rainbow', extent=extent, origin='lower')
        ax3.scatter(stLon, stLat, marker="^", s=50, color='white', zorder=101)
        ax3.set_xlabel('Longitude ($^\circ$)', fontsize=12)
        ax3.set_ylabel('Latitude ($^\circ$)', fontsize=12)
        ax3.set_xlim(xlimRange)
        ax3.set_ylim(ylimRange)
        ax3.scatter(snapSrcLon, snapSrcLat, marker="o",
                    facecolors='none', edgecolors='white', linewidths=2,
                    s=200, color='white', zorder=111)
        ax3.scatter(lonps, latps, marker="o",
                    facecolors='none', edgecolors='grey', linewidths=2,
                    s=200, color='grey', zorder=111)
        ax3.scatter(eventLon, eventLat, marker="o",
                    facecolors='none', edgecolors='black', linewidths=2,
                    s=200, color='black', zorder=111, label='SSANP')
        ax3.set_aspect('auto')
    
    
        cbar = plt.colorbar(imagePS, ax=ax3)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label(label='Normalized brightness', fontsize=20)
        fig.tight_layout()
        plt.savefig("{0}/brightnessMap2D{1}_{2}.png".format(
            outputdir,iSnapShot,refertime,dpi=360))
    
        plt.show()
    
    
        snapShot2dYOZ = snapShot3dPS[:,srcYdegIdx,:]/np.max(snapShot3dPS[srcXdegIdx,:,:])
        snapShot2dXOY = snapShot3dPS[:,:,wantedDep[idx]]/np.max(snapShot3dPS[:,:,wantedDep[idx]])
        snapShot2dXOZ = snapShot3dPS[srcXdegIdx,:,:]/np.max(snapShot3dPS[:,srcYdegIdx,:])
        shape_yoz = np.shape(snapShot2dYOZ)
        shape_xoy = np.shape(snapShot2dXOY)
        shape_xoz = np.shape(snapShot2dXOZ)
        print( 'shape_yoz =', shape_yoz)
        print( 'shape_xoy =', shape_xoy)
        print( 'shape_xoz =', shape_xoz)
    
        fig = plt.figure( constrained_layout=True, figsize=(6,6))
        fig.subplots_adjust(hspace=0.0)
        fig.subplots_adjust(wspace=0.0)
        gs0 = fig.add_gridspec( 9, 9 )
        ax1 = fig.add_subplot(gs0[0:5, 0:3])
        ax2 = fig.add_subplot(gs0[0:5, 4:9])
        ax3 = fig.add_subplot(gs0[6:9, 4:9])
    
    
        #-- yoz slice
        extent=[ studydepth[0], studydepth[1], minLat, maxLat  ]
        imageYOZ = ax1.imshow( snapShot2dYOZ, interpolation='none',
                             cmap='rainbow', extent=extent, origin='lower' )
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.set_xlabel('Depth (km)', fontsize=12)
        ax1.set_ylabel('Latitude ($^\circ$)', fontsize=12)
        ax1.set_xticks(np.arange( studydepth[0], studydepth[1], step=10))
        ax1.set_yticks(np.arange( minLat, maxLat, step=0.15 ))
        ax1.set_xlim( deplimRange )
        ax1.set_ylim( ylimRange )
        ax1.set_xlim(ax1.get_xlim()[::-1])
        ax1.set_aspect('auto')
        ax1.scatter(snapSrcDep, snapSrcLat, marker="o",
                    facecolors='none', edgecolors='white', linewidths=2,
                    s=200, color='white', zorder=111)
        ax1.scatter(depps, latps, marker="o",
                    facecolors='none', edgecolors='grey', linewidths=2,
                    s=200, color='grey', zorder=111)
        ax1.scatter( eventDep, eventLat, marker="o",
                     facecolors='none', edgecolors='black', linewidths=2,
                     s=200, color='black', zorder=111 )
    
        #-- xoy slice
        extent=[ minLon, maxLon, minLat, maxLat]
        imageXOY = ax2.imshow( snapShot2dXOY, interpolation='none',
                             cmap='rainbow', extent=extent, origin='lower' )
        ax2.scatter( stLon, stLat, marker="^", s=50, color='white', zorder=101 )
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlim( xlimRange )
        ax2.set_ylim( ylimRange )
        ax2.scatter(snapSrcLon, snapSrcLat, marker="o",
                    facecolors='none', edgecolors='white', linewidths=2,
                    s=200, color='white', zorder=111, label='SSANP')
        ax2.scatter(lonps, latps, marker="o",
                    facecolors='none', edgecolors='grey', linewidths=2,
                    s=200, color='grey', zorder=111)
        ax2.scatter( eventLon, eventLat, marker="o",
                     facecolors='none', edgecolors='black', linewidths=2,
                     s=200, color='black', zorder=111, label='SSANP' )
        ax2.set_aspect('auto')
    
        #-- xoz slice
        extent=[ minLon, maxLon, studydepth[0], studydepth[1] ]
        imageXOZ= ax3.imshow( snapShot2dXOZ.T, interpolation='none',
                             cmap='rainbow',  extent=extent, origin='lower' )
        ax3.tick_params(axis='both', which='major', labelsize=12)
        ax3.set_xlabel('Longitude ($^\circ$)', fontsize=12)
        ax3.set_ylabel('Depth (km)', fontsize=12)
        ax3.set_xticks(np.arange( minLon, maxLon, step=0.15 ))
        ax3.set_yticks(np.arange( studydepth[0], studydepth[1], step=10))
    
    
        ax3.set_xlim( xlimRange )
        ax3.set_ylim( deplimRange )
        ax3.set_ylim(ax3.get_ylim()[::-1])
        ax3.set_aspect('auto')
        ax3.scatter(snapSrcLon, snapSrcDep, marker="o",
                    facecolors='none', edgecolors='white', linewidths=2,
                    s=200, color='white', zorder=111)
        ax3.scatter(lonps, depps, marker="o",
                    facecolors='none', edgecolors='grey', linewidths=2,
                    s=200, color='grey', zorder=111)
        ax3.scatter( eventLon, eventDep, marker="o",
                     facecolors='none', edgecolors='black', linewidths=2,
                     s=200, color='black', zorder=111 )
    
    
        plt.tight_layout()
        plt.savefig( "{0}/brightnessMap3D{1}_{2}.png".format(
                    outputdir,iSnapShot,refertime,dpi=360 ))
        plt.show()
   
#%% 
if __name__=="__main__":        
    dataPath = './catalog_GCMT_2010-03-01_2010-05-13_Mw6.0-8.0_50-300km/2010-03-04-22-39-29/'
    peaktimes = [1768,1780,1786]
    plotbrmap(dataPath,peaktimes)

