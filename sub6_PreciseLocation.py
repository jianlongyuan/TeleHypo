#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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


Please cite:
Yuan, J., Kao, H., and Yu, J. (2020). Depth-scanning algorithm: Accurate, automatic,
and efficient determination of focal depths for local and regional earthquakes.
Journal of Geophysical Research: Solid Earth 125, e2020JB019430
    
Jianlong Yuan, Huilian Ma, Jiashun Yu, Zixuan Liu and Shaojie Zhang. (2025). An approach 
for teleseismic location by automatically matching depth phase. Front. Earth Sci. (Under revirew)
  

Any questions or advices? Please contact at:
    jianlongyuan@cdut.edu.cn (Jianlong Yuan)
    1334631943@qq.com (Huilian Ma)
    j.yu@cdut.edu.cn  (Jiashun Yu)
    2751017165@qq.com (Zixuan Liu)
    1716136870@qq.com (Shaojie Zhang)
     
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


#%%-- subroutine: load input parameters from 'DSA_SETTINGS.txt'
def load_settings():
    '''
     PARAMETER          DESCRIPTION
     
     par1    Data directory, including wavefroms and velocity model 
     par2    Velocity model name (this string should not include '.nd')    
     par3    Tolerance between the observed and predicted differential travel times (second)
     par4    Cross-correlation coefficient threshold
     par5    Minimal frequency used for band-pass filter of vertical component (Hz)
     par6    Maximal frequency used for band-pass filter of vertical component (Hz)
     par7    Minimal frequency used for band-pass filter of horizontal components (Hz)
     par8    Maximal frequency used for band-pass filter of horizontal components (Hz)
     par9    Radius above the focal depth reported by catalog (interger, km)
     par10   Radius below the focal depth reported by catalog (interger, km)
     par11   For monitoring: 1 -> active,  0 -> inactive
     par12   Plot Steps 1 and 2 of DSA: 1 -> active,  0 -> inactive
    '''
    
    try:
        SETTINGS = pd.read_csv('./SETTINGS.txt', delim_whitespace=True, index_col='PARAMETER')
        par1 = SETTINGS.VALUE.loc['catalogPath']
        par2 = SETTINGS.VALUE.loc['velModel']
        par3 = float( SETTINGS.VALUE.loc['arrTimeDiffTole'] )
        par4 = float( SETTINGS.VALUE.loc['ccThreshold']  )
        par5 = float( SETTINGS.VALUE.loc['vFrequencyFrom']  )
        par6 = float( SETTINGS.VALUE.loc['vFrequencyTo']  )
        par7 = float( SETTINGS.VALUE.loc['hFrequencyFrom']  )
        par8 = float( SETTINGS.VALUE.loc['hFrequencyTo']  )
        par9 =   int( SETTINGS.VALUE.loc['depthRadiusAbove']  )
        par10 =  int( SETTINGS.VALUE.loc['depthRadiusBelow']  )
        par11 =  int( SETTINGS.VALUE.loc['verboseFlag']  )
        par12 =  int( SETTINGS.VALUE.loc['plotSteps1n2Flag']  )
        
        return  par1, par2, par3, par4, par5, par6, par7, par8, par9, par10, par11, par12
    
    except:
        sys.exit("Errors in 'SETTINGS.txt' !\n")
        

#%%-- subroutine: cross-correlation
@nb.jit(nopython=True)
def xcorrssl( scanTimeBeg, scanTimeEnd, tem, tra ):
    
    temLeng = len(tem)
    traLeng = len(tra)
    time_lags= traLeng - temLeng + 1
    
    #-- demean for the template
    b = tem - np.mean(tem)
    corr_norm_val = []
    
    for k in range( time_lags ):
        if ( k >= scanTimeBeg and k <=scanTimeEnd ):
            # demean for the trace
            a = tra[k:(k+temLeng)] - np.mean(tra[k:(k+temLeng)])
            stdev = (np.sum(a**2)) ** 0.5 * (np.sum(b**2)) ** 0.5
            if stdev != 0:
                corr = np.sum(a*b)/stdev
            else:
                corr = 0
            corr_norm_val.append(corr)
        else:
            corr_norm_val.append(0.)
            
    return corr_norm_val



#-- subroutine: arrival time forward modelling kernel
def subArrivalTimeForward( velModel, srcDepth, recDisInDeg, phaList, recDepth ):
    model = TauPyModel(model= velModel )
    
    try:
        arrivals = model.get_travel_times(source_depth_in_km=srcDepth,
                                           distance_in_degree=recDisInDeg,
                                           phase_list= phaList,
                                           receiver_depth_in_km=recDepth)

    except: # avoid TauP error
        srcDepth += 0.1
        arrivals = model.get_travel_times(source_depth_in_km=srcDepth,
                                           distance_in_degree=recDisInDeg,
                                           phase_list= phaList,
                                           receiver_depth_in_km=recDepth) 
    return  arrivals











def subCalSteps1and2OneStation( args ):
    ist              = args[0]
    evla             = args[1]
    evlo             = args[2]
    evdp             = args[3]
    epTime           = args[4]
    velModel         = args[5]
    stRawE           = args[6]
    stRawN           = args[7]
    stRawZ           = args[8]
    numScanDepth     = args[9]
    srcDepthScanBeg  = args[10]
    srcDepthScanEnd  = args[11]
    arrTimeDiffTole  = args[12]
    ccThreshold      = args[13]
    vFrequencyFrom   = args[14]
    vFrequencyTo     = args[15]
    hFrequencyFrom   = args[16]
    hFrequencyTo     = args[17]
    invPath          = args[18]
    outfilePath      = args[19]
    # diffTime         = args[20]
    verboseFlag      = args[21]          
    #%%-- Allocate memory
    totNumMatPhaOneSta = np.zeros(( numScanDepth))
    avgArrTimeDiffResOneStaSum = np.zeros(( numScanDepth))
    avgArrTimeDiffResOneStaSum.fill(9999) # initial array with a high value
    depthCandidateArrGlobalZ = [[] for i in range(numScanDepth)]
    depthCandidateArrGlobalT = [[] for i in range(numScanDepth)]
    depthCandidatePhaOrgNameGlobalZ = [[] for i in range(numScanDepth)]
    depthCandidatePhaOrgNameGlobalT = [[] for i in range(numScanDepth)]
    
    #%%########################################################
    # Step 1: Automatic generation of synthetic waveforms for #
    #         all possible depth phases                       #
    ###########################################################
    #%%-- Get some key infomation
    #print('stRawZ[0].stats', stRawZ[0].stats )
    '''
    evla  = iev.origins[0].latitude
    evlo  = iev.origins[0].longitude
    evdp  = iev.origins[0].depth/1000.0 
    '''
    network = stRawZ[0].stats.network
    station = stRawZ[0].stats.station
    location= stRawZ[0].stats.location
    
    starttime = UTCDateTime(epTime)

    try:
        inv = read_inventory( "{0}{1}.{2}.xml".format( invPath, network, station ) )
    except:
        print( 'Can not find {0}{1}.{2}.xml'.format( invPath, network, station ) )
        return 0
    
    net = inv[0]
    sta = net[0]
    stla = sta.latitude
    stlo = sta.longitude
    epi_dist, azimuth, baz = gps2dist_azimuth(evla, evlo, stla, stlo )
    recDisInKm  = epi_dist/1000.0
    recDisInDeg = kilometer2degrees(recDisInKm)
    recDepth = 0  # station's depth( default 0 km)
    #%%-- Remove mean value and trend
    stWantedE0 = stRawE.copy()
    stWantedN0 = stRawN.copy()
    stWantedZ0 = stRawZ.copy()
    stWantedE0[0].detrend( type='demean')
    stWantedN0[0].detrend( type='demean')
    stWantedZ0[0].detrend( type='demean')
    stWantedE0[0].detrend( type='simple')
    stWantedN0[0].detrend( type='simple')
    stWantedZ0[0].detrend( type='simple')
    
    print( 'Detrend done!\n')
#    stWantedE0[0].plot()
#    stWantedN0[0].plot()
#    stWantedZ0[0].plot()
    #%%-- Remove response
    try:
        pre_filt = (0.001, 0.005, 50.0, 60.0)
        stWantedE0[0].remove_response(inventory=inv, output="DISP",
                  pre_filt=pre_filt, plot=False )
        stWantedN0[0].remove_response(inventory=inv, output="DISP",
                  pre_filt=pre_filt, plot=False )
        stWantedZ0[0].remove_response(inventory=inv, output="DISP",
                  pre_filt=pre_filt, plot=False )
        
        print( 'Remove response done!\n')
#        stWantedE0[0].plot()
#        stWantedN0[0].plot()
#        stWantedZ0[0].plot()
        
    except:
        print( 'No response file! Maybe synthetic data?\n')
        stT0 = stWantedN0.copy()        
        stZ0 = stWantedZ0.copy()        
        
        #%% -- do ratation from Z12 to ZNE
    try:
        if stRawE[0].stats.channel == 'HH1' or stRawE[0].stats.channel == 'BH1':
            stZ12 = stWantedZ0 + stWantedE0 + stWantedN0
            stZ12.rotate( method='->ZNE', inventory=inv )
            stWantedZ0 = stZ12.select(component="Z")
            stWantedN0 = stZ12.select(component="N")
            stWantedE0 = stZ12.select(component="E")
        
        #%%-- do ratation: NE to RT using back-azimuth angle        
        stNE = stWantedN0 + stWantedE0
        stNE.rotate( method='NE->RT', back_azimuth=baz )
        stT0 = stNE.select(component="T")        
        stZ0 = stWantedZ0.copy()

        print( 'Rotation done!\n')
#        stZ0.plot()
#        stT0.plot()        
        
        
        #%%-- taper before filtering
#        stZ0[0] = stZ0[0].taper(max_percentage=0.1, side='left')
#        stT0[0] = stT0[0].taper(max_percentage=0.1, side='left')

    except:
        print (network+'.'+station+str(': rotation failed!'))
        #stR0 = stWantedE0.copy()
        stT0 = stWantedN0.copy()        
        stZ0 = stWantedZ0.copy()
    
    
    
    #%%-- frequency filtering
    stZ0[0] = stZ0[0].filter('bandpass', freqmin=vFrequencyFrom, freqmax=vFrequencyTo,
                                         corners=4, zerophase=False)

    stT0[0] = stT0[0].filter('bandpass', freqmin=hFrequencyFrom, freqmax=hFrequencyTo,
                                         corners=4, zerophase=False)

    print( 'Bandpass filtering done!\n')
    
    #%%-- 重采样为 10 Hz
    resamRate = 10 # Hz
    stZ0.interpolate( sampling_rate=resamRate, method="lanczos",
                      a=12, window="blackman" )

    stT0.interpolate( sampling_rate=resamRate, method="lanczos",
                      a=12, window="blackman" )
    DT = stZ0[0].stats.delta    
    print( 'Reampling done!\n')


    #%%-- check waveform
    if verboseFlag == 1:
        print('\n Check Z, R, and T waveforms: \n')
        #stZ0.plot()
        #stT0.plot()
    
    #%%-- Waveform scanning window used for DSA, choosing 2 mins before
    # theoretical onset of the direct P and 10 mins after S
    try:

        refDepth = evdp # focal depth (adopt from Catalog) for calculting onset time of direct wave (km)        
        phaList = [ 'P', 'S', 'ScS' ]
        arrivals = subArrivalTimeForward( velModel, refDepth, recDisInDeg, phaList, recDepth )
        calOnsetP   = arrivals[0].time
        calOnsetS   = arrivals[1].time
        calOnsetScS = arrivals[2].time
    except:
        print( network+'.'+station+str(': theoretical onset time failed!') )
        return 0
         
    #-- 选择直达P前两分钟和ScS波后十分钟的数据段进行扫描, 用ScS的原因是可以去除面波干扰
    timeBeforeP = 2*60 # sec
    timeAfterS  = 2*60 # sec
          
    #%%-- Extract wavefroms according to the wanted time window
    timeWinBeg = calOnsetP-timeBeforeP
    timeWinEnd = calOnsetScS+timeAfterS
    stRawT = stT0.trim( starttime+timeWinBeg, starttime+timeWinEnd )
    stRawZ = stZ0.trim( starttime+timeWinBeg, starttime+timeWinEnd )

    #%%-- 如果选择不用R分量，则接下来的匹配过程将R分量替换为Z（暂定，可改动）
    #stRawR = stRawZ.copy()
    
    #%%-- check waveform
    if verboseFlag == 1:
        print('\n Check Z, R, and T waveforms: \n')
        #stRawZ.plot()
        #stRawT.plot()
        
        
    #%%-- Select phase with large amplitude using the distribution
    # of amplitude peak and drop
    try:    
        stNorZ = stRawZ[0].data / max( np.fabs(stRawZ[0].data))
        stNorT = stRawT[0].data / max( np.fabs(stRawT[0].data))
    except:
        print( network+'.'+station+str(': data normalization failed!') )
        return 0    

    extremaMinIdxZ = signal.argrelextrema( np.array( stNorZ ), np.less)
    extremaMinIdxT = signal.argrelextrema( np.array( stNorT ), np.less)
    extremaMaxIdxZ = signal.argrelextrema( np.array( stNorZ ), np.greater)
    extremaMaxIdxT = signal.argrelextrema( np.array( stNorT ), np.greater)      
    extremaIdxZ = np.concatenate( (extremaMinIdxZ, extremaMaxIdxZ), axis=1 )
    extremaIdxT = np.concatenate( (extremaMinIdxT, extremaMaxIdxT), axis=1 )
    
    histZ = sorted(  stNorZ[extremaIdxZ][0] )
    histT = sorted(  stNorT[extremaIdxT][0] )
    
    meanZ = np.mean(histZ)
    meanT = np.mean(histT)
    stdZ  = np.std(histZ)
    stdT  = np.std(histT)
    
    ratioStd1 = 2.0
    leftBoundry1Z  = meanZ - stdZ * ratioStd1
    leftBoundry1T  = meanT - stdT * ratioStd1
    rightBoundry1Z = meanZ + stdZ * ratioStd1
    rightBoundry1T = meanT + stdT * ratioStd1
                        
               
    #%%-- plot histogram
    if verboseFlag == -1:
        print( '\n Histogram of peaks/troughs below:')
        #-- Z
        plt.figure( constrained_layout=True, figsize=(3,4))
        plt.hist( histZ, density=False, bins=11, orientation='horizontal')
        plt.text( 10, -0.5, r'$\mu-{0}\sigma$'.format( format(ratioStd1, '.0f') ),
                 fontsize=14, rotation=0 )
        plt.text( 10,  0.5, r'$\mu+{0}\sigma$'.format( format(ratioStd1, '.0f') ),
                 fontsize=14, rotation=0 )  
        plt.xscale('log')
        plt.xlim(1e0, 1e3)
        plt.ylim(-1.1, 1.1)
        plt.xlabel('Number of peaks/troughs', fontsize=12, labelpad=6)
        plt.ylabel('Amplitude (Z-component)', fontsize=12, labelpad=6)
        plt.axhspan(leftBoundry1Z, rightBoundry1Z, alpha=0.4, color='black')
        plt.show()
        
        #-- T
        plt.figure( constrained_layout=True, figsize=(3,4))
        plt.hist( histT, density=False, bins=11, orientation='horizontal')
        plt.text( 10, -0.5, r'$\mu-{0}\sigma$'.format( format(ratioStd1, '.0f') ),
                 fontsize=14, rotation=0 )
        plt.text( 10,  0.5, r'$\mu+{0}\sigma$'.format( format(ratioStd1, '.0f') ),
                 fontsize=14, rotation=0 )      
        plt.xscale('log')
        plt.xlim(1e0, 1e3)
        plt.ylim(-1.1, 1.1)
        plt.xlabel('Number of peaks/troughs', fontsize=12, labelpad=6)
        plt.ylabel('Amplitude (T-component)', fontsize=12, labelpad=6)
        plt.axhspan(leftBoundry1T, rightBoundry1T, alpha=0.4, color='black')
        plt.show()





                    
    #-- filter out the small amplitudes
    largeAmpZ = np.zeros( len(stNorZ) )
    largeAmpT = np.zeros( len(stNorT) )
    for i in range (len(stNorZ) ):
        if stNorZ[i] <= leftBoundry1Z or stNorZ[i] >= rightBoundry1Z:
            largeAmpZ[ i ] = stNorZ[i]
             
    for i in range (len(stNorT) ):
        if stNorT[i] <= leftBoundry1T or stNorT[i] >= rightBoundry1T:
            largeAmpT[ i ] = stNorT[i]


    #%%-- Get P and S onset using kurtosis of scipy 
    kurWinBegP = int( (calOnsetP-timeWinBeg-120)/DT ) # 120 sec before P
    kurWinEndP = int( (calOnsetP-timeWinBeg+60)/DT ) # 120 sec after P
    kurWinBegS = int( (calOnsetS-timeWinBeg-120)/DT ) # 120 sec before S
    kurWinEndS = int( (calOnsetS-timeWinBeg+60)/DT ) # 120 sec after S
    
        
    #-- Z
    timeLengThr = 50
    numSamRoll = int( timeLengThr/DT ) # 50 sec for rolling computation of kurtosis
    dfZ = pd.DataFrame()   
    dfZ['Z'] = stNorZ[kurWinBegP:kurWinEndP]
    dfZ['kurtosisZ'] = dfZ['Z'].rolling(numSamRoll).apply(kurt, raw=True)       
    kurtosisZ = dfZ[ 'kurtosisZ' ]
    kurtosisZ = kurtosisZ / np.max( np.fabs( kurtosisZ ))
    #-- 由于kurtosis峰值对应时刻会比直达波实际起跳时刻延迟,因此需要改进,即对kurtosis
    # 的结果计算梯度( 参考Craig(2019) )
    dfKurZ = pd.DataFrame()   
    dfKurZ['kurZ'] = kurtosisZ
    dfKurZ['kurtosis2Z'] = dfKurZ['kurZ'].rolling(numSamRoll).apply(kurt, raw=True)       
    kurtosis2Z = dfKurZ[ 'kurtosis2Z' ]
    kurtosis2Z = kurtosis2Z / np.max( np.fabs( kurtosis2Z ))
    #-- 计算kurtosis函数的梯度    
    grad2Z  = np.zeros( len(kurtosis2Z) )
    for idx in range( len(kurtosis2Z)-10 ):
        val = kurtosis2Z[idx+10]-kurtosis2Z[idx]
        if val > 0:
            grad2Z[idx] = val   
    onsetBegIdx1Z = np.argmax( grad2Z )
    print( 'onsetBegIdx1Z=', onsetBegIdx1Z)
    prelimiOnsetZ = onsetBegIdx1Z*DT
    '''
    pltDebug.figure(figsize=(12,2))
    pltDebug.plot( kurtosisZ+2 )
    pltDebug.plot( kurtosis2Z+1 )
    pltDebug.plot( grad2Z )
    pltDebug.margins(0)
    pltDebug.show()
    '''
    
    #-- T
    dfT = pd.DataFrame()   
    dfT['T'] = stNorT[kurWinBegS:kurWinEndS]
    dfT['kurtosisT'] = dfT['T'].rolling(numSamRoll).apply(kurt, raw=True)       
    kurtosisT = dfT[ 'kurtosisT' ]
    kurtosisT = kurtosisT / np.max( np.fabs( kurtosisT ))
    #-- 计算kurtosis函数的梯度        
    dfKurT = pd.DataFrame()   
    dfKurT['kurT'] = kurtosisT
    dfKurT['kurtosis2T'] = dfKurT['kurT'].rolling(numSamRoll).apply(kurt, raw=True)       
    kurtosis2T = dfKurT[ 'kurtosis2T' ]
    kurtosis2T = kurtosis2T / np.max( np.fabs( kurtosis2T ))

    grad2T  = np.zeros( len(kurtosis2T) )
    for idx in range( len(kurtosis2T)-10 ):
        val = kurtosis2T[idx+10]-kurtosis2T[idx]
        if val > 0:
            grad2T[idx] = val   
    onsetBegIdx2T = np.argmax( grad2T )
    print( 'onsetBegIdx2T=', onsetBegIdx2T)
    prelimiOnsetT = ( kurWinBegS + onsetBegIdx2T)*DT
    '''
    pltDebug.figure(figsize=(12,2))
    pltDebug.plot( kurtosisT+2 )
    pltDebug.plot( kurtosis2T+1 )
    pltDebug.plot( grad2T )
    pltDebug.margins(0)
    pltDebug.show()    
    '''
    



    #%%-- plot kurtosis function
    if verboseFlag == 1:
        print('\n Kurtosis picking:')
        kurNorZ = kurtosisZ / np.max(np.fabs(kurtosisZ))
        tKur = np.arange( 0, len(kurtosisZ), 1)*DT+kurWinBegP*DT
        tT0  = np.arange( 0, len(stNorZ), 1)*DT
        pltDebug.figure(figsize=(12,2))
        pltDebug.tick_params(axis='both', which='major', labelsize=10)
        pltDebug.xlabel('Time (s)', fontsize=12)
        pltDebug.ylabel('Normalized Amp.', fontsize=12)
        pltDebug.title( 'T: Large amplitudes (grey) and kurtosis (blue)', fontsize=12 )
        pltDebug.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
        pltDebug.plot( tKur, kurNorZ, lw=2, color='blue' )
        pltDebug.plot( tT0, stNorZ*1.0+1, lw=0.5, color='gray', label='Raw record' )
        pltDebug.plot( tT0, largeAmpZ*1.0+1, lw=1.0, color='black', label='Large amplitude' )
        pltDebug.scatter( prelimiOnsetZ, 0.25, marker="o", s=100, label='Picked onset',
                          facecolor='none', edgecolor='black', lw=1, zorder=101 )
        pltDebug.scatter( calOnsetP-timeWinBeg, 0.5, marker="o", s=100, label='Theoretical onset', 
                          facecolor='none', edgecolor='red', lw=1, zorder=101 )
        pltDebug.xlim( 0, len(stNorZ)*DT )
        pltDebug.margins(0)
        pltDebug.legend(prop={"size":10}, loc='upper right')
        
        
        kurNorT = kurtosisT / np.max(np.fabs(kurtosisT))              
        tKur = np.arange( 0, len(kurtosisT), 1)*DT+kurWinBegS*DT
        tT0  = np.arange( 0, len(stNorT), 1)*DT
        pltDebug.figure(figsize=(12,2))
        pltDebug.tick_params(axis='both', which='major', labelsize=10)
        pltDebug.xlabel('Time (s)', fontsize=12)
        pltDebug.ylabel('Normalized Amp.', fontsize=12)
        pltDebug.title( 'T: Large amplitudes (grey) and kurtosis (blue)', fontsize=12 )
        pltDebug.ticklabel_format(style='sci',scilimits=(-3,4),axis='both')
        pltDebug.plot( tKur, kurNorT , lw=2, color='blue' )
        pltDebug.plot( tT0, stNorT*1.0+1, lw=0.5, color='gray', label='Raw record' )
        pltDebug.plot( tT0, largeAmpT*1.0+1, lw=1.0, color='black', label='Large amplitude' )
        pltDebug.scatter( prelimiOnsetT, 0.25, marker="o", s=100, label='Picked onset',
                          facecolor='none', edgecolor='black', lw=1, zorder=101 )
        pltDebug.scatter( calOnsetS-timeWinBeg, 0.5, marker="o", s=100, label='Theoretical onset', 
                          facecolor='none', edgecolor='red', lw=1, zorder=101 )
        pltDebug.xlim( 0, len(stNorT)*DT )
        pltDebug.margins(0)
        pltDebug.legend(prop={"size":10}, loc='upper left')
        
        plt.show()






    #%%-- when arrival time difference between the observed direct phases and
    # the theoretical ones is larger than 30 s, then skip current station
    # arrival time, then skip current station
    if np.fabs( prelimiOnsetZ+timeWinBeg-calOnsetP ) > 30 or\
       np.fabs( prelimiOnsetT+timeWinBeg-calOnsetS ) > 30:
           print ( network+'.'+station+str(': arrival time failed!') )
           return 0


    #%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #-------------------------------------------------------------
    # 功能: 获取直达波
    #-------------------------------------------------------------
    """
    beforeOnsetZ = prelimiOnsetZ + 5#-- 5 sec before prelimiOnsetZ of Z component
    afterOnsetZ  = prelimiOnsetZ + timeLengThr #-- 50 sec after P onset of Z component
    beforeOnsetT = prelimiOnsetT + 5
    afterOnsetT  = prelimiOnsetT + timeLengThr
    temBegIdxZ   = int( beforeOnsetZ/DT )
    temEndIdxZ   = int( afterOnsetZ/DT )
    temBegIdxT   = int( beforeOnsetT/DT )
    temEndIdxT   = int( afterOnsetT/DT )
    temDataZ     = largeAmpZ[ temBegIdxZ:temEndIdxZ ]
    temDataT     = largeAmpT[ temBegIdxT:temEndIdxT ]
    #-- 寻找波峰和波谷的索引位置
    posPeakIdxZ, _ = find_peaks( temDataZ, height=0.1, distance=int(0.5/DT) )
    posPeakIdxT, _ = find_peaks( temDataT, height=0.1, distance=int(0.5/DT) )
    negPeakIdxZ, _ = find_peaks( temDataZ*-1, height=0.1, distance=int(0.5/DT) )
    negPeakIdxT, _ = find_peaks( temDataT*-1, height=0.1, distance=int(0.5/DT) )
    #-- 第一个波峰和第一个波谷之间的时差作为直达波的半个周期
    try:
        halfCircleTimeZ = np.fabs( (posPeakIdxZ[0]-negPeakIdxZ[0])*DT )
        halfCircleTimeT = np.fabs( (posPeakIdxT[0]-negPeakIdxT[0])*DT )
    except:
        return
        
    #-- 直达波截取的参考起始时间:弱振幅滤波后的第一个强震幅出现的时刻
    tP = prelimiOnsetZ # beforeOnsetZ+posPeakIdxZ[0]*DT
    tS = prelimiOnsetT # beforeOnsetT+posPeakIdxT[0]*DT
    #-- 直达波最终截取的起始时间:以tP为中心,向前1.5个周期,向后3.5个周期
    finalOnsetBegP = tP - 0.5 * halfCircleTimeZ
    finalOnsetEndP = tP + 3.5 * halfCircleTimeZ
    finalOnsetBegS = tS - 0.5 * halfCircleTimeT
    finalOnsetEndS = tS + 3.5 * halfCircleTimeT
    #-- 直达波起止时间的索引
    finalOnsetBegIdxP = int( finalOnsetBegP/DT )
    finalOnsetEndIdxP = int( finalOnsetEndP/DT )
    finalOnsetBegIdxS = int( finalOnsetBegS/DT )
    finalOnsetEndIdxS = int( finalOnsetEndS/DT )
    print( "finalOnsetBegP, finalOnsetEndP =", finalOnsetBegP, finalOnsetEndP )
    print( "finalOnsetBegS, finalOnsetEndS =", finalOnsetBegS, finalOnsetEndS )


    #--寻找振幅值接近于0的直达波起止点，让直达波看着更加合理, 做法是：1. 以上一步的直达波
    # 候选者的起始时间t0和截至时间t1为准，从t0向前截取一个周期长度的数据d0，并从t1向后截取
    # 一个周期的数据d1；2.对d0从t0向前搜索满足振幅小于等于阈值e的时刻t00，对d1从t1向后
    # 搜索满足振幅小于等于阈值e的时刻t11; 将时刻t00-t11的数据段作为最终直达波模板.如果t00
    # 或t11不存在，则将 t0-t11 或 t00-t1 数据段作为最终直达波模板.如果t00和t11均
    # 不存在， 则将原来的t0-t1数据段作为最终直达波模板.
    finalOnset1BegIdxP = int( (finalOnsetBegP-2*halfCircleTimeZ)/DT )
    finalOnset1EndIdxP = int( (finalOnsetEndP+2*halfCircleTimeZ)/DT )
    beforeTemplateZ = stNorZ[ finalOnset1BegIdxP:finalOnsetBegIdxP ] 
    afterTemplateZ  = stNorZ[ finalOnsetEndIdxP:finalOnset1EndIdxP ]
    beforeTemplateZ = beforeTemplateZ / np.max( np.fabs(beforeTemplateZ) )
    afterTemplateZ  = afterTemplateZ / np.max( np.fabs(afterTemplateZ) )
    
    for isam, ival in enumerate( reversed(beforeTemplateZ) ):
        if np.fabs(ival) <= 0.1:
            finalOnsetBegIdxP -= isam
            break
    for isam, ival in enumerate( afterTemplateZ ):
        if np.fabs(ival) <= 0.1:
            finalOnsetEndIdxP += isam
            break
    
    finalOnset1BegIdxS = int( (finalOnsetBegS-2*halfCircleTimeT)/DT )
    finalOnset1EndIdxS = int( (finalOnsetEndS+2*halfCircleTimeT)/DT )
    beforeTemplateT = stNorT[ finalOnset1BegIdxS:finalOnsetBegIdxS ] 
    afterTemplateT  = stNorT[ finalOnsetEndIdxS:finalOnset1EndIdxS ]
    beforeTemplateT = beforeTemplateT / np.max( np.fabs(beforeTemplateT) )
    afterTemplateT  = afterTemplateT / np.max( np.fabs(afterTemplateT) )
    
    for isam, ival in enumerate( reversed(beforeTemplateT) ):
        if np.fabs(ival) <= 0.1:
            finalOnsetBegIdxS -= isam
            break
    for isam, ival in enumerate( afterTemplateT ):
        if np.fabs(ival) <= 0.1:
            finalOnsetEndIdxS += isam
            break 
    
    
    #-- 最终直达波的起跳时刻
    finalOnsetBegP = finalOnsetBegIdxP*DT
    finalOnsetBegS = finalOnsetBegIdxS*DT
    templateZ = stNorZ[ finalOnsetBegIdxP:finalOnsetEndIdxP ]
    templateT = stNorT[ finalOnsetBegIdxS:finalOnsetEndIdxS ]
    
    
    
    #%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #-- 此部分用于调试:::开始点
    pltDebug.figure(figsize=(12,2))
    pltDebug.plot( temDataZ )
    pltDebug.scatter( posPeakIdxZ, np.ones( len(posPeakIdxZ) ), color='red' )
    pltDebug.scatter( negPeakIdxZ, np.ones( len(negPeakIdxZ) ), color='black' )
    pltDebug.margins(0)
    pltDebug.show()
    print( posPeakIdxZ[0], negPeakIdxZ[0], halfCircleTimeZ, 'DT=', DT )
    print( '5*halfCircleTimeZ = ', 5*halfCircleTimeZ )

    pltDebug.figure(figsize=(12,2))
    pltDebug.plot( temDataT )
    pltDebug.scatter( posPeakIdxT, np.ones( len(posPeakIdxT) ), color='red' )
    pltDebug.scatter( negPeakIdxT, np.ones( len(negPeakIdxT) ), color='black' )
    pltDebug.margins(0)
    pltDebug.show()
    print( posPeakIdxT[0], negPeakIdxT[0], halfCircleTimeT, 'DT=', DT )
    print( '5*halfCircleTimeT = ', 5*halfCircleTimeT )    
    #-- 此部分用于调试:::终止点
    #--!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    
    
    
    #%%-- first time to roughly chose direct wave
    beforeOnsetZ = prelimiOnsetZ - 5  #-- 5 sec before P onset of Z component
    afterOnsetZ  = prelimiOnsetZ + 10 #-- 10 sec after P onset of Z component
    temZtBegIdx  = int( beforeOnsetZ/DT )
    temZtEndIdx  = int( afterOnsetZ/DT )
    templateZ    = stNorZ[ temZtBegIdx:temZtEndIdx ]
    

    beforeOnsetT = prelimiOnsetT - 5
    afterOnsetT  = prelimiOnsetT + 10
    temTtBegIdx  = int( beforeOnsetT/DT )
    temTtEndIdx  = int( afterOnsetT/DT )
    templateT    = stNorT[ temTtBegIdx:temTtEndIdx ]

    #%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    '''#-- 此部分用于调试:::开始点
    pltDebug.figure(figsize=(5,2))
    pltDebug.plot( templateZ )
    pltDebug.margins(0)
    pltDebug.show()

    pltDebug.figure(figsize=(5,2))
    pltDebug.plot( templateT )
    pltDebug.margins(0)
    pltDebug.show()  
   ''' #-- 此部分用于调试:::终止点
    #--!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    
    try:
        if max(templateZ) == 0 or max(templateT) == 0:
            print (  network+'.'+station+str(': template failed!') )
            return 0
    except:
        print (  network+'.'+station+str(': template failed!') )
        return 0
    #-- 在视窗内搜寻振幅最大和最小值所对应的时差作为直达波的半个周期
    minAmpTemValZ = np.min(templateZ)
    minAmpTemIdxZ = np.argmin(templateZ)
    minAmpTemTimeZ = minAmpTemIdxZ * DT + beforeOnsetZ
    maxAmpTemValZ = np.max(templateZ)
    maxAmpTemIdxZ = np.argmax(templateZ)
    maxAmpTemTimeZ = maxAmpTemIdxZ * DT + beforeOnsetZ
    
    minAmpTemValT = np.min(templateT)
    minAmpTemIdxT = np.argmin(templateT)
    minAmpTemTimeT = minAmpTemIdxT * DT + beforeOnsetT
    maxAmpTemValT = np.max(templateT)
    maxAmpTemIdxT = np.argmax(templateT)
    maxAmpTemTimeT =  maxAmpTemIdxT * DT + beforeOnsetT
    
    halfCircleTimeZ = np.fabs( minAmpTemTimeZ - maxAmpTemTimeZ )
    halfCircleTimeT = np.fabs( minAmpTemTimeT - maxAmpTemTimeT )

    """
    if np.fabs(minAmpTemValZ) > np.fabs(maxAmpTemValZ):
        tP = minAmpTemTimeZ
    else:
        tP = maxAmpTemTimeZ
    
    if np.fabs(minAmpTemValT) > np.fabs(maxAmpTemValT):
        tS = minAmpTemTimeT
    else:
        tS = maxAmpTemTimeT
    """

    
    tP = prelimiOnsetZ
    tS = prelimiOnsetT

    #%%-- using x times periods as the time length of direct wave     
    finalOnsetBegP = tP - 0.5 * halfCircleTimeZ    
    finalOnsetEndP = tP + 4.5 * halfCircleTimeZ    
    finalOnsetBegS = tS - 0.5 * halfCircleTimeT 
    finalOnsetEndS = tS + 4.5 * halfCircleTimeT
    finalOnsetBegIdxP = int( finalOnsetBegP/DT )
    finalOnsetEndIdxP = int( finalOnsetEndP/DT )
    finalOnsetBegIdxS = int( finalOnsetBegS/DT )
    finalOnsetEndIdxS = int( finalOnsetEndS/DT )
    #print( "finalOnsetBegP, finalOnsetEndP =", finalOnsetBegP, finalOnsetEndP )
    #print( "finalOnsetBegS, finalOnsetEndS =", finalOnsetBegS, finalOnsetEndS )

    templateZ = stNorZ[ finalOnsetBegIdxP:finalOnsetEndIdxP ]
    templateT = stNorT[ finalOnsetBegIdxS:finalOnsetEndIdxS ]
    

    #--寻找振幅值接近于0的直达波起止点，让直达波看着更加合理, 做法是：1. 以上一步的直达波
    # 候选者的起始时间t0和截至时间t1为准，从t0向前截取一个周期长度的数据d0，并从t1向后截取
    # 一个周期的数据d1；2.对d0从t0向前搜索满足振幅小于等于阈值e的时刻t00，对d1从t1向后
    # 搜索满足振幅小于等于阈值e的时刻t11; 将时刻t00-t11的数据段作为最终直达波模板.如果t00
    # 或t11不存在，则将 t0-t11 或 t00-t1 数据段作为最终直达波模板.如果t00和t11均
    # 不存在， 则将原来的t0-t1数据段作为最终直达波模板.
    """
    finalOnset1BegIdxP = int( (finalOnsetBegP-2*halfCircleTimeZ)/DT )
    finalOnset1EndIdxP = int( (finalOnsetEndP+2*halfCircleTimeZ)/DT )
    beforeTemplateZ = stNorZ[ finalOnset1BegIdxP:finalOnsetBegIdxP ] 
    afterTemplateZ  = stNorZ[ finalOnsetEndIdxP:finalOnset1EndIdxP ]
    beforeTemplateZ = beforeTemplateZ / np.max( np.fabs(beforeTemplateZ) )
    afterTemplateZ  = afterTemplateZ / np.max( np.fabs(afterTemplateZ) )
    
    for isam, ival in enumerate( reversed(beforeTemplateZ) ):
        if np.fabs(ival) <= 0.1:
            finalOnsetBegIdxP -= isam
            break
    for isam, ival in enumerate( afterTemplateZ ):
        if np.fabs(ival) <= 0.1:
            finalOnsetEndIdxP += isam
            break
    
    finalOnset1BegIdxS = int( (finalOnsetBegS-2*halfCircleTimeT)/DT )
    finalOnset1EndIdxS = int( (finalOnsetEndS+2*halfCircleTimeT)/DT )
    beforeTemplateT = stNorT[ finalOnset1BegIdxS:finalOnsetBegIdxS ] 
    afterTemplateT  = stNorT[ finalOnsetEndIdxS:finalOnset1EndIdxS ]
    beforeTemplateT = beforeTemplateT / np.max( np.fabs(beforeTemplateT) )
    afterTemplateT  = afterTemplateT / np.max( np.fabs(afterTemplateT) )
    
    for isam, ival in enumerate( reversed(beforeTemplateT) ):
        if np.fabs(ival) <= 0.1:
            finalOnsetBegIdxS -= isam
            break
    for isam, ival in enumerate( afterTemplateT ):
        if np.fabs(ival) <= 0.1:
            finalOnsetEndIdxS += isam
            break 
    
    
    #-- 最终直达波的起跳时刻
    finalOnsetBegP = finalOnsetBegIdxP*DT
    finalOnsetBegS = finalOnsetBegIdxS*DT
    templateZ = stNorZ[ finalOnsetBegIdxP:finalOnsetEndIdxP ]
    templateT = stNorT[ finalOnsetBegIdxS:finalOnsetEndIdxS ]
    """

    
    
    #%%-- plot direct-wave templates
    if verboseFlag == 1:       
        print('\n The selected direct phases:')
        pltDebug.figure(figsize=(5,2))
        t = np.arange( 0, len( templateZ ), 1 )*DT
        pltDebug.axhline( 0, linewidth=0.5, color='gray' )
        pltDebug.plot( t, templateZ/np.max( np.fabs(templateZ) ) )
        pltDebug.title( 'P template (Z)' )
        pltDebug.xlabel('Time (s)', fontsize=12)
        pltDebug.ylabel('Amplitude', fontsize=12)
        pltDebug.margins(0)
        
        
        pltDebug.figure( figsize=(5,2) )
        t = np.arange( 0, len( templateT ), 1 )*DT
        pltDebug.axhline( 0, linewidth=0.5, color='gray' )
        pltDebug.plot( t, templateT/np.max( np.fabs(templateT) ) )
        pltDebug.title( 'S template (T)' )
        pltDebug.xlabel('Time (s)', fontsize=12)
        pltDebug.ylabel('Amplitude', fontsize=12)
        pltDebug.margins(0)
        pltDebug.show()




    #%%##############################################################
    # Step 2: Match-filtering of all possible depth phases by using #
    #         1) phase shifting and 2) match-flitering              #
    #################################################################
    
    phaseShiftStart = -180
    phaseShiftEnd   = 180
    PhaseShittInc   = 10
    numPhase = int( (phaseShiftEnd-phaseShiftStart)/PhaseShittInc )
    #print("Number of phase shift=", numPhase)
    scanTimeBegZ = finalOnsetBegIdxP
    scanTimeEndZ = finalOnsetEndIdxS # 因为Z基本没有S部分或者S部分振幅非常弱, 所以扫描到S即可
    scanTimeBegT = finalOnsetBegIdxS
    scanTimeEndT = len(stNorT)
    print("Z scanning from",scanTimeBegZ*DT, "to", scanTimeEndZ*DT, "sec")
    print("T scanning from",scanTimeBegT*DT, "to", scanTimeEndT*DT, "sec")

    #-- calculate cross-correlation coefficient (CC) on Z component
    count = 0
    temLengZ = len(templateZ)
    traLengZ = len(stNorZ)
    corrLengZ= traLengZ - temLengZ + 1
    corrValZ = np.zeros((numPhase, corrLengZ))
    
    for phaseShift in range(phaseShiftStart, phaseShiftEnd, PhaseShittInc):
        #-- Phase shift using Hilbert transform
        tmp = hilbert( templateZ )
        tmp = np.real( np.abs(tmp) * np.exp((np.angle(tmp) +\
                        (phaseShift)/180.0 * np.pi) * 1j) )      
        #-- cross-corelation
        corrValZ[count] = xcorrssl( scanTimeBegZ, scanTimeEndZ, tmp, stNorZ )
        count += 1

        #-- plotting for paper
        if phaseShift == 60:
            templateZ60 = tmp
        if phaseShift == 120:
            templateZ120 = tmp
        if phaseShift == 170:
            templateZ170 = tmp        

    
    print( 'CC calculating of Z component done!' )


    
    #-- calculate cross-correlation coefficient (CC) on T component        
    count = 0
    temLengT = len(templateT)
    traLengT = len(stNorT)
    corrLengT= traLengT - temLengT + 1
    corrValT = np.zeros((numPhase, corrLengT))
   
    for phaseShift in range(phaseShiftStart, phaseShiftEnd, PhaseShittInc):
        #-- Phase shift using Hilbert transform
        tmp = hilbert( templateT )
        tmp = np.real( np.abs(tmp) * np.exp((np.angle(tmp) +\
                        (phaseShift)/180.0 * np.pi) * 1j) )          
        #-- cross-corelation
        corrValT[count] = xcorrssl( scanTimeBegT, scanTimeEndT, tmp, stNorT )
        count += 1
        
        #-- plotting for paper
        if phaseShift == 60:
            templateT60 = tmp
        if phaseShift == 120:
            templateT120 = tmp
        if phaseShift == 170:
            templateT170 = tmp
            
    print( 'CC calculating of T component done!' )    

 
    #%%-- Get the maximum cross-correlation value of each template
    PickCorrZ = np.amax( corrValZ, axis=0 )
    PickCorrT = np.amax( corrValT, axis=0 )

    
    #%%-- Get the time lag showing the peak value of cross-correlation
    peaksIdxZ, _ = find_peaks(PickCorrZ, height=ccThreshold, distance=int(temLengZ/2))
    peaksIdxT, _ = find_peaks(PickCorrT, height=ccThreshold, distance=int(temLengT/2))
    pickedArrTimeZ = peaksIdxZ * DT
    pickedArrTimeT = peaksIdxT * DT

    

       
    
    
    #%%-- To show the deep-phase waveform corresponding to the peak/troughthat
    # meets the CC threshold, we keep the waveform within a time-window 
    # centering on the peak/trough amplitude.
    largeAmpZ = np.zeros( len(stNorZ) )
    largeAmpT = np.zeros( len(stNorT) )
    
    for i in range (len(stNorZ) ):
        if stNorZ[i] <= leftBoundry1Z or stNorZ[i] >= rightBoundry1Z:
            for j in range( int(temLengZ/1) ):
                if (i-j) >=0 and (i+j) < len(stNorZ):
                    largeAmpZ[ i-j ] = 1.
             
    for i in range (len(stNorT) ):
        if stNorT[i] <= leftBoundry1T or stNorT[i] >= rightBoundry1T:
            for j in range( int(temLengT/1) ):
                if (i-j) >=0 and (i+j) < len(stNorT):
                    largeAmpT[ i-j ] = 1.
                             
        
    #%%-- Select CC for the phase with large amplitude
    peaksCurveZ = np.zeros( len(PickCorrZ) )
    for i in range( len(peaksIdxZ) ):
        peaksCurveZ[ peaksIdxZ[i] ] = PickCorrZ[ peaksIdxZ[i] ] 
    

    peaksCurveT = np.zeros( len(PickCorrT) )        
    for i in range( len(peaksIdxT) ):
        peaksCurveT[ peaksIdxT[i] ] = PickCorrT[ peaksIdxT[i] ]

        
        
    pickedCorrLargeAmpZ = np.zeros( len(PickCorrZ) )      
    for i in range( len(PickCorrZ) ):
        pickedCorrLargeAmpZ[i] = largeAmpZ[i] * peaksCurveZ[i]
    peaksIdxZ, _ = find_peaks(pickedCorrLargeAmpZ, height=ccThreshold, distance=1)
    pickedArrTimeLargeAmpZ = peaksIdxZ * DT
    
    
    pickedCorrLargeAmpT = np.zeros( len(PickCorrT) )
    for i in range( len(PickCorrT) ):
        pickedCorrLargeAmpT[i] = largeAmpT[i] * peaksCurveT[i] 
    peaksIdxT, _ = find_peaks(pickedCorrLargeAmpT, height=ccThreshold, distance=1)
    pickedArrTimeLargeAmpT = peaksIdxT * DT
           


   


    #%%-- Evaluate the quality of data by using a condition:
    # If only the direct phase has CC, then skip the current station.
    if (len(pickedArrTimeLargeAmpZ) < 2) or\
       (len(pickedArrTimeLargeAmpT) < 2):
           print( network+'.'+station+str(': CC failed!') )
           return 0
        

    
    #%%-- calculate arrival time differences between the selected phases and
    # the direct waves
    pickedArrTimeDiffZ = pickedArrTimeLargeAmpZ - pickedArrTimeZ[0]
    pickedArrTimeDiffT = pickedArrTimeLargeAmpT - pickedArrTimeT[0]
    print( 'pickedArrTimeDiffZ =', pickedArrTimeDiffZ )  
    print( 'pickedArrTimeDiffT =', pickedArrTimeDiffT )
    
    #%%-- save the phase-shifting angle of the picked CC
    phaShiftAngZ = []
    phaShiftAngIdxZ = []
    phaShiftAngTimeZ = []
    for i in range( len(peaksIdxZ) ):
        for j in range(numPhase):
            if PickCorrZ[ peaksIdxZ[i] ] == corrValZ[j][ peaksIdxZ[i] ]:
                phaShiftAngZ.append( j*PhaseShittInc+phaseShiftStart )
                phaShiftAngIdxZ.append( peaksIdxZ[i] )
                phaShiftAngTimeZ.append( peaksIdxZ[i]*DT )

    phaShiftAngT = []
    phaShiftAngIdxT = []
    phaShiftAngTimeT = []
    for i in range( len(peaksIdxT) ):
        for j in range(numPhase):
            if PickCorrT[ peaksIdxT[i] ] == corrValT[j][ peaksIdxT[i] ]:
                phaShiftAngT.append( j*PhaseShittInc+phaseShiftStart )
                phaShiftAngIdxT.append( peaksIdxT[i] )
                phaShiftAngTimeT.append( peaksIdxT[i]*DT )
                
    #%%-- plot filtered-out and selected large amplitudes and their CC
    if verboseFlag == 1:
        print( 'The selected large amplitudes:')
        #-- Z
        plt.figure( constrained_layout=True, figsize=(12,2))       
        data  = stNorZ
        largeAmp = data * largeAmpZ
        axisX = np.arange( 0, len(data),  1)*DT
        tZcc  = np.arange( 0, corrLengZ,  1)*DT
        plt.plot( axisX, data, color='lightgray' )
        plt.plot( axisX, largeAmp, label='Large Amp. (Z)' )
        plt.plot( tZcc, peaksCurveZ-2, color='lightgray')
        plt.plot( tZcc, pickedCorrLargeAmpZ-2, label='CC of large Amp.' )
        plt.axhline(ccThreshold-2, linewidth=1, linestyle='--', color='gray')
        plt.ylim(-2, 1.1)
        plt.margins(x=0)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('CC and Amp.', fontsize=12)
        plt.legend(prop={"size":10}, loc='upper right')
        plt.show()
        
        
        #-- T
        plt.figure( constrained_layout=True, figsize=(12,2))     
        data  = stNorT
        largeAmp = data * largeAmpT
        axisX = np.arange( 0, len(data),  1)*DT
        tTcc  = np.arange( 0, corrLengT,  1)*DT
        plt.plot( axisX, data, color='lightgray' )
        plt.plot( axisX, largeAmp, label='Large Amp. (T)' )
        plt.plot( tTcc, peaksCurveT-2, color='lightgray')
        plt.plot( tTcc, pickedCorrLargeAmpT-2, label='CC of large Amp.' )
        plt.axhline(ccThreshold-2, linewidth=1, linestyle='--', color='gray')
        plt.ylim(-2, 1.1)
        plt.margins(x=0)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('CC and Amp.', fontsize=12)
        plt.legend(prop={"size":10}, loc='upper left')
        plt.show()



    #%%#####################################################
    # Step 3: Preliminary determination of the focal depth #
    ########################################################     
    #-- search for focal depth using TauP
    scanDepthMembers = []
    indexCounter = 0
    srcDepthScanInc = 1
    
    for tmp_depth in np.arange( srcDepthScanBeg, srcDepthScanEnd, srcDepthScanInc):
        taupOriginalName = []
        arrivalsTimeDiffP = []
        arrivalsTimeDiffS = []
        srcDepth= tmp_depth  # km
        scanDepthMembers.append( tmp_depth )
        
        if tmp_depth%5 == 0:
            print('Scanning depth = ', tmp_depth, 'km')
        
#        phaList = [ 'S' ]
#        arrivals = subArrivalTimeForward( velModel, srcDepth, recDisInDeg, phaList, recDepth )
#        onsetCalS = arrivals[0].time
#    
#        phaList = [ "P" ]
#        arrivals = subArrivalTimeForward( velModel, srcDepth, recDisInDeg, phaList, recDepth )
#        onsetCalP = arrivals[0].time
        
        phaList = [ 'P', 'pP', 'sP', 'S', 'sS' ]       
        arrivals = subArrivalTimeForward( velModel, srcDepth, recDisInDeg, phaList, recDepth )
        
        #%% calculate time differences of P-wave and S-wave
        for i in range(len(arrivals)):
            if arrivals[i].name == 'P':
                onsetCalP = arrivals[i].time
            elif arrivals[i].name == 'S':
                onsetCalS = arrivals[i].time
                
        for i in range(len(arrivals)):
            if arrivals[i].name != 'P' and arrivals[i].name != 'S':         
                taupOriginalName.append( arrivals[i].name )
                arrivalsTimeDiffP.append( arrivals[i].time - onsetCalP) # P-wave
                arrivalsTimeDiffS.append( arrivals[i].time - onsetCalS) # S-wave
                #because Taup will give several same phase names, here use number to identify each phase
                arrivals[i].name = i
        
        #%%-- matching arrival time differences of observed data and that of
        # synthetic data calculated by using TauP pakadge.
        
        matchedTaupPhaseOrgNameT  = []
        matchedTaupPhaseDigNameT  = []
        matchedTaupTimeDiffT   = []
        matchedPickedTimeT = []
        matchedPickedTimeDiffT = []

        
        matchedTaupPhaseOrgNameZ  = []
        matchedTaupPhaseDigNameZ  = []
        matchedTaupTimeDiffZ   = []
        matchedPickedTimeZ = []
        matchedPickedTimeDiffZ = []

        #%%-- get the number of matched phases (Z component)
        tmpArrivalsTimeDiffP    = arrivalsTimeDiffP
        tmpPickedPhaseTimeDiffZ = list( pickedArrTimeDiffZ )
        tmpPickedPhaseTimeZ     = list( pickedArrTimeLargeAmpZ )
        loopFlag = 1
        while( loopFlag == 1 ):    
            for i in range(len(tmpArrivalsTimeDiffP)):
                for x in range(len(tmpPickedPhaseTimeDiffZ)):
                    if( tmpArrivalsTimeDiffP[i]>=0.0 and tmpPickedPhaseTimeDiffZ[x] >=0.0 ): # match phases except direct p
                        if math.fabs( tmpArrivalsTimeDiffP[i] - tmpPickedPhaseTimeDiffZ[x] ) <= arrTimeDiffTole:                
                            matchedPickedTimeDiffZ.append(tmpPickedPhaseTimeDiffZ[x])
                            matchedTaupTimeDiffZ.append(tmpArrivalsTimeDiffP[i])
                            matchedTaupPhaseDigNameZ.append(arrivals[i].name)
                            matchedTaupPhaseOrgNameZ.append(taupOriginalName[i])
                            matchedPickedTimeZ.append(tmpPickedPhaseTimeZ[x])
                            tmpPickedPhaseTimeDiffZ.remove( tmpPickedPhaseTimeDiffZ[x] )
                            tmpPickedPhaseTimeZ.remove( tmpPickedPhaseTimeZ[x] )
                            break
            loopFlag = 0
            
            
        #%%-- get the number of matched phases (T component)
        tmpArrivalsTimeDiffS    = arrivalsTimeDiffS
        tmpPickedPhaseTimeDiffT = list( pickedArrTimeDiffT )
        tmpPickedPhaseTimeT     = list( pickedArrTimeLargeAmpT )
        loopFlag = 1
        while( loopFlag == 1 ):    
            for i in range(len(tmpArrivalsTimeDiffS)):
                for x in range(len(tmpPickedPhaseTimeDiffT)):
                    if( tmpArrivalsTimeDiffS[i] >=0.0 and tmpPickedPhaseTimeDiffT[x] >=0.0):# match phases except direct s
                        if math.fabs( tmpArrivalsTimeDiffS[i] - tmpPickedPhaseTimeDiffT[x] ) <= arrTimeDiffTole:                
                            matchedPickedTimeDiffT.append(tmpPickedPhaseTimeDiffT[x])
                            matchedTaupTimeDiffT.append(tmpArrivalsTimeDiffS[i])
                            matchedTaupPhaseDigNameT.append(arrivals[i].name)
                            matchedTaupPhaseOrgNameT.append(taupOriginalName[i])
                            matchedPickedTimeT.append(tmpPickedPhaseTimeT[x])
                            tmpPickedPhaseTimeDiffT.remove( tmpPickedPhaseTimeDiffT[x] )
                            tmpPickedPhaseTimeT.remove( tmpPickedPhaseTimeT[x] )
                            break
            loopFlag = 0
            
        
        
            
            
                        

        #%%-- Calculate the total number of matched phases having a unique name
        # (at least one depth-phase matches)
        if( len(matchedTaupTimeDiffZ) >= 1 and\
            len(matchedTaupTimeDiffT) >= 1 ):

            totNumMatPhaOneSta[indexCounter] = len( set(matchedTaupPhaseDigNameZ) ) +\
                len( set(matchedTaupPhaseDigNameT) )   
       
        #-- Calculate the RMS of arrival time differences of preliminary solution 
        # (at least one depth-phase matches)
        if( len(matchedTaupTimeDiffZ) >= 1 ):
            avgArrTimeDiffResEachStationZ = np.sum( np.fabs( np.array(matchedPickedTimeDiffZ) - np.array(matchedTaupTimeDiffZ) ) )/len(matchedTaupTimeDiffZ)
        
        
        if( len(matchedTaupTimeDiffT) >= 1 ):
            avgArrTimeDiffResEachStationT = np.sum( np.fabs( np.array(matchedPickedTimeDiffT) - np.array(matchedTaupTimeDiffT) ) )/len(matchedTaupTimeDiffT)
        
        if( len(matchedTaupTimeDiffZ) >= 1 and\
            len(matchedTaupTimeDiffT) >= 1 ):
            avgArrTimeDiffResOneStaSum[indexCounter] = ( avgArrTimeDiffResEachStationZ +\
                                      avgArrTimeDiffResEachStationT )/2.
                                                              
                                      
        depthCandidateArrGlobalZ[indexCounter].append( matchedPickedTimeZ )
        depthCandidateArrGlobalT[indexCounter].append( matchedPickedTimeT ) 
        depthCandidatePhaOrgNameGlobalT[indexCounter].append( matchedTaupPhaseOrgNameT )
        depthCandidatePhaOrgNameGlobalZ[indexCounter].append( matchedTaupPhaseOrgNameZ )        
        indexCounter += 1
        
        

 
    #%%-- 创建字典保存台站信息
    if np.max(totNumMatPhaOneSta) > 0:
        oneStationInfo = {}
        
        #-- share by Z and T components
        oneStationInfo[ 'ist' ]                 = ist
        oneStationInfo[ 'network' ]             = network
        oneStationInfo[ 'station' ]             = station
        oneStationInfo[ 'location' ]            = location    
        oneStationInfo[ 'Lat(deg)' ]            = stla
        oneStationInfo[ 'Lon(deg)' ]            = stlo
        oneStationInfo[ 'Az(deg)' ]             = azimuth
        oneStationInfo[ 'EpDis(deg)' ]          = recDisInDeg   
        oneStationInfo[ 'ScanDep(km)' ]         = scanDepthMembers
        oneStationInfo[ 'NumMatPha' ]           = totNumMatPhaOneSta
        oneStationInfo[ 'TimeRes(s)' ]          = avgArrTimeDiffResOneStaSum
        oneStationInfo[ 'ratioStd1' ]           = ratioStd1 
        oneStationInfo[ 'DT' ]                  = DT
        
        #-- Z
        oneStationInfo[ 'templateZ0' ]          = templateZ
        oneStationInfo[ 'templateZ60' ]         = templateZ60
        oneStationInfo[ 'templateZ120' ]        = templateZ120
        oneStationInfo[ 'templateZ170' ]        = templateZ170
        oneStationInfo[ 'largeAmpZ' ]           = stNorZ * largeAmpZ
        oneStationInfo[ 'stNorZ' ]              = stNorZ
        oneStationInfo[ 'peaksCurveZ' ]         = peaksCurveZ
        oneStationInfo[ 'pickedCorrLargeAmpZ' ] = pickedCorrLargeAmpZ
        oneStationInfo[ 'peaksIdxZ' ]           = peaksIdxZ
        oneStationInfo[ 'corrLengZ' ]           = corrLengZ
        oneStationInfo[ 'pickedArrTimeLargeAmpZ' ]   = pickedArrTimeLargeAmpZ
        oneStationInfo[ 'phaShiftAngZ' ]             = phaShiftAngZ
        oneStationInfo[ 'phaShiftAngTimeZ' ]         = phaShiftAngTimeZ
        oneStationInfo[ 'finalOnsetBegP' ]           = finalOnsetBegP
        oneStationInfo[ 'leftBoundry1Z' ]            = leftBoundry1Z
        oneStationInfo[ 'rightBoundry1Z' ]           = rightBoundry1Z
        oneStationInfo[ 'histZ' ]                    = histZ
        oneStationInfo[ 'depthCandidateArrGlobalZ' ]        = depthCandidateArrGlobalZ
        oneStationInfo[ 'depthCandidatePhaOrgNameGlobalZ' ] = depthCandidatePhaOrgNameGlobalZ

        
        #-- T
        oneStationInfo[ 'templateT0' ]          = templateT
        oneStationInfo[ 'templateT60' ]         = templateT60
        oneStationInfo[ 'templateT120' ]        = templateT120
        oneStationInfo[ 'templateT170' ]        = templateT170
        oneStationInfo[ 'largeAmpT' ]           = stNorT * largeAmpT
        oneStationInfo[ 'stNorT' ]              = stNorT
        oneStationInfo[ 'peaksCurveT' ]         = peaksCurveT
        oneStationInfo[ 'pickedCorrLargeAmpT' ] = pickedCorrLargeAmpT
        oneStationInfo[ 'peaksIdxT' ]           = peaksIdxT
        oneStationInfo[ 'corrLengT' ]           = corrLengT
        oneStationInfo[ 'pickedArrTimeLargeAmpT' ]   = pickedArrTimeLargeAmpT
        oneStationInfo[ 'phaShiftAngT' ]             = phaShiftAngT
        oneStationInfo[ 'phaShiftAngTimeT' ]         = phaShiftAngTimeT
        oneStationInfo[ 'finalOnsetBegS' ]           = finalOnsetBegS
        oneStationInfo[ 'leftBoundry1T' ]            = leftBoundry1T
        oneStationInfo[ 'rightBoundry1T' ]           = rightBoundry1T
        oneStationInfo[ 'histT' ]                    = histT
        oneStationInfo[ 'depthCandidateArrGlobalT' ]        = depthCandidateArrGlobalT
        oneStationInfo[ 'depthCandidatePhaOrgNameGlobalT' ] = depthCandidatePhaOrgNameGlobalT


        #-- 输出台站信息（数据类型：字典，输出格式：二进制）   
        fileName = str(outfilePath)+'/'+str(network)+'.'+\
                   str(station)+'.'+str(location)+'oneStationInfo.bin'
        outFile  = open( fileName, 'wb') 
        pickle.dump( oneStationInfo, outFile ) 
        outFile.close() 
    
        #-- 输出台站信息（数据类型：字典，输出格式：ASCII）
        fileName = str(outfilePath)+'/'+str(network)+'.'+\
                   str(station)+'.'+str(location)+'oneStationInfo.txt'
        outFile  = open( fileName, 'wt')  
        outFile.write(str(oneStationInfo)) 
        outFile.close()

        print( network+'.'+station+': done!' )
    else:
        print( network+'.'+station+': matches failed!' )












#%%----------------------------------------------------------------------------
#-- 子程序3： 绘制匹配过程的所有图形
#------------------------------------------------------------------------------
def subPlotMatchingProcess( data, idx, FLAG, outfilePath, ccThreshold ):
    
    station   = data[ 'station' ]
    network   = data[ 'network' ]
    DT        = data[ 'DT' ]
    ratioStd1 = data[ 'ratioStd1' ]  
      
    if FLAG == 'Z':
        template0               = data[ 'templateZ0' ]
        template60              = data[ 'templateZ60' ]
        template120             = data[ 'templateZ120' ]
        template170             = data[ 'templateZ170' ]
        largeAmp                = data[ 'largeAmpZ' ]
        stNor                   = data[ 'stNorZ' ]
        peaksCurve              = data[ 'peaksCurveZ' ]
        pickedCorrLargeAmp      = data[ 'pickedCorrLargeAmpZ' ]
        peaks                   = data[ 'peaksIdxZ' ]
        corrLeng                = data[ 'corrLengZ' ]
        pickedArrTimeLargeAmp   = data[ 'pickedArrTimeLargeAmpZ' ]
        phaShiftAng             = data[ 'phaShiftAngZ' ]
        phaShiftAngTime         = data[ 'phaShiftAngTimeZ' ]
        finalOnsetBeg           = data[ 'finalOnsetBegP' ]
        leftBoundry1            = data[ 'leftBoundry1Z' ]
        rightBoundry1           = data[ 'rightBoundry1Z' ]
        hist                    = data[ 'histZ' ]
        matchedPickedTime       = data[ 'depthCandidateArrGlobalZ' ][idx][0]
        matchedTaupPhaseOrgName = data[ 'depthCandidatePhaOrgNameGlobalZ' ][idx][0]
        
    elif FLAG == 'T':
        template0               = data[ 'templateT0' ]
        template60              = data[ 'templateT60' ]
        template120             = data[ 'templateT120' ]
        template170             = data[ 'templateT170' ]
        largeAmp                = data[ 'largeAmpT' ]
        stNor                   = data[ 'stNorT' ]
        peaksCurve              = data[ 'peaksCurveT' ]
        pickedCorrLargeAmp      = data[ 'pickedCorrLargeAmpT' ]
        peaks                   = data[ 'peaksIdxT' ]
        corrLeng                = data[ 'corrLengT' ]
        pickedArrTimeLargeAmp   = data[ 'pickedArrTimeLargeAmpT' ]
        phaShiftAng             = data[ 'phaShiftAngT' ]
        phaShiftAngTime         = data[ 'phaShiftAngTimeT' ]
        finalOnsetBeg           = data[ 'finalOnsetBegS' ]
        leftBoundry1            = data[ 'leftBoundry1T' ]
        rightBoundry1           = data[ 'rightBoundry1T' ]
        hist                    = data[ 'histT' ]
        matchedPickedTime       = data[ 'depthCandidateArrGlobalT' ][idx][0]
        matchedTaupPhaseOrgName = data[ 'depthCandidatePhaOrgNameGlobalT' ][idx][0]
        
    else:
        print( '\n\n\n')
        print( 'subPlotMatchingProcess: FLAG Error!')
        print( '\n\n\n')
        
    #######################################################################
    # plot wavefrom, templates, cc, depth-phase matches, phase-shifted angles 
    #######################################################################    
    # set figure layout
    fig = plt.figure( figsize=(8,4))
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.05)
    gs0 = fig.add_gridspec(1, 2, width_ratios=[8,1] )
    gs00 = gs0[1].subgridspec(6,1)
    gs01 = gs0[0].subgridspec(10,1)
    
    ax0 = fig.add_subplot(gs00[0:2, 0])
    ax1 = fig.add_subplot(gs00[2, 0])
    ax2 = fig.add_subplot(gs00[3, 0])
    ax3 = fig.add_subplot(gs00[4, 0])
    ax4 = fig.add_subplot(gs00[5, 0])
    ax5 = fig.add_subplot(gs01[0:3, 0:])
    ax6 = fig.add_subplot(gs01[3:6, 0:])
    ax7 = fig.add_subplot(gs01[6, 0:])
    ax8 = fig.add_subplot(gs01[7:10, 0:])
    
    # plot data
    t1 = np.arange( 0, len(template0), 1)*DT
    t2 = np.arange( 0, len(stNor),  1)*DT
    tcc= np.arange( 0, corrLeng,   1)*DT
    
    ax0.hist( hist, bins=11, orientation='horizontal')
    ax1.plot( t1, template0, color='orange' )
    ax2.plot( t1, template60 )
    ax3.plot( t1, template120 )
    ax4.plot( t1, template170 )
    ax5.plot( t2, stNor )
    ax6.plot( t2, stNor, color='lightgray' )
    ax6.plot( t2, largeAmp )
    ax7.plot( tcc, peaksCurve, color='lightgray')
    ax7.plot( tcc, pickedCorrLargeAmp )
    
    # add templates
    tTemp = np.arange( 0, len(template0), 1)* DT+finalOnsetBeg
    ax5.plot( tTemp, template0, color='orange' )
    # add texts
    ax0.text( 2,   -0.4, r'$\mu-{0}\sigma$'.format( format(ratioStd1, '.0f') ),
             fontsize=10, rotation=0 )
    ax0.text( 2,    0.3, r'$\mu+{0}\sigma$'.format( format(ratioStd1, '.0f') ),
             fontsize=10, rotation=0 )
    ax1.text( 0.01, 0.45, '0 °', fontsize=10, color='black')
    ax2.text( 0.01, 0.45, '60 °', fontsize=10, color='black')
    ax3.text( 0.01, 0.45, '120 °', fontsize=10, color='black')
    ax4.text( 0.01, 0.45, '170 °', fontsize=10, color='black')
    
    # add matched phase and arrival time
    # find the phases sharing same arrival
    uniFinalArr = list(set(matchedPickedTime))
    for i in range( len( uniFinalArr ) ):
        phaNameForPlot = []
        count = 1
        for j in range( len( matchedTaupPhaseOrgName ) ):
            if matchedPickedTime[j] == uniFinalArr[i]:
                if (len(phaNameForPlot)) > 0:
                    phaNameForPlot.append( "{0}{1}".format( matchedTaupPhaseOrgName[j], count ) )
                    if matchedTaupPhaseOrgName[j] == "S":
                        phaNameForPlot[-1] = "S"
                    count += 1
                else:
                    phaNameForPlot.append( "{0}".format( matchedTaupPhaseOrgName[j] ) )
            
        nameAmpOffset = -0.7
        ax6.axvline( uniFinalArr[i], ymin=0, ymax=0.5, linewidth=1, color='black', linestyle='--')
        for k in range( len( phaNameForPlot ) ):
            if k>=0 and k < len( phaNameForPlot ) -1:
                ax6.text( uniFinalArr[i]-0.65, nameAmpOffset+0.5*k, "{0} + ".format( phaNameForPlot[k] ),
                             fontsize=11, color='black', rotation=90)
            else:
                ax6.text( uniFinalArr[i]-0.65, nameAmpOffset+0.5*k, "{0}".format( phaNameForPlot[k] ),
                             fontsize=11, color='black', rotation=90)
            
    ax7.plot( pickedArrTimeLargeAmp, pickedCorrLargeAmp[ peaks ], "o",
             color='black', markersize=4, zorder=101)
    
    # plot phase-shifting angles
    ax8.plot( tcc, pickedCorrLargeAmp, alpha=0 ) # just use its time axis
    ax8.scatter( phaShiftAngTime, phaShiftAng,
                s=20, color='black', zorder=101 )
    
    #plot CC and phase-shifting angle of matched phases 
    for i in range( len( uniFinalArr ) ):
        for j in range( len( phaShiftAng ) ):
            if uniFinalArr[i] == phaShiftAngTime[j]:
                    ax8.scatter( uniFinalArr[i], phaShiftAng[j],
                                s=20, color='red', zorder=101 )
        for j in range( len( peaks ) ):
            if uniFinalArr[i] == pickedArrTimeLargeAmp[j]:
                    ax7.plot( uniFinalArr[i], pickedCorrLargeAmp[ peaks[j] ], "o",
                             color='red',markersize=4, zorder=101)
    #set grid
    ax8.grid(True, linestyle='--', linewidth=0.5)
    
    # set title
    ax1.tick_params(axis='both', which='major', labelsize=10)
    #
    ax0.set_xscale('log')
    
    #set lim
    ax0.set_xlim(1e0, 1e4)
    ax5.set_xlim( finalOnsetBeg-50, finalOnsetBeg+200 )
    ax6.set_xlim( finalOnsetBeg-50, finalOnsetBeg+200 )
    ax7.set_xlim( finalOnsetBeg-50, finalOnsetBeg+200 )
    ax8.set_xlim( finalOnsetBeg-50, finalOnsetBeg+200 )
    ax0.set_ylim(-1.1, 1.1)
    ax1.set_ylim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    ax4.set_ylim(-1.2, 1.2)
    ax5.set_ylim(-1.2, 1.2)
    ax6.set_ylim(-1.3, 1.1)
    ax7.set_ylim(0, 1.2)
    ax8.set_ylim(-200, 200)
    ax8.set_yticks(np.arange(-180, 190, step=60))
    
    # set xticks
    ax0.xaxis.set_ticks_position('top')
    ax5.xaxis.set_ticks_position('top')
    ax6.xaxis.set_ticks_position('top')
    ax5.yaxis.set_ticks_position('left')
    ax6.yaxis.set_ticks_position('left')
    ax7.yaxis.set_ticks_position('left')
    ax8.yaxis.set_ticks_position('left')
    
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])
    ax4.set_xticks([])
    ax6.set_xticklabels([])
    ax7.set_xticks([])
    ax8.set_xticks([])
    
    ax0.set_yticks([])
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])
    ax4.set_yticks([])        
    
    #remove axis margins
    ax1.margins(x=0)
    ax2.margins(x=0)
    ax3.margins(x=0)
    ax4.margins(x=0)
    ax5.margins(x=0)
    ax6.margins(x=0)
    ax7.margins(x=0)
    ax8.margins(x=0)
    
    # remove some spines
    ax6.spines['bottom'].set_visible(False)
    ax7.spines['top'].set_visible(False)
    
    # set labels
    ax0.xaxis.set_label_position('top')
    ax1.xaxis.set_label_position('top')
    ax5.xaxis.set_label_position('top')
    ax5.yaxis.set_label_position('left')
    ax6.yaxis.set_label_position('left')
    ax7.yaxis.set_label_position('left')
    ax8.yaxis.set_label_position('left')
    
    #%%
    ax5.set_title( network+'.'+station, fontsize=10, loc='left', pad=-10 ) #-- 绘制台站名
    ax0.set_xlabel('Number', fontsize=12, labelpad=6)
    ax5.set_xlabel('Time (s)', fontsize=12, labelpad=8)
    ax5.set_ylabel('Amp.', fontsize=12)
    ax6.set_ylabel('Amp.', fontsize=12)
    ax7.set_ylabel('CC', fontsize=12, labelpad=15)
    ax8.set_ylabel('Shifted (°)', fontsize=12)
    
    # set zero lines
    ax7.axhline(ccThreshold, linewidth=0.8, linestyle='--', color='gray')
    
    # set span
    ax0.axhspan(leftBoundry1, rightBoundry1, facecolor='0.5', alpha=0.3)
    ax5.axhspan(leftBoundry1, rightBoundry1, facecolor='0.5', alpha=0.3)
         
    # plot figure number
    ax5.text( finalOnsetBeg-50+2, 0.7, FLAG, fontsize=12, color='black' )
    
    #show
    plt.tight_layout()
    plt.savefig( str(outfilePath)+'/'+str(network)+'.'+\
                 str(station)+'.'+FLAG+'_Steps1_2.png', dpi=360 )
    plt.close()
    
    
    
    

#%%
      
#%%-- load input parameters
catalogPath, velModel, arrTimeDiffTole, ccThreshold,\
vFrequencyFrom, vFrequencyTo, hFrequencyFrom, hFrequencyTo,\
depthRadiusAbove, depthRadiusBelow, verboseFlag, plotSteps1n2Flag = load_settings()

#------------------------------------------------------------------------------
#  获取 dataPath 的所有目录总个数
#------------------------------------------------------------------------------
dirs = []
dirsAndFiles = sorted( os.listdir(catalogPath), reverse=True )
#-- 筛选出目录名字（因为可能含有非目录的文件存在）
try:
    print( "\n All directories in the path: "+str(catalogPath) )
    for iMem in dirsAndFiles:
        if os.path.isdir( catalogPath+iMem ):
            dirs.append( iMem )
            #print( '\t'+dirs[-1] )
except:
    sys.exit( 'No directories in the path: '+catalogPath ) 
Ndirs = len( dirs )
print( '目录个数 =', Ndirs )


#--查看每个子目录是否为空目录
for idx, idir in enumerate( dirs ):
  if idx >= 0:
    print('\n\n\n')
    print('------------------------------------------------------------------')
    print( 'Event:', idx, ', directory name:', idir )
    dataPath = catalogPath+idir+'/'
    print( 'Path: ', dataPath )
    if os.path.exists( dataPath+'ssnapresults' ):
        ssnapPath = dataPath+'ssnapresults/'
    else:
        print( '\t No ssnapresults!!!' )
        continue
    #读取ssnap定位信息       
    eventfile = fnmatch.filter(os.listdir(ssnapPath), '*cataloghigh*')
    eventfile = pd.read_csv(ssnapPath+eventfile[0], delim_whitespace=True)
    eventfile = list(eventfile)
    epTime = eventfile[0]
    epLat = float(eventfile[1])
    epLon = float(eventfile[2])
    evDp  = float(eventfile[3])
    epMag = float(eventfile[6])
    evPars = epTime, epLat, epLon, evDp, epMag
    
    print( '\n\n\n=============================')
    print( 'evPars   = ', evPars )
    
    #--存放仪器响应文件的目录
    invPath = dataPath+'/inventory/'
    #-- create output file directory
    eventDirectoryName = dataPath
    outfilePath = eventDirectoryName+'DSA_results'
    if not os.path.exists( str(outfilePath) ):
        os.mkdir( str(outfilePath) )
    else:
        print( '\n Warning: '+outfilePath+' already exists!\n')
        shutil.rmtree( str(outfilePath) )
        os.mkdir( str(outfilePath) )
    
    
    #%%-- get the number of waveform files
    #useDataName = eventDirectoryName + '/afterMergeWfData/'
    useDataName = eventDirectoryName
    data = pd.read_csv( str(dataPath)+'0_StationWithHighSNRforDSA_OrgTimeGCMT_DepthGCMT.csv' )   
    filePathE = data['filePathE']
    filePathN = data['filePathN']
    filePathZ = data['filePathZ']
    numSt = len(filePathE)

    srcDepthScanBeg = evDp-depthRadiusAbove
    if srcDepthScanBeg < 5: # 目前发现0-4 km的TauP理论深度震相异常多，这里强制从5 km开始扫描
        srcDepthScanBeg = 5
    srcDepthScanEnd = evDp+depthRadiusBelow
    numScanDepth    = int(srcDepthScanEnd-srcDepthScanBeg)
    
    #%%-- print key information
    print( '\n-------------------- INPUT PARAMETERS -----------------------\n')
    print( 'dataPath           =', dataPath )
    print( 'ssnapPath           =', ssnapPath )
    print( 'eventDirectoryName =', eventDirectoryName )
    print( 'invPath            =', invPath )
    print( 'outfilePath        =', outfilePath )
    print( 'velModel           =', velModel)
    print( 'arrTimeDiffTole    =', arrTimeDiffTole)
    print( 'ccThreshold        =', ccThreshold )
    print( 'vFrequencyFrom     =', vFrequencyFrom )
    print( 'vFrequencyTo       =', vFrequencyTo )
    print( 'hFrequencyFrom     =', hFrequencyFrom )
    print( 'hFrequencyTo       =', hFrequencyTo )
    print( 'depthRadiusAbove   =', depthRadiusAbove )
    print( 'depthRadiusBelow   =', depthRadiusBelow )
    print( 'verboseFlag        =', verboseFlag )
    print( 'plotSteps1n2Flag   =', plotSteps1n2Flag )
    print( 'Number of numSt    =', numSt )
    print( 'srcDepthScanBeg    =', srcDepthScanBeg )
    print( 'srcDepthScanEnd    =', srcDepthScanEnd )
    print( 'numScanDepth       =', numScanDepth )
    print( '\n-------------------------------------------------------------\n')
    
    
    
    
    #---------------------#
    #-- 串行匹配台站数据   --#
    #---------------------#
    begin_timer = timeit.default_timer()
    
    #-- 生成TauP速度模型
    taup_create.build_taup_model(catalogPath+'/velocityModel/'+velModel+'.nd')
    
    #%%-- 读取文件名
    for ist in range( len(filePathE) ):
        if Path(filePathE[ist]).is_file() == True and Path(filePathN[ist]).is_file() == True and Path(filePathZ[ist]).is_file() == True:
          if ist >= 0:
            infileE = open( filePathE[ist] )
            infileN = open( filePathN[ist] )
            infileZ = open( filePathZ[ist] )
            stRawE  = read(infileE.name, debug_headers=True)
            stRawN  = read(infileN.name, debug_headers=True)
            stRawZ  = read(infileZ.name, debug_headers=True)
    
    
        print('\n\n\n========================================' )
        print('Now processing:', ist+1, '/', numSt, 'stations' )
        print( '\t Station:', stRawZ[0].stats.station )
        print( stRawE )
        print( stRawN )
        print( stRawZ )
        if verboseFlag == 1:
            stRawE.plot()
            stRawN.plot()
            stRawZ.plot()
    
        #-- 传输参数到匹配模块
        args = [ ist, epLat, epLon, evDp, epTime,
                 velModel, stRawE, stRawN, stRawZ,
                 numScanDepth, srcDepthScanBeg, srcDepthScanEnd,
                 arrTimeDiffTole, ccThreshold,\
                 vFrequencyFrom, vFrequencyTo,\
                 hFrequencyFrom, hFrequencyTo,\
                 invPath, outfilePath, 0, verboseFlag ]
        subCalSteps1and2OneStation( args )
    
    
    #%%######################################################
    # Step 3: Preliminary solution                          #
    #########################################################
    #%%-- 汇总所有台站的匹配结果
    scanDepthMembers                = np.zeros((numScanDepth))
    totNumMatPhaEachStation         = np.zeros((numSt, numScanDepth))
    avgArrTimeDiffResEachStationSum = np.zeros((numSt, numScanDepth))
    avgArrTimeDiffResEachStationSum.fill(9999) # initial array with a high value
    sumAvgArrTimeDiffResGlobal = np.zeros((numScanDepth))
    sumAvgArrTimeDiffResGlobal.fill(9999) # initial array with a high value
    depthCandidatePhaDigNameGlobalZ = [[] for j in range(numSt)]
    depthCandidatePhaDigNameGlobalR = [[] for j in range(numSt)]
    depthCandidatePhaDigNameGlobalT = [[] for j in range(numSt)]
    
    
    #%%-- 读取台站信息（数据类型：字典，读入格式：二进制）
    resultFiles = fnmatch.filter( sorted(os.listdir(outfilePath)), '*oneStationInfo.bin')
    numFiles = len(resultFiles)
    print( 'Number of stations results:', numFiles )
    print( 'resultFiles:', resultFiles )
    
    for idx, iFile in enumerate( resultFiles ):
        with open( outfilePath+'/'+iFile, 'rb') as reader:
            data = pickle.loads( reader.read() )
            scanDepthMembers                     = data[ 'ScanDep(km)' ]
            totNumMatPhaEachStation[idx]         = data[ 'NumMatPha' ]
            avgArrTimeDiffResEachStationSum[idx] = data[ 'TimeRes(s)' ]
    
    #-- Calculate the total number of matched phases
    sumGlobal = totNumMatPhaEachStation.sum(axis=0)
    
    #-- sum the valid arrival time difference
    for idepth in range( len(scanDepthMembers) ):
        tmpCount = 0
        tmpSum   = 0.
        for ist in range( numSt ):
            if( avgArrTimeDiffResEachStationSum[ist][idepth] < 9999 ):
                tmpSum   += avgArrTimeDiffResEachStationSum[ist][idepth]
                tmpCount += 1
        if tmpCount > 0:
            sumAvgArrTimeDiffResGlobal[idepth] = tmpSum / tmpCount
    
    
    #-- the depth exceeds the given threshold
    thresholdMaxNum = np.max(sumGlobal)*0.9
    
    prelimCandidatesGlobal = []
    for i in range( len(scanDepthMembers) ):
        if sumGlobal[i] >= thresholdMaxNum:
            candidates = sumAvgArrTimeDiffResGlobal[i], scanDepthMembers[i], i
            prelimCandidatesGlobal.append( candidates )
    
    tmp = min(prelimCandidatesGlobal)  # get the candidate with the minimal average rms
    prelimSolution    = tmp[1]  # get the depth with the minimal average rms
    prelimSolutionIdx = tmp[2] # 获取初步解在scanDepthMembers数组里面的索引，以便后续成图
    print('prelimSolution=', prelimSolution, 'km', 'idx =', prelimSolutionIdx )
    
    #-- 输出所有台站的总和结果
    if len(resultFiles) > 0:
        with open( str(outfilePath)+'/'+'0_PreliminarySolution.csv',
                   mode='w', newline='' ) as outFile:
            writer = csv.writer( outFile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow( [ 'ScanDep(km)', 'NumMatPha', 'TimeRes(s)', 'Solution(km)' ] )
            for idx in range( len(scanDepthMembers) ):
                writer.writerow(['{0}'.format( scanDepthMembers[idx] ),
                                 '{0}'.format( sumGlobal[idx] ),
                                 '{0}'.format( sumAvgArrTimeDiffResGlobal[idx] ),
                                 '{0}'.format( prelimSolution ) ])
        outFile.close()
    
    
    
    
    #%%######################################################
    # Plot matching results                                 #
    #########################################################
    #%%-- 读取台站信息（数据类型：字典，读入格式：二进制）
    resultFiles = fnmatch.filter( sorted(os.listdir(outfilePath)),
                                  '*oneStationInfo.bin')
    numFiles = len(resultFiles)
    print( 'Number of stations results:', numFiles )
    print( 'resultFiles:', resultFiles )
    
    for idx, iFile in enumerate( resultFiles ):
        with open( outfilePath+'/'+iFile, 'rb') as reader:
            data = pickle.loads( reader.read() )
    
        #-- 筛选台站用于绘制震相匹配结果，筛选标准：单个台站震相匹配数最大值对应的
        #-  深度位于初步解（prelimSolution）附近1 km范围内
    
    
        #-- 绘制Z分量匹配过程的图像
        PLOTFLAG = 'Z'
        subPlotMatchingProcess( data, prelimSolutionIdx, PLOTFLAG, outfilePath, ccThreshold )
        PLOTFLAG = 'T'
        subPlotMatchingProcess( data, prelimSolutionIdx, PLOTFLAG, outfilePath, ccThreshold )
    
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
    fig = plt.figure( constrained_layout=True, figsize=(8,3))
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
    ax0.plot(t, sumGlobal3090, color="black", linewidth=1.5, alpha=1) 
    ax0.set_xlim( plotXlim )
    ax0.set_ylim( 0, max(sumGlobal3090)*1.3 )
      
    # set labels
    ax0.set_ylabel('Number of matches', fontsize=16)
    ax0.set_xlabel('Depth (km)', fontsize=16)
    ax0.grid(True, linestyle='--', linewidth=0.25)
    ax0.axvline( prelimiSolution3090, lw=1.6, color='black', ls='--') 
    ax0.text( prelimiSolution3090+0.25, max(sumGlobal3090)*1.1, 'DSA',
              color='black', fontsize=14 )
    ax0.tick_params(axis='both', which='major', labelsize=14)
    
    #-- set axis
    ax0.margins(x=0)
    tmp = plotXlim[1] - plotXlim[0]
    if tmp <= 50:
        depthStep = 5
    elif tmp <= 200:
        depthStep = 10
    else:
        depthStep = 20
    ax0.set_xticks( np.arange( plotXlim[0], plotXlim[1], step = depthStep ) )
    
    #-- save    
    plt.tight_layout()
    figNamePng = outfilePath+'/DSA_STEP3.png'
    plt.savefig( figNamePng, dpi=100 )
    plt.show()    
    
    #%% calculate computing time
    end_timer = timeit.default_timer()
    elapsedTime = end_timer - begin_timer
    print('Elapsed time: ', format( elapsedTime, '.1f'),
      'sec = ', format( elapsedTime/60.0, '.1f'), 'min' )


