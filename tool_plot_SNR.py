#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Function: Function: Calculate signal to noise ratio of station 131A
    
"""

from obspy.taup import TauPyModel, taup_create
import matplotlib.pyplot as plt
from obspy.geodetics.base import kilometer2degrees
from obspy.core import UTCDateTime
import math
import numpy as np
from obspy import read, read_inventory
from obspy.geodetics import gps2dist_azimuth
import pandas as pd
import os, fnmatch, sys
import timeit
import shutil
import csv
import gc
plt.rcParams["font.family"] = "Times New Roman"


#%%-- subroutine: load input parameters from 'SETTINGS.txt'
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
        SETTINGS = pd.read_csv('./SETTINGS.txt',
                               delim_whitespace=True, index_col='PARAMETER')
        
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
        
        return par1, par2, par3, par4, par5, par6, par7, par8, par9, par10, par11, par12     
        gc.collect()  
    except:
        sys.exit("Errors in 'SETTINGS.txt' !\n")
        


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
    gc.collect()




def subCalSteps1and2OneStation( args ):
    ist              = args[0]
    iev              = args[1]
    velModel         = args[2]
    selectedfileE    = args[3]
    selectedfileN    = args[4]
    selectedfileZ    = args[5]
    stRawE           = args[6]
    stRawN           = args[7]
    stRawZ           = args[8]
#    numScanDepth     = args[9]
#    srcDepthScanBeg  = args[10]
#    srcDepthScanEnd  = args[11]
#    arrTimeDiffTole  = args[12]
#    ccThreshold      = args[13]
    vFrequencyFrom   = args[14]
    vFrequencyTo     = args[15]
    hFrequencyFrom   = args[16]
    hFrequencyTo     = args[17]
    invPath          = args[18]
    outfilePath      = args[19]
    figListName      = args[20]
    stationListHighSN= args[21]
    epLat = args[22]
    epLon = args[23]
    evDp =  args[24]   

    #%%-- Get some key infomation
    evla  = iev.origins[0].latitude
    evlo  = iev.origins[0].longitude
    evdp  = iev.origins[0].depth/1000.0 

    network = stRawZ[0].stats.network
    station = stRawZ[0].stats.station
    wavestart = UTCDateTime(stRawZ[0].stats.starttime)
    originTime = iev.origins[0].time
    print('originTime = ',originTime)
    
    
    network  = stRawZ[0].stats.network
    station  = stRawZ[0].stats.station
    location = stRawZ[0].stats.location
    legendName = network+'.'+station+'.'+location
    
    
    
    try:
        inv = read_inventory( "{0}{1}.{2}.xml".format( invPath, network, station ) )
    except:
        print( 'Can not find {0}{1}.{2}.xml'.format( invPath, network, station ) )
        gc.collect()
        return(0)
    
    net = inv[0]
    sta = net[0]
    stla = sta.latitude
    stlo = sta.longitude
    elev = sta.elevation
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
    # stWantedE0[0].plot()
    # stWantedN0[0].plot()
    # stWantedZ0[0].plot()
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
        stR0 = stWantedE0.copy()
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
        
        #-- do ratation: NE to RT using back-azimuth angle        
        stNE = stWantedN0 + stWantedE0
        stNE.rotate( method='NE->RT', back_azimuth=baz )
        stR0 = stNE.select(component="R")
        stT0 = stNE.select(component="T")        
        stZ0 = stWantedZ0.copy()

        print( 'Rotation done!\n')
#        stZ0.plot()
#        stR0.plot()
#        stT0.plot()        
        
        
        #-- taper before filtering
#        stZ0[0] = stZ0[0].taper(max_percentage=0.1, side='left')
#        stR0[0] = stR0[0].taper(max_percentage=0.1, side='left')
#        stT0[0] = stT0[0].taper(max_percentage=0.1, side='left')

    except:
        print (network+'.'+station+str(': rotation failed!'))
        stR0 = stWantedE0.copy()
        stT0 = stWantedN0.copy()        
        stZ0 = stWantedZ0.copy()
    
    
    
    #%%-- frequency filtering
    stZ0[0] = stZ0[0].filter('bandpass', freqmin=vFrequencyFrom, freqmax=vFrequencyTo,
                                         corners=4, zerophase=False)
    stR0[0] = stR0[0].filter('bandpass', freqmin=vFrequencyFrom, freqmax=vFrequencyTo,
                                         corners=4, zerophase=False)
    stT0[0] = stT0[0].filter('bandpass', freqmin=hFrequencyFrom, freqmax=hFrequencyTo,
                                         corners=4, zerophase=False)

    print( 'Bandpass filtering done!\n')
    
    resamRate = 10 # Hz
    stZ0.interpolate( sampling_rate=resamRate, method="lanczos",
                      a=12, window="blackman" )
    stR0.interpolate( sampling_rate=resamRate, method="lanczos",
                      a=12, window="blackman" )
    stT0.interpolate( sampling_rate=resamRate, method="lanczos",
                      a=12, window="blackman" )
    DT = stZ0[0].stats.delta    
    print( 'Reampling done!\n')

    stRawE0 = stRawE.copy()
    stRawN0 = stRawN.copy()
    stRawZ0 = stRawZ.copy()
    stRawE0.interpolate( sampling_rate=resamRate, method="lanczos",
                      a=12, window="blackman" )
    stRawN0.interpolate( sampling_rate=resamRate, method="lanczos",
                      a=12, window="blackman" )
    stRawZ0.interpolate( sampling_rate=resamRate, method="lanczos",
                      a=12, window="blackman" )    


    #%%-- check waveform
    if verboseFlag == 1:
        print('\n Check Z, R, and T waveforms: \n')
        stZ0.plot()
        stR0.plot()
        stT0.plot()
    
    Z0CP1 = stZ0.copy()  
    Z0CP2 = stZ0.copy() 
    R0CP1 = stR0.copy()  
    R0CP2 = stR0.copy() 
    T0CP1 = stT0.copy()  
    T0CP2 = stT0.copy()    
    
    
    
    
    #%%-- Waveform scanning window used for DSA, choosing 2 mins before
    # theoretical onset of the direct P and 10 mins after S
    try:

        refDepth = evdp # focal depth (adopt from Catalog) for calculting onset time of direct wave (km)        
        phaList = [ "P", 'pP', 'sP' ]
        arrivalsP = subArrivalTimeForward( velModel, refDepth, recDisInDeg, phaList, recDepth )
        calOnsetP   = arrivalsP[0].time
        

        phaList = [ 'S', 'sS'  ]
        arrivalsS = subArrivalTimeForward( velModel, refDepth, recDisInDeg, phaList, recDepth )
        calOnsetS = arrivalsS[0].time
        

        phaList = [ 'ScS'  ]
        arrivalsScS = subArrivalTimeForward( velModel, refDepth, recDisInDeg, phaList, recDepth )
        calOnsetScS = arrivalsScS[0].time
        
    except:
        gc.collect()
        sys.exit( network+'.'+station+str(': theoretical onset time failed!') )


    print( arrivalsP+arrivalsS )

    timeBeforeP = 2*60 # sec
    timeAfterS  = 2*60 # sec
          
    #%%-- Extract wavefroms according to the wanted time window
    try:
        timeWinEnd = calOnsetScS+timeAfterS
        stR1 = stR0.trim( originTime, originTime+timeWinEnd )
        stT1 = stT0.trim( originTime, originTime+timeWinEnd )
        stZ1 = stZ0.trim( originTime, originTime+timeWinEnd )
        #normalization
        dataR0 = stR1[0].data / max( np.fabs(stR1[0].data))
        dataT0 = stT1[0].data / max( np.fabs(stT1[0].data))
        dataZ0 = stZ1[0].data / max( np.fabs(stZ1[0].data))
        
        #-- 对原始数据也进行重采样，用于成图比较
        stRawE1 = stRawE0.trim( originTime, originTime+timeWinEnd )
        stRawN1 = stRawN0.trim( originTime, originTime+timeWinEnd )
        stRawZ1 = stRawZ0.trim( originTime, originTime+timeWinEnd )
        #normalization
        dataRawE = stRawE1[0].data / max( np.fabs(stRawE1[0].data))
        dataRawN = stRawN1[0].data / max( np.fabs(stRawN1[0].data))
        dataRawZ = stRawZ1[0].data / max( np.fabs(stRawZ1[0].data))
        
    except:
        print( network+'.'+station+str(': cut waveforms failed!') )
        gc.collect()
        return 0
    
    try:
        # Signal to noise ratio (S/N) is defined as TW2/TW1, where TW1 is the 
        # recording from the time window between 40s to 10s before the P 
        # arrival (Marked purple in Figure 3B), and TW2 is defined as the 
        # recording from the 30s width window directly after the P arrival 
        # (Marked green in Figure 3B). The 10s time-interval before the P 
        # arrival is to avoid the direct wave’s involvement in TW1 due to velocity model errors.
        timeBeforeP = 30 # sec
        timeAfterP  = 30
        timeBeforeS = 30
        timeAfterS  = 30 # sec
              
        
        #%%-- Extract wavefroms according to the wanted time window
        noiseTimeWinBegZR  = calOnsetP-timeBeforeP-10
        noiseTimeWinEndZR  = calOnsetP-10
        signalTimeWinBegZR = noiseTimeWinEndZR+10
        signalTimeWinEndZR = signalTimeWinBegZR+timeAfterP
    
        noiseTimeWinBegT  = calOnsetS-timeBeforeS-10
        noiseTimeWinEndT  = calOnsetS-10
        signalTimeWinBegT = noiseTimeWinEndT+10
        signalTimeWinEndT = signalTimeWinBegT+timeAfterS
    
    
        noiseZ = Z0CP1.trim( originTime+noiseTimeWinBegZR, originTime+noiseTimeWinEndZR )
        signalZ= Z0CP2.trim( originTime+signalTimeWinBegZR, originTime+signalTimeWinEndZR )        
    
        noiseR = R0CP1.trim( originTime+noiseTimeWinBegZR, originTime+noiseTimeWinEndZR )
        signalR= R0CP2.trim( originTime+signalTimeWinBegZR, originTime+signalTimeWinEndZR )
        
    
        noiseT = T0CP1.trim( originTime+noiseTimeWinBegT, originTime+noiseTimeWinEndT )
        signalT= T0CP2.trim( originTime+signalTimeWinBegT, originTime+signalTimeWinEndT )
        
        
        
        #calculate signal-to-noise ratio
        noiseZ  = np.max( np.fabs( noiseZ[0].data ) )
        signalZ = np.max( np.fabs( signalZ[0].data ) )
        noiseR  = np.max( np.fabs( noiseR[0].data ) )
        signalR = np.max( np.fabs( signalR[0].data ) )
        noiseT  = np.max( np.fabs( noiseT[0].data ) )
        signalT = np.max( np.fabs( signalT[0].data ) )
         
        snrZ = signalZ/noiseZ
        snrR = signalR/noiseR
        snrT = signalT/noiseT
        
        
        print('\n======================')
        print( 'snrZ =', snrZ )
        print( 'snrR =', snrR )
        print( 'snrT =', snrT )
        print('======================\n')
    except:
        print( network+'.'+station+str(': SN calculation failed!') )
        gc.collect()
        return 0    

        
    #%%--
    if ist >= 0 :
        # set figure layout
        fig = plt.figure( figsize=(10,5))
        fig.subplots_adjust(hspace=0.1)
        fig.subplots_adjust(wspace=0.0)
        gs0 = fig.add_gridspec(3, 1 )
        
        ax0 = fig.add_subplot(gs0[0, 0])
        ax1 = fig.add_subplot(gs0[1, 0])
        ax2 = fig.add_subplot(gs0[2, 0])
            
        # plot data
        tZRT = np.arange( 0, len(dataZ0), 1)*DT
        ax0.plot( tZRT, dataZ0, lw=0.5, c='black' )
        ax0.plot( calOnsetP, -0., 'o', color='red',   mfc='none', markersize=5 )
        
        tZRT = np.arange( 0, len(dataR0), 1)*DT
        ax1.plot( tZRT, dataR0, label='R', lw=0.5, c='black' )
        ax1.plot( calOnsetP, -0., 'o', color='red',   mfc='none', markersize=5 )
     
        tZRT = np.arange( 0, len(dataT0), 1)*DT
        ax2.plot( tZRT, dataT0, lw=0.5, c='black' )
        ax2.plot( calOnsetS, -0., 'o', color='red', mfc='none', markersize=5 )
    
        ax0.axvspan( noiseTimeWinBegZR, noiseTimeWinEndZR, alpha=0.1, color='blue')   
        ax0.axvspan( signalTimeWinBegZR, signalTimeWinEndZR, alpha=0.1, color='lime') 
        ax1.axvspan( noiseTimeWinBegZR, noiseTimeWinEndZR, alpha=0.1, color='blue')   
        ax1.axvspan( signalTimeWinBegZR, signalTimeWinEndZR, alpha=0.1, color='lime') 
        ax2.axvspan( noiseTimeWinBegT, noiseTimeWinEndT, alpha=0.1, color='blue')   
        ax2.axvspan( signalTimeWinBegT, signalTimeWinEndT, alpha=0.1, color='lime') 
        
        ax0.margins(0)
        ax1.margins(0)
        ax2.margins(0)
        ax0.tick_params(axis='both', which='major', labelsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax0.xaxis.set_ticks_position('top')
        ax0.xaxis.set_label_position('top')
            
        ax0.text( noiseTimeWinEndZR-10, -1.4, 'S/N: '+str(format( snrZ, '.1f')),
                  fontsize=14, color='black' )
        ax1.text( noiseTimeWinEndZR-10, -1.4, 'S/N: '+str(format( snrR, '.1f')),
                  fontsize=14, color='black' )
        ax2.text( noiseTimeWinEndT-10, -1.4, 'S/N: '+str(format( snrT, '.1f')),
                  fontsize=14, color='black' )
        
        ax0.text(noiseTimeWinBegT+50, 0.4, legendName+'.BHZ\n'+'Distance: '+
                 str( format( recDisInDeg, '.1f' ) )+r'$\degree$', fontsize=14 )
        ax1.text(noiseTimeWinBegT+50, 0.4, legendName+'.BHR\n'+'Distance: '+
                 str( format( recDisInDeg, '.1f' ) )+r'$\degree$', fontsize=14 )
        ax2.text(noiseTimeWinBegZR-40,  0.4, legendName+'.BHT\n'+'Distance: '+
                 str( format( recDisInDeg, '.1f' ) )+r'$\degree$', fontsize=14 )
        
        ax1.set_xticks([])
        ax0.set_xlim( noiseTimeWinBegZR-50, noiseTimeWinBegT+150 )
        ax1.set_xlim( noiseTimeWinBegZR-50, noiseTimeWinBegT+150 )
        ax2.set_xlim( noiseTimeWinBegZR-50, noiseTimeWinBegT+150 )
        ax0.set_ylim( -1.5, 1.5 )
        ax1.set_ylim( -1.5, 1.5 )
        ax2.set_ylim( -1.5, 1.5 )
        ax0.set_xlabel( 'Time (s)', fontsize=14 )
        ax2.set_xlabel( 'Time (s)', fontsize=14 )
        ax0.set_ylabel( 'Amp.', fontsize=14 )
        ax1.set_ylabel( 'Amp.', fontsize=14 )
        ax2.set_ylabel( 'Amp', fontsize=14 )

        plt.tight_layout()
        figName = "{0}.{1}_SNR.pdf".format( network, station )
        figName1 = "{0}.{1}_SNR.png".format( network, station )
        plt.savefig( str(outfilePath)+'/'+str(figName), dpi=200 )
        plt.savefig( str(outfilePath)+'/'+str(figName1), dpi=200 )
        plt.show()
        plt.close()



#%%-- Main
if __name__ == "__main__":
     
    #%%-- load input parameters
    catalogPath, velModel, arrTimeDiffTole, ccThreshold,\
    vFrequencyFrom, vFrequencyTo, hFrequencyFrom, hFrequencyTo,\
    depthRadiusAbove, depthRadiusBelow, verboseFlag, plotSteps1n2Flag = load_settings()

    from obspy.core.event import read_events
    try:
        catalog = read_events( catalogPath+'catalog.xml', format="QUAKEML"  )
        print( catalog )
    except:
        sys.exit( "No 'catalog.xml' in the catalog path: " + catalogPath ) 

    for idx, iev in enumerate( catalog ):
        epTime = iev.origins[0].time
        epLat  = iev.origins[0].latitude
        epLon  = iev.origins[0].longitude
        evDp   = iev.origins[0].depth /1000.0
        epMag  = iev.magnitudes[0].mag
        print( '\n\n\n=============================')
        print( 'Number of events =', len(catalog) )
        print( 'Now processing event: ', idx+1, '/', len(catalog) )
        print( 'epTime  = ', epTime )
        print( 'epLat   = ', epLat )
        print( 'epLon   = ', epLon )
        print( 'evDp    = ', evDp )
        print( 'epMag   = ', epMag )

        date   = iev.origins[0].time.datetime.date()
        hour   = iev.origins[0].time.hour
        minute = iev.origins[0].time.minute
        second = iev.origins[0].time.second
        eventDirectoryName = str(date)+'-'+str(hour)+'-'+str(minute)+'-'+str(second)
        print( 'eventDirectoryName', eventDirectoryName)
        
        if not os.path.exists( catalogPath+str(eventDirectoryName) ):
            print('Warning: The directory < '+catalogPath+str(eventDirectoryName)+\
                  ' >  not found!' )
            continue

        wfPath = catalogPath+str(eventDirectoryName)+'/'
        invPath = catalogPath+str(eventDirectoryName)+'/'+'inventory'+'/'
        velPath = catalogPath+'velocityModel'+'/'
    
        wfFiles = fnmatch.filter( sorted(os.listdir(wfPath)), 'TA.131A*.mseed')
        numMseed  = len(wfFiles)
        if numMseed == 0:
            print( "--- No station waveform for this event! ---" )
            continue

        selectedfileE = []
        selectedfileN = []
        selectedfileZ = []
        for ist in range( numMseed-2 ):
            try:
                wfFile1 = str( wfPath )+str( wfFiles[ist+0] )
                wfFile2 = str( wfPath )+str( wfFiles[ist+1] )
                wfFile3 = str( wfPath )+str( wfFiles[ist+2] )
                
                #--          
                infileE = open( wfFile1 )
                infileN = open( wfFile2 )
                infileZ = open( wfFile3 )
                stRawE = read(infileE.name, debug_headers=True)
                stRawN = read(infileN.name, debug_headers=True)
                stRawZ = read(infileZ.name, debug_headers=True)            
    
                if( stRawE[0].stats.channel == 'BHE' and
                    stRawN[0].stats.channel == 'BHN' and
                    stRawZ[0].stats.channel == 'BHZ' and
                    stRawE[0].stats.station == stRawN[0].stats.station and
                    stRawE[0].stats.station == stRawZ[0].stats.station and
                    stRawN[0].stats.station == stRawZ[0].stats.station ):
                    selectedfileE.append( wfFile1 )
                    selectedfileN.append( wfFile2 )
                    selectedfileZ.append( wfFile3 )
                    print( 'Found:', stRawE[0].stats.network+'.'+stRawE[0].stats.station )
            except:
                print( "Station error!" )
                continue
            
        numSt = len(selectedfileZ)
        print( '\n Total selected stations:', numSt )
        

        
        #%%-- create output file directory
        outfilePath = str('./outputFigures')
        if not os.path.exists(outfilePath):
            os.mkdir(outfilePath)
        else:
            print( '\n Warning: outfilePath already exists!\n')
            
        #%%--                       
        if evDp < 0 or evDp > 5000:
            sys.exit("\n Current event's depth error, stop!\n")
        if (evDp-depthRadiusAbove) < 0:
            print( "\n Warning: 'depthRadiusAbove' should be <=",
                  math.floor(evDp), ', now set it to be 5 \n' )
            depthRadiusAboveNew = math.floor(evDp)-5 #-- 0-4 km will give unstable results
        else:
            depthRadiusAboveNew = depthRadiusAbove

        srcDepthScanBeg = math.floor( evDp-depthRadiusAboveNew )
        srcDepthScanEnd = math.floor( evDp+depthRadiusBelow )
        numScanDepth    = int(srcDepthScanEnd-srcDepthScanBeg)
     
        stationListHighSN = str(wfPath)+'/'+'0_StationWithHighSNR.csv'
        
        #%%-- print key information
        print( '\n-------------------- INPUT PARAMETERS -----------------------\n')
        print( 'catalogPath        =', catalogPath )
        print( 'wfPath             =', wfPath )
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
        print( 'Number of mseed    =', numMseed )
        print( 'depthRadiusAboveNew=', depthRadiusAboveNew )
        print( 'srcDepthScanBeg    =', srcDepthScanBeg )
        print( 'srcDepthScanEnd    =', srcDepthScanEnd )
        print( 'numScanDepth       =', numScanDepth )
        print( '\n-------------------------------------------------------------\n')    


        begin_timer = timeit.default_timer()
        taup_create.build_taup_model( str( velPath )+str(velModel)+'.nd', str( velPath ) )
        
        for ist in range( numSt ):
            infileE = open( selectedfileE[ist] )
            infileN = open( selectedfileN[ist] )
            infileZ = open( selectedfileZ[ist] )
            stRawE = read(infileE.name, debug_headers=True)
            stRawN = read(infileN.name, debug_headers=True)
            stRawZ = read(infileZ.name, debug_headers=True)
            print('\n\n\n========================================' )
            print('Event: ', idx+1, '/', len(catalog) )
            print('Now processing:', ist+1, '/', numSt, 'stations' )

            cpyVelModel = str( velModel )+'-'+str( ist-ist )
            shutil.copy( str( velPath )+str(velModel)+'.npz',
                         str( velPath )+str(cpyVelModel)+'.npz' ) 
            cpyVelModel = str( velModel )
            args = [ ist, iev, cpyVelModel,
                     selectedfileE, selectedfileN, selectedfileZ,
                     stRawE, stRawN, stRawZ,
                     numScanDepth, srcDepthScanBeg, srcDepthScanEnd,
                     arrTimeDiffTole, ccThreshold,\
                     vFrequencyFrom, vFrequencyTo,\
                     hFrequencyFrom, hFrequencyTo,\
                     invPath, outfilePath, catalogPath, stationListHighSN, 
                     epLat, epLon, evDp]
            subCalSteps1and2OneStation( args )