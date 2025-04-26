#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Function: Calculate signal to noise ratio

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
from PyPDF2 import PdfFileMerger

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
        #-- free  memmory
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

    #-- free  memmory
    #del arrivals
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
    epTime = args[25]
    #%%-- Allocate memory
#    totNumMatPhaOneSta = np.zeros(( numScanDepth))
#    avgArrTimeDiffResOneStaSum = np.zeros(( numScanDepth))
#    avgArrTimeDiffResOneStaSum.fill(9999) # initial array with a high value
#    depCanPhaDigNameOneStaZ = [[] for i in range(numScanDepth)]
#    depCanPhaDigNameOneStaR = [[] for i in range(numScanDepth)]
#    depCanPhaDigNameOneStaT = [[] for i in range(numScanDepth)]
    
    
    #%%########################################################
    # Step 1: Automatic generation of synthetic waveforms for #
    #         all possible depth phases                       #
    ###########################################################
    #%%-- Get some key infomation   
    starttime = epTime
    network = stRawZ[0].stats.network
    station = stRawZ[0].stats.station
#    location= stRawZ[0].stats.location
    wavestart = UTCDateTime(stRawZ[0].stats.starttime)

    
    print('starttime = ',starttime)
    try:
        inv = read_inventory( "{0}{1}.{2}.xml".format( invPath, network, station ) )
    except:
        print( 'Can not find {0}{1}.{2}.xml'.format( invPath, network, station ) )
        #-- free  memmory
        gc.collect()
        return(0)

    net = inv[0]
    sta = net[0]
    stla = sta.latitude
    stlo = sta.longitude
    elev = sta.elevation

    epi_dist, azimuth, baz = gps2dist_azimuth(epLat, epLon, stla, stlo )
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


    #%%-- check waveform
    if verboseFlag == 1:
        print('\n Check Z, R, and T waveforms: \n')
        stZ0.plot()
        stR0.plot()
        stT0.plot()
    
  
    #-- 
    Z0CP1 = stZ0.copy()  
    Z0CP2 = stZ0.copy() 
    R0CP1 = stR0.copy()  
    R0CP2 = stR0.copy() 
    T0CP1 = stT0.copy()  
    T0CP2 = stT0.copy()    
    
    
    
    
    
    #%%-- Waveform scanning window used for DSA, choosing 2 mins before
    # theoretical onset of the direct P and 10 mins after S
    try:

        refDepth = evDp # focal depth (adopt from Catalog) for calculting onset time of direct wave (km)        
        phaList = [ "P", 'pP', 'sP', 'PcP' ]
        arrivalsP = subArrivalTimeForward( velModel, refDepth, recDisInDeg, phaList, recDepth )
        calOnsetP   = arrivalsP[0].time
        

        phaList = [ 'S', 'sS',  'ScS'  ]
        arrivalsS = subArrivalTimeForward( velModel, refDepth, recDisInDeg, phaList, recDepth )
        calOnsetS = arrivalsS[0].time
        

        phaList = [ 'ScS'  ]
        arrivalsScS = subArrivalTimeForward( velModel, refDepth, recDisInDeg, phaList, recDepth )
        calOnsetScS = arrivalsScS[0].time
        
    except:
        
        gc.collect()
        sys.exit( network+'.'+station+str(': theoretical onset time failed!') )


    print( arrivalsP+arrivalsS )

         
    #-- 选择直达P前两分钟和ScS波后十分钟的数据段进行扫描, 用ScS的原因是可以去除面波干扰
    timeBeforeP = 2*60 # sec
    timeAfterS  = 2*60 # sec
          
    #%%-- Extract wavefroms according to the wanted time window
    try:
        timeWinBeg = calOnsetP-timeBeforeP
        timeWinEnd = calOnsetScS+timeAfterS
        stR1 = stR0.trim( starttime+timeWinBeg, starttime+timeWinEnd )
        stT1 = stT0.trim( starttime+timeWinBeg, starttime+timeWinEnd )
        stZ1 = stZ0.trim( starttime+timeWinBeg, starttime+timeWinEnd )
        #normalization
        dataZ0 = stZ1[0].data / max( np.fabs(stZ1[0].data))
        dataR0 = stR1[0].data / max( np.fabs(stR1[0].data))
        dataT0 = stT1[0].data / max( np.fabs(stT1[0].data))
        
    except:
        print( network+'.'+station+str(': cut waveforms failed!') )

        gc.collect()
        return 0
 
    

    #%%########################################################################
    #-- calculation s/n
    ###########################################################################
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
        signalTimeWinBegZR = calOnsetP
        signalTimeWinEndZR = calOnsetP+timeAfterP
    
        noiseTimeWinBegT  = calOnsetS-timeBeforeS-10
        noiseTimeWinEndT  = calOnsetS-10
        signalTimeWinBegT = calOnsetS
        signalTimeWinEndT = calOnsetS+timeAfterS
    
    
        noiseZ = Z0CP1.trim( starttime+noiseTimeWinBegZR, starttime+noiseTimeWinEndZR )
        signalZ= Z0CP2.trim( starttime+signalTimeWinBegZR, starttime+signalTimeWinEndZR )        
    
        noiseR = R0CP1.trim( starttime+noiseTimeWinBegZR, starttime+noiseTimeWinEndZR )
        signalR= R0CP2.trim( starttime+signalTimeWinBegZR, starttime+signalTimeWinEndZR )
        
    
        noiseT = T0CP1.trim( starttime+noiseTimeWinBegT, starttime+noiseTimeWinEndT )
        signalT= T0CP2.trim( starttime+signalTimeWinBegT, starttime+signalTimeWinEndT )
        
        
        
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



    if ist < 10:
        # set figure layout
        fig = plt.figure( constrained_layout=True, figsize=(10,5))
        fig.subplots_adjust(hspace=0.1)
        fig.subplots_adjust(wspace=0.0)
        gs0 = fig.add_gridspec(3, 1 )
        
        ax0 = fig.add_subplot(gs0[0, 0])
        ax1 = fig.add_subplot(gs0[1, 0])
        ax2 = fig.add_subplot(gs0[2, 0])
            
        # plot data
        tZRT = np.arange( 0, len(dataZ0), 1)*DT
        ax0.plot( tZRT, dataZ0, label='Z' )
        ax0.plot( calOnsetP-timeWinBeg, -0., 'o', color='red',   mfc='none', markersize=10 )
        ax0.plot( calOnsetS-timeWinBeg, -0., 'o', color='black', mfc='none', markersize=10 )
        
        tZRT = np.arange( 0, len(dataR0), 1)*DT
        ax1.plot( tZRT, dataR0, label='R' )
        ax1.plot( calOnsetP-timeWinBeg, -0., 'o', color='red',   mfc='none', markersize=10 )
        ax1.plot( calOnsetS-timeWinBeg, -0., 'o', color='black', mfc='none', markersize=10 )
     
        tZRT = np.arange( 0, len(dataT0), 1)*DT
        ax2.plot( tZRT, dataT0, label='T' )
        ax2.plot( calOnsetS-timeWinBeg, -0., 'o', color='black', mfc='none', markersize=10 )
    
        
        for i in range (len(arrivalsP)):
            ax0.axvline( arrivalsP[i].time-timeWinBeg, linewidth=1, color='black', linestyle='--')
            ax0.text( arrivalsP[i].time-timeWinBeg+2, 0.5, arrivalsP[i].name,
                      fontsize=12, color='black', rotation=90)
            ax1.axvline( arrivalsP[i].time-timeWinBeg, linewidth=1, color='black', linestyle='--')
            ax1.text( arrivalsP[i].time-timeWinBeg+2, 0.5, arrivalsP[i].name,
                      fontsize=12, color='black', rotation=90)
            
        for i in range (len(arrivalsS)):
            ax2.axvline( arrivalsS[i].time-timeWinBeg, linewidth=1, color='black', linestyle='--')
            ax2.text( arrivalsS[i].time-timeWinBeg+2, 0.5, arrivalsS[i].name,
                      fontsize=12, color='black', rotation=90)
      
    #    ax0.axvspan( 0, timeBeforeP, alpha=0.1, color='black')   
    #    ax0.axvspan( timeBeforeP, timeBeforeP+timeAfterP, alpha=0.1, color='red')      
        
        ax0.margins(0)
        ax1.margins(0)
        ax2.margins(0)
        ax0.tick_params(axis='both', which='major', labelsize=10)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        ax0.xaxis.set_ticks_position('top')
        ax0.xaxis.set_label_position('top')
        ax0.text( 0+2, 0.6, "Z"+'  '+'SN:'+str(format( snrZ, '.2f')),
                  fontsize=12, color='black' )
        ax1.text( 0+2, 0.6, "R"+'  '+'SN:'+str(format( snrR, '.2f')),
                  fontsize=12, color='black' )
        ax2.text( 350+2, 0.6, "T"+'  '+'SN:'+str(format( snrT, '.2f')),
                  fontsize=12, color='black' )
        ax1.set_xticks([])
        ax0.set_xlim( 0, 300 )
        ax1.set_xlim( 0, 300 )
        ax2.set_xlim( 350, len(dataT0)*DT )
        ax0.set_ylim( -1, 1 )
        ax1.set_ylim( -1, 1 )
        ax2.set_ylim( -1, 1 )
        ax0.set_xlabel( 'Time (s)', fontsize=10 )
        ax2.set_xlabel( 'Time (s)', fontsize=10 )
        ax0.set_title( str(network)+'.'+str(station)+'\t'+ \
                       str( format( recDisInDeg, '.2f' ) )+r'$\degree$',
                       fontsize=14,
                       loc = 'left')
        plt.tight_layout()
        #--保存
        figName = "{0}.{1}_WaveformsPhasesSNR.pdf".format( network, station )
        plt.savefig( str(outfilePath)+'/'+str(figName), dpi=100 )
    
        # Clear the current axes.
        plt.cla() 
        # Clear the current figure.
        fig.clear()
        plt.clf() 
        # Closes all the figure windows.
        plt.close('all')   
        plt.close(fig)
        gc.collect()
    
    
        with open( figListName, mode='a', newline='' ) as outFile:
            writer = csv.writer( outFile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['{0}'.format( ist ),
                             '{0}'.format( network ),
                             '{0}'.format( station ),
                             '{0}'.format( epLat ),
                             '{0}'.format( epLon ),
                             '{0}'.format( evDp ),
                             '{0}'.format( wavestart ),
                             '{0}'.format( stla ),
                             '{0}'.format( stlo ),
                             '{0}'.format( elev ),
                             '{0}'.format( azimuth ),
                             '{0}'.format( format( recDisInDeg, '.2f' ) ),
                             '{0}'.format( format( snrZ, '.2f' ) ),
                             '{0}'.format( format( snrR, '.2f' ) ),
                             '{0}'.format( format( snrT, '.2f' ) ),
                             '{0}'.format( figName ) ])
        outFile.close()




    SNRTHR = 3.0
    if( snrZ >= SNRTHR and snrT >= SNRTHR ):
        print( selectedfileE[ist] )
        print( selectedfileN[ist] )
        print( selectedfileZ[ist] )

        #-- 输出信噪比高的台站列表
        with open( stationListHighSN, mode='a', newline='' ) as outFile:
            writer = csv.writer( outFile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['{0}'.format( ist ),
                             '{0}'.format( network ),
                             '{0}'.format( station ),
                             '{0}'.format( epLat ),
                             '{0}'.format( epLon ),
                             '{0}'.format( evDp ),
                             '{0}'.format( wavestart ),                              
                             '{0}'.format( stla ),
                             '{0}'.format( stlo ),
                             '{0}'.format( elev ),
                             '{0}'.format( azimuth ),
                             '{0}'.format( format( recDisInDeg, '.2f' ) ),
                             '{0}'.format( format( snrZ, '.2f' ) ),
                             '{0}'.format( format( snrR, '.2f' ) ),
                             '{0}'.format( format( snrT, '.2f' ) ),
                             '{0}'.format( selectedfileE[ist] ),
                             '{0}'.format( selectedfileN[ist] ),
                             '{0}'.format( selectedfileZ[ist] ) ])
        outFile.close()
    gc.collect()









 
#%%-- Main
if __name__ == "__main__":
     
    #%%-- load input parameters
    catalogPath, velModel, arrTimeDiffTole, ccThreshold,\
    vFrequencyFrom, vFrequencyTo, hFrequencyFrom, hFrequencyTo,\
    depthRadiusAbove, depthRadiusBelow, verboseFlag, plotSteps1n2Flag = load_settings()

    #%%--
    from obspy.core.event import read_events
    readcsvflag = False
    try:
        catalog = read_events( catalogPath+'catalog.xml', format="QUAKEML"  )
        print( catalog )
    except:
        readcsvflag = True
        print( "No 'catalog.xml' in the catalog path: " + catalogPath ) 
    if readcsvflag:
        print( 'now try to read table_results.csv')
        eventdata = pd.read_csv(catalogPath+'/table_results.csv')
        catalog = eventdata['starttimeGCMT']
        stla = eventdata['LatGCMT(deg)']
        stlo = eventdata['LonGCMT(deg)']
        elev = eventdata['DepGCMT(km)']
    

    #-- 
    for idx, iev in enumerate( catalog ):
     #if idx < (len(catalog)/2) and idx >= 17:
        if readcsvflag:
            #-- print some key information
            epTime = UTCDateTime(catalog[idx])
            print('epTime = ',epTime)
            epLat  = stla[idx]
            epLon  = stlo[idx]
            evDp   = elev[idx]
            print( '\n\n\n=============================')
            print( 'Number of events =', len(catalog) )
            print( 'Now processing event: ', idx+1, '/', len(catalog) )
            print( 'epTime  = ', epTime )
            print( 'epLat   = ', epLat )
            print( 'epLon   = ', epLon )
            print( 'evDp    = ', evDp )
            date   = str(epTime)[0:10]
            hour   = int(str(epTime)[11:13])
            minute = int(str(epTime)[14:16])
            second = int(str(epTime)[17:19])
        else:
            #-- print some key information
            epTime = iev.origins[0].time
            epLat  = iev.origins[0].latitude
            epLon  = iev.origins[0].longitude
            evDp   = iev.origins[0].depth /1000.0
            #epMag  = iev.magnitudes[0].mag
            print( '\n\n\n=============================')
            print( 'Number of events =', len(catalog) )
            print( 'Now processing event: ', idx+1, '/', len(catalog) )
            print( 'epTime  = ', epTime )
            print( 'epLat   = ', epLat )
            print( 'epLon   = ', epLon )
            print( 'evDp    = ', evDp )
            #print( 'epMag   = ', epMag )

            date   = iev.origins[0].time.datetime.date()
            hour   = iev.origins[0].time.hour
            minute = iev.origins[0].time.minute
            second = iev.origins[0].time.second
            
        #%%-- 
        eventDirectoryName = str(date)+'-'+str(hour)+'-'+str(minute)+'-'+str(second)
        print( 'eventDirectoryName', eventDirectoryName)
        
        if not os.path.exists( catalogPath+str(eventDirectoryName) ):
            print('Warning: The directory < '+catalogPath+str(eventDirectoryName)+\
                  ' >  not found!' )
            continue

        #--
        wfPath = catalogPath+str(eventDirectoryName)+'/'
        #--
        invPath = catalogPath+str(eventDirectoryName)+'/'+'inventory'+'/'
        #--
        velPath = catalogPath+'velocityModel'+'/'
    
        #%%--
        wfFiles = fnmatch.filter( sorted(os.listdir(wfPath)), '*.mseed')
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
                
                #-- 判断该台站是否含有三分量数据                
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
        if not os.path.exists(str(wfPath)+'figures_SNR'):
            os.mkdir(str(wfPath)+'figures_SNR')
        else:
            print( '\n Warning: "figures_SNR" already exists!\n')
            
        outfilePath = str(wfPath)+'figures_SNR'
              
        #%%--                       
        if evDp < 0 or evDp > 5000:
            sys.exit("\n Current event's depth error, stop!\n")
        if (evDp-depthRadiusAbove) < 0:
            print( "\n Warning: 'depthRadiusAbove' should be <=",
                  math.floor(evDp), ', now set it to be 5 \n' )
            depthRadiusAboveNew = math.floor(evDp)-5 #-- 0-4 km will give unstable results
        else:
            depthRadiusAboveNew = depthRadiusAbove
        
        #%%-- Step 1 to 3 of DSA
        srcDepthScanBeg = math.floor( evDp-depthRadiusAboveNew )
        srcDepthScanEnd = math.floor( evDp+depthRadiusBelow )
        numScanDepth    = int(srcDepthScanEnd-srcDepthScanBeg)



        #%%-- 
        figListName = str(outfilePath)+'/'+'0_FigListOfWaveformsAndPhasesAndSNR.csv'
        with open( figListName, mode='w', newline='' ) as outFile:
            writer = csv.writer( outFile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow( [ 'Idx', 'Net', 'Sta', 'epLat', 'epLon', 'evDp', 'epTime', 'Lat(deg)',  'Lon(deg)','elev',
                               'Az(deg)', 'EpDis(deg)', 'snrZ', 'snrR', 'snrT',
                               'figName' ] )

        #%%--
        stationListHighSN = str(wfPath)+'/'+'0_StationWithHighSNR.csv'
        with open( stationListHighSN, mode='w', newline='' ) as outFile:
            writer = csv.writer( outFile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow( [ 'Idx', 'Net', 'Sta', 'epLat', 'epLon', 'evDp', 'epTime', 'Lat(deg)',  'Lon(deg)', 'elev',
                               'Az(deg)', 'EpDis(deg)', 'snrZ', 'snrR', 'snrT',
                               'filePathE', 'filePathN', 'filePathZ' ] )
       
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


 
    
        #%%-- 
        #---------------------#
        #--                 --#
        #---------------------#
        begin_timer = timeit.default_timer()
        
        #-- TauP
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
#            print( stRawE )
#            print( stRawN )
#            print( stRawZ )
#            stRawE.plot()
#            stRawN.plot()
#            stRawZ.plot()
            

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
                     invPath, outfilePath, figListName, stationListHighSN, 
                     epLat, epLon, evDp, epTime]
            subCalSteps1and2OneStation( args )

        
 




        
        #%%##########################################
        #-- 
        #############################################
        
        data = pd.read_csv( str(outfilePath)+'/'+'0_FigListOfWaveformsAndPhasesAndSNR.csv' )
        figName = data[ 'figName' ]
        EpDis   = data[ 'EpDis(deg)' ]
        numFigs = len( figName )
        
        #-- 30-50、50-70、70-90
        figDis3050 = []
        figDis5070 = []
        figDis7090 = []
        for iFig in range( numFigs ):
            tmp = EpDis[iFig], figName[iFig]
            if EpDis[iFig] >= 30 and EpDis[iFig] < 50:
                figDis3050.append( tmp )
            elif EpDis[iFig] >= 50 and EpDis[iFig] < 70:
                figDis5070.append( tmp )
            elif EpDis[iFig] >= 70 and EpDis[iFig] <= 90:
                figDis7090.append( tmp )
        
        #-- 
        figDis3050.sort(key=lambda item: item[0] )
        figDis5070.sort(key=lambda item: item[0] )
        figDis7090.sort(key=lambda item: item[0] )
        
        
        #-- merge pdf
        from PyPDF2 import PdfMerger
        merger3050 = PdfMerger()
        merger5070 = PdfMerger()
        merger7090 = PdfMerger()
        for idx, iMem in enumerate( figDis3050 ):
            merger3050.append( str(outfilePath)+'/'+iMem[1] )
        for idx, iMem in enumerate( figDis5070 ):
            merger5070.append( str(outfilePath)+'/'+iMem[1] )
        for idx, iMem in enumerate( figDis7090 ):
            merger7090.append( str(outfilePath)+'/'+iMem[1] )            
        
        if len(figDis3050)>0:
            merger3050.write( str(outfilePath)+'/'+'0_dist30-50_MergedStationsSNR.pdf' )
        if len(figDis5070)>0:
            merger5070.write( str(outfilePath)+'/'+'0_dist50-70_MergedStationsSNR.pdf' )
        if len(figDis7090)>0:
            merger7090.write( str(outfilePath)+'/'+'0_dist70-90_MergedStationsSNR.pdf' )
        merger3050.close()
        merger5070.close()
        merger7090.close()
        
        del figName, EpDis, numFigs
        del figDis3050, figDis5070, figDis7090
        del merger3050, merger5070, merger7090
        
        
        #%% calculate computing time
        end_timer = timeit.default_timer()
        elapsedTime = end_timer - begin_timer
        print('Elapsed time: ', format( elapsedTime, '.1f'),
          'sec = ', format( elapsedTime/60.0, '.1f'), 'min' )

        gc.collect()
        

        
