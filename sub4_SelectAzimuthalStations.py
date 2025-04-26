#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Function:
    The coverage is divided into 36 regions at intervals of 10 ° in azimuth.
    A maximum of 5 high S/N stations are selected from each region. 
    Since the number of high S/N stations in some azimuthal areas can be less than 5,
    the total number of high S/N stations in all azimuth areas is usually less than 180,
    which also means that the number of depth phases with high quality will be less than 180. 
     
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import time
import gc, os, sys
plt.rcParams["font.family"] = "Times New Roman"
begin_timer = time.perf_counter()  # Start timer    
    


#%%############################################################################
def subSelectStationsFrom36AzimuthalRanges( dataPath ) :
        
    #%%##########################################
    #-- 
    #############################################
    try:
        data = pd.read_csv( str(dataPath)+'0_StationWithHighSNR.csv' )
        azimuth = data[ 'Az(deg)' ]
        snrZ    = data[ 'snrZ' ]
        snrT    = data[ 'snrT' ]
        epLat    = data[ 'epLat' ]
        epLon    = data[ 'epLon' ]
        evDp     = data[ 'evDp' ]
        epTime = data['epTime']
        Lat    = data[ 'Lat(deg)' ]
        Lon    = data[ 'Lon(deg)' ]
        elev     = data[ 'elev' ]
        fileE   = data[ 'filePathE' ]
        fileN   = data[ 'filePathN' ]
        fileZ   = data[ 'filePathZ' ]
        numAvailableSt = len( azimuth )    
        print( 'numAvailableSt =', numAvailableSt )
    except:
        print( '\t No 0_StationWithHighSNR.csv!!!' )
        return(0)
    
    
    #%%################################################
    #--   
    ##################################################
    azRange = np.arange( 0, 360, 10 )
    numStOneAzRangeThr    = 5
    totalSelectedStations = len(azRange)*numStOneAzRangeThr
    print( '\n Total needed stations      =', totalSelectedStations)
    print( '\n Number of azimuthal ranges =', len(azRange) )
    
    selectedStationListENZ = []
    
    #%%-- 
    if numAvailableSt == 0:
        print( '\n No available stations! \n' )
        return(0)
    else:        
        numAzimuthalRange = len( azRange )
        azimuthalRangeStation = [[] for i in range(numAzimuthalRange)]  
        
        #-- 
        azInc = azRange[1]-azRange[0]
        for iAz in range( numAzimuthalRange ):
            for ist in range( numAvailableSt ):
                if azimuth[ist] >= azRange[iAz] and azimuth[ist] < azRange[iAz]+azInc:
                    totSNR = snrZ[ist] + snrT[ist]
                    tmp = totSNR, ist
                    azimuthalRangeStation[iAz].append( tmp )
            numStCurrentRange = len(azimuthalRangeStation[iAz])
            
            print( '\n numStCurrentRange =', numStCurrentRange )
            print( '\t [minAz, maxAz)  =', '[', azRange[iAz], ',', azRange[iAz]+azInc, ')' )
            print( '\t azimuthalRangeStation =', azimuthalRangeStation[iAz] )
    
            
    
            #%%--
            if numStCurrentRange > numStOneAzRangeThr:
                #-- 
                azimuthalRangeStation[iAz] = sorted(azimuthalRangeStation[iAz],
                                              key=lambda l:l[0], reverse=True)
                #print( '\t Sorted_azimuthalRangeStation =', azimuthalRangeStation[iAz] )
                #%%
                for iMem in range( numStOneAzRangeThr ):
                    totSNR = azimuthalRangeStation[iAz][iMem][0]  # [0]代表台站信噪比
                    stIdx  = azimuthalRangeStation[iAz][iMem][1]  # [1]代表台站索引
                    tmp = totSNR, epLat[stIdx], epLon[stIdx], evDp[stIdx], epTime[stIdx], Lat[stIdx], Lon[stIdx], elev[stIdx], fileE[stIdx], fileN[stIdx], fileZ[stIdx]
                    selectedStationListENZ.append( tmp )                

            elif numStCurrentRange >= 1 and numStCurrentRange <= numStOneAzRangeThr:
                for iMem in range( numStCurrentRange ):
                    totSNR = azimuthalRangeStation[iAz][iMem][0]  # [0] for snr
                    stIdx  = azimuthalRangeStation[iAz][iMem][1]  # [1] for index
                    tmp = totSNR, epLat[stIdx], epLon[stIdx], evDp[stIdx], epTime[stIdx], Lat[stIdx], Lon[stIdx], elev[stIdx], fileE[stIdx], fileN[stIdx], fileZ[stIdx]
                    selectedStationListENZ.append( tmp )  
    
        
       
        numSelectedStation = len(selectedStationListENZ)
        print("\n Total final selected stations =", numSelectedStation)
        #for ist in range( numSelectedStation ):
        #    print( selectedStationListENZ[ist] ) 
        
        

        stationListHighSN = str(dataPath)+'0_StationWithHighSNRforDSA_OrgTimeGCMT_DepthGCMT.csv'
        with open( stationListHighSN, mode='w', newline='' ) as outFile:
            writer = csv.writer( outFile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow( [ 'totSNR', 'epLat','epLon','evDp','epTime','lat','lon','elev','filePathE', 'filePathN', 'filePathZ' ] )
            for ist in range( numSelectedStation ):
                writer.writerow( [ selectedStationListENZ[ist][0],
                                   selectedStationListENZ[ist][1],
                                   selectedStationListENZ[ist][2],
                                   selectedStationListENZ[ist][3], 
                                   selectedStationListENZ[ist][4],
                                   selectedStationListENZ[ist][5],
                                   selectedStationListENZ[ist][6],
                                   selectedStationListENZ[ist][7], 
                                   selectedStationListENZ[ist][8],
                                   selectedStationListENZ[ist][9],
                                   selectedStationListENZ[ist][10]])

   
    
    
    
    
    
    
    
    

#%%--
# user defined parameters
#----
def load_settings():
    '''
     PARAMETER          DESCRIPTION   
     par1    Data directory, including wavefroms and velocity model 
    '''
    
    try:
        SETTINGS = pd.read_csv('./SETTINGS.txt',
                               delim_whitespace=True, index_col='PARAMETER')
        par1 = SETTINGS.VALUE.loc['catalogPath'] 
        return par1
        gc.collect()  
    except:
        sys.exit("Errors in 'SETTINGS.txt' !\n")
        
mainCatalogPath= load_settings()



#------------------------------------------------------------------------------
#  
#------------------------------------------------------------------------------
dirs = []
dirsAndFiles = sorted( os.listdir(mainCatalogPath), reverse=True )
try:
    print( "\n All directories in the path: "+str(mainCatalogPath) )
    for iMem in dirsAndFiles:
        if os.path.isdir( mainCatalogPath+iMem ):
            dirs.append( iMem )
            #print( '\t'+dirs[-1] )
except:
    sys.exit( 'No directories in the path: '+mainCatalogPath ) 
Ndirs = len( dirs )
print( 'Ndirs =', Ndirs )


#--
for idx, idir in enumerate( dirs ):
  if idx >= 0:
    print('\n\n\n')
    print('------------------------------------------------------------------')
    print( 'Event:', idx, ', directory name:', idir )
    dataPath = mainCatalogPath+idir+'/'
    print( 'Path: ', dataPath )
    subSelectStationsFrom36AzimuthalRanges( dataPath )    
    


end_timer = time.perf_counter()    # End timer
elapsedTime = end_timer - begin_timer
print('Elapsed time of downloading: ', format( elapsedTime, '.1f'),
      'sec = ', format( elapsedTime/60.0, '.1f'), 'min' )