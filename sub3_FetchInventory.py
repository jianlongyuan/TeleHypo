#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Function: Download instrument response from IRIS
      
"""


import matplotlib.pyplot as plt
from obspy.core import UTCDateTime
import numpy as np
import os, fnmatch, sys
import fnmatch
from obspy import read
from obspy.clients.fdsn import Client
import pandas as pd
import gc
plt.rcParams["font.family"] = "Times New Roman"


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

#%%-- user defined parameters
mainCatalogPath= load_settings()
#------------------------------------------------------------------------------
#   dataPath 
#------------------------------------------------------------------------------
dirs = []
dirsAndFiles = sorted( os.listdir(mainCatalogPath), reverse=True )
#-- 
try:
    print( "\n All directories in the path: "+str(mainCatalogPath) )
    for iMem in dirsAndFiles:
        if os.path.isdir( mainCatalogPath+iMem ):
            dirs.append( iMem )
            #print( '\t'+dirs[-1] )
except:
    sys.exit( 'No directories in the path: '+mainCatalogPath ) 
Ndirs = len( dirs )
print( '目录个数 =', Ndirs )


#--
for idx, idir in enumerate( dirs ):
    
  if idx >= 0:
    print( 'Event', idx, idir )
    existedStation = fnmatch.filter( sorted(os.listdir(mainCatalogPath+idir)), '*BHZ*.mseed')
    numExiStation  = len(existedStation)
    print( '\t Number of existed stations:', numExiStation )
    
    if numExiStation == 0:
        print( "--- No station waveform for this directory! ---" )
        continue

    #-- 
    invPath = mainCatalogPath+'/'+str(idir)+'/'+'inventory'+'/'
    
    #-- 
    for idxSt, iStFile in enumerate(existedStation):
      if idxSt >= 0:
        #-- 
        try:
            stRawZ = read(str(mainCatalogPath+idir)+'/'+iStFile, debug_headers=True)
            print ('\n---------------------------------------------------------\n')
            print("idx     = ", idx, ' / ', Ndirs )
            print("Station = ", idxSt+1, ' / ', numExiStation )
            print("Network = ", stRawZ[0].stats.network )
            print("Station = ", stRawZ[0].stats.station )
            print("stZ[0].stats.starttime", stRawZ[0].stats.starttime)
            print("stZ[0].stats.endtime  ", stRawZ[0].stats.endtime)
        
            #-- 
            currentInvFile = "{0}/{1}.{2}.xml".format( invPath,
                                                       stRawZ[0].stats.network,
                                                       stRawZ[0].stats.station )
        except:
            print ('Error in:', stRawZ[0].stats.station)
            continue
            
        if os.path.isfile( currentInvFile ):
            print("File already exists, now skip this station! \n")
            continue
        else:
            print("No such file, now downlaoding...")
    
            # -- download the inventory file
            client = Client("IRIS")
            try:
                inv = client.get_stations( network = stRawZ[0].stats.network,
                                           station = stRawZ[0].stats.station,
                                           loc     = "**",
                                           channel = "BHN,BHE,BHZ",
                                           starttime = stRawZ[0].stats.starttime,
                                           endtime   = stRawZ[0].stats.endtime,
                                           level     = "response" )
                inv.write("{0}/{1}.{2}.xml".format( invPath,
                                                stRawZ[0].stats.network,
                                                stRawZ[0].stats.station ),
                                                format="STATIONXML")
                #print(inv)
                print ('Write inventory file done! \n')
            except:
                print ('No Inventory file for:', stRawZ[0].stats.station)
                continue

