#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Function: Fetch catalog and waveforms from GCMT

Note: Due to the large number of stations that need to be downloaded, 
      sometimes the download is incomplete at once, and the script needs
      to be run multiple times (see parameter 'DownloadTimes') to ensure
      the integrity of the downloaded stations

"""
import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
import os, fnmatch, sys
from obspy import read
import time
from obspy.geodetics import gps2dist_azimuth
from obspy.geodetics.base import kilometer2degrees
import random
from obspy.clients.fdsn.mass_downloader import CircularDomain, \
    Restrictions, MassDownloader

# May need to run multiple times to ensure the integrity of the downloaded stations
DownloadTimes = 2
for iii in range( DownloadTimes ):    
    #%%-- user defined parameters
    
    minDisInDeg, maxDisInDeg = 30, 90 #-- unit: Degree
    timeBeforeOrg = -1800 #-- time length before origin time (sec)
    timeAfterOrg  = 1800 #-- time length after origin time (sec)
    minMagnitude  = 6.0
    maxMagnitude  = 8.0
    minDepth      = 50
    maxDepth      = 200
    startDate     = '2010-03-04'
    endDate       = '2010-05-13'
    startTime     = UTCDateTime( str(startDate)+'T00:00:00' )
    endTime       = UTCDateTime( str(endDate)  +'T23:59:59' )
    dataPath      = './'
    
    
    #-- create catalog path
    mainCatalogPath = str(dataPath)+'catalog_GCMT_'+str(startDate)+'_'+str(endDate)+\
                                    '_Mw'+str(minMagnitude)+'-'+str(maxMagnitude)+\
                                    '_'+str(minDepth)+'-'+str(maxDepth)+'km'
    
    if not os.path.exists( mainCatalogPath ):
        os.mkdir( mainCatalogPath )
    else:
        print('Warning: The directory < '+str(mainCatalogPath)+\
              ' > already exists! ' )   
    
    client      = Client("IRIS")
    catalogName = "GCMT"
    catalog = client.get_events( starttime    = startTime,
                                 endtime      = endTime,
                                 minmagnitude = minMagnitude,
                                 maxmagnitude = maxMagnitude,
                                 mindepth     = minDepth,
                                 maxdepth     = maxDepth,
                                 catalog      = catalogName )
    
    #%%
    print( catalog )
    # #-- plot in screen
    # catalog.plot( )
    # #-- output catalog
    # catalog.plot( resolution = 'l',
    #               projection = "global",
    #               color      = 'depth',
    #               outfile    = mainCatalogPath+'/'+'catalog.png' )
    catalog.write( mainCatalogPath+'/'+'catalog.xml', format="QUAKEML" )
    catalog.write( mainCatalogPath+'/'+'catalog.cnv', format="CNV" )
    
    
    
    
    #%%-- fetch station data
    bulk = []
    elapsedTimeAll = []
    
    for idx, iev in enumerate( catalog ):
      if idx > int( len(catalog)/2 ):
        begin_timer = time.perf_counter()  # Start timer
        
        try:
             #-- print some key information
            epTime = iev.origins[0].time
            epLat  = iev.origins[0].latitude
            epLon  = iev.origins[0].longitude
            epDep  = iev.origins[0].depth /1000.0
            epMag  = iev.magnitudes[0].mag
            print( '\n\n\n=============================')
            print( 'Number of events =', len(catalog) )
            print( 'Now processing event: ', idx+1, '/', len(catalog) )
            print( 'epTime  = ', epTime )
            print( 'epLat   = ', epLat )
            print( 'epLon   = ', epLon )
            print( 'epDep   = ', epDep )
            print( 'epMag   = ', epMag )
    
            date   = iev.origins[0].time.datetime.date()
            hour   = iev.origins[0].time.hour
            minute = iev.origins[0].time.minute
            second = iev.origins[0].time.second
            eventDirectoryName = str(date)+'-'+str(hour)+'-'+str(minute)+'-'+str(second)
            print( 'eventDirectoryName', eventDirectoryName)
            
            if not os.path.exists( mainCatalogPath+'/'+str(eventDirectoryName) ):
                os.mkdir( mainCatalogPath+'/'+str(eventDirectoryName) )
            else:
                print('Warning: The directory < '+mainCatalogPath+'/'+str(eventDirectoryName)+\
                      ' > already exists! Now replacing the old files!' )
            #--waveforms path
            eventPath = mainCatalogPath+'/'+str(eventDirectoryName)+'/'
            #--inventory path
            invPath = mainCatalogPath+'/'+str(eventDirectoryName)+'/'+'inventory'+'/'            
                
            #%%################################################
            #-- fetch all available station data
            ##################################################
            from obspy.clients.fdsn import RoutingClient
            client = RoutingClient("iris-federator")
            inv = client.get_stations(
                    network   = '*',
                    station   = '*',
                    location  = '--,00',
                    channel   = 'BHE,BHN,BHZ',
                    level     = 'channel',
                    includeavailability = True,
                    matchtimeseries = True,
                    starttime = iev.origins[0].time,
                    endtime   = iev.origins[0].time+10,
                    latitude  = iev.origins[0].latitude,
                    longitude = iev.origins[0].longitude,
                    minradius = minDisInDeg,
                    maxradius = maxDisInDeg )
            
            
            print( '\n Total number of available stations:' )
            #inv.plot( projection='global', resolution='c', marker='v', size=50 )         
            
            numNetworks = len(inv.networks)
            print( '\n Total number of networks =',  numNetworks )
            print( '\n Total number of channels =',  len( inv.networks[0].stations[0].channels) )
            
            #%%--  three-component data
            stationList = []
            for inet in range( numNetworks ):
                print( '\n Number of available stations of', inet+1,
                       'network =', len( inv.networks[inet] ) ) 
                for ist in range( len( inv.networks[inet] ) ):
                    if len( inv.networks[inet].stations[ist].channels) == 3:
                        net = inv.networks[inet].code
                        name= inv.networks[inet][ist].code
                        lat = inv.networks[inet][ist].latitude
                        lon = inv.networks[inet][ist].longitude
                        ele = inv.networks[inet][ist].elevation
    
                        dist, az, baz = gps2dist_azimuth(epLat, epLon, lat, lon )
                        distInDeg = kilometer2degrees( dist/1000.0 )
                        
                        stInfo = net, name, lat, lon, ele, az, baz, distInDeg
                        
                        stationList.append( stInfo )
                        
                        #print( inv.networks[inet].stations[ist] )
            
            numAvailableSt = len(stationList)
            print( '\nTotal available stations with N/E/Z =', numAvailableSt )        
    
    
            azRange = np.arange( 0, 360, 10 )
            numStInEachAzRange    = 500
            totalSelectedStations = len(azRange)*numStInEachAzRange
            print( '\n Total needed stations      =', totalSelectedStations)
            print( '\n Number of azimuthal ranges =', len(azRange) )
            
            selectedStationList = []
            
    
            if numAvailableSt == 0:
                sys.exit( '\n No available stations! \n' )
            elif numAvailableSt >= 1 and numAvailableSt <= totalSelectedStations:
                    selectedStationList = stationList
            elif numAvailableSt > totalSelectedStations:
    
                restStationList = list(stationList) 
    
                numAzimuthalRange = len( azRange )
                azimuthalRangeStation = [[] for i in range(numAzimuthalRange)]  
     
                azInc = azRange[1]-azRange[0]
                for iAz in range( numAzimuthalRange ):
                    for ist in range( numAvailableSt ):
                        if stationList[ist][5] >= azRange[iAz] and stationList[ist][5] < azRange[iAz]+azInc:
                            azimuthalRangeStation[iAz].append( ist )
                    numStCurrentRange = len(azimuthalRangeStation[iAz])
                    
                    print( '\n numStCurrentRange =', numStCurrentRange )
                    print( '\t [minAz, maxAz)  =', '[', azRange[iAz], ',', azRange[iAz]+azInc, ')' )
                    print( '\t azimuthalRangeStation =', azimuthalRangeStation[iAz] )
    
                    
                    if numStCurrentRange > numStInEachAzRange:
                        randomStationIdx = random.sample( range( 0, numStCurrentRange ), numStInEachAzRange )
                        print( 'randomStationIdx =', randomStationIdx )
                        for iMem in randomStationIdx:
                            stIdx = azimuthalRangeStation[iAz][iMem]
                            selectedStationList.append( stationList[stIdx] )
                            restStationList.remove( stationList[stIdx] )
                    elif numStCurrentRange >= 1 and numStCurrentRange <= numStInEachAzRange:
                        for iMem in range( numStCurrentRange ):
                            stIdx = azimuthalRangeStation[iAz][iMem]
                            selectedStationList.append( stationList[stIdx] )
                            restStationList.remove( stationList[stIdx] )

                numSelectedSt = len(selectedStationList)
                print("numSelectedSt (all azimuthal ranges )=", numSelectedSt)
                stillNeededNum = np.int( totalSelectedStations - numSelectedSt )
                print("stillNeededNum =", stillNeededNum)
                if stillNeededNum > 0:
                    numRestSt = len(restStationList)
                    randomStationIdx = random.sample( range( 0, numRestSt ), stillNeededNum )
                    print( 'randomStationIdx =', randomStationIdx )
                    for iMem in randomStationIdx:
                        selectedStationList.append( restStationList[iMem] )
            numSelectedStation = len(selectedStationList)
            print("\n Total selected stations =", numSelectedStation)
      
        
            try:
                t1 = iev.origins[0].time+0
                t2 = iev.origins[0].time+10
                for ist in selectedStationList:
                    network  = ist[0]
                    station  = ist[1]
                    location  = '--,00'
                    channel   = 'BHZ'
                    tmp = ( network, station, location, channel, t1, t2 )
                    bulk.append( tmp )  
                selectedInv = client.get_stations_bulk(bulk)
                selectedInv.plot( projection='global', resolution='c', 
                                  marker='v', size=50,
                                  outfile = eventPath+'000_stationsMap.png' )
            except:
                print( '\n client.get_stations_bulk: failed! \n' )
                
                
                
            existedStationList = []
            finalStationList   = []
            
            try:    
                existedStation = fnmatch.filter( sorted(os.listdir(eventPath)), '*BHZ*.mseed')
                numExiStation  = len(existedStation)
                print( '\n Number of existed stations:', numExiStation )
                for ist in existedStation:
                    infileZ = open( eventPath+ist )
                    stZ = read( infileZ.name, debug_headers=True)
                    existedStationList.append( stZ[0].stats.station )
            except:
                print( '\n No existed stations! \n' )
            
            
            networkList = []
            stationList = []
            for ist in selectedStationList:
                networkList.append( ist[0] )
                stationList.append( ist[1] )
            networkList = list( set(networkList) )
            stationList = list( set(stationList) )
            
            if numExiStation == 0:
                finalStationList = stationList
            else:
                finalStationList = list(set(stationList).difference(set(existedStationList)))
            
            numFinalStation  = len(finalStationList)
            print( "\n Number of stations needed to be downloaded = ", len(finalStationList) )
            
            if numFinalStation == 0:
                print('\n\n\n')
                print( "-----------------------------------------" )
                print( "  No stations needed to be downloaded!   " )
                print( "-----------------------------------------" )
                print('\n\n\n')
                continue

            networkList      = ",".join( networkList )
            finalStationList = ",".join( finalStationList )
                    
            
            
            try:    
                #%%
                # Circular domain around the epicenter. This will download all data between
                # 70 and 90 degrees distance from the epicenter. This module also offers
                # rectangular and global domains. More complex domains can be defined by
                # inheriting from the Domain class.
                domain = CircularDomain(latitude =iev.origins[0].latitude,
                                        longitude=iev.origins[0].longitude,
                                        minradius=minDisInDeg,
                                        maxradius=maxDisInDeg )
                
                restrictions = Restrictions(
                    # Get data from 5 minutes before the event to one hour after the
                    # event. This defines the temporal bounds of the waveform data.
                    starttime = iev.origins[0].time+timeBeforeOrg,
                    endtime   = iev.origins[0].time+timeAfterOrg,
                    # You might not want to deal with gaps in the data. If this setting is
                    # True, any trace with a gap/overlap will be discarded.
                    reject_channels_with_gaps = True,
                    # And you might only want waveforms that have data for at least 95 % of
                    # the requested time span. Any trace that is shorter than 95 % of the
                    # desired total duration will be discarded.
                    minimum_length = 0.95,
                    # No two stations should be closer than 10 km to each other. This is
                    # useful to for example filter out stations that are part of different
                    # networks but at the same physical station. Settings this option to
                    # zero or None will disable that filtering.
                    minimum_interstation_distance_in_m = 0,
                    network = networkList,
                    station = finalStationList,
                    channel = 'BHE,BHN,BHZ',
                    # Location codes are arbitrary and there is no rule as to which
                    # location is best. Same logic as for the previous setting.
                    location_priorities = [ "", "00" ] )
                
                # No specified providers will result in all known ones being queried.
                mdl = MassDownloader()
                # The data will be downloaded to the ``./waveforms/`` and ``./stations/``
                # folders with automatically chosen file names.
                # Control how many threads are used to download data in parallel
                # per data center - 3 is a value in agreement with some data centers.
                mdl.download(domain, restrictions, threads_per_client=3,
                             mseed_storage=eventPath,
                             stationxml_storage=eventPath+"inventory")
            except:
    
                end_timer = time.perf_counter()    # End timer
                elapsedTime = end_timer - begin_timer
                elapsedTimeAll.append( elapsedTime )
                print('Elapsed time of downloading: ', format( elapsedTime, '.1f'),
                      'sec = ', format( elapsedTime/60.0, '.1f'), 'min' )
                
                print( '\n mdl.download: failed! \n' )
                continue
            
            

            end_timer = time.perf_counter()    # End timer
            elapsedTime = end_timer - begin_timer
            elapsedTimeAll.append( elapsedTime )
            print('Elapsed time of downloading: ', format( elapsedTime, '.1f'),
                  'sec = ', format( elapsedTime/60.0, '.1f'), 'min' )
            
        
        except:
            end_timer = time.perf_counter()    # End timer
            elapsedTime = end_timer - begin_timer
            elapsedTimeAll.append( elapsedTime )
            print('Elapsed time of downloading: ', format( elapsedTime, '.1f'),
                  'sec = ', format( elapsedTime/60.0, '.1f'), 'min' )
            
            print( 'No available stations!' ) 
            continue
    
    
    print('\n\n\n Total elapsed time of downloading of each event:' )
    for idx, elapsedTime in enumerate( elapsedTimeAll ):
        print('\t Event '+str(idx+1), ': ',
              format( elapsedTime, '.1f'), 'sec = ',
              format( elapsedTime/60.0, '.1f'), 'min' )
    
    totalElapsedTime = sum(elapsedTimeAll)
    print('Total elapsed times: ',
          format( totalElapsedTime, '.1f'), 'sec =',
          format( totalElapsedTime/60.0, '.1f'), 'min =',
          format( totalElapsedTime/3600.0, '.1f'), 'hrs' )