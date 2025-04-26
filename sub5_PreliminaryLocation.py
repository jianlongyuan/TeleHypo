"""
Framework:
  1. Scans the continuous waveform using the SSA method (Kao and Shan, 2004).
  2. Automatically pick up the first arrival time of P and S through the kurtosis function (Baillard et al., 2014). 
  3. the results obtained by MAXI (Font et al., 2004) are used as a preliminary solution.

Input parameters can be seen in:
  SETTINGS.txt


Please cite:
Tan, F., Kao, H., Nissen, E., and Eaton, D. (2019). Seismicity-scanning based on
navigated automatic phase-picking. Journal of Geophysical Research: Solid Earth 124, 3802–3818

Jianlong Yuan, Huilian Ma, Jiashun Yu, Zixuan Liu and Shaojie Zhang. (2025). An approach 
for teleseismic location by automatically matching depth phase. Front. Earth Sci. (Under revirew)


Any questions or advices? Please contact at:
    jianlongyuan@cdut.edu.cn (Jianlong Yuan)
    1334631943@qq.com (Huilian Ma)
    j.yu@cdut.edu.cn  (Jiashun Yu)
    2751017165@qq.com (Zixuan Liu)
    1716136870@qq.com (Shaojie Zhang)
     
"""

import numpy as np
import pickle
import taupz
import csv
from obspy import read
from pathlib import Path
import distance
import math
from psseparation import psseparation
import copy
import ssa
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
from detect_peaks import detect_peaks
from operator import itemgetter
import picking
import locatePS
import calc_mag
import timeit
import os,shutil,sys
import pandas as pd
start = timeit.default_timer()

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']



def load_settings():
    
    try:
        SETTINGS = pd.read_csv('./SETTINGS.txt', delim_whitespace=True, index_col='PARAMETER')
        
        par1 = SETTINGS.VALUE.loc['catalogPath']
        par5 = SETTINGS.VALUE.loc['nametag']
        par6 = SETTINGS.VALUE.loc['area']     
        par7 = float( SETTINGS.VALUE.loc['studydepthbottom'] )
        par8 = float( SETTINGS.VALUE.loc['studydepthtop']  )
        par9 = float( SETTINGS.VALUE.loc['studydepth_for_locatebottom']  )
        par10 = float( SETTINGS.VALUE.loc['studydepth_for_locatetop']  )
        par11 = float( SETTINGS.VALUE.loc['depgrid']  )
        par12 = float( SETTINGS.VALUE.loc['latgrid']  )
        par13 = float( SETTINGS.VALUE.loc['longrid']  )
        par14 = int( SETTINGS.VALUE.loc['process_unit']  )
        par15 = int( SETTINGS.VALUE.loc['overlap']  )  
        par16 = float( SETTINGS.VALUE.loc['recDisScanInc']  )        
        par17 = float( SETTINGS.VALUE.loc['srcDepthScanInc']  )                    
        par36 =  int( SETTINGS.VALUE.loc['sr']  )        
        par37 = float( SETTINGS.VALUE.loc['scanlowf']  )
        par38 = float( SETTINGS.VALUE.loc['scanhighf']  )        
        par39 =  int( SETTINGS.VALUE.loc['root']  )
        par40 =  int( SETTINGS.VALUE.loc['win']  )       
        par41 =  int( SETTINGS.VALUE.loc['step']  )  
        par42 =  float( SETTINGS.VALUE.loc['DetectionThreshold']  )       
        par43 =  int( SETTINGS.VALUE.loc['kurwindow']  )        
        par44 = float( SETTINGS.VALUE.loc['picklf1']  )
        par45 = float( SETTINGS.VALUE.loc['pickhf1']  )
        par46 = float( SETTINGS.VALUE.loc['picklf2']  )
        par47 = float( SETTINGS.VALUE.loc['pickhf2']  )
        par48 =  int( SETTINGS.VALUE.loc['dstop']  )     
        par49 = float( SETTINGS.VALUE.loc['highq'] )
        par50 = float( SETTINGS.VALUE.loc['lowq']  )
        par51 = float( SETTINGS.VALUE.loc['terr']  )   
        par52 = float( SETTINGS.VALUE.loc['Q_threshold'] )
        par53 = float( SETTINGS.VALUE.loc['outlier']  )
        par54 = float( SETTINGS.VALUE.loc['mindepgrid']  )
        par55 = float( SETTINGS.VALUE.loc['minimprove']  )
        par56 = SETTINGS.VALUE.loc['tableP']  
        par57 = SETTINGS.VALUE.loc['tableS']        
        
        
        
        return  par1,par5, par6, par7, par8, par9, \
                par10, par11, par12, par13, par14, par15, par16, par17, \
                par36, par37, par38, par39, \
                par40, par41, par42, par43, par44, par45, par46, par47, par48, par49, \
                par50, par51, par52, par53, par54, par55, par56, par57
               
    
    except:
        sys.exit("Errors in 'SETTINGS.txt' !\n")
        

#%%-- load input parameters
catalogPath,nametag,area,\
studydepthbottom,studydepthtop,studydepth_for_locatebottom,\
studydepth_for_locatetop,depgrid,latgrid,longrid,process_unit,overlap,\
recDisScanInc, srcDepthScanInc, sr,scanlowf,scanhighf,root,win,step,DetectionThreshold,\
kurwindow,picklf1,pickhf1,picklf2,pickhf2,dstop,highq,lowq,terr,Q_threshold,\
outlier,mindepgrid,minimprove,table_P,table_S = load_settings()

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
print( 'Ndirs =', Ndirs )


#--查看每个子目录是否为空目录
for idx, idir in enumerate( dirs ):
  if idx >= 0:
    print('\n\n\n')
    print('------------------------------------------------------------------')
    print( 'Event:', idx, ', directory name:', idir )
    dataPath = catalogPath+idir+'/'
    print( 'Path: ', dataPath )
    try:
        data = pd.read_csv( str(dataPath)+'0_StationWithHighSNRforDSA_OrgTimeGCMT_DepthGCMT.csv' )   
    except:
        print( '\t No 0_StationWithHighSNRforDSA_OrgTimeGCMT_DepthGCMT.csv!!!' )
        continue
    
    
    #%% creating output location ###################################
    outputdir = dataPath+'ssnapresults/'
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    else:
        print( outputdir+' already exists!')
        shutil.rmtree(outputdir)
        os.mkdir(outputdir)
        

    #读取csv文件
    filePathE = data['filePathE']
    filePathN = data['filePathN']
    filePathZ = data['filePathZ']
    filelat = data['lat']
    filelon = data['lon']
    filelev = data['elev']
    evLatList = data['epLat']
    evLonList = data['epLon']
    evDepList = data['evDp']
    epTime = data['epTime']
    eventLat = evLatList[0]
    eventLon = evLonList[0]
    eventDep = evDepList[0]
    time_begin = epTime[0]
    
    #%%-- 读取台站列表和相应的仪器响应
    stlas=[]
    stlos=[]
    stz=[]

    v=[]
    h1=[]
    h2=[]
    xmlfiles_v=[]
    xmlfiles_h1=[]
    xmlfiles_h2=[]
    #%%-- 读取地震文件
    for i in range(len(filePathE)):
        if Path(filePathE[i]).is_file() == True and Path(filePathN[i]).is_file() == True and Path(filePathZ[i]).is_file() == True:
            channelz=read(filePathE[i])
            channele=read(filePathN[i])
            channeln=read(filePathZ[i])
            channelz[0].detrend( type='demean')
            channele[0].detrend( type='demean')
            channeln[0].detrend( type='demean')
            channelz[0].detrend( type='simple')
            channele[0].detrend( type='simple')
            channeln[0].detrend( type='simple')

            v.append(channelz)
            h1.append(channele)
            h2.append(channeln)
            stlas.append(float(filelat[i]))
            stlos.append(float(filelon[i]))
            stz.append(float(filelev[i]))

    #%% 框定扫描网格（前后左右各一度）&input parameters ###################################################
    lonFrom = eventLon-1
    lonTo = eventLon+1
    latFrom = eventLat-1
    latTo =  eventLat+1
    studyarea=[ latFrom, lonFrom, latTo-latFrom, lonTo-lonFrom] # degree
    studydepth = [studydepthbottom , studydepthtop] # km
    studydepth_for_locate = [studydepth_for_locatebottom, studydepth_for_locatetop ] #km

    points=int(sr*(process_unit+overlap)*60)
    psseparate = 'off'
    savefigure = 'on'
    
    beginyear = time_begin[0:4]
    beginmonth = time_begin[5:7]
    beginday = time_begin[8:10]
    
    beginhour = time_begin[11:13]
    beginmin = time_begin[14:16]
    beginsecond = time_begin[17:19]
    
    time_begin=dict([
        ('year',beginyear),
        ('month',beginmonth),
        ('day',beginday),
        ('hour',beginhour),
        ('min',beginmin),
        ('second', beginsecond)
    ])
    
    
    time_end=dict([
        ('year',beginyear),
        ('month',beginmonth),
        ('day',beginday),
        ('hour',str(int(beginhour)+1)),
        ('min',beginmin),
        ('second', beginsecond)
    ])
    
    
    
    
    #%%-- 生成用于空间搜索网格点（SSA）
    lats = np.arange( studyarea[0],  studyarea[0]+studyarea[2], latgrid)
    lons = np.arange( studyarea[1],  studyarea[1]+studyarea[3], longrid)
    deps = np.arange( studydepth[0], studydepth[1], depgrid)
    
    
    #%%-- 读取P和S的走时表
    with open(str(catalogPath)+'tableP&S/'+str(table_P), 'rb') as f:
        comein=pickle.load(f)
    tableP = comein[0]
    
    with open(str(catalogPath)+'tableP&S/'+str(table_S), 'rb') as f:
        comein=pickle.load(f)
    tableS = comein[0]
    
    
    studygrids=[]
    for i in lats:
        for j in lons:
            for k in deps:
                studygrids.append( [i,j,k] )
    
    print('len(studygrids) =', len(studygrids))
    print( 'lats =', lats )
    print( 'lons =', lons )
    pickle.dump([studygrids],open(outputdir+'studygrids.p','wb'))
    
    
    
    #%%-- 生成用于空间搜索网格点（MAXI）
    deps_for_locate=np.arange(studydepth_for_locate[0],studydepth_for_locate[1], depgrid)
    studygrids_for_locate=[]
    for i in lats:
        for j in lons:
            for k in deps_for_locate:
                studygrids_for_locate.append([i,j,k])
    
    
    #%%-- 计算需要扫描的天数
    print('Scanning now:')
    totalday = int(float(time_end['day']) - float(time_begin['day']) + 1)
    for date in range(0, totalday):
    
        day = int(float(time_begin['day']) + date)
        if day < 10:
            day='0' + str(day)
        else:
            day=str(day)
  
        #%%-- 计算每个台站到每个空间搜索网格点的距离（SSA）
        traveldis=[]
        for i in studygrids:
            a=[]
            for j, k in zip( stlas, stlos ):
                a.append( distance.dis( j, k, i[0], i[1]) ) #计算大圆距离
            traveldis.append(a) # 每个 traveldis 成员包含了所有台站到某一网格点的距离
        '''
        maxt = 0
        for ins in traveldis:
            for i in range(len(ins)):
                if ins[i] >maxt:
                    maxt = ins[i]
                    print(maxt)
        '''
        #%%-- 计算每个台站到每个空间搜索网格点的距离（MAXI）
        traveldis_for_locate=[]
        for i in studygrids_for_locate:
            a=[]
            for j, k in zip( stlas, stlos):
                a.append( distance.dis( j, k, i[0], i[1] ) ) #计算大圆距离
            traveldis_for_locate.append(a) # 每个 traveldis_for_locate 成员包含了所有台站到某一网格点的距离
    
    
    
        #%%-- 计算每个台站到每个空间搜索网格点的走时（SSA）
        ptraveltimes=[]
        straveltimes=[]
        for i in range(0,len(traveldis)): # len(traveldis) = 网格点总数
            if i%500==0:
                print( 'Creating travel time tables for SSA', i, '/', len(traveldis))
            a=[]
            b=[]
            for j in range(0, len(traveldis[i])):
                timeP = taupz.taupz(tableP, tableS, studygrids[i][2], traveldis[i][j],
                                    'P', stz[j], recDisScanInc, srcDepthScanInc)
                a.append(timeP)
                timeS = taupz.taupz(tableP, tableS, studygrids[i][2], traveldis[i][j],
                                    'S', stz[j], recDisScanInc, srcDepthScanInc)
                b.append(timeS)
    
            ptraveltimes.append(a) # 每个 ptraveltimes 成员包含了所有台站到某一网格点的走时
            straveltimes.append(b)
    
        ptraveltimes = np.array(ptraveltimes)
        straveltimes = np.array(straveltimes)
        
        #%% calculate computing time
        stop = timeit.default_timer()
        elapsedTime1 = stop - start
        print('Elapsed time (Creating travel time tables for SSA): ',
              format( elapsedTime1, '.1f'), 'sec = ',
              format( elapsedTime1/60.0, '.1f'), 'min' )
    
        #%%--debug
        ''' 
        plt.plot( ptraveltimes)
        plt.show()
        plt.plot(straveltimes)
        plt.show()
        '''
    
        #%%-- 计算每个台站到每个空间搜索网格点的走时（MAXI）
        ptraveltimes_for_locate=[]
        straveltimes_for_locate=[]
        for i in range(0,len(traveldis_for_locate)):
            if i%2500==0:
                print( 'Creating travel time tables for MAXI', i, '/', len(traveldis_for_locate))
            a=[]
            b=[]
            for j in range(0, len(traveldis_for_locate[i])):
                timeP = taupz.taupz(tableP, tableS, studygrids_for_locate[i][2], 
                                    traveldis_for_locate[i][j],'P', stz[j], recDisScanInc, srcDepthScanInc)
                a.append(timeP)
                timeS = taupz.taupz(tableP, tableS, studygrids_for_locate[i][2],
                                    traveldis_for_locate[i][j],'S', stz[j], recDisScanInc, srcDepthScanInc)
                b.append(timeS)
    
            ptraveltimes_for_locate.append(a) # 每个 ptraveltimes 成员包含了所有台站到某一网格点的走时
            straveltimes_for_locate.append(b)
    
        ptraveltimes_for_locate = np.array(ptraveltimes_for_locate)
        straveltimes_for_locate = np.array(straveltimes_for_locate)
    
    
        #%% calculate computing time
        stop = timeit.default_timer()
        elapsedTime1 = stop - start
        print('Elapsed time (Creating travel time tables for MAXI): ',
              format( elapsedTime1, '.1f'), 'sec = ',
              format( elapsedTime1/60.0, '.1f'), 'min' )   
        
        
        print( 'Creating travel time tables for SSA and MAXI is done!')
        #%%-- interpolate wavefrom according to the sampling rate
        nsta=len(v)
        for i in range(0,nsta):
            if len(v[i])>1:
                for j in range(0,len(v[i])):
                    v[i][j].stats.sampling_rate=round(v[i][j].stats.sampling_rate)
                v[i].merge(method=1, fill_value='interpolate')
            v[i]=v[i][0]
            print(v[i], v[i][0])
        
            '''
            #v[i]=v[i][0]
            if v[i].stats.sampling_rate != sr:
                #v[i].interpolate(sr)
                v[i].data = signal.resample_poly(v[i].data, up=2, down=5)
                v[i].stats.sampling_rate = sr
            '''
    
            if len(h1[i])>1:
                for j in range(0,len(h1[i])):
                    h1[i][j].stats.sampling_rate=round(h1[i][j].stats.sampling_rate)
                h1[i].merge(method=1, fill_value='interpolate')
            h1[i]=h1[i][0]
    
            '''
            #h1[i]=h1[i][0]
            if h1[i].stats.sampling_rate != sr:
                h1[i].resample(sr)
            '''
    
            if len(h2[i])>1:
                for j in range(0,len(h2[i])):
                    h2[i][j].stats.sampling_rate=round(h2[i][j].stats.sampling_rate)
                h2[i].merge(method=1, fill_value='interpolate')
            h2[i]=h2[i][0]
    
            '''
            #h2[i]=h2[i][0]
            if h2[i].stats.sampling_rate != sr:
                h2[i].resample(sr)
            '''
    
    
    
        #-- 不同时间长度的情况  
        if day == time_end['day']:
            duration = int(float(time_end['hour']))
        elif day == time_begin['day']:
            duration = 24 - int(float(time_begin['hour']))
        else:
            duration = 24
    
        #-- 如果是最后一天或者只有一天的情况
        if day == time_end['day'] and day == time_begin['day']:
            
            if int(float(time_end['hour']))>int(float(time_begin['hour'])):
                duration = int(float(time_end['hour'])) - int(float(time_begin['hour']))
            else:
                duration = int(float(time_begin['hour'])) - int(float(time_end['hour']))+24
            
            print(int(float(time_begin['hour'])),int(float(time_end['hour'])),duration)
        #%%
        print('duration =', duration )
        #%%-- 对每一个小时的数据进行扫描定位
        for runhour in range( 0, duration ):
            if day == time_begin['day']:
                runh = runhour
            else:
                runh = 24 - int(float(time_begin['hour'])) + (date-1)*24 + runhour
    
            #-- 如果扫描的数据是最后一个小时的情况
            if int(float(time_begin['hour'])+runh)%24 == 23:
                vborrow = []
                h1borrow = []
                h2borrow = []
                if int(float(day)+1) < 10:
                    dayp = '0' + str(int(float(day)+1))
                else:
                    dayp = str(int(float(day)+1))
                for i in range(len(filePathE)):
                    
                    if Path(filePathE[i]).is_file() == True and Path(filePathN[i]).is_file() == True and Path(filePathZ[i]).is_file() == True:
                        channelz = read(filePathE[i])
                        channele = read(filePathN[i])
                        channeln = read(filePathZ[i])
                        vborrow.append(channelz[0])
                        h1borrow.append(channele[0])
                        h2borrow.append(channeln[0])
    
            t1 = UTCDateTime(time_begin['year']+'-'+
                             time_begin['month']+'-'+
                             str(int(float(time_begin['day'])+
                                     np.floor( float(time_begin['hour'])+runh ) /24 ) )+'T'+
                             str(int(float(time_begin['hour'])+runh)%24)+':'+time_begin['min']+':'+time_begin['second'])
    
            #%%-- 对三分量波形数据进行滤波
            vfilter = []
            h1filter = []
            h2filter = []
    
            for i in v:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour'])+runh)%24 == 23: #-- 如果扫描的数据是最后一个小时的情况
                    for look in vborrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look

                            break
                #%%-- 截取时间长度为 (process_unit + overlap) 的数据并找出中值
                try:
                    ii.trim(t1,  t1+(process_unit + overlap) * 60, pad=True, fill_value=0)
    
                    fill = np.median(ii.data)
                    zz = np.nonzero(ii.data == 0)
                    for find in zz[0]:
                        ii.data[find] = fill
                    ii.data = ii.data[0:len(ii.data)-1]
                    if ii.stats.sampling_rate != sr:
                        ii.resample(sr)
                    ii.filter(type='bandpass', freqmin=scanlowf, freqmax=scanhighf, corners=3, zerophase=True)
                    ii.detrend()
                    vfilter.append(ii.data)
                except NotImplementedError:
                    vfilter.append(np.ones(points))
    
    
            for i in h1:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour']) + runh) % 24 == 23: #-- 如果扫描的数据是最后一个小时的情况
                    for look in h1borrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.trim(t1, t1 + (process_unit + overlap) * 60, pad=True, fill_value=0)
                    fill = np.median(ii.data)
                    zz = np.nonzero(ii.data == 0)
                    for find in zz[0]:
                        ii.data[find] = fill
                    ii.data = ii.data[0:len(ii.data) - 1]
                    if ii.stats.sampling_rate != sr:
                        ii.resample(sr)
                    ii.filter(type='bandpass', freqmin=scanlowf, freqmax=scanhighf, corners=3, zerophase=True)
                    ii.detrend()
                    h1filter.append(ii.data)
                except NotImplementedError:
                    h1filter.append(np.ones(points))
    
    
            for i in h2:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour']) + runh) % 24 == 23: #-- 如果扫描的数据是最后一个小时的情况
                    for look in h2borrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.trim(t1, t1 + (process_unit + overlap) * 60, pad=True, fill_value=0)
                    fill = np.median(ii.data)
                    zz = np.nonzero(ii.data == 0)
                    for find in zz[0]:
                        ii.data[find] = fill
                    ii.data = ii.data[0:len(ii.data) - 1]
                    if ii.stats.sampling_rate != sr:
                        ii.resample(sr)
                    ii.filter(type='bandpass', freqmin=scanlowf, freqmax=scanhighf, corners=3, zerophase=True)
                    ii.detrend()
                    h2filter.append(ii.data)
                except NotImplementedError:
                    h2filter.append(np.ones(points))
    
            #%%-- 
            vn = []
            hn = []
            if psseparate == 'on': #-- 在定位之前就分离P和S波形，判断是否存在异常情况，如果有则对数据道充1，无则直接计算叠加能量
                onlyP=[]
                onlyS=[]
                number=0
                for i,j,k in zip(h1filter, h2filter, vfilter):
                    number = number + 1
                    print(number)
                    [P , S, LIN , COS] = psseparation(np.array([i,j,k]), 80)
                    onlyP.append(np.array(P))
                    onlyS.append(np.array(S))
    
    
                for i in onlyP:
                    i=i-np.mean(i)
                    a=abs(i)
                    if math.isnan(np.median(a)) == True or np.median(a) == 0:
                        a = np.ones(points)
                    if len(a) < points:
                        a = np.ones(points)
                    if len(a) > points:
                        a=a[0: points]
    
                    for j in range(0,process_unit+overlap):
                        a[j *60* sr:(j + 1)*60*sr]=a[j*60*sr:(j+1)*60*sr]/np.median(a[j*60*sr:(j+1)*60*sr])
                    a=a**(1/root)
                    vn.append(a)
    
                for i in onlyS:
                    i=i-np.mean(i)
                    a=abs(i)
                    if math.isnan(np.median(a)) == True or np.median(a) == 0:
                        a = np.ones(points)
                    if len(a) < points:
                        a = np.ones(points)
                    if len(a) > points:
                        a=a[0: points]
    
                    for j in range(0,process_unit+overlap):
                        a[j *60* sr:(j + 1)*60*sr]=a[j*60*sr:(j+1)*60*sr]/np.median(a[j*60*sr:(j+1)*60*sr])
                    a=a**(1/root)
                    hn.append(a)
            else: #-- 对每一分钟的数据进行判断是否有异常情况，如果有则对数据道充1，无则直接计算叠加能量
                count=0
                for i in vfilter:
                    a=abs(i)
                    if math.isnan(np.median(a)) == True or np.median(a) == 0:
                        a = np.ones(points)
                    if len(a) < points:
                        a = np.ones(points)
                    if len(a) > points:
                        a=a[0: points]
                    #-- 计算每一分钟的能量（Tan et al. 2019的公式4）
                    for j in range(0,process_unit+overlap):
                        a[j*60*sr:(j+1)*60*sr]=a[j*60*sr:(j+1)*60*sr]/np.median(a[j*60*sr:(j+1)*60*sr])
                    a=a**(1/root)
                    vn.append(a)
    
                for i,j in zip(h1filter, h2filter):
    
                    if math.isnan(np.median(i)) == True or np.median(i) == 0:
                        i = np.ones(points)
                    if len(i) < points:
                        i = np.ones(points)
                    if len(i) > points:
                        i=i[0: points]
    
                    if math.isnan(np.median(j)) == True or np.median(j) == 0:
                        j = np.ones(points)
                    if len(j) < points:
                        j = np.ones(points)
                    if len(j) > points:
                        j=j[0: points]
    
                    #-- 计算每一分钟的能量（两个水平分量的贡献为0.5）
                    a=(i**2+j**2)**0.5
    
                    for j in range(0,process_unit+overlap):
                        a[j *60* sr:(j + 1)*60*sr]=a[j*60*sr:(j+1)*60*sr]/np.median(a[j*60*sr:(j+1)*60*sr])
                    a=a**(1/root)
                    hn.append(a)
    
                    
    
    
            #%%-- Step 1: Preliminary Source Scanning （通过改进版的SSA）
            print('Step 1: Preliminary Source Scanning...')
            #-- 计算叠加能量分布
            print('\t P part ...')
            brp = ssa.med_scan(ptraveltimes, np.array(vn), sr, win, step)
            brmapp = ((brp / (win * sr)) ** 0.5 / nsta) ** root
            print('\t S part ...')
            brs = ssa.med_scan(straveltimes, np.array(hn), sr, win, step)
            brmaps = ((brs / (win * sr)) ** 0.5 / nsta) ** root
            
            pickle.dump([brmapp], open(outputdir+day+str(int(float(time_begin['hour'])+runh)%24)+'ssabrmap_P.p', 'wb'))
            pickle.dump([brmaps], open(outputdir+day+str(int(float(time_begin['hour'])+runh)%24)+'ssabrmap_S.p', 'wb'))
             
            #-- 计算总的叠加能量分布（P和S，文章里面的公式5）
            shape_s = np.shape(brmaps)
            brmapp = brmapp[:, 0:shape_s[1]]
            brmap = np.multiply(brmaps, brmapp)
    
    
    
            #%% calculate computing time
            stop = timeit.default_timer()
            elapsedTime1 = stop - start
            print('Elapsed time (Step 1: Preliminary Source Scanning): ',
                  format( elapsedTime1, '.1f'), 'sec = ',
                  format( elapsedTime1/60.0, '.1f'), 'min' )
            
            
            #%%-- 获取每一分钟数据段的能量最大值（亮点），图像横坐标为每一分钟的计数，纵坐标为能量
            brmax = []
            for i in range(0, len(brmap[0])):
                brmax.append(max(brmap[:, i]))
            plt.figure( constrained_layout=True, figsize=(8,4))
            plt.clf()
            xBr = np.arange( 0, len(brmax), 1 )
            plt.semilogy( xBr*step, brmax, color='black', linestyle='-', lw=1)
            plt.margins(x=0)
            plt.axhline(y=DetectionThreshold, color='red')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.xlabel('Scanning time (s)', fontsize=14)
            plt.ylabel('Maximum brightness', fontsize=14)
            plt.savefig(outputdir+day+str(int(float(time_begin['hour'])+runh)%24)+'brmax.png', dpi=600)
            #plt.savefig(outputdir+day+str(int(float(time_begin['hour'])+runh)%24)+'brmax.svg', dpi=600)
            plt.show()
            
            brmaxFileName = str(outputdir)+'0_brmax.csv'
            with open( brmaxFileName, mode='w', newline='' ) as outFile:
                writer = csv.writer( outFile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                writer.writerow( [ 'sampleIdx', 'dt(s)', 'time(s)', 'brmax' ] )
                for i in range( len(brmax) ):
                    writer.writerow(['{0}'.format( i ),
                                     '{0}'.format( format(   step,  '.2f' ) ),
                                     '{0}'.format( format( i*step,  '.2f' ) ),
                                     '{0}'.format( format( brmax[i],'.6f')) ])
            outFile.close()
 

            
            #%% 获取能量亮点曲线满足阈值mph的峰值，其位置索引记为出现的时间（以分钟为单位）
            peaktimes = detect_peaks(x=brmax, mph=DetectionThreshold, mpd=0)
            print('peaktimes [index]', peaktimes)
    
            peakheight = []
            for i in peaktimes:
                peakheight.append((brmax[i], i))
    
            order = sorted(peakheight, key=itemgetter(0), reverse=True)
    
            peaks = []
            for i in order:
                peaks.append( brmap[:, i[1]] )
    
            pp = []
            for i in peaks:
                m = max(i)
                p = [j for j, k in enumerate(i) if k == m]
                pp.append(p)
            print("Event candidates's index = ", pp )

            #-- get event candidates
            evlas = []
            evlos = []
            evdeps = []
            for i in pp:
                evlas.append(studygrids[i[0]][0])
                evlos.append(studygrids[i[0]][1])
                evdeps.append(studygrids[i[0]][2])
    
            print(evlas,evlos,evdeps)
    
    
    
            #%%-- Step 2: Kurtosis-based phase picking
            print('Step 2: Kurtosis picking...')
            vfilter = []
            h1filter = []
            h2filter = []
    
            vfilter2 = []
            h1filter2 = []
            h2filter2 = []
    
            for i in v:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour'])+runh)%24 == 23:
                    for look in vborrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.trim(t1, t1 + (process_unit + overlap) * 60, pad=True, fill_value=0)
                    fill = np.median(ii.data)
                    zz = np.nonzero(ii.data == 0)
                    for find in zz[0]:
                        ii.data[find] = fill
                    ii.data = ii.data[0:len(ii.data) - 1]
                    if ii.stats.sampling_rate != sr:
                        ii.resample(sr)
                    ii.filter(type='bandpass', freqmin=picklf2, freqmax=pickhf2, corners=3, zerophase=True)
                    ii.detrend()
                    vfilter.append(ii.data)
                except NotImplementedError:
                    vfilter.append(np.ones(points))
    
            for i in h1:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour'])+runh)%24 == 23:
                    for look in h1borrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.trim(t1, t1 + (process_unit + overlap) * 60, pad=True, fill_value=0)
                    fill = np.median(ii.data)
                    zz = np.nonzero(ii.data == 0)
                    for find in zz[0]:
                        ii.data[find] = fill
                    ii.data = ii.data[0:len(ii.data) - 1]
                    if ii.stats.sampling_rate != sr:
                        ii.resample(sr)
                    ii.filter(type='bandpass', freqmin=picklf2, freqmax=pickhf2, corners=3, zerophase=True)
                    ii.detrend()
                    h1filter.append(ii.data)
                except NotImplementedError:
                    h1filter.append(np.ones(points))
    
            for i in h2:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour'])+runh)%24 == 23:
                    for look in h2borrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.trim(t1, t1 + (process_unit + overlap) * 60, pad=True, fill_value=0)
                    fill = np.median(ii.data)
                    zz = np.nonzero(ii.data == 0)
                    for find in zz[0]:
                        ii.data[find] = fill
                    ii.data = ii.data[0:len(ii.data) - 1]
                    if ii.stats.sampling_rate != sr:
                        ii.resample(sr)
                    ii.filter(type='bandpass', freqmin=picklf2, freqmax=pickhf2, corners=3, zerophase=True)
                    ii.detrend()
                    h2filter.append(ii.data)
                except NotImplementedError:
                    h2filter.append(np.ones(points))
    
            for i in v:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour'])+runh)%24 == 23:
                    for look in vborrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.trim(t1, t1 + (process_unit + overlap) * 60, pad=True, fill_value=0)
                    fill = np.median(ii.data)
                    zz = np.nonzero(ii.data == 0)
                    for find in zz[0]:
                        ii.data[find] = fill
                    ii.data = ii.data[0:len(ii.data) - 1]
                    if ii.stats.sampling_rate != sr:
                        ii.resample(sr)
                    ii.filter(type='bandpass', freqmin=picklf1, freqmax=pickhf1, corners=3, zerophase=True)
                    ii.detrend()
                    vfilter2.append(ii.data)
                except NotImplementedError:
                    vfilter2.append(np.ones(points))
    
            for i in h1:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour'])+runh)%24 == 23:
                    for look in h1borrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.trim(t1, t1 + (process_unit + overlap) * 60, pad=True, fill_value=0)
                    fill = np.median(ii.data)
                    zz = np.nonzero(ii.data == 0)
                    for find in zz[0]:
                        ii.data[find] = fill
                    ii.data = ii.data[0:len(ii.data) - 1]
                    if ii.stats.sampling_rate != sr:
                        ii.resample(sr)
                    ii.filter(type='bandpass', freqmin=picklf1, freqmax=pickhf1, corners=3, zerophase=True)
                    ii.detrend()
                    h1filter2.append(ii.data)
                except NotImplementedError:
                    h1filter2.append(np.ones(points))
    
            for i in h2:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour'])+runh)%24 == 23:
                    for look in h2borrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.trim(t1, t1 + (process_unit + overlap) * 60, pad=True, fill_value=0)
                    fill = np.median(ii.data)
                    zz = np.nonzero(ii.data == 0)
                    for find in zz[0]:
                        ii.data[find] = fill
                    ii.data = ii.data[0:len(ii.data) - 1]
                    if ii.stats.sampling_rate != sr:
                        ii.resample(sr)
                    ii.filter(type='bandpass', freqmin=picklf1, freqmax=pickhf1, corners=3, zerophase=True)
                    ii.detrend()
                    h2filter2.append(ii.data)
                except NotImplementedError:
                    h2filter2.append(np.ones(points))
    
            #-- pick phase
            [ponsets_final, sonsets_final] = picking.pick(points,
                outputdir+day+str(int(float(time_begin['hour'])+runh)%24)+'_',nametag,
                nsta, order, pp, ptraveltimes, straveltimes, step, win, kurwindow,
                sr, dstop, vfilter, h1filter, h2filter, vfilter2, h1filter2, h2filter2,
                savefigure, overlap)
    
            
            #%% calculate computing time
            stop = timeit.default_timer()
            elapsedTime1 = stop - start
            print('Elapsed time (Step 2: Kurtosis picking): ',
                  format( elapsedTime1, '.1f'), 'sec = ',
                  format( elapsedTime1/60.0, '.1f'), 'min' )
    
    
            
            #%%-- Step 3: MAXI locating
            print('Step 3: MAXI locating...')
            [events, MAXI, catalog] = locatePS.MAXI_locate(ponsets_final, sonsets_final,
                ptraveltimes_for_locate, straveltimes_for_locate, stlas, stlos, stz, lowq,
                highq, studygrids_for_locate, Q_threshold, terr, outlier, tableP, tableS, 
                latgrid, longrid, depgrid, mindepgrid, minimprove, outputdir, recDisScanInc, srcDepthScanInc)

            #-- 对第二步够拾取数的每一个地震进行计算交叉点数，每一列或者每一行对应一个地震（得测试），按照studygrid_for_locate得网格点顺序
            pickle.dump([MAXI],open(outputdir +day+str(int(float(time_begin['hour'])+runh)%24)+ 'MAXIvalue' + nametag +'.p','wb'))

            #%% calculate computing time
            stop = timeit.default_timer()
            elapsedTime1 = stop - start
            print('Elapsed time (Step 3: MAXI locating): ',
                  format( elapsedTime1, '.1f'), 'sec = ',
                  format( elapsedTime1/60.0, '.1f'), 'min' ) 
    
    
            
            vraw=[]
            h1raw=[]
            h2raw=[]
            for i in v:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour'])+runh)%24 == 23:
                    for look in vborrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.trim(t1, t1 + (process_unit + overlap) * 60, pad=True, fill_value=0)
                    fill = np.median(ii.data)
                    zz = np.nonzero(ii.data == 0)
                    for find in zz[0]:
                        ii.data[find] = fill
                    ii.data = ii.data[0:len(ii.data) - 1]
                    if ii.stats.sampling_rate != sr:
                        ii.resample(sr)
                    ii.detrend()
                except NotImplementedError:
                    ii.data=np.zeros(points)
    
                vraw.append(ii)
    
    
            for i in h1:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour'])+runh)%24 == 23:
                    for look in h1borrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.trim(t1, t1 + (process_unit + overlap) * 60, pad=True, fill_value=0)
                    fill = np.median(ii.data)
                    zz = np.nonzero(ii.data == 0)
                    for find in zz[0]:
                        ii.data[find] = fill
                    ii.data = ii.data[0:len(ii.data) - 1]
                    if ii.stats.sampling_rate != sr:
                        ii.resample(sr)
                    ii.detrend()
                except NotImplementedError:
                    ii.data = np.zeros(points)
    
                h1raw.append(ii)
    
            for i in h2:
                ii = copy.deepcopy(i)
                if int(float(time_begin['hour'])+runh)%24 == 23:
                    for look in h2borrow:
                        if look.stats.station == ii.stats.station:
                            ii = ii + look
                            break
                try:
                    ii.trim(t1, t1 + (process_unit + overlap) * 60, pad=True, fill_value=0)
                    fill = np.median(ii.data)
                    zz = np.nonzero(ii.data == 0)
                    for find in zz[0]:
                        ii.data[find] = fill
                    ii.data = ii.data[0:len(ii.data) - 1]
                    if ii.stats.sampling_rate != sr:
                        ii.resample(sr)
                    ii.detrend()
                except NotImplementedError:
                    ii.data = np.zeros(points)
    
                h2raw.append(ii)
    
    
    
            #%%-- Step 4: Calculate Magnitude
            magnitude = calc_mag.local_magnitude(sr, stz, stlas, stlos, xmlfiles_v, xmlfiles_h1, xmlfiles_h2, catalog, events, sonsets_final, vraw, h1raw, h2raw, points)

            
            high = []
            low = []
            for i in magnitude:
                if len(i) == 8 and i[6] >= highq * 2:
                    high.append(i)
                else:
                    low.append(i)
    
            cataloghigh = sorted(high, key=itemgetter(0), reverse=False)
            cataloglow = sorted(low, key=itemgetter(0), reverse=False)
            for i in range(0, len(cataloghigh)):
                #cataloghigh[i][0] = str(int(np.floor(cataloghigh[i][0] / 60))) + ':' + str(cataloghigh[i][0] % 60)
                
                evOnset = UTCDateTime(time_begin['year']+'-'+
                                      time_begin['month']+'-'+
                                      time_begin['day']+'T'+
                                      time_begin['hour']+'-'+
                                      time_begin['min']+'-'+
                                      time_begin['second'])+cataloghigh[i][0]
                cataloghigh[i][0] = str(evOnset)
                
            for i in range(0, len(cataloglow)):    
                #cataloglow[i][0] = str(int(np.floor(cataloglow[i][0] / 60))) + ':' + str(cataloglow[i][0] % 60)
                evOnset = UTCDateTime(time_begin['year']+'-'+
                                      time_begin['month']+'-'+
                                      time_begin['day']+'T'+
                                      time_begin['hour']+'-'+
                                      time_begin['min']+'-'+
                                      time_begin['second'])+cataloglow[i][0]
                cataloglow[i][0] = str(evOnset) 
            f = open(outputdir + day + str(int(float(time_begin['hour'])+runh)%24) + 'cataloghigh_' + nametag + '.txt', 'w')
            for i in cataloghigh:
                for j in i:
                    f.write(str(j))
                    f.write(' ')
                f.write('\n')
            f.close()
    
            f = open(outputdir + day + str(int(float(time_begin['hour'])+runh)%24) + 'cataloglow_' + nametag + '.txt', 'w')
            for i in cataloglow:
                for j in i:
                    f.write(str(j))
                    f.write(' ')
                f.write('\n')
            f.close()

    
    
    
            #%% calculate computing time
            stop = timeit.default_timer()
            elapsedTime1 = stop - start
            print('Elapsed time (total): ',
                  format( elapsedTime1, '.1f'), 'sec = ',
                  format( elapsedTime1/60.0, '.1f'), 'min' )         
        #%% plot brmap slices
        #plotbrmap(dataPath,peaktimes)
        # plotMAXI(dataPath)



    
    
    