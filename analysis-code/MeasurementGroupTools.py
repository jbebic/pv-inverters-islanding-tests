#!/usr/bin/env python

import numpy as np    
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as dpdf

from numpy import cos, sin, arctan2, zeros, ones, pi, sqrt, mod

def ProcessResults(mypath, TestLog,
                   VMAG = {'icsLvl': 3.0, 'delta':0.1, 'low':0.1, 'uImagLim':10.0},
                   NCY = {'pre':1., 'post': 0.2}, # {'pre': 3., 'post': 5.}
                   MISC = {'tol':0.001},
                   AMP = {'L_IntRate':5856.0, 'Pcutin':50.0, 'Pcutout':5.0}, # switching frequency and Pcutout
                   BASE = {'Vln':277.128, 'freq':60.},
                   LIMIT = {'iWmin': 0.9, 'iWmax': 1.1} ):
    """ Reads and merges TDMS files, 
        calculates additional signals, 
        detects time to island formation to cessasion, 
        calculates scalar atributes of the islanding tests        
        plots grouped results into multipage Results.pdf, and 
        saves grouped results into Results.h5

    Input: Directory with the test result files, e.g.: "aLabView\\20150306"

    """
#   import pdb # debugger
    import datetime 
    import pandas as pd # multidimensional data analysis

    from os import listdir
    from os.path import isfile, join

    from nptdms import TdmsFile
    # from numpy import array
    from pandas import concat, ExcelWriter, rolling_mean, Timestamp
    from scipy import fftpack

    CONFIG = {'WriteAllDataToExcel':False, # Will take forever ~5min for 27GB
              'WriteLimitedDataToExcel':False, # Only data from island creation to cessation
              'WriteSummaryToExcel':True, # Summary from TDMS files
              'WriteDataToHDF5':True, # Test results consolidated in h5 format
              'ValidateCTs':False, # Plot pages that validate CT scaling and orientation
              'UseRAF':True, # Use Rolling average filtering on current and voltage signals
              'UseIslandContactor':False,
              'UseIslandVmag':False,
              'PlotFullRange': False} # Add a page with full time range of data

    # BASE = {'Vln':480/sqrt(3)} # Voltage base

    # B2 LC1 CT group was reversed during calibration
    B2LC1SIGN = -1.0 #  
    
    # LC1 B1 CT was reversed on 20150311, restored, then reversed again during PG&E CT calibration
    B1LC1SIGN = -1.0 # reversed CT. Use +1 for correct polarity 
    # Limiting plot range of acquired signals
    # Islanding detection works by comparing: 'Island Contactor status' > icsLvl # abs(uVmag-iVmag)>delta
    # Collapse detection works by comparing: iVmag<low
    # VMAG = {'icsLvl': 3, 'delta':0.1, 'low':0.1} # island contactor status Level, Signal magnitudes in p.u. to limit plot range
    # NCY = {'pre':3, 'post': 5} # Number of cycles to show pre-islanding and post-collapse

    # mypath = 'aLabView\\20150311' # Now a function parameter
    myfiles = [f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    # filtering for .tdms extension
    tdmsfiles_list = [f for f in myfiles if f.endswith(".tdms")]

    # empty dictionaries
    ch_data   = {} # interim results
    sig_info  = {} # fname -> DataFrame 
    sig_data  = {} # fname -> DataFrame time,signal,...,signal
    file_info = {} # fname -> DataFrame Comment

    fiComment = [] # list to collect all file comments
    fiProgRev = [] # ditto but program version
    fitStart  = [] # TDMS tStart
    fitEnd    = [] # TDMS tStart

    # Cycling through files to figure out how many are concatenated
    for fname in tdmsfiles_list:
        tdms_file = TdmsFile(mypath + "\\" + fname)
        # fetching comments
        tdms_root = tdms_file.object()
        fiComment = fiComment + [tdms_root.property(u'Comment')]
        fiProgRev = fiProgRev + [tdms_root.property(u'Program Revision')]

        # groups_list = tdms_file.groups() # There is only one group 'Analog'
        channels_list = tdms_file.group_channels(u'Analog') # u for unicode.
        # pdb.set_trace() # Debugger stop if desired
        # ch_names = [ch.property(u'NI_ChannelName') for ch in channels_list]
        ch_names = [ch.path.split("/")[2].strip("' ") for ch in channels_list]
        ch_slope = [ch.property(u'NI_Scale[0]_Linear_Slope') for ch in channels_list]
        ch_icept = [ch.property(u'NI_Scale[0]_Linear_Y_Intercept') for ch in channels_list]
        ch_tstrt = [ch.property(u'wf_start_time').replace(tzinfo=None) for ch in channels_list]
        ch_tincr = [ch.property(u'wf_increment') for ch in channels_list]
        ch_tsamp = [ch.property(u'wf_samples') for ch in channels_list]
        ch_toffs = [ch.property(u'wf_start_offset') for ch in channels_list]
        ch_tend  = [ch.property(u'wf_start_time').replace(tzinfo=None) +
                    datetime.timedelta(
                        seconds=ch.property(u'wf_increment')*
                                ch.property(u'wf_samples'))
                    for ch in channels_list]
        ch_scld  = [ch.property(u'NI_Scaling_Status')!=u'unscaled' for ch in channels_list]
        # pack all this into a dataframe
        sig_info[fname] = pd.DataFrame({
                'chName':   ch_names,
                'chScaled': ch_scld,
                'chScale':  ch_slope,
                'chIcept':  ch_icept,
                'chTstart': ch_tstrt,
                'chTend':   ch_tend,
                'chTincr':  ch_tincr},
                columns=['chName',
                         'chScaled',
                         'chScale',
                         'chIcept',
                         'chTstart',
                         'chTend',
                         'chTincr'])

        ch_data['Time'] = ch.time_track()
        for ch in channels_list:
            # ch_data[ch.property(u'NI_ChannelName')] = ch.data*ch.property(u'NI_Scale[0]_Linear_Slope')
            ch_data[ch.path.split("/")[2].strip("' ")] = ch.data*ch.property(u'NI_Scale[0]_Linear_Slope')

        sig_data[fname] = pd.DataFrame(ch_data,columns=['Time']+ch_names)

    file_info = pd.DataFrame({
                'fiComment': fiComment,
                'fiProgRev': fiProgRev},
                columns=['fiComment',
                         'fiProgRev'],
                index=tdmsfiles_list)

    # Concatenating files that have a matching chTstart chTend
    keys = sorted(sig_info.keys())
    flast = keys[0]
    df1 = sig_info[flast]
    tStartLast = df1[df1.chName == u'Utility Bus V A'].chTstart.values[0]
    tEndLast   = df1[df1.chName == u'Utility Bus V A'].chTend.values[0]
    for fname in keys[1:]:
        df1 = sig_info[fname]
        tStart = df1[df1.chName == u'Utility Bus V A'].chTstart.values[0]
        tEnd   = df1[df1.chName == u'Utility Bus V A'].chTend.values[0]
        if(tEndLast == tStart):
            # merge files
            print tEndLast
            print tStart
            print fname + " continues " + flast
            dtLast = tEndLast-tStartLast
            sig_data[fname].Time += dtLast / np.timedelta64(1, 's')
            # sig_data[fname].Time += datetime.timedelta.total_seconds(tEndLast-tStartLast)
            sig_data[flast] = concat([sig_data[flast],sig_data[fname]],ignore_index=True)
            del sig_data[fname] # removes object from dictionary
            sig_info[flast].chTend = sig_info[fname].chTend
            del sig_info[fname]
            file_info = file_info.drop(fname)
            tEndLast = tEnd
        else:
            tStartLast = tStart
            tEndLast = tEnd
            flast = fname

    if CONFIG['WriteSummaryToExcel']:
        writer = ExcelWriter(mypath + '\\MergeSummary.xlsx')
        file_info.to_excel(writer,'file_info')
        if True: # error due to time zone awareness in LabView time stamps
            for fname in file_info.index.values.tolist():
                sig_info[fname].to_excel(writer,fname)
        writer.save()

    if CONFIG['WriteAllDataToExcel']: # This takes forever -- 5min for ~27GB
        writer = ExcelWriter(mypath + '\\AllData.xlsx')
        for fname in file_info.index.values.tolist():
            temp = sig_data[fname].replace(tzinfo=None)
            temp.to_excel(writer,fname)
            # sig_data[fname].to_excel(writer,fname)
        writer.save()

    # Only the data from island formation to cessation
    if CONFIG['WriteLimitedDataToExcel']: # file is open here, but written from within the plot loop
        print "Opening: LimitedData.xlsx"
        writer = ExcelWriter(mypath + '\\LimitedData.xlsx')

    if CONFIG['WriteDataToHDF5']: # file is open here, but written from within the plot loop
        print "Opening: Results.h5"
        h5results = pd.HDFStore(mypath + '\\Results.h5')

    # Preparing for plot output
    print "Opening: Results.pdf"
    pltPdf = dpdf.PdfPages(mypath + '\\Results.pdf')

    # Processing and plotting files
    # list of files to consider
    file_list = file_info.index.values.tolist();
    for fname in file_list:
    # for fname in [file_list[0]]:
        print "Processing: " + fname
        # Looking for file number in TestLog
        fnum = fname.split(" ")[0].lstrip("0")
        fnumix = TestLog[TestLog['LVFiNums'].str.contains(fnum)].index
        if fnumix.size == 0:
            # file not found
            print "Warning: " + fname + " is not in the TestLog\n"
            continue # skips the result files not listed in the TestLog
#        elif fnumix.size > 1:
#            # more than one file found
#            print "Warning: " + fname + " was found " + str(fnumix.size) + " times in TestLog\n"
        else:
            TestLog.loc[fnumix, ('lvComment')] = file_info['fiComment'][fname]
            TestLog.loc[fnumix, ('lvProgRev')] = file_info['fiProgRev'][fname]
            TestLog.loc[fnumix, ('TDMSfname')] = fname
            TestLog.loc[fnumix, ('TDMSfnum')]  = fnum
            TestLog.loc[fnumix, ('tStart')]    = Timestamp(sig_info[fname].chTstart.values[0])
            TestLog.loc[fnumix, ('tEnd')]      = Timestamp(sig_info[fname].chTend.values[0])
            TestLog.loc[fnumix, ('ixflags')]   = 0

        # Utility voltage magnitude: alpha beta -> mag
        if not CONFIG['UseRAF']:
            uVa=sig_data[fname][u'Utility Bus V A'].values
            uVb=sig_data[fname][u'Utility Bus V B'].values
            uVc=sig_data[fname][u'Utility Bus V C'].values
        else:
            # rolling average filtering to suppress 5856Hz switching frequency
            tinc = sig_info[fname]['chTincr'][sig_info[fname]['chName']==u'Utility Bus V A'].values[0]
            uVa = rolling_mean(sig_data[fname][u'Utility Bus V A'],1./AMP['L_IntRate']*2./tinc).values
            uVb = rolling_mean(sig_data[fname][u'Utility Bus V B'],1./AMP['L_IntRate']*2./tinc).values
            uVc = rolling_mean(sig_data[fname][u'Utility Bus V C'],1./AMP['L_IntRate']*2./tinc).values
        
        uVal = uVa - 0.5 * (uVb + uVc)
        uVbe = sqrt(3.)/2. * (uVb - uVc)
        uVmag = 2./3.*sqrt(uVal*uVal+uVbe*uVbe)
        sig_data[fname][u'Utility Vmag'] = pd.Series(uVmag,index=sig_data[fname].index)

        # Island voltage magnitude: alpha beta -> mag
        if not CONFIG['UseRAF']:
            iVa=sig_data[fname][u'Island Bus V A'].values
            iVb=sig_data[fname][u'Island Bus V B'].values
            iVc=sig_data[fname][u'Island Bus V C'].values
        else:
            # rolling average filtering to suppress 5856Hz switching frequency
            tinc = sig_info[fname]['chTincr'][sig_info[fname]['chName']==u'Island Bus V A'].values[0]
            iVa = rolling_mean(sig_data[fname][u'Island Bus V A'],1./AMP['L_IntRate']*2./tinc).values
            iVb = rolling_mean(sig_data[fname][u'Island Bus V B'],1./AMP['L_IntRate']*2./tinc).values
            iVc = rolling_mean(sig_data[fname][u'Island Bus V C'],1./AMP['L_IntRate']*2./tinc).values

        iVal = iVa - 0.5 * (iVb + iVc)
        iVbe = sqrt(3.)/2. * (iVb - iVc)
        iVmag = 2./3.*sqrt(iVal*iVal+iVbe*iVbe)
        sig_data[fname][u'Island Val'] = pd.Series(iVal,index=sig_data[fname].index)
        sig_data[fname][u'Island Vbe'] = pd.Series(iVbe,index=sig_data[fname].index)
        sig_data[fname][u'Island Vmag'] = pd.Series(iVmag,index=sig_data[fname].index)

        # Island voltage frequency calculations using PLL. Must execute in a for loop, can't vectorize
        L_VlnIn = 480*sqrt(2.)/sqrt(3.)
        Pll_BW = 4.0*377
        GmPllWn = .725*Pll_BW
        GmPllPrpGn = Pll_BW/L_VlnIn
        GmPllIntGn = GmPllWn*GmPllWn/L_VlnIn
        GmPllWInt = 377.
        GmPllWIntMx =  2.5*GmPllWInt
        GmPllWIntMn = -0.5*GmPllWInt
        GmPllW = 377.
        L_DelTm1 = sig_info[fname].chTincr.values[0] # Taking the first channel's time increment
        if not CONFIG['UseRAF']:
            ix0PLL = 0
        else:
            ix0PLL = int(1./AMP['L_IntRate']*2./tinc)+1
        GmAngElecFbk = -arctan2(iVbe[ix0PLL],iVal[ix0PLL])
        iVx   = zeros(iVa.shape) # setting output arrays to zero
        iVy   = zeros(iVa.shape)
        iWpll = ones(iVa.shape)*377.0

        for i in range(ix0PLL,iVa.shape[0]):
            # calculate angle
            GmPllDelAng  = L_DelTm1*GmPllW;
            GmAngElecFbk = mod(GmAngElecFbk + GmPllDelAng, 2*pi)
            
            # Calculate voltage transform
            iVx[i] =  iVal[i]*cos(GmAngElecFbk) + iVbe[i]*sin(GmAngElecFbk)
            iVy[i] = -iVal[i]*sin(GmAngElecFbk) + iVbe[i]*cos(GmAngElecFbk)
            # calculate voltage error
            GmPllVyErr = -iVy[i]
            # Calculate integral term, clamp
            GmPllWInt = GmPllWInt + GmPllIntGn*L_DelTm1*GmPllVyErr
            if (GmPllWInt > GmPllWIntMx): 
                GmPllWInt = GmPllWIntMx
            if (GmPllWInt < GmPllWIntMn):
                GmPllWInt = GmPllWIntMn
            # Calculate PLL frequency, clamp
            GmPllW = GmPllWInt + GmPllVyErr*GmPllPrpGn;
            if (GmPllW > GmPllWIntMx): 
                GmPllW = GmPllWIntMx
            if (GmPllW < GmPllWIntMn):
                GmPllW = GmPllWIntMn
            iWpll[i] = GmPllWInt

        sig_data[fname][u'Island Vx']   = pd.Series(iVx,  index=sig_data[fname].index)
        sig_data[fname][u'Island Vy']   = pd.Series(iVy,  index=sig_data[fname].index)
        sig_data[fname][u'Island Wpll'] = pd.Series(iWpll,index=sig_data[fname].index)
            
        # Island voltage rms values using rolling_mean of squared signals
        iVa2 = iVa*iVa
        iVb2 = iVb*iVb
        iVc2 = iVc*iVc
        sig_data[fname][u'Island Va^2'] = pd.Series(iVa2,index=sig_data[fname].index)
        sig_data[fname][u'Island Vb^2'] = pd.Series(iVb2,index=sig_data[fname].index)
        sig_data[fname][u'Island Vc^2'] = pd.Series(iVc2,index=sig_data[fname].index)

        tinc = sig_info[fname]['chTincr'][sig_info[fname]['chName']==u'Island Bus V A'].values[0]
        Varms = sqrt(rolling_mean(sig_data[fname][u'Island Va^2'],1./60./tinc).values)
        Vbrms = sqrt(rolling_mean(sig_data[fname][u'Island Vb^2'],1./60./tinc).values)
        Vcrms = sqrt(rolling_mean(sig_data[fname][u'Island Vc^2'],1./60./tinc).values)
        iFr1cyRAF = rolling_mean(sig_data[fname][u'Island Wpll']/(2*pi),1./60./tinc).values
        sig_data[fname][u'Island Varms'] = pd.Series(Varms,index=sig_data[fname].index)
        sig_data[fname][u'Island Vbrms'] = pd.Series(Vbrms,index=sig_data[fname].index)
        sig_data[fname][u'Island Vcrms'] = pd.Series(Vcrms,index=sig_data[fname].index)
        sig_data[fname][u'Island freq']  = pd.Series(iFr1cyRAF,index=sig_data[fname].index)
        
        if False: # Incorrect calculations
            # Island voltage sequence components based on rms values
            Vposx = Varms - Vbrms*cos(pi/3)*cos(2*pi/3) + Vbrms*sin(pi/3)*sin(2*pi/3) - Vcrms*cos(pi/3)*cos(4*pi/3) - Vcrms*sin(pi/3)*sin(4*pi/3)
            Vposy =       - Vbrms*cos(pi/3)*sin(2*pi/3) - Vbrms*sin(pi/3)*cos(2*pi/3) - Vcrms*cos(pi/3)*sin(4*pi/3) + Vcrms*sin(pi/3)*cos(4*pi/3)
            Vpos = sqrt(Vposx*Vposx+Vposy*Vposy)/3
            Vnegx = Varms - Vbrms*cos(pi/3)*cos(4*pi/3) + Vbrms*sin(pi/3)*sin(4*pi/3) - Vcrms*cos(pi/3)*cos(2*pi/3) - Vcrms*sin(pi/3)*sin(2*pi/3)
            Vnegy =       - Vbrms*cos(pi/3)*sin(4*pi/3) - Vbrms*sin(pi/3)*cos(4*pi/3) - Vcrms*cos(pi/3)*sin(2*pi/3) + Vcrms*sin(pi/3)*cos(2*pi/3)
            Vneg = sqrt(Vnegx*Vnegx+Vnegy*Vnegy)/3
            Vzerx = Varms - Vbrms*cos(pi/3) - Vcrms*cos(pi/3)
            Vzery =       - Vbrms*sin(pi/3) + Vcrms*sin(pi/3)
            Vzer  = sqrt(Vzerx*Vzerx+Vzery*Vzery)/3
            sig_data[fname][u'Island Vpos'] = pd.Series(Vpos,index=sig_data[fname].index)
            sig_data[fname][u'Island Vneg'] = pd.Series(Vneg,index=sig_data[fname].index)
            sig_data[fname][u'Island Vzer'] = pd.Series(Vzer,index=sig_data[fname].index)

        if True: # Correct calaculations
            iV1mag = np.ones_like(iVa)*(-1.)*sqrt(2.)*BASE['Vln']
            iV2mag = np.ones_like(iVa)*(-1.)
            iV0mag = np.ones_like(iVa)*(-1.)
            # iV1ang = np.zeroes_like(iVa)
            # iV2ang = np.zeroes_like(iVa)
            # iV0ang = np.zeroes_like(iVa)

            for i in range(ix0PLL,iVa.shape[0]):
                # limiting iWpll[i]
                iWmax = 2.*np.pi*BASE['freq']*LIMIT['iWmax']
                iWmin = 2.*np.pi*BASE['freq']*LIMIT['iWmin']
                temp1 = min(max(iWmin,iWpll[i]),iWmax)
                temp2 = 2.*np.pi/temp1/tinc
                N1cy = int(np.around(temp2/2.))*2 # makes N1cy the closest even number
                if N1cy > i:
                    # print 'Limiting N1cy', N1cy, 'to i', i
                    N1cy = int(min(i,N1cy)/2)*2 # down-rounding to an even number
                    if N1cy < 256:
                        # print 'Less than 256 points in N1cy, skipping'
                        continue
                Va = iVa[i-N1cy:i]
                Vb = iVb[i-N1cy:i]
                Vc = iVc[i-N1cy:i]

                Va_fft = fftpack.fft(Va)
                Vb_fft = fftpack.fft(Vb)
                Vc_fft = fftpack.fft(Vc)
                # sample_freq=fftpack.fftfreq(Va.size, d=tinc)
                # pidxs  = np.where(sample_freq > 0)

                # sequence calculations using first harmonic
                Va1 = 2./N1cy*Va_fft[1]
                Vb1 = 2./N1cy*Vb_fft[1]
                Vc1 = 2./N1cy*Vc_fft[1]
                aa = np.e**complex(0,2.*np.pi/3.)
                V0 = 1./3.*(Va1 + Vb1 + Vc1)
                V1 = 1./3.*(Va1 + aa*Vb1 + aa*aa*Vc1)
                V2 = 1./3.*(Va1 + aa*aa*Vb1 + aa*Vc1)
                iV0mag[i]=np.abs(V0)
                iV1mag[i]=np.abs(V1)
                iV2mag[i]=np.abs(V2)
                # iV0ang[i]=np.angle(V0)
                # iV1ang[i]=np.angle(V1)
                # iV2ang[i]=np.angle(V2)
        
            sig_data[fname][u'Island Vpos'] = pd.Series(iV1mag,index=sig_data[fname].index)
            sig_data[fname][u'Island Vneg'] = pd.Series(iV2mag,index=sig_data[fname].index)
            sig_data[fname][u'Island Vzer'] = pd.Series(iV0mag,index=sig_data[fname].index)

        # Utility currents
        if not CONFIG['UseRAF']:
            uIa=sig_data[fname][u'Utility I A'].values
            uIb=sig_data[fname][u'Utility I B'].values
            uIc=sig_data[fname][u'Utility I C'].values
        else:
            # rolling average filtering to suppress 5856Hz switching frequency
            tinc = sig_info[fname]['chTincr'][sig_info[fname]['chName']==u'Utility I A'].values[0]
            uIa = rolling_mean(sig_data[fname][u'Utility I A'],1./AMP['L_IntRate']*2./tinc).values
            uIb = rolling_mean(sig_data[fname][u'Utility I B'],1./AMP['L_IntRate']*2./tinc).values
            uIc = rolling_mean(sig_data[fname][u'Utility I C'],1./AMP['L_IntRate']*2./tinc).values

        uIal = uIa - 0.5 * (uIb + uIc)
        uIbe = sqrt(3.)/2. * (uIb - uIc)
        uImag = sqrt(uIal*uIal + uIbe*uIbe)
        sig_data[fname][u'uIal'] = pd.Series(uIal,index=sig_data[fname].index)
        sig_data[fname][u'uIbe'] = pd.Series(uIbe,index=sig_data[fname].index)
        sig_data[fname][u'uImag'] = pd.Series(uImag,index=sig_data[fname].index)

        # Utility Power calculations kW
        uP = (uVa*uIa+uVb*uIb+uVc*uIc)/1000
        uQ = ((uVb-uVc)*uIa+(uVa-uVb)*uIc+(uVc-uVa)*uIb)/sqrt(3)/1000
        sig_data[fname][u'P Utility'] = pd.Series(uP,index=sig_data[fname].index)
        sig_data[fname][u'Q Utility'] = pd.Series(uQ,index=sig_data[fname].index)

        # RLC currents
        if not CONFIG['UseRAF']:
            rIa=sig_data[fname][u'RLC Passive Load I A'].values
            rIb=sig_data[fname][u'RLC Passive Load I B'].values
            rIc=sig_data[fname][u'RLC Passive Load I C'].values
        else:
            # rolling average filtering to suppress 5856Hz switching frequency
            tinc = sig_info[fname]['chTincr'][sig_info[fname]['chName']==u'RLC Passive Load I A'].values[0]
            rIa = rolling_mean(sig_data[fname][u'RLC Passive Load I A'],1./AMP['L_IntRate']*2./tinc).values
            rIb = rolling_mean(sig_data[fname][u'RLC Passive Load I B'],1./AMP['L_IntRate']*2./tinc).values
            rIc = rolling_mean(sig_data[fname][u'RLC Passive Load I C'],1./AMP['L_IntRate']*2./tinc).values

        # RLC power calcuations
        rP = (iVa*rIa+iVb*rIb+iVc*rIc)/1000
        rQ = ((iVb-iVc)*rIa+(iVa-iVb)*rIc+(iVc-iVa)*rIb)/sqrt(3)/1000
        sig_data[fname][u'P RLC'] = pd.Series(rP,index=sig_data[fname].index)
        sig_data[fname][u'Q RLC'] = pd.Series(rQ,index=sig_data[fname].index)

        # Amplifier currents
        if not CONFIG['UseRAF']:
            ampIa=sig_data[fname][u'GE Load I A'].values
            ampIb=sig_data[fname][u'GE Load I B'].values
            ampIc=sig_data[fname][u'GE Load I C'].values
        else:
            # rolling average filtering to suppress 5856Hz switching frequency
            tinc = sig_info[fname]['chTincr'][sig_info[fname]['chName']==u'GE Load I A'].values[0]
            ampIa = rolling_mean(sig_data[fname][u'GE Load I A'],1./AMP['L_IntRate']*2./tinc).values
            ampIb = rolling_mean(sig_data[fname][u'GE Load I B'],1./AMP['L_IntRate']*2./tinc).values
            ampIc = rolling_mean(sig_data[fname][u'GE Load I C'],1./AMP['L_IntRate']*2./tinc).values

        # Amplifier power calculations
        ampP = (iVa*ampIa+iVb*ampIb+iVc*ampIc)/1000
        ampQ = ((iVb-iVc)*ampIa+(iVa-iVb)*ampIc+(iVc-iVa)*ampIb)/sqrt(3)/1000
        sig_data[fname][u'P AMP'] = pd.Series(ampP,index=sig_data[fname].index)
        sig_data[fname][u'Q AMP'] = pd.Series(ampQ,index=sig_data[fname].index)

        # B2 currents
        if not CONFIG['UseRAF']:
            b2Ia=B2LC1SIGN*sig_data[fname][u'B2 LC1 I A'].values
            b2Ib=B2LC1SIGN*sig_data[fname][u'B2 LC1 I B'].values
            b2Ic=B2LC1SIGN*sig_data[fname][u'B2 LC1 I C'].values
        else:
            # rolling average filtering to suppress 5856Hz switching frequency
            tinc = sig_info[fname]['chTincr'][sig_info[fname]['chName']==u'B2 LC1 I A'].values[0]
            b2Ia = rolling_mean(B2LC1SIGN*sig_data[fname][u'B2 LC1 I A'],1./AMP['L_IntRate']*2./tinc).values
            b2Ib = rolling_mean(B2LC1SIGN*sig_data[fname][u'B2 LC1 I B'],1./AMP['L_IntRate']*2./tinc).values
            b2Ic = rolling_mean(B2LC1SIGN*sig_data[fname][u'B2 LC1 I C'],1./AMP['L_IntRate']*2./tinc).values

        # B2 Power calculations
        b2P = (iVa*b2Ia+iVb*b2Ib+iVc*b2Ic)/1000
        b2Q = ((iVb-iVc)*b2Ia+(iVa-iVb)*b2Ic+(iVc-iVa)*b2Ib)/sqrt(3)/1000
        sig_data[fname][u'P B2'] = pd.Series(b2P,index=sig_data[fname].index)
        sig_data[fname][u'Q B2'] = pd.Series(b2Q,index=sig_data[fname].index)

        # B1 currents
        if not CONFIG['UseRAF']:
            b1LC1=B1LC1SIGN*sig_data[fname][u'B1 LC1 I'].values
            b1LC2=sig_data[fname][u'B1 LC2 I'].values
            b1LC3=sig_data[fname][u'B1 LC3 I'].values
        else:
            # rolling average filtering to suppress 5856Hz switching frequency
            tinc = sig_info[fname]['chTincr'][sig_info[fname]['chName']==u'B1 LC1 I'].values[0]
            b1LC1 = rolling_mean(B1LC1SIGN*sig_data[fname][u'B1 LC1 I'],1./AMP['L_IntRate']*2./tinc).values
            b1LC2 = rolling_mean(sig_data[fname][u'B1 LC2 I'],1./AMP['L_IntRate']*2./tinc).values
            b1LC3 = rolling_mean(sig_data[fname][u'B1 LC3 I'],1./AMP['L_IntRate']*2./tinc).values
        
        b1Ia = b1LC1 - b1LC2
        b1Ib = b1LC3 - b1LC1
        b1Ic = b1LC2 - b1LC3
        sig_data[fname][u'b1Ia'] = pd.Series(b1Ia,index=sig_data[fname].index)
        sig_data[fname][u'b1Ib'] = pd.Series(b1Ib,index=sig_data[fname].index)
        sig_data[fname][u'b1Ic'] = pd.Series(b1Ic,index=sig_data[fname].index)
        
        # B1 Power calculations
        b1P = (iVa*b1Ia+iVb*b1Ib+iVc*b1Ic)/1000
        b1Q = ((iVb-iVc)*b1Ia+(iVa-iVb)*b1Ic+(iVc-iVa)*b1Ib)/sqrt(3)/1000
        sig_data[fname][u'P B1'] = pd.Series(b1P,index=sig_data[fname].index)
        sig_data[fname][u'Q B1'] = pd.Series(b1Q,index=sig_data[fname].index)

        # Total PV calculations (banks 1 and 2)
        pvIa = b1Ia + b2Ia
        pvIb = b1Ib + b2Ib
        pvIc = b1Ic + b2Ic
        sig_data[fname][u'pvIa'] = pd.Series(pvIa,index=sig_data[fname].index)
        sig_data[fname][u'pvIb'] = pd.Series(pvIb,index=sig_data[fname].index)
        sig_data[fname][u'pvIc'] = pd.Series(pvIc,index=sig_data[fname].index)

        pvIal = pvIa - 0.5 * (pvIb + pvIc)
        pvIbe = sqrt(3.)/2. * (pvIb - pvIc)
        sig_data[fname][u'pvIal'] = pd.Series(pvIal,index=sig_data[fname].index)
        sig_data[fname][u'pvIbe'] = pd.Series(pvIbe,index=sig_data[fname].index)
                
        # Penetration calculations
        # penB1 = where(iVmag/sqrt(2)/BASE['Vln'] > VMAG['low'],b1P/rP,NaN)
        # penB2 = where(iVmag/sqrt(2)/BASE['Vln'] > VMAG['low'],b2P/rP,NaN)
        # penPV = where(iVmag/sqrt(2)/BASE['Vln'] > VMAG['low'],(b1P+b2P)/rP,NaN)
        penB1 = np.where(sig_data[fname][u'Island Contactor status'] > VMAG['icsLvl'],b1P/(rP+ampP),np.nan)
        penB2 = np.where(sig_data[fname][u'Island Contactor status'] > VMAG['icsLvl'],b2P/(rP+ampP),np.nan)
        penPV = np.where(sig_data[fname][u'Island Contactor status'] > VMAG['icsLvl'],(b1P+b2P)/(rP+ampP),np.nan)
        sig_data[fname][u'B1 pen'] = pd.Series(penB1,index=sig_data[fname].index)
        sig_data[fname][u'B2 pen'] = pd.Series(penB2,index=sig_data[fname].index)
        sig_data[fname][u'B1+B2 pen'] = pd.Series(penPV,index=sig_data[fname].index)

        # Selecting a region of interest: island creation to cessation
        df1 = sig_data[fname]
        if CONFIG['UseIslandContactor']:
            # ix1 = df1[abs(df1[u'Utility Vmag']-df1[u'Island Vmag'])/sqrt(2)/BASE['Vln'] > VMAG['delta']].index.values[0]
            if df1[abs(df1[u'Island Contactor status']) < VMAG['icsLvl']].empty:
                ix1 = df1.index.values[-1]/2
                TestLog.loc[fnumix, ('ixflags')] += 1
            else:
                ix1 = df1[abs(df1[u'Island Contactor status']) < VMAG['icsLvl']].index.values[0]
                TestLog.loc[fnumix, ('ixflags')] += 2
        else:
            # using magnitude of utility current
            if df1[df1[u'uImag'] < VMAG['uImagLim']].empty:
                ix1 = df1.index.values[-1]/2
                TestLog.loc[fnumix, ('ixflags')] += 4
            else:
                ix1 = df1[df1[u'uImag'] < VMAG['uImagLim']].index.values[0]
                TestLog.loc[fnumix, ('ixflags')] += 8
        
        if CONFIG['UseIslandVmag']:
            if df1[df1[u'Island Vmag']/sqrt(2)/BASE['Vln'] < VMAG['low']].empty:
                ix2 = df1.index.values[-1]/2
                TestLog.loc[fnumix, ('ixflags')] += 16
            else:
                ix2 = df1[df1[u'Island Vmag']/sqrt(2)/BASE['Vln'] < VMAG['low']].index.values[0]
                TestLog.loc[fnumix, ('ixflags')] += 32
        else:
            # using amplifier+RLC load power level
            if df1[df1[u'P AMP']+df1[u'P RLC'] < AMP['Pcutout']].empty:
                ix2 = df1.index.values[-1]/2
                TestLog.loc[fnumix, ('ixflags')] += 64
            else:
                ix2 = df1[df1[u'P AMP']+df1[u'P RLC'] < AMP['Pcutout']].index.values[0]
                TestLog.loc[fnumix, ('ixflags')] += 128
        if ix2 < ix1:
            ix1 = int((ix2+ix1)/2)
            ix2 = ix1+1
            TestLog.loc[fnumix, ('ixflags')] += 256
                    

        # Collect attributes
        TestLog.loc[fnumix, ('ix1')] = ix1 # TestLog['ix1'][fnumix] = ix1
        TestLog.loc[fnumix, ('ix2')] = ix2 # TestLog['ix2'][fnumix] = ix2
        TestLog.loc[fnumix, ('tIslStart')] = df1['Time'][ix1] # TestLog['tIslStart'][fnumix] = df1['Time'][ix1]
        TestLog.loc[fnumix, ('tIslEnd')] = df1['Time'][ix2] # TestLog['tIslEnd'][fnumix]   = df1['Time'][ix2]
        TestLog.loc[fnumix, ('tIslDur')] = df1['Time'][ix2] - df1['Time'][ix1] # TestLog['tIslDur'][fnumix]   = df1['Time'][ix2] - df1['Time'][ix1]

        TestLog.loc[fnumix, ('VposMax')]= df1[u'Island Vpos'][ix1:ix2].max()
        TestLog.loc[fnumix, ('VnegMax')]= df1[u'Island Vneg'][ix1:ix2].max()
        TestLog.loc[fnumix, ('VzerMax')]= df1[u'Island Vzer'][ix1:ix2].max()
        TestLog.loc[fnumix, ('VmagMax')]= df1[u'Island Vmag'][ix1:ix2].max()/sqrt(2)/BASE['Vln']

        TestLog.loc[fnumix, ('fIslMean')]= df1[u'Island freq'][ix1:ix2].mean()
        TestLog.loc[fnumix, ('fIslStd')] = df1[u'Island freq'][ix1:ix2].std()
        TestLog.loc[fnumix, ('fIslMin')] = df1[u'Island freq'][ix1:ix2].min()
        TestLog.loc[fnumix, ('fIslMax')] = df1[u'Island freq'][ix1:ix2].max()
        TestLog.loc[fnumix, ('fIslSkw')] = df1[u'Island freq'][ix1:ix2].skew()

        tinc = sig_info[fname]['chTincr'][sig_info[fname]['chName']==u'Utility Bus V A'].values[0]
        left = int(NCY['pre']*1./60./tinc)
        right = int(NCY['post']*1./60./tinc)
        ix1 = max([ix1-left,0])
        ix2 = min([ix2+right,df1.index.values[-1]])
        df2 = df1[(df1.index > ix1) & (df1.index < ix2)]
        # Collect attributes
        TestLog.loc[fnumix, ('ix1left')] = ix1 # TestLog['ix1left'][fnumix]   = ix1
        TestLog.loc[fnumix, ('ix2right')] = ix2 # TestLog['ix2right'][fnumix]  = ix2
        
        if CONFIG['WriteLimitedDataToExcel']: # Only the data from island formation to cessation
            df2.to_excel(writer,fname) # data is written here

        if CONFIG['WriteDataToHDF5']: # All data
            h5results.put('df1_'+fnum,df1)

        # Output plot pages 
        label= file_info[file_info.index==fname][['fiComment']].values[0][0]
        OutputAlBePage(pltPdf, df2, fname, label) # Alpha Beta voltages and currents
        # OutputPLLPage(pltPdf, df2, fname) # PLL variables
        if CONFIG['ValidateCTs']: # Plots pages with CT signals to determine polarity
            OutputCTsValidationPages(pltPdf, df2, fname)
        OutputVfIPage(pltPdf, df2, fname)
        OutputPQVPage(pltPdf, df2, fname)
        if CONFIG['PlotFullRange']: # Plots a page with entire length of captured signals
            OutputAlBePage(pltPdf, df1, fname, label) # Alpha Beta voltages and currents
            # OutputPLLPage(pltPdf, df1, fname) # PLL variables
            if CONFIG['ValidateCTs']: # Plots pages with CT signals to determine polarity
                OutputCTsValidationPages(pltPdf, df1, fname)
            OutputVfIPage(pltPdf, df1, fname)
            OutputPQVPage(pltPdf, df1, fname)

    print "Closing: Results.pdf"
    pltPdf.close() # Close the pdf file

    if CONFIG['WriteLimitedDataToExcel']: # Close excel file
        print "Writing: LimitedData.xlsx"
        writer.save() # file is saved here

    if CONFIG['WriteDataToHDF5']: # Close h5 file
        print "Writing: Results.h5"
        h5results.close()
    
    return
    
def OutputAlBePage(pltPdf, df2, fname, label,
                   BASE = {'Vln':277.128, 'freq':60.}):

    # Fig1: Utility voltage
    fig, (ax0, ax1)= plt.subplots(nrows=2, ncols=1, figsize=(8.5,11))
    fig.suptitle(fname) # This titles the figure
    # File info output to page top
    ax0.annotate(label,
                 xy=(0.5/8.5, 10.5/11), # (0.5,-0.25)inch from top left corner
                 xycoords='figure fraction',
                 horizontalalignment='left',
                 verticalalignment='top',
                 fontsize=10)
    # fig.subplots_adjust(top=9./11.)
    
    # Alpha/Beta plots
    ax0.set_title('Island Voltage Al/Be')
    ax0.plot(df2['Island Val']/1.5/sqrt(2.)/BASE['Vln'], df2['Island Vbe']/1.5/sqrt(2.)/BASE['Vln'])
    ax0.set_xlim([-1.5,1.5])
    ax0.set_ylim([-1.2,1.2])
    ax0.grid(True, which='both')
    ax0.set_aspect('equal')

    ax1.set_title('Currents Al/Be')
    ax1.plot(df2['pvIal']/1.5, df2['pvIbe']/1.5)
    ax1.set_xlim([-300,300])
    ax1.set_ylim([-240,240])
    ax1.grid(True, which='both')
    ax1.set_aspect('equal')
    # ax1.set_title('Island Voltage Al/Be')
    # ax1.plot(df2['Time'], df2['Island Val']/1.5/sqrt(2)/BASE['Vln'])
    # ax1.plot(df2['Time'], df2['Island Vbe']/1.5/sqrt(2)/BASE['Vln'])
    # ax1.set_ylim([-1.2,1.2])
    # ax1.grid(True, which='both')

    pltPdf.savefig() # saves fig to pdf
    plt.close() # Closes fig to clean up memory

    return

def OutputPLLPage(pltPdf, df2, fname):
    fig, (ax0,ax1,ax2,ax3) = plt.subplots(nrows=4, ncols=1,
                                              figsize=(8.5,11),
                                              sharex=True)
    fig.suptitle(fname) # This titles the figure

    ax0.set_title('Utility Bus Vabc')
    ax0.plot(df2['Time'], df2[u'Utility Bus V A'])
    ax0.plot(df2['Time'], df2[u'Utility Bus V B'])
    ax0.plot(df2['Time'], df2[u'Utility Bus V C'])
    ax0.set_ylim([-500,500])
    ax0.grid(True, which='both')

    ax1.set_title('Island Bus Vabc')
    ax1.plot(df2['Time'], df2[u'Island Bus V A'])
    ax1.plot(df2['Time'], df2[u'Island Bus V B'])
    ax1.plot(df2['Time'], df2[u'Island Bus V C'])
    ax1.plot(df2['Time'], df2[u'Island Vmag'])
    # ax1.set_ylim([-500,500])
    ax1.grid(True, which='both')

    ax2.set_title('Island Bus Frequency')
    ax2.plot(df2['Time'], df2[u'Island Wpll']/(2*pi))
    ax2.plot(df2['Time'], df2[u'Island freq'])
    # ax2.set_ylim([-100,100])
    ax2.grid(True, which='both')

    ax3.set_title('Island Bus Vx, Vy')
    ax3.plot(df2['Time'], df2[u'Island Vx'])
    ax3.plot(df2['Time'], df2[u'Island Vy'])
    # ax3.set_ylim([-100,100])
    ax3.grid(True, which='both')

    pltPdf.savefig() # Saves fig to pdf
    plt.close() # Closes fig to clean up memory
    return
    
def OutputCTsValidationPages(pltPdf, df2, fname):

    # CTs Page 1: 
    fig, (ax0,ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=6, ncols=1,
                                              figsize=(8.5,11),
                                              sharex=True)
    fig.suptitle(fname) # This titles the figure

    ax0.set_title('Phase A CTs: rlc_Ia = u_Ia + pv_Ia')
    ax0.plot(df2['Time'], df2[u'RLC Passive Load I A'])
    ax0.plot(df2['Time'], df2[u'Utility I A']+df2[u'pvIa'])
    # ax0.set_ylim([-50,50])
    ax0.grid(True, which='both')

    ax1.set_title('Phase B CTs: rlc_Ib = u_Ib + pv_Ib')
    ax1.plot(df2['Time'], df2[u'RLC Passive Load I B'])
    ax1.plot(df2['Time'], df2[u'Utility I B']+df2[u'pvIb'])
    # ax1.set_ylim([-50,50])
    ax1.grid(True, which='both')

    ax2.set_title('Phase C CTs: rlc_Ic = u_Ic + pv_Ic')
    ax2.plot(df2['Time'], df2[u'RLC Passive Load I C'])
    ax2.plot(df2['Time'], df2[u'Utility I C']+df2[u'pvIc'])
    # ax2.set_ylim([-50,50])
    ax2.grid(True, which='both')

    ax3.set_title('Phase A CTs: u_Ia = rlc_Ia - pv_Ia, b2Ia')
    ax3.plot(df2['Time'], df2[u'Utility I A'])
    ax3.plot(df2['Time'], df2[u'RLC Passive Load I A']-df2[u'pvIa'])
    ax3.plot(df2['Time'], df2[u'B2 LC1 I A'])
    # ax3.set_ylim([-25,25])
    ax3.grid(True, which='both')

    ax4.set_title('Phase B CTs: u_Ib = rlc_Ib - pv_Ib, b2Ib')
    ax4.plot(df2['Time'], df2[u'Utility I B'])
    ax4.plot(df2['Time'], df2[u'RLC Passive Load I B']-df2[u'pvIb'])
    ax4.plot(df2['Time'], df2[u'B2 LC1 I B'])
    # ax4.set_ylim([-25,25])
    ax4.grid(True, which='both')

    ax5.set_title('Phase C CTs: u_Ic = rlc_Ic - pv_Ic, b2Ic')
    ax5.plot(df2['Time'], df2[u'Utility I C'])
    ax5.plot(df2['Time'], df2[u'RLC Passive Load I C']-df2[u'pvIc'])
    ax5.plot(df2['Time'], df2[u'B2 LC1 I C'])
    # ax5.set_ylim([-25,25])
    ax5.grid(True, which='both')

    pltPdf.savefig() # Saves fig to pdf
    plt.close() # Closes fig to clean up memory


    # CTs Page 2: 
    fig, (ax0,ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=6, ncols=1,
                                              figsize=(8.5,11),
                                              sharex=True)
    fig.suptitle(fname) # This titles the figure

    ax0.set_title('Phase A CTs: rlc_Ia = u_Ia + pv_Ia')
    ax0.plot(df2['Time'], df2[u'RLC Passive Load I A'])
    ax0.plot(df2['Time'], df2[u'Utility I A']+df2[u'pvIa'])
    # ax0.set_ylim([-50,50])
    ax0.grid(True, which='both')

    ax1.set_title('Phase B CTs: rlc_Ib = u_Ib + pv_Ib')
    ax1.plot(df2['Time'], df2[u'RLC Passive Load I B'])
    ax1.plot(df2['Time'], df2[u'Utility I B']+df2[u'pvIb'])
    # ax1.set_ylim([-50,50])
    ax1.grid(True, which='both')

    ax2.set_title('Phase C CTs: rlc_Ic = u_Ic + pv_Ic')
    ax2.plot(df2['Time'], df2[u'RLC Passive Load I C'])
    ax2.plot(df2['Time'], df2[u'Utility I C']+df2[u'pvIc'])
    # ax2.set_ylim([-50,50])
    ax2.grid(True, which='both')

    ax3.set_title('Phase A CTs: pv_Ia = rlc_Ia - u_Ia, b1Ia, b2Ia')
    ax3.plot(df2['Time'], df2[u'RLC Passive Load I A']-df2[u'Utility I A'])
    ax3.plot(df2['Time'], df2[u'pvIa'])
    ax3.plot(df2['Time'], df2[u'b1Ia'])
    ax3.plot(df2['Time'], df2[u'B2 LC1 I A'])
    # ax3.set_ylim([-50,50])
    ax3.grid(True, which='both')

    ax4.set_title('Phase B CTs: pv_Ib = rlc_Ib - u_Ib, b1Ib, b2Ib')
    ax4.plot(df2['Time'], df2[u'RLC Passive Load I B']-df2[u'Utility I B'])
    ax4.plot(df2['Time'], df2[u'pvIb'])
    ax4.plot(df2['Time'], df2[u'b1Ib'])
    ax4.plot(df2['Time'], df2[u'B2 LC1 I B'])
    # ax4.set_ylim([-50,50])
    ax4.grid(True, which='both')

    ax5.set_title('Phase C CTs: pv_Ic = rlc_Ic - u_Ic, b1Ic, b2Ic')
    ax5.plot(df2['Time'], df2[u'RLC Passive Load I C']-df2[u'Utility I C'])
    ax5.plot(df2['Time'], df2[u'pvIc'])
    ax5.plot(df2['Time'], df2[u'b1Ic'])
    ax5.plot(df2['Time'], df2[u'B2 LC1 I C'])
    # ax5.set_ylim([-50,50])
    ax5.grid(True, which='both')

    pltPdf.savefig() # Saves fig to pdf
    plt.close() # Closes fig to clean up memory
    return
    
def OutputVfIPage(pltPdf, df2, fname):

    fig, (ax0,ax1,ax2,ax3,ax4) = plt.subplots(nrows=5, ncols=1,
                                              figsize=(8.5,11),
                                              sharex=True)
    fig.suptitle(fname) # This titles the figure

    # ax0.set_title('Utility Bus Vabc')
    # ax0.plot(df2['Time'], df2[u'Utility Bus V A'])
    # ax0.plot(df2['Time'], df2[u'Utility Bus V B'])
    # ax0.plot(df2['Time'], df2[u'Utility Bus V C'])
    # ax0.set_ylim([-500,500])
    # ax0.grid(True, which='both')

    ax0.set_title('Island Bus Vabc')
    ax0.plot(df2['Time'], df2[u'Island Bus V A'])
    ax0.plot(df2['Time'], df2[u'Island Bus V B'])
    ax0.plot(df2['Time'], df2[u'Island Bus V C'])
    ax0.plot(df2['Time'], df2[u'Island Vmag'])
    # ax0.set_ylim([-500,500])
    ax0.grid(True, which='both')

    ax1.set_title('Island Bus Frequency')
    # ax1.plot(df2['Time'], df2[u'Island Wpll']/(2*pi))
    ax1.plot(df2['Time'], df2[u'Island freq'])
    # ax1.set_ylim([50, 70])
    ax1.grid(True, which='both')

    ax2.set_title('Total Load Current Iabc')
    ax2.plot(df2['Time'], df2[u'RLC Passive Load I A']+df2[u'GE Load I A'])
    ax2.plot(df2['Time'], df2[u'RLC Passive Load I B']+df2[u'GE Load I B'])
    ax2.plot(df2['Time'], df2[u'RLC Passive Load I C']+df2[u'GE Load I C'])
    # ax2.set_ylim([-100,100])
    ax2.grid(True, which='both')

    ax3.set_title('B1+B2 Iabc')
    ax3.plot(df2['Time'], df2[u'pvIa'])
    ax3.plot(df2['Time'], df2[u'pvIb'])
    ax3.plot(df2['Time'], df2[u'pvIc'])
    # ax3.set_ylim([-100,100])
    ax3.grid(True, which='both')

    ax4.set_title('Utility Iabc')
    ax4.plot(df2['Time'], df2[u'Utility I A'])
    ax4.plot(df2['Time'], df2[u'Utility I B'])
    ax4.plot(df2['Time'], df2[u'Utility I C'])
    # ax4.set_ylim([-100,100])
    ax4.grid(True, which='both')

    pltPdf.savefig() # Saves fig to pdf
    plt.close() # Closes fig to clean up memory
    return
    
def OutputPQVPage(pltPdf, df2, fname,
                  BASE = {'Vln':277.128}):
    # Fig4: 
    fig, (ax0,ax1,ax2,ax3,ax4) = plt.subplots(nrows=5, ncols=1,
                                              figsize=(8.5,11),
                                              sharex=True)
    fig.suptitle(fname) # This titles the figure

    ax0.set_title('P[kW]: Utility, Load, PV')
    ax0.plot(df2['Time'], df2[u'P Utility'])
    ax0.plot(df2['Time'], df2[u'P RLC']+df2[u'P AMP'])
    ax0.plot(df2['Time'], df2[u'P B1']+df2[u'P B2'])
    # ax0.set_ylim([-50,250])
    ax0.grid(True, which='both')

    ax1.set_title('Q[kVAr]: Utility, Load, PV')
    ax1.plot(df2['Time'], df2[u'Q Utility'])
    ax1.plot(df2['Time'], df2[u'Q RLC']+df2[u'Q AMP'])
    ax1.plot(df2['Time'], df2[u'Q B1']+df2[u'Q B2'])
    # ax1.set_ylim([-80,80])
    ax1.grid(True, which='both')

    ax2.set_title('Island Vpos, pu penetration')
    ax2.plot(df2['Time'], df2[u'Island Vpos']/sqrt(2.)/BASE['Vln'])
    ax2.plot(df2['Time'], df2[u'B1+B2 pen'])
    ax2.set_ylim([0,1.5])
    ax2.grid(True, which='both')

    ax3.set_title('Island Vneg, Vzero')
    ax3.plot(df2['Time'], df2[u'Island Vneg']/sqrt(2.)/BASE['Vln'])
    ax3.plot(df2['Time'], df2[u'Island Vzer']/sqrt(2.)/BASE['Vln'])
    # ax3.set_ylim([0,0.25])
    ax3.grid(True, which='both')

    ax4.set_title('Island Vrms abc')
    ax4.plot(df2['Time'], df2[u'Island Varms']/BASE['Vln'])
    ax4.plot(df2['Time'], df2[u'Island Vbrms']/BASE['Vln'])
    ax4.plot(df2['Time'], df2[u'Island Vcrms']/BASE['Vln'])
    # ax4.set_ylim([0,1.25])
    ax4.grid(True, which='both')

    pltPdf.savefig() # Saves fig to pdf
    plt.close() # Closes fig to clean up memory
    return

def AddSeqComp( mysubdirpath, TestLog, dname,
                VMAG = {'icsLvl': 3.0, 'delta':0.1, 'low':0.1, 'uImagLim':10.0},
                NCY = {'pre':1., 'post': 0.2}, # {'pre': 3., 'post': 5.}
                MISC = {'tol':0.001},
                AMP = {'L_IntRate':5856.0, 'Pcutout':4.0}, # switching frequency and Pcutout
                BASE = {'Vln':277.128, 'freq':60.},
                LIMIT = {'iWmin': 0.9, 'iWmax': 1.1}):
    import pandas as pd
    import numpy as np

    from scipy import fftpack

    CONFIG = {'WriteAllDataToExcel':False, # Will take forever ~5min for 27GB
              'WriteLimitedDataToExcel':False, # Only data from island creation to cessation
              'WriteSummaryToExcel':True, # Summary from TDMS files
              'WriteDataToHDF5':True, # Test results consolidated in h5 format
              'ValidateCTs':False, # Plot pages that validate CT scaling and orientation
              'UseRAF':True, # Use Rolling average filtering on current and voltage signals
              'UseIslandContactor':False,
              'UseIslandVmag':False,
              'PlotFullRange': False} # Add a page with full time range of data

    # Open pdf file to plot results
    print "Opening: Results1.pdf"
    pltPdf = dpdf.PdfPages(mysubdirpath + '\\Results1.pdf')
    h5in   = pd.HDFStore(mysubdirpath + "\\Results.h5")
    h5out  = pd.HDFStore(mysubdirpath + "\\Results1.h5")

    for fnum in sorted(TestLog[TestLog[u'DirName']==dname][u'TDMSfnum']):
        fnumix = TestLog[TestLog[u'TDMSfnum']==fnum].index
        key = "df1_" + str(fnum)
        if key in h5in:
            tlogs3 = TestLog[TestLog[u'TDMSfnum']==fnum]
            df1 = h5in[key]
            df3 = pd.DataFrame(df1['Time'],columns=['Time'])
            # print dname, fnum, key, df1.shape
            fname = tlogs3[u'TDMSfname'].values[0]
            print fname

            tinc = df1[u'Time'][1]-df1[u'Time'][0]
            ix1 = int(tlogs3[u'ix1'].values[0])
            ix2 = int(tlogs3[u'ix2'].values[0])
            ix1left = int(tlogs3[u'ix1left'].values[0])
            ix2right = int(tlogs3[u'ix2right'].values[0])
            iWpll = df1[u'Island Wpll']
            iVa = df1[u'Island Bus V A']
            iVb = df1[u'Island Bus V B']
            iVc = df1[u'Island Bus V C']

            iV1mag = np.ones_like(iVa)*(-1.)*sqrt(2.)*BASE['Vln']
            iV2mag = np.ones_like(iVa)*(-1.)
            iV0mag = np.ones_like(iVa)*(-1.)
            # iV1ang = np.zeroes_like(iVa)
            # iV2ang = np.zeroes_like(iVa)
            # iV0ang = np.zeroes_like(iVa)

            for i in range(ix1left,iVa.shape[0]):
                # limiting iWpll[i]
                iWmax = 2.*np.pi*BASE['freq']*LIMIT['iWmax']
                iWmin = 2.*np.pi*BASE['freq']*LIMIT['iWmin']
                temp1 = min(max(iWmin,iWpll[i]),iWmax)
                temp2 = 2.*np.pi/temp1/tinc
                N1cy = int(np.around(temp2/2.))*2 # makes N1cy the closest even number
                if N1cy > i:
                    # print 'Limiting N1cy', N1cy, 'to i', i
                    N1cy = int(min(i,N1cy)/2)*2 # down-rounding to an even number
                    if N1cy < 256:
                        # print 'Less than 256 points in N1cy, skipping'
                        continue
                Va = iVa[i-N1cy:i]
                Vb = iVb[i-N1cy:i]
                Vc = iVc[i-N1cy:i]

                Va_fft = fftpack.fft(Va)
                Vb_fft = fftpack.fft(Vb)
                Vc_fft = fftpack.fft(Vc)
                # sample_freq=fftpack.fftfreq(Va.size, d=tinc)
                # pidxs  = np.where(sample_freq > 0)

                # sequence calculations using first harmonic
                Va1 = 2./N1cy*Va_fft[1]
                Vb1 = 2./N1cy*Vb_fft[1]
                Vc1 = 2./N1cy*Vc_fft[1]
                aa = np.e**complex(0,2.*np.pi/3.)
                V0 = 1./3.*(Va1 + Vb1 + Vc1)
                V1 = 1./3.*(Va1 + aa*Vb1 + aa*aa*Vc1)
                V2 = 1./3.*(Va1 + aa*aa*Vb1 + aa*Vc1)
                iV0mag[i]=np.abs(V0)
                iV1mag[i]=np.abs(V1)
                iV2mag[i]=np.abs(V2)
                # iV0ang[i]=np.angle(V0)
                # iV1ang[i]=np.angle(V1)
                # iV2ang[i]=np.angle(V2)

            df3[u'Island V0mag'] = pd.Series(iV0mag,index=df3.index)
            df3[u'Island V1mag'] = pd.Series(iV1mag,index=df3.index)
            df3[u'Island V2mag'] = pd.Series(iV2mag,index=df3.index)
            if '/' + key in h5out.keys():
                h5out.remove(key)
            h5out.put(key, df3)

            # Downselecting data to plot a page
            df1a = df1[(df1.index > ix1left) & (df1.index < ix2right)]
            df3a = df3[(df3.index > ix1left) & (df3.index < ix2right)]
            RevisePQVPage(pltPdf, df1a, df3a, fname)
            TestLog.loc[fnumix, ('V0magMax')] = df3[u'Island V0mag'][ix1:ix2].max()
            TestLog.loc[fnumix, ('V1magMax')] = df3[u'Island V1mag'][ix1:ix2].max()
            TestLog.loc[fnumix, ('V2magMax')] = df3[u'Island V2mag'][ix1:ix2].max()

        else:
            print key, " is not in the h5data"

    h5in.close()
    h5out.close()
    print "Closing: Results1.pdf"
    pltPdf.close() # Close the pdf file

    return

def RevisePQVPage(pltPdf, 
                  df2, df3, 
                  fname,
                  BASE = {'Vln':277.128, 'freq':60.}):

    import matplotlib.pyplot as plt

    # Fig4: 
    fig, (ax0,ax1,ax2,ax3,ax4) = plt.subplots(nrows=5, ncols=1,
                                              figsize=(8.5,11),
                                              sharex=True)
    fig.suptitle(fname) # This titles the figure

    ax0.set_title('P[kW]: Utility, Load, PV')
    ax0.plot(df2['Time'], df2[u'P Utility'])
    ax0.plot(df2['Time'], df2[u'P RLC']+df2[u'P AMP'])
    ax0.plot(df2['Time'], df2[u'P B1']+df2[u'P B2'])
    # ax0.set_ylim([-50,250])
    ax0.grid(True, which='both')

    ax1.set_title('Q[kVAr]: Utility, Load, PV')
    ax1.plot(df2['Time'], df2[u'Q Utility'])
    ax1.plot(df2['Time'], df2[u'Q RLC']+df2[u'Q AMP'])
    ax1.plot(df2['Time'], df2[u'Q B1']+df2[u'Q B2'])
    # ax1.set_ylim([-80,80])
    ax1.grid(True, which='both')

    ax2.set_title('Island Vpos, V1mag, pu penetration')
    ax2.plot(df2['Time'], df2[u'Island Vpos']/BASE['Vln'])
    ax2.plot(df3['Time'], df3[u'Island V1mag']/BASE['Vln']/sqrt(2.))
    ax2.plot(df2['Time'], df2[u'B1+B2 pen'])
    ax2.set_ylim([0,1.5])
    ax2.grid(True, which='both')

    ax3.set_title('Island Vneg, V2mag, Vzero, V0mag')
    ax3.plot(df2['Time'], df2[u'Island Vneg']/BASE['Vln'])
    ax3.plot(df3['Time'], df3[u'Island V2mag']/BASE['Vln']/sqrt(2.))
    ax3.plot(df2['Time'], df2[u'Island Vzer']/BASE['Vln'])
    ax3.plot(df3['Time'], df3[u'Island V0mag']/BASE['Vln']/sqrt(2.))
    # ax3.set_ylim([0,0.25])
    ax3.grid(True, which='both')

    ax4.set_title('Island Vrms abc')
    ax4.plot(df2['Time'], df2[u'Island Varms']/BASE['Vln'])
    ax4.plot(df2['Time'], df2[u'Island Vbrms']/BASE['Vln'])
    ax4.plot(df2['Time'], df2[u'Island Vcrms']/BASE['Vln'])
    # ax4.set_ylim([0,1.25])
    ax4.grid(True, which='both')

    pltPdf.savefig() # Saves fig to pdf
    plt.close() # Closes fig to clean up memory
    return

def ReplotResults( mysubdirpath, TestLog, 
                VMAG = {'icsLvl': 3.0, 'delta':0.1, 'low':0.1, 'uImagLim':10.0},
                NCY = {'pre':1., 'post': 0.2}, # {'pre': 3., 'post': 5.}
                MISC = {'tol':0.001},
                AMP = {'L_IntRate':5856.0, 'Pcutout':4.0}, # switching frequency and Pcutout
                BASE = {'Vln':277.128, 'freq':60.},
                LIMIT = {'iWmin': 0.9, 'iWmax': 1.1}):
    import pandas as pd
    import numpy as np

    CONFIG = {'WriteAllDataToExcel':False, # Will take forever ~5min for 27GB
              'WriteLimitedDataToExcel':False, # Only data from island creation to cessation
              'WriteSummaryToExcel':True, # Summary from TDMS files
              'WriteDataToHDF5':True, # Test results consolidated in h5 format
              'ValidateCTs':False, # Plot pages that validate CT scaling and orientation
              'UseRAF':True, # Use Rolling average filtering on current and voltage signals
              'UseIslandContactor':False,
              'UseIslandVmag':False,
              'PlotFullRange': True} # Add a page with full time range of data

    # Open pdf file to plot results
    print "Opening: Results1.pdf"
    pltPdf = dpdf.PdfPages(mysubdirpath + '\\Results1.pdf')
    h5in   = pd.HDFStore(mysubdirpath + "\\Results.h5")
    for fnum in sorted(TestLog[u'TDMSfnum']):
        fnumix = TestLog[TestLog[u'TDMSfnum']==fnum].index
        key = "df1_" + str(fnum)
        if key in h5in:
            tlogs3 = TestLog[TestLog[u'TDMSfnum']==fnum]
            fname = tlogs3[u'TDMSfname'].values[0]
            print fname
            df1 = h5in[key]

            ix1left = int(tlogs3[u'ix1left'].values[0])
            ix2right = int(tlogs3[u'ix2right'].values[0])
            df2 = df1[(df1.index > ix1left) & (df1.index < ix2right)]

            # Output plot pages 
            label= tlogs3[u'lvComment'].values[0]
            OutputAlBePage(pltPdf, df2, fname, label) # Alpha Beta voltages and currents
            # OutputPLLPage(pltPdf, df2, fname) # PLL variables
            if CONFIG['ValidateCTs']: # Plots pages with CT signals to determine polarity
                OutputCTsValidationPages(pltPdf, df2, fname)
            OutputVfIPage(pltPdf, df2, fname)
            OutputPQVPage(pltPdf, df2, fname)
            if CONFIG['PlotFullRange']: # Plots a page with entire length of captured signals
                OutputAlBePage(pltPdf, df1, fname, label) # Alpha Beta voltages and currents
                # OutputPLLPage(pltPdf, df1, fname) # PLL variables
                if CONFIG['ValidateCTs']: # Plots pages with CT signals to determine polarity
                    OutputCTsValidationPages(pltPdf, df1, fname)
                OutputVfIPage(pltPdf, df1, fname)
                OutputPQVPage(pltPdf, df1, fname)

        else:
            print key, " is not in the h5data"

    h5in.close()
    print "Closing: Results1.pdf"
    pltPdf.close() # Close the pdf file

    return
