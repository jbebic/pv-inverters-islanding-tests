#!/usr/bin/env python
from pylab import *

def MergeSavePlotTDMS( mypath,
               VMAG = {'delta':0.1, 'low':0.1},
               NCY = {'pre':3, 'post': 5},
               BASE = {'Vln':277.128} ):
    """ Merges TDMS files, crops time to island formation to cessasion, saves in excel

    Input: Directory with the test result files, e.g.: "aLabView\\20150306"

    MergeTDMS maintains several Excel files in the target directory:
        MergeSummary.xlsx
        SignalsInfo.xlsx
        CroppedData.xlsx
    """
#   import pdb # debugger
    import datetime 
    import pandas as pd # multidimensional data analysis
#   import xlsxwriter
#   import numpy as np

    # Matplotlib ===
    # import matplotlib
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # from matplotlib.backends.backend_pdf import PdfPages
    # Wei's advice ===
    import matplotlib.backends.backend_pdf as dpdf

    from os import listdir
    from os.path import isfile, join

    from nptdms import TdmsFile
    # from numpy import array
    from numpy import cos, sin, arctan2
    from pandas import concat, ExcelWriter, rolling_mean

    CONFIG = {'WriteAllDataToExcel':False, # Will take forever ~5min for 27GB
              'WriteLimitedDataToExcel':False, # Only data from island creation to cessation
              'WriteSummaryToExcel':True, # Summary from TDMS files
              'ValidateCTs':False, # Plot pages that validate CT scaling and orientation
              'PlotFullRange': False} # Add a page with full time range of data

    # BASE = {'Vln':480/sqrt(3)} # Voltage base

    # B2 LC1 CT group was reversed during calibration
    B2LC1SIGN = -1.0 #  
    
    
    # LC1 B1 CT was reversed on 20150311, restored, then reversed again during PG&E CT calibration
    B1LC1SIGN = -1.0 # reversed CT. Use +1 for correct polarity 
    # Limiting plot range of acquired signals
    # Islanding detection works by comparing: 'Island Contactor status' > icsLvl # abs(uVmag-iVmag)>delta
    # Collapse detection works by comparing: iVmag<low
    VMAG = {'icsLvl': 3, 'delta':0.1, 'low':0.1} # island contactor status Level, Signal magnitudes in p.u. to limit plot range
    NCY = {'pre':3, 'post': 5} # Number of cycles to show pre-islanding and post-collapse

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
        ch_tstrt = [ch.property(u'wf_start_time') for ch in channels_list]
        ch_tincr = [ch.property(u'wf_increment') for ch in channels_list]
        ch_tsamp = [ch.property(u'wf_samples') for ch in channels_list]
        ch_toffs = [ch.property(u'wf_start_offset') for ch in channels_list]
        ch_tend  = [ch.property(u'wf_start_time') +
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
            sig_data[fname].Time += datetime.timedelta.total_seconds(tEndLast-tStartLast)
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
        if False: # error due to time zone awareness in LabView time stamps
            for fname in file_info.index.values.tolist():
                sig_info[fname].to_excel(writer,fname)
        writer.save()

    if CONFIG['WriteAllDataToExcel']: # This takes forever -- 5min for ~27GB
        writer = ExcelWriter(mypath + '\\AllData.xlsx')
        for fname in file_info.index.values.tolist():
            sig_data[fname].to_excel(writer,fname)
        writer.save()

    # Only the data from island formation to cessation
    if CONFIG['WriteLimitedDataToExcel']: # file is open here, but written from within the plot loop
        print "Opening: LimitedData.xlsx"
        writer = ExcelWriter(mypath + '\\LimitedData.xlsx')

    # Plotting results
    # Open pdf file
    print "Opening: Results.pdf"
    pltPdf = dpdf.PdfPages(mypath + '\\Results.pdf')
    # prepare a list of files to plot
    file_list = file_info.index.values.tolist();
    for fname in file_list:
    # for fname in [file_list[0]]:
        print "Processing: " + fname
        # Utility voltage magnitude: alpha beta -> mag
        uVa=sig_data[fname][u'Utility Bus V A'].values
        uVb=sig_data[fname][u'Utility Bus V B'].values
        uVc=sig_data[fname][u'Utility Bus V C'].values
        uVal = uVa - 0.5 * (uVb + uVc)
        uVbe = sqrt(3.)/2. * (uVb - uVc)
        uVmag = 2./3.*sqrt(uVal*uVal+uVbe*uVbe)
        sig_data[fname][u'Utility Vmag'] = pd.Series(uVmag,index=sig_data[fname].index)

        # Island voltage magnitude: alpha beta -> mag
        iVa=sig_data[fname][u'Island Bus V A'].values
        iVb=sig_data[fname][u'Island Bus V B'].values
        iVc=sig_data[fname][u'Island Bus V C'].values
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
        GmAngElecFbk = -arctan2(iVbe[0],iVal[0])
        iVx   = zeros(iVa.shape) # setting output arrays to zero
        iVy   = zeros(iVa.shape)
        iWpll = ones(iVa.shape)*377.0

        for i in range(0,iVa.shape[0]):
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
        sig_data[fname][u'Island Varms'] = pd.Series(Varms,index=sig_data[fname].index)
        sig_data[fname][u'Island Vbrms'] = pd.Series(Vbrms,index=sig_data[fname].index)
        sig_data[fname][u'Island Vcrms'] = pd.Series(Vcrms,index=sig_data[fname].index)

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

        # Utility currents
        uIa=sig_data[fname][u'Utility I A'].values
        uIb=sig_data[fname][u'Utility I B'].values
        uIc=sig_data[fname][u'Utility I C'].values

        uIal = uIa - 0.5 * (uIb + uIc)
        uIbe = sqrt(3.)/2. * (uIb - uIc)
        sig_data[fname][u'uIal'] = pd.Series(uIal,index=sig_data[fname].index)
        sig_data[fname][u'uIbe'] = pd.Series(uIbe,index=sig_data[fname].index)

        # Utility Power calcuations kW
        uP = (uVa*uIa+uVb*uIb+uVc*uIc)/1000
        uQ = ((uVb-uVc)*uIa+(uVa-uVb)*uIc+(uVc-uVa)*uIb)/sqrt(3)/1000
        sig_data[fname][u'P Utility'] = pd.Series(uP,index=sig_data[fname].index)
        sig_data[fname][u'Q Utility'] = pd.Series(uQ,index=sig_data[fname].index)

        # RLC currents
        rIa=sig_data[fname][u'RLC Passive Load I A'].values
        rIb=sig_data[fname][u'RLC Passive Load I B'].values
        rIc=sig_data[fname][u'RLC Passive Load I C'].values

        # RLC power calcuations
        rP = (iVa*rIa+iVb*rIb+iVc*rIc)/1000
        rQ = ((iVb-iVc)*rIa+(iVa-iVb)*rIc+(iVc-iVa)*rIb)/sqrt(3)/1000
        sig_data[fname][u'P RLC'] = pd.Series(rP,index=sig_data[fname].index)
        sig_data[fname][u'Q RLC'] = pd.Series(rQ,index=sig_data[fname].index)

        # Amplifier currents
        ampIa=sig_data[fname][u'GE Load I A'].values
        ampIb=sig_data[fname][u'GE Load I B'].values
        ampIc=sig_data[fname][u'GE Load I C'].values

        # Amplifier power calculations
        ampP = (iVa*ampIa+iVb*ampIb+iVc*ampIc)/1000
        ampQ = ((iVb-iVc)*ampIa+(iVa-iVb)*ampIc+(iVc-iVa)*ampIb)/sqrt(3)/1000
        sig_data[fname][u'P AMP'] = pd.Series(ampP,index=sig_data[fname].index)
        sig_data[fname][u'Q AMP'] = pd.Series(ampQ,index=sig_data[fname].index)

        # B2 currents
        b2Ia=B2LC1SIGN*sig_data[fname][u'B2 LC1 I A'].values
        b2Ib=B2LC1SIGN*sig_data[fname][u'B2 LC1 I B'].values
        b2Ic=B2LC1SIGN*sig_data[fname][u'B2 LC1 I C'].values

        # B2 Power calculations
        b2P = (iVa*b2Ia+iVb*b2Ib+iVc*b2Ic)/1000
        b2Q = ((iVb-iVc)*b2Ia+(iVa-iVb)*b2Ic+(iVc-iVa)*b2Ib)/sqrt(3)/1000
        sig_data[fname][u'P B2'] = pd.Series(b2P,index=sig_data[fname].index)
        sig_data[fname][u'Q B2'] = pd.Series(b2Q,index=sig_data[fname].index)

        # B1 currents
        b1LC1=B1LC1SIGN*sig_data[fname][u'B1 LC1 I'].values
        b1LC2=sig_data[fname][u'B1 LC2 I'].values
        b1LC3=sig_data[fname][u'B1 LC3 I'].values
        
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
        penB1 = where(sig_data[fname][u'Island Contactor status'] > VMAG['icsLvl'],b1P/(rP+ampP),NaN)
        penB2 = where(sig_data[fname][u'Island Contactor status'] > VMAG['icsLvl'],b2P/(rP+ampP),NaN)
        penPV = where(sig_data[fname][u'Island Contactor status'] > VMAG['icsLvl'],(b1P+b2P)/(rP+ampP),NaN)
        sig_data[fname][u'B1 pen'] = pd.Series(penB1,index=sig_data[fname].index)
        sig_data[fname][u'B2 pen'] = pd.Series(penB2,index=sig_data[fname].index)
        sig_data[fname][u'B1+B2 pen'] = pd.Series(penPV,index=sig_data[fname].index)

        # Selecting a region of interest: island creation to cessation
        df1 = sig_data[fname]
        # ix1 = df1[abs(df1[u'Utility Vmag']-df1[u'Island Vmag'])/sqrt(2)/BASE['Vln'] > VMAG['delta']].index.values[0]
        if df1[abs(df1[u'Island Contactor status']) < VMAG['icsLvl']].empty:
            ix1 = df1.index.values[-1]/2
        else:
            ix1 = df1[abs(df1[u'Island Contactor status']) < VMAG['icsLvl']].index.values[0]
        if df1[abs(df1[u'Island Vmag'])/sqrt(2)/BASE['Vln'] < VMAG['low']].empty:
            ix2 = df1.index.values[-1]/2
        else:
            ix2 = df1[abs(df1[u'Island Vmag'])/sqrt(2)/BASE['Vln'] < VMAG['low']].index.values[0]

        tinc = sig_info[fname]['chTincr'][sig_info[fname]['chName']==u'Utility Bus V A'].values[0]
        left = int(NCY['pre']*1./60./tinc)
        right = int(NCY['post']*1./60./tinc)
        ix1 = max([ix1-left,0])
        ix2 = min([ix2+right,df1.index.values[-1]])
        df2 = df1[(df1.index > ix1) & (df1.index < ix2)]

        
        if CONFIG['WriteLimitedDataToExcel']: # Only the data from island formation to cessation
            df2.to_excel(writer,fname) # data is written here

        if True: # Place to try new things
            # Fig1: Utility voltage
            fig, (ax0, ax1)= plt.subplots(nrows=2, ncols=1, figsize=(8.5,11))
            fig.suptitle(fname) # This titles the figure
            # File info output to page top
            label= file_info[file_info.index==fname][['fiComment']].values[0][0]
            ax0.annotate(label,
                         xy=(0.5/8.5, 10.5/11), # (0.5,-0.25)inch from top left corner
                         xycoords='figure fraction',
                         horizontalalignment='left',
                         verticalalignment='top',
                         fontsize=10)
            subplots_adjust(top=9./11.)
            
            # Alpha/Beta plots
            ax0.set_title('Island Voltage Al/Be')
            ax0.plot(df2['Island Val']/1.5/sqrt(2)/BASE['Vln'], df2['Island Vbe']/1.5/sqrt(2)/BASE['Vln'])
            ax0.set_xlim([-1.5,1.5])
            ax0.set_ylim([-1.2,1.2])
            ax0.grid(True, which='both')
            ax0.set_aspect('equal')

            ax1.set_title('Currents Al/Be')
            ax1.plot(df2['pvIal']/1.5, df2['pvIbe']/1.5)
            # ax1.set_ylim([-1.2,1.2])
            ax1.grid(True, which='both')
            ax1.set_aspect('equal')
            # ax1.set_title('Island Voltage Al/Be')
            # ax1.plot(df2['Time'], df2['Island Val']/1.5/sqrt(2)/BASE['Vln'])
            # ax1.plot(df2['Time'], df2['Island Vbe']/1.5/sqrt(2)/BASE['Vln'])
            # ax1.set_ylim([-1.2,1.2])
            # ax1.grid(True, which='both')

            pltPdf.savefig() # saves fig to pdf
            plt.close() # Closes fig to clean up memory

        if False: # Adding a chart with PLL variables
            # Fig1a: 
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
            # ax2.set_ylim([-100,100])
            ax2.grid(True, which='both')

            ax3.set_title('Island Bus Vx, Vy')
            ax3.plot(df2['Time'], df2[u'Island Vx'])
            ax3.plot(df2['Time'], df2[u'Island Vy'])
            # ax3.set_ylim([-100,100])
            ax3.grid(True, which='both')

            pltPdf.savefig() # Saves fig to pdf
            plt.close() # Closes fig to clean up memory


        if CONFIG['PlotFullRange']: # Plots a page with entire length of captured signals
            # Fig2: 
            fig, (ax0,ax1,ax2,ax3,ax4) = plt.subplots(nrows=5, ncols=1,
                                                      figsize=(8.5,11),
                                                      sharex=True)
            # plt.title(fname) # this has no effect

            # ax0.set_title('Utility Bus Vabc')
            # ax0.plot(sig_data[fname]['Time'], sig_data[fname][u'Utility Bus V A'])
            # ax0.plot(sig_data[fname]['Time'], sig_data[fname][u'Utility Bus V B'])
            # ax0.plot(sig_data[fname]['Time'], sig_data[fname][u'Utility Bus V C'])
            # ax0.set_ylim([-500,500])
            # ax0.grid(True, which='both')

            ax0.set_title('Island Bus Vabc')
            ax0.plot(sig_data[fname]['Time'], sig_data[fname][u'Island Bus V A'])
            ax0.plot(sig_data[fname]['Time'], sig_data[fname][u'Island Bus V B'])
            ax0.plot(sig_data[fname]['Time'], sig_data[fname][u'Island Bus V C'])
            ax0.plot(sig_data[fname]['Time'], sig_data[fname][u'Island Vmag'])
            ax0.set_ylim([-500,500])
            ax0.grid(True, which='both')

            ax1.set_title('Island Bus Frequency')
            ax1.plot(df2['Time'], df2[u'Island Wpll']/(2*pi))
            ax1.set_ylim([-120,120])
            ax1.grid(True, which='both')

            ax2.set_title('RLC Load Current Iabc')
            ax2.plot(sig_data[fname]['Time'], sig_data[fname][u'RLC Passive Load I A'])
            ax2.plot(sig_data[fname]['Time'], sig_data[fname][u'RLC Passive Load I B'])
            ax2.plot(sig_data[fname]['Time'], sig_data[fname][u'RLC Passive Load I C'])
            ax2.set_ylim([-100,100])
            ax2.grid(True, which='both')

            ax3.set_title('B1+B2 Iabc')
            ax3.plot(sig_data[fname]['Time'], sig_data[fname][u'pvIa'])
            ax3.plot(sig_data[fname]['Time'], sig_data[fname][u'pvIb'])
            ax3.plot(sig_data[fname]['Time'], sig_data[fname][u'pvIb'])
            ax3.set_ylim([-100,100])
            ax3.grid(True, which='both')
 
            ax4.set_title('Utility Iabc')
            ax4.plot(sig_data[fname]['Time'], sig_data[fname][u'Utility I A'])
            ax4.plot(sig_data[fname]['Time'], sig_data[fname][u'Utility I B'])
            ax4.plot(sig_data[fname]['Time'], sig_data[fname][u'Utility I C'])
            ax4.set_ylim([-100,100])
            ax4.grid(True, which='both')
 
            pltPdf.savefig() # Saves fig to pdf
            plt.close() # Closes fig to clean up memory

        if CONFIG['ValidateCTs']: # Plots a page to validate CT reads and orientation
            # FigX: 
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

        # Fig3: 
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
        ax1.plot(df2['Time'], df2[u'Island Wpll']/(2*pi))
        ax1.set_ylim([-60, 180])
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
        ax2.plot(df2['Time'], df2[u'Island Vpos']/BASE['Vln'])
        ax2.plot(df2['Time'], df2[u'B1+B2 pen'])
        ax2.set_ylim([0,1.5])
        ax2.grid(True, which='both')

        ax3.set_title('Island Vneg, Vzero')
        ax3.plot(df2['Time'], df2[u'Island Vneg']/BASE['Vln'])
        ax3.plot(df2['Time'], df2[u'Island Vzer']/BASE['Vln'])
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

    print "Closing: Results.pdf"
    pltPdf.close() # Close the pdf file

    if CONFIG['WriteLimitedDataToExcel']: # Close excel file
        print "Writing: LimitedData.xlsx"
        writer.save() # file is saved here
    
    return