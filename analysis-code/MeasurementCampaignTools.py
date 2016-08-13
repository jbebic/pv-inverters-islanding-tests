# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 18:43:10 2015

@author: 200010679
"""

def PopulateMasterTestLogTable(mypath):
    """ Calls MergeTestLogs to iterate over all subdirs of mypath and 
    read in TestLogSnn Excel files and build a dataframe with all recorded
    test conditions. 
    Then, calls ProcessResults to extract scalar properties of 
    islanding test results and save them into placeholders in the 
    master table.
    Finally, saves the master table as h5 and Excel files

    Input:  Directory with the test result directories, e.g.: "aLabView2"

    Output: TestLogsAll.h5, TestLogsAll.xlsx
    Note:   ProcessResults generate Results.h5 and Results.pdf files in 
            the data subdirs.
    """
    from os import listdir
    from os.path import isdir, join

    from pandas import HDFStore, ExcelWriter
    import MeasurementGroupTools as mgt

    TestLog = MergeTestLogs(mypath)    

    mydirs = [d for d in listdir(mypath) if isdir(join(mypath,d)) ]

    # print mydirs + mydirs[1:2]
    for dname in mydirs:
        mysubdirpath = mypath + "\\" + dname
        print "Processing: " + dname
        mgt.ProcessResults(mysubdirpath, TestLog)

    h5store = HDFStore(mypath + "\\" + 'TestLogsAll.h5')
    h5store.put('TestLogsAll',TestLog)
    h5store.close()

    writer = ExcelWriter(mypath + "\\" + 'TestLogsAll.xlsx')
    TestLog.to_excel(writer,'TestLogsAll') # the second argument defines sheet name
    writer.save()

    return

def MergeTestLogs(mypath):
    """ Reads TestLogSnn Excel files from all subdirs of mypath and builds
    a dataframe with all relevant test conditions. This dataframe is later
    used to associate tdms files with test conditions by searching the file 
    numbers.

    Input: Directory with the test result directories, e.g.: "aLabView2"
    Output: Master dataframe with merged TestLog records
    """
    import pandas as pd # multidimensional data analysis
    import numpy as np  # python numerical library
    from os import listdir
    from os.path import isdir, isfile, join

    TLCFG = {'OperatorCol': 0, # TestPlan sheet
             'DGCodeCol'  : 1,
             'P3phCol'    : 2,
             'PwallCol'   : 3,
             'PgenCol'    : 4,
             'PrcntPenCol': 5,
             'RLCpwrCol'  : 6,
             'RTDSfileCol': 7,
             'QCLoadCol'  : 8,
             'PFLoadCol'  : 9,
             'LVFiNumsCol':10,
             'NrmlFlgCol' :11,
             'CommentCol' :12,
             'GEAmpPRow'  : 3, # Definitions sheet
             'GEAmpQRow'  : 1,
             'LabViewPRow': 0,
             'LabViewQRow': 4,
             'RtdsPRow'   : 7,
             'RtdsQRow'   : 8}


    # TestLog File entries for parsing into a dataframe
    # TestPlan Sheet
    # tlCompCode  = []
    tlOperator  = []
    tlDGCode    = []
    tlP3phB2    = []
    tlPwallB1   = []
    tlPgen      = []
    tlPrcntPen  = []
    tlRLCpwr    = []
    tlRTDSFile  = []
    tlQCload    = []
    tlPFload    = []
    tlLVFiNums  = []
    tlNrmlFlg   = []
    tlComment   = []
    # Definitions Sheet
    tlGEAmpQ    = []
    tlGEAmpP    = []
    tlLabViewP  = []
    tlLabViewQ  = []
    tlRtdsP     = []
    tlRtdsQ     = []

    TestLog = pd.DataFrame({
                'Operator': tlOperator,
                'DGCode': tlDGCode,
                'P3phB2': tlP3phB2,
                'PwallB1': tlPwallB1,
                'Pgen': tlPgen,
                'PrcntPen': tlPrcntPen,
                'RLCpwr': tlRLCpwr,
                'RTDSFile': tlRTDSFile,
                'QCload': tlQCload,
                'PFload': tlPFload,
                'LVFiNums': tlLVFiNums,
                'NrmlFlg': tlNrmlFlg,
                'Comment': tlComment},
                index=None)

    mydirs = [d for d in listdir(mypath) if isdir(join(mypath,d)) ]

    for dname in mydirs:
        mysubdirpath = mypath + "\\" + dname
        myfiles = [f for f in listdir(mysubdirpath) if isfile(join(mysubdirpath,f)) ]
        # filtering for xlsx extension
        TestLog_list = [f for f in myfiles if f.endswith(".xlsx") and f.startswith("TestLog")]
    
        if len(TestLog_list) < 1:
            print 'There is no TestLog file in the directory: ', mypath
            return
        
        if len(TestLog_list) > 2:
            print 'There is more than one TestLog file in the directory: ', mypath
            return
    
        print 'Reading: '+dname+'\\'+TestLog_list[0]

        TestLogData = {}
        # For when Sheet1's format differs from Sheet2
        xls = pd.ExcelFile(mysubdirpath + "\\" + TestLog_list[0])
        TestLogData['TestPlan'] = xls.parse('TestPlan', index_col=None, header=None, na_values=['NA'])
        TestLogData['Definitions'] = xls.parse('Definitions', index_col=None, header=None)
        
        NrmlFlags = TestLogData['TestPlan'].iloc[:,TLCFG['NrmlFlgCol']]
        for i in range(0,NrmlFlags.shape[0]):
            if NrmlFlags[i] == 'y' or NrmlFlags[i] == '?':
                tlOperator  = tlOperator + [TestLogData['TestPlan'].iloc[i,TLCFG['OperatorCol']]]
                tlDGCode    = tlDGCode   + [TestLogData['TestPlan'].iloc[i,TLCFG['DGCodeCol']]]
                tlP3phB2    = tlP3phB2   + [TestLogData['TestPlan'].iloc[i,TLCFG['P3phCol']]]
                tlPwallB1   = tlPwallB1  + [TestLogData['TestPlan'].iloc[i,TLCFG['PwallCol']]]
                tlPgen      = tlPgen     + [TestLogData['TestPlan'].iloc[i,TLCFG['PgenCol']]]
                tlPrcntPen  = tlPrcntPen + [TestLogData['TestPlan'].iloc[i,TLCFG['PrcntPenCol']]]
                tlRLCpwr    = tlRLCpwr   + [TestLogData['TestPlan'].iloc[i,TLCFG['RLCpwrCol']]]
                tlRTDSFile  = tlRTDSFile + [TestLogData['TestPlan'].iloc[i,TLCFG['RTDSfileCol']]]
                tlQCload    = tlQCload   + [TestLogData['TestPlan'].iloc[i,TLCFG['QCLoadCol']]]
                tlPFload    = tlPFload   + [TestLogData['TestPlan'].iloc[i,TLCFG['PFLoadCol']]]
                tlLVFiNums  = tlLVFiNums + [str(TestLogData['TestPlan'].iloc[i,TLCFG['LVFiNumsCol']])]
                tlNrmlFlg   = tlNrmlFlg  + [TestLogData['TestPlan'].iloc[i,TLCFG['NrmlFlgCol']]]
                tlComment   = tlComment  + [TestLogData['TestPlan'].iloc[i,TLCFG['CommentCol']]]

        temp = pd.DataFrame({
                'Operator': tlOperator,
                'DGCode': tlDGCode,
                'P3phB2': tlP3phB2,
                'PwallB1': tlPwallB1,
                'Pgen': tlPgen,
                'PrcntPen': tlPrcntPen,
                'RLCpwr': tlRLCpwr,
                'RTDSFile': tlRTDSFile,
                'QCload': tlQCload,
                'PFload': tlPFload,
                'LVFiNums': tlLVFiNums,
                'NrmlFlg': tlNrmlFlg,
                'Comment': tlComment},
                index=None)
        temp['DirName']  = dname
        temp['FileName'] = TestLog_list[0]
        temp['CompCode'] = TestLogData['TestPlan'].iloc[0,0]

        # print 'Definitions sheet has: ', TestLogData['Definitions'].iloc[:,0].size, ' values'
        if TestLogData['Definitions'].iloc[:,0].size >=8:
            temp['GEAmpP'] = TestLogData['Definitions'].iloc[TLCFG['GEAmpPRow'],0]
            temp['GEAmpQ'] = TestLogData['Definitions'].iloc[TLCFG['GEAmpQRow'],0]
            temp['LabViewP'] = TestLogData['Definitions'].iloc[TLCFG['LabViewPRow'],0]
            temp['LabViewQ'] = TestLogData['Definitions'].iloc[TLCFG['LabViewQRow'],0]
            temp['RtdsP'] = TestLogData['Definitions'].iloc[TLCFG['RtdsPRow'],0]
            temp['RtdsQ'] = TestLogData['Definitions'].iloc[TLCFG['RtdsQRow'],0]

        TestLog = TestLog.append(temp, ignore_index=True)

        del tlOperator[:]
        del tlDGCode[:]
        del tlP3phB2[:]
        del tlPwallB1[:]
        del tlPgen[:]
        del tlPrcntPen[:]
        del tlRLCpwr[:]
        del tlRTDSFile[:]
        del tlQCload[:]
        del tlPFload[:]
        del tlLVFiNums[:]
        del tlNrmlFlg[:]
        del tlComment[:]

    TestLog['lvComment'] = 'n/a'
    TestLog['lvProgRev'] = 'n/a'
    TestLog['TDMSfname'] = 'n/a'
    TestLog['TDMSfnum']  = 'n/a'
    TestLog['tStart']    = np.nan
    TestLog['tEnd']      = np.nan
    # Attributes
    TestLog['tIslStart'] = np.nan
    TestLog['tIslEnd']   = np.nan
    TestLog['tIslDur']   = np.nan
    TestLog['ix1']       = np.nan
    TestLog['ix1left']   = np.nan
    TestLog['ix2']       = np.nan
    TestLog['ix2right']  = np.nan
    TestLog['ixflags']   = -1

    TestLog['fIslStd']   = np.nan
    TestLog['fIslMean']  = np.nan
    TestLog['fIslMin']   = np.nan
    TestLog['fIslMax']   = np.nan
    TestLog['fIslSkw']   = np.nan

    TestLog['VmagMax']   = np.nan

    TestLog['VposMax']   = np.nan # placeholders for original sequence comp
    TestLog['VnegMax']   = np.nan
    TestLog['VzerMax']   = np.nan

    TestLog['V0magMax']  = np.nan # placeholders for corrected sequence comp
    TestLog['V1magMax']  = np.nan
    TestLog['V2magMax']  = np.nan
    
    return TestLog
    
def AddSeqComp(mypath):
    """ Loads TestLogAll.h5 from the specified path, then calls 
    MeasurementGroupTools.AddSeqComp to recalculate seq components using FFT  

    Input:  Directory of the measurment campaign, e.g.: "aLabView2"
    Output: Results1.h5, Results1.pdf in the data subdirs.
    """
    from pandas import HDFStore, ExcelWriter
    import MeasurementGroupTools as mgt

    h5logs = HDFStore(mypath + "\\" + 'TestLogsAll.h5')
    TestLog = h5logs['TestLogsAll']

    dirs = TestLog[u'DirName'].unique()
    for dname in dirs:
        mysubdirpath = mypath + "\\" + dname
        print "Processing: " + dname
        mgt.AddSeqComp(mysubdirpath, TestLog, dname)

    h5logs.put('TestLogsAll',TestLog)
    h5logs.close()

    writer = ExcelWriter(mypath + "\\" + 'TestLogsAll.xlsx')
    TestLog.to_excel(writer,'TestLogsAll') # the second argument defines sheet name
    writer.save()

    return

def ReplotAllResults(mypath):
    """ Loads TestLogAll.h5 from the specified path, then calls 
    MeasurementGroupTools.ReplotResults to replot

    Input:  Directory of the measurment campaign, e.g.: "aLabView2"
    Output: Results1.pdf in the data subdirs.
    """
    from pandas import HDFStore
    import MeasurementGroupTools as mgt

    h5logs = HDFStore(mypath + "\\" + 'TestLogsAll.h5')
    TestLog = h5logs['TestLogsAll']

    dirs = TestLog[u'DirName'].unique()
    for dname in dirs:
        mysubdirpath = mypath + "\\" + dname
        print "Processing: " + dname
        tlogs1 = TestLog[TestLog[u'DirName'] == dname]
        mgt.ReplotResults(mysubdirpath, tlogs1)

    h5logs.close()

    return
