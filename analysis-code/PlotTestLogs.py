# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 18:43:10 2015

@author: 200010679
"""

def PlotTestLogsAll(myfile):
    """ Reads the specifed TestLogsAll.HDF5 file and calculates and plots relationships. 
    
    Input: File name with the results, e.g.: "aLabView2\\TestLogsAll.h5"

    Output: Plots in the same directory
    """
    import pandas as pd # multidimensional data analysis
    import numpy as np  # python numerical library
    # from os import listdir
    # from os.path import isdir, isfile, join
    
    import matplotlib.pyplot as plt
    # from matplotlib.backends.backend_pdf import PdfPages
    # Wei's advice ===
    import matplotlib.backends.backend_pdf as dpdf

    from pandas import read_hdf, HDFStore, ExcelWriter

    CONFIG = {'PlotIslDurHist':True, # Island durations histogram
              'PlotCorDur2Pen':True, # Correlation of duration to penetration
              'PlotCorDur2fStd':True, # Correlation of duration to standard deviation of frequency
             }

    if myfile.endswith(".h5"):
        mypdffile = "".join(myfile.split(".")[0:-1] + ['.pdf'])
    else:
        mypdffile = myfile + '.pdf'

    print "Opening: " + myfile
    h5store = HDFStore(myfile)
    TestLog = h5store.get('TestLogsAll')

    print "Opening: " + mypdffile
    pltPdf = dpdf.PdfPages(mypdffile)

    # Filtering TestLog into df1
    df1 = TestLog[(TestLog['tIslDur'] > 0.) & 
                  (TestLog['NrmlFlg'] == 'y') & 
                  (TestLog['FileName'] != 'TestLogMotorBr.xlsx') & 
                  (TestLog['FileName'] != 'TestLogSummer01.xlsx')]
    # Adding details to df1
    df1['QCload0']=df1['QCload']
    df1.loc[(df1[df1['QCload0']<0].index), ('QCload0')] =0.0 # df1['QCload0'][df1['QCload0']<0] = 0.0
    df1['PFact']=df1['LabViewP']/(df1['LabViewP']**2 + (-df1['GEAmpQ']+df1['QCload0'])**2).apply(np.sqrt)
    df1['PFactsign']='ind'
    df1.loc[(df1[df1['QCload0']>df1['GEAmpQ']].index), ('PFactsign')]='cap' # df1['PFactsign'][df1['QCload0']>df1['GEAmpQ']]='cap'
    df1s = df1[df1['FileName'].str.contains('Summer')]
    df1w = df1[df1['FileName'].str.contains('Winter')]
    
    if CONFIG['PlotIslDurHist']: # Island duration histogram
        # Fig: Island duration histogram
        fig, (ax0)= plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        # provision for a label
        # fig.suptitle(myfile) # This titles the figure
        # File info output to page top
        # label= file_info[file_info.index==fname][['fiComment']].values[0][0]
        # label = myfile
        # ax0.annotate(label,
        #              xy=(0.2/6.4, 4.6/4.8), # (0.2,-0.2)inch from top left corner
        #              xycoords='figure fraction',
        #              horizontalalignment='left',
        #              verticalalignment='top',
        #              fontsize=10)
        # subplots_adjust(top=4./4.8)

        df2a = TestLog['tIslDur'][TestLog['tIslDur'] > 0.]
        ax0.set_title('Island Duration Histogram')
        df2a.plot(kind='hist', bins=20, ax=ax0, alpha=0.5) # legend=True
        # df2.plot(kind='hist', bins=20, ax=ax0, alpha=0.5, legend=True)
        # ax0.set_xlim([-1.5,1.5])
        # ax0.set_ylim([-1.2,1.2])
        ax0.grid(True, which='both')
        ax0.set_xlabel('Island duration (sec)')
        ax0.set_ylabel('Number of observations')
        # ax0.set_aspect('equal')

        # ax1.set_title('Currents Al/Be')
        # ax1.plot(df2['pvIal']/1.5, df2['pvIbe']/1.5)
        # ax1.set_xlim([-300,300])
        # ax1.set_ylim([-240,240])
        # ax1.grid(True, which='both')
        # ax1.set_aspect('equal')
        # ax1.set_title('Island Voltage Al/Be')
        # ax1.plot(df2['Time'], df2['Island Val']/1.5/sqrt(2)/BASE['Vln'])
        # ax1.plot(df2['Time'], df2['Island Vbe']/1.5/sqrt(2)/BASE['Vln'])
        # ax1.set_ylim([-1.2,1.2])
        # ax1.grid(True, which='both')

        pltPdf.savefig() # saves fig to pdf
        plt.close() # Closes fig to clean up memory

        # Fig: PF actual histogram
        fig, (ax0)= plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        df2a = df1['PFact']
        ax0.set_title('Load PF actual')
        df2a.plot(kind='hist', bins=20, ax=ax0, alpha=0.5) # legend=True
        ax0.grid(True, which='both')
        ax0.set_xlabel('PF actual')
        ax0.set_ylabel('Number of observations')
        pltPdf.savefig() # saves fig to pdf
        plt.close() # Closes fig to clean up memory


    if CONFIG['PlotCorDur2Pen']: # Island duration corelations 
        # Fig: duration to penetration
        fig, (ax0)= plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        ax0.set_title('Island Duration vs. Penetration')
        df2a = TestLog[['tIslDur','PrcntPen']][(TestLog['tIslDur'] > 0.)&(TestLog['NrmlFlg'] == 'y') & (TestLog['FileName'] != 'TestLogMotorBr.xlsx')]
        df2b = TestLog[['tIslDur','PrcntPen']][(TestLog['tIslDur'] >  0.) & (TestLog['NrmlFlg'] == 'y') & (TestLog['FileName'] == 'TestLogMotorBr.xlsx')]
        df2a1 = df2a[['tIslDur','PrcntPen']][(df2a['PrcntPen'] > 0.95) & (df2a['PrcntPen'] < 1.05)]
        df2a2 = df2a[['tIslDur','PrcntPen']][(df2a['PrcntPen'] < 0.95) | (df2a['PrcntPen'] > 1.05)]
        ax0.plot(df2a2['PrcntPen'], df2a2['tIslDur'], 'bo', markersize=4, markeredgecolor='none', label='CMPLDs')
        ax0.plot(df2a1['PrcntPen']-df2a1['PrcntPen'].mean() + 1.0, df2a1['tIslDur'], 'bo', markersize=4, markeredgecolor='none', label='')
        ax0.plot(df2b['PrcntPen'], df2b['tIslDur'], 'ro', markersize=4, markeredgecolor='none', label='MotorB')
        ax0.grid(True, which='both')
        ax0.set_xlabel('PV Penetration (pu)')
        ax0.set_ylabel('Island Duration (sec)')
        # Now add the legend with some customizations.
        legend = ax0.legend(loc='upper left', shadow=False)
        for label in legend.get_texts():
            label.set_fontsize('small')
        # plt.legend()

        pltPdf.savefig() # saves fig to pdf
        plt.close() # Closes fig to clean up memory

        # Fig: duration to penetration with PF as a parameter
        fig, (ax0)= plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        ax0.set_title('Island Duration vs. Penetration')
        df2a = df1[['tIslDur','PrcntPen']][(df1['PFact'] < 0.97) & (df1['PFactsign'] == 'ind')]
        df2a1 = df2a[['tIslDur','PrcntPen']][(df2a['PrcntPen'] > 0.95) & (df2a['PrcntPen'] < 1.05)]
        df2a2 = df2a[['tIslDur','PrcntPen']][(df2a['PrcntPen'] < 0.95) | (df2a['PrcntPen'] > 1.05)]
        df2b = df1[['tIslDur','PrcntPen']][(df1['PFact'] < 0.99) & (df1['PFact'] >= 0.97) & (df1['PFactsign'] == 'ind')]
        df2b1 = df2b[['tIslDur','PrcntPen']][(df2b['PrcntPen'] > 0.95) & (df2b['PrcntPen'] < 1.05)]
        df2b2 = df2b[['tIslDur','PrcntPen']][(df2b['PrcntPen'] < 0.95) | (df2b['PrcntPen'] > 1.05)]
        df2c = df1[['tIslDur','PrcntPen']][(df1['PFact'] > 0.99)]
        df2c1 = df2c[['tIslDur','PrcntPen']][(df2c['PrcntPen'] > 0.95) & (df2c['PrcntPen'] < 1.05)]
        df2c2 = df2c[['tIslDur','PrcntPen']][(df2c['PrcntPen'] < 0.95) | (df2c['PrcntPen'] > 1.05)]
        df2d = df1[['tIslDur','PrcntPen']][(df1['PFact'] < 0.99) & (df1['PFactsign'] == 'cap')]
        df2d1 = df2d[['tIslDur','PrcntPen']][(df2d['PrcntPen'] > 0.95) & (df2d['PrcntPen'] < 1.05)]
        df2d2 = df2d[['tIslDur','PrcntPen']][(df2d['PrcntPen'] < 0.95) | (df2d['PrcntPen'] > 1.05)]
        print df2a['tIslDur'].count()
        print df2b['tIslDur'].count()
        print df2c['tIslDur'].count()
        print df2d['tIslDur'].count()
        ax0.plot(df2d1['PrcntPen']-df2d1['PrcntPen'].mean()+1.0, df2d1['tIslDur'], 'ko', markersize=4, markeredgecolor='none', label='')
        ax0.plot(df2d2['PrcntPen'], df2d2['tIslDur'], 'ko', markersize=4, markeredgecolor='none', label='PF ~ 0.98cap')
        ax0.plot(df2c1['PrcntPen']-df2c1['PrcntPen'].mean()+1.0, df2c1['tIslDur'], 'ro', markersize=4, markeredgecolor='none', label='')
        ax0.plot(df2c2['PrcntPen'], df2c2['tIslDur'], 'ro', markersize=4, markeredgecolor='none', label='PF ~ 1.0')
        ax0.plot(df2b1['PrcntPen']-df2b1['PrcntPen'].mean()+1.0, df2b1['tIslDur'], 'go', markersize=4, markeredgecolor='none', label='')
        ax0.plot(df2b2['PrcntPen'], df2b2['tIslDur'], 'go', markersize=4, markeredgecolor='none', label='PF ~ 0.98ind')
        ax0.plot(df2a1['PrcntPen']-df2a1['PrcntPen'].mean()+1.0, df2a1['tIslDur'], 'bo', markersize=4, markeredgecolor='none', label='')
        ax0.plot(df2a2['PrcntPen'], df2a2['tIslDur'], 'bo', markersize=4, markeredgecolor='none', label='PF ~ 0.95ind')
        ax0.grid(True, which='both')
        ax0.set_xlabel('PV Penetration (pu)')
        ax0.set_ylabel('Island Duration (sec)')
        ax0.set_ylim([0,0.6])
        # Now add the legend with some customizations.
        legend = ax0.legend(loc='upper left', shadow=False)
        for label in legend.get_texts():
            label.set_fontsize('small')
        # plt.legend()

        pltPdf.savefig() # saves fig to pdf
        plt.close() # Closes fig to clean up memory

        # Fig: duration to penetration at PF with season as parameter
        fig, (ax0)= plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        ax0.set_title('Island Duration vs. Penetration at PF ~ 0.95ind')
        df2a = df1s[['tIslDur','PrcntPen']][(df1s['PFact'] < 0.97) & (df1s['PFactsign'] == 'ind')]
        df2a1 = df2a[['tIslDur','PrcntPen']][(df2a['PrcntPen'] > 0.95) & (df2a['PrcntPen'] < 1.05)]
        df2a2 = df2a[['tIslDur','PrcntPen']][(df2a['PrcntPen'] < 0.95) | (df2a['PrcntPen'] > 1.05)]
        df2b = df1w[['tIslDur','PrcntPen']][(df1w['PFact'] < 0.97) & (df1w['PFactsign'] == 'ind')]
        df2b1 = df2b[['tIslDur','PrcntPen']][(df2b['PrcntPen'] > 0.95) & (df2b['PrcntPen'] < 1.05)]
        df2b2 = df2b[['tIslDur','PrcntPen']][(df2b['PrcntPen'] < 0.95) | (df2b['PrcntPen'] > 1.05)]
        print "Summer: " + str(df2a['tIslDur'].count())
        print "Winter: " + str(df2b['tIslDur'].count())
        ax0.plot(df2b1['PrcntPen']-df2b1['PrcntPen'].mean()+1.0, df2b1['tIslDur'], 'bo', markersize=4, markeredgecolor='none', label='')
        ax0.plot(df2b2['PrcntPen'], df2b2['tIslDur'], 'bo', markersize=4, markeredgecolor='none', label='winter')
        ax0.plot(df2a1['PrcntPen']-df2a1['PrcntPen'].mean()+1.0, df2a1['tIslDur'], 'ro', markersize=4, markeredgecolor='none', label='')
        ax0.plot(df2a2['PrcntPen'], df2a2['tIslDur'], 'ro', markersize=4, markeredgecolor='none', label='summer')
        ax0.grid(True, which='both')
        ax0.set_xlabel('PV Penetration (pu)')
        ax0.set_ylabel('Island Duration (sec)')
        ax0.set_ylim([0,0.6])
        # Now add the legend with some customizations.
        legend = ax0.legend(loc='upper left', shadow=False)
        for label in legend.get_texts():
            label.set_fontsize('small')
        # plt.legend()

        pltPdf.savefig() # saves fig to pdf
        plt.close() # Closes fig to clean up memory

        # Fig: duration to penetration at PF with season as parameter
        fig, (ax0)= plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        ax0.set_title('Island Duration vs. Penetration at PF ~ 0.98ind')
        df2a = df1s[['tIslDur','PrcntPen']][(df1s['PFact'] > 0.97) & (df1s['PFact'] < 0.99) & (df1s['PFactsign'] == 'ind')]
        df2a1 = df2a[['tIslDur','PrcntPen']][(df2a['PrcntPen'] > 0.95) & (df2a['PrcntPen'] < 1.05)]
        df2a2 = df2a[['tIslDur','PrcntPen']][(df2a['PrcntPen'] < 0.95) | (df2a['PrcntPen'] > 1.05)]
        df2b = df1w[['tIslDur','PrcntPen']][(df1w['PFact'] > 0.97) & (df1w['PFact'] < 0.99)  & (df1w['PFactsign'] == 'ind')]
        df2b1 = df2b[['tIslDur','PrcntPen']][(df2b['PrcntPen'] > 0.95) & (df2b['PrcntPen'] < 1.05)]
        df2b2 = df2b[['tIslDur','PrcntPen']][(df2b['PrcntPen'] < 0.95) | (df2b['PrcntPen'] > 1.05)]
        print "Summer: " + str(df2a['tIslDur'].count())
        print "Winter: " + str(df2b['tIslDur'].count())
        ax0.plot(df2b1['PrcntPen']-df2b1['PrcntPen'].mean()+1.0, df2b1['tIslDur'], 'bo', markersize=4, markeredgecolor='none', label='')
        ax0.plot(df2b2['PrcntPen'], df2b2['tIslDur'], 'bo', markersize=4, markeredgecolor='none', label='winter')
        ax0.plot(df2a1['PrcntPen']-df2a1['PrcntPen'].mean()+1.0, df2a1['tIslDur'], 'ro', markersize=4, markeredgecolor='none', label='')
        ax0.plot(df2a2['PrcntPen'], df2a2['tIslDur'], 'ro', markersize=4, markeredgecolor='none', label='summer')
        ax0.grid(True, which='both')
        ax0.set_xlabel('PV Penetration (pu)')
        ax0.set_ylabel('Island Duration (sec)')
        ax0.set_ylim([0,0.6])
        # Now add the legend with some customizations.
        legend = ax0.legend(loc='upper left', shadow=False)
        for label in legend.get_texts():
            label.set_fontsize('small')
        # plt.legend()

        pltPdf.savefig() # saves fig to pdf
        plt.close() # Closes fig to clean up memory

        # Fig: duration to penetration at PF with season as parameter
        fig, (ax0)= plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        ax0.set_title('Island Duration vs. Penetration at PF ~ 1.0')
        df2a = df1s[['tIslDur','PrcntPen']][(df1s['PFact'] > 0.99)]
        df2a1 = df2a[['tIslDur','PrcntPen']][(df2a['PrcntPen'] > 0.95) & (df2a['PrcntPen'] < 1.05)]
        df2a2 = df2a[['tIslDur','PrcntPen']][(df2a['PrcntPen'] < 0.95) | (df2a['PrcntPen'] > 1.05)]
        df2b = df1w[['tIslDur','PrcntPen']][(df1w['PFact'] > 0.99)]
        df2b1 = df2b[['tIslDur','PrcntPen']][(df2b['PrcntPen'] > 0.95) & (df2b['PrcntPen'] < 1.05)]
        df2b2 = df2b[['tIslDur','PrcntPen']][(df2b['PrcntPen'] < 0.95) | (df2b['PrcntPen'] > 1.05)]
        print "Summer: " + str(df2a['tIslDur'].count())
        print "Winter: " + str(df2b['tIslDur'].count())
        ax0.plot(df2b1['PrcntPen']-df2b1['PrcntPen'].mean()+1.0, df2b1['tIslDur'], 'bo', markersize=4, markeredgecolor='none', label='')
        ax0.plot(df2b2['PrcntPen'], df2b2['tIslDur'], 'bo', markersize=4, markeredgecolor='none', label='winter')
        ax0.plot(df2a1['PrcntPen']-df2a1['PrcntPen'].mean()+1.0, df2a1['tIslDur'], 'ro', markersize=4, markeredgecolor='none', label='')
        ax0.plot(df2a2['PrcntPen'], df2a2['tIslDur'], 'ro', markersize=4, markeredgecolor='none', label='summer')
        ax0.grid(True, which='both')
        ax0.set_xlabel('PV Penetration (pu)')
        ax0.set_ylabel('Island Duration (sec)')
        ax0.set_ylim([0,0.6])
        # Now add the legend with some customizations.
        legend = ax0.legend(loc='upper left', shadow=False)
        for label in legend.get_texts():
            label.set_fontsize('small')
        # plt.legend()

        pltPdf.savefig() # saves fig to pdf
        plt.close() # Closes fig to clean up memory

        # Fig: duration to penetration at PF with season as parameter
        fig, (ax0)= plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        ax0.set_title('Island Duration vs. Penetration at PF ~ 0.98cap')
        df2a = df1s[['tIslDur','PrcntPen']][(df1s['PFact'] > 0.97) & (df1s['PFact'] < 0.99) & (df1s['PFactsign'] == 'cap')]
        df2a1 = df2a[['tIslDur','PrcntPen']][(df2a['PrcntPen'] > 0.95) & (df2a['PrcntPen'] < 1.05)]
        df2a2 = df2a[['tIslDur','PrcntPen']][(df2a['PrcntPen'] < 0.95) | (df2a['PrcntPen'] > 1.05)]
        df2b = df1w[['tIslDur','PrcntPen']][(df1w['PFact'] > 0.97) & (df1w['PFact'] < 0.99)  & (df1w['PFactsign'] == 'cap')]
        df2b1 = df2b[['tIslDur','PrcntPen']][(df2b['PrcntPen'] > 0.95) & (df2b['PrcntPen'] < 1.05)]
        df2b2 = df2b[['tIslDur','PrcntPen']][(df2b['PrcntPen'] < 0.95) | (df2b['PrcntPen'] > 1.05)]
        print "Summer: " + str(df2a['tIslDur'].count())
        print "Winter: " + str(df2b['tIslDur'].count())
        ax0.plot(df2b1['PrcntPen']-df2b1['PrcntPen'].mean()+1.0, df2b1['tIslDur'], 'bo', markersize=4, markeredgecolor='none', label='')
        ax0.plot(df2b2['PrcntPen'], df2b2['tIslDur'], 'bo', markersize=4, markeredgecolor='none', label='winter')
        ax0.plot(df2a1['PrcntPen']-df2a1['PrcntPen'].mean()+1.0, df2a1['tIslDur'], 'ro', markersize=4, markeredgecolor='none', label='')
        ax0.plot(df2a2['PrcntPen'], df2a2['tIslDur'], 'ro', markersize=4, markeredgecolor='none', label='summer')
        ax0.grid(True, which='both')
        ax0.set_xlabel('PV Penetration (pu)')
        ax0.set_ylabel('Island Duration (sec)')
        ax0.set_ylim([0,0.6])
        # Now add the legend with some customizations.
        legend = ax0.legend(loc='upper left', shadow=False)
        for label in legend.get_texts():
            label.set_fontsize('small')
        # plt.legend()

        pltPdf.savefig() # saves fig to pdf
        plt.close() # Closes fig to clean up memory



    if False: # 
        # Fig2: 
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

    print "Closing: " + mypdffile
    pltPdf.close() # Close the pdf file
    
    
    h5store.close()
    return

PlotTestLogsAll("aLabView2\\TestLogsAll_20160809.h5")
