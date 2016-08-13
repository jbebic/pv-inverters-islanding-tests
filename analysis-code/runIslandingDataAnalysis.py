# -*- coding: utf-8 -*-
"""
Created on Sun Aug 07 19:58:04 2016

@author: 200010679
"""

import MeasurementCampaignTools as mct
# import MeasurementGroupTools as mgt

dirname = 'aLabView1a3'
mct.PopulateMasterTestLogTable(dirname)
# mct.AddSeqComp(dirname)
mct.ReplotAllResults(dirname)
