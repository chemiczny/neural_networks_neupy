#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:51:35 2019

@author: michal
"""

import sys
from SOFMgenerator import SOFMgenerator
from dataPreprocessor import DataPreprocessor
from os.path import basename


inputName = sys.argv[1]

fileId = basename(inputName)
fileId = inputName.replace(".inp", "")
fileId = inputName.replace("/", "")

data = DataPreprocessor()
sofm = SOFMgenerator(data, fileId)

inputFile = open(inputName, 'r')

line = inputFile.readline()

while line:
    args = line.split()
    sofm.runSOFM( int(args[0]), args[1], args[2], args[3], args[4] )
    
    line = inputFile.readline()


inputFile.close()