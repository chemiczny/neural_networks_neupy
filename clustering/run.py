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



inputFile = open(inputName, 'r')

line = inputFile.readline()
i = 0
while line:
    args = line.split()
    columns2remove = line.split("|")[1]
    columns2remove = columns2remove.split(":")
    
    data = DataPreprocessor()
    data.removeFromInputcolumns(columns2remove)
    
    initResFile = True
    if i > 0:
        initResFile = False
    sofm = SOFMgenerator(data, fileId, initResFile)
    sofm.runSOFM( int(args[0]), int(args[1]), int(args[2]), args[3], args[4] )
    
    line = inputFile.readline()
    i += 1


inputFile.close()