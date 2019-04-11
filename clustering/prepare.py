#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:34:42 2019

@author: michal
"""

from os.path import isdir, join
from os import mkdir, remove
from glob import glob
from SOFMgenerator import SOFMgenerator
from dataPreprocessor import DataPreprocessor

inputDir = "inputs"

if not isdir(inputDir):
    mkdir(inputDir)
    
for inputFile in glob(join( inputDir, "*.inp" )):
    remove(inputFile)
    

dataObj = DataPreprocessor()
sofm = SOFMgenerator(dataObj)

dataPerFile = 4

nRows = range(5, 11)
nCols = range(5, 11)

repeatNetwork = 3

nnCounter = 0

actualI = 0
actualFile = open( join( inputDir, str(actualI)+".inp" ),'w' )
dataInActualFile = 0

Rthreshold = [  0.9, 0.875, 0.85]

for repeat in range(repeatNetwork):
    for rt in Rthreshold:
        
        for nr in nRows:
            for nc in nCols:
                for lr in sofm.learningRadius:
                    if lr >= max(nr, nc):
                        continue
                    
                    for wi in sofm.weightInit:
                        for gt in sofm.gridType:
                            if gt == "hexagon" and wi == "init_pca":
                                continue
                            
                            data = [ str(nr) , str(nc), str(lr), wi, gt, str(rt) ]
                            data = " ".join(data) + "\n"
                            actualFile.write(data)
                            
                            dataInActualFile += 1
                            nnCounter += 1
                            
                            if dataInActualFile >= dataPerFile:
                                dataInActualFile = 0
                                actualFile.close()
                                actualI += 1
                                
                                actualFile = open( join( inputDir, str(actualI)+".inp" ),'w' )
                        
actualFile.close()
print(nnCounter, actualI)