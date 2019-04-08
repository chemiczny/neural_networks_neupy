#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:34:42 2019

@author: michal
"""

from os.path import isdir, join
from os import mkdir, remove
from glob import glob
from neuralNetworkGenerator import NeuralNetworkManager
from dataPreprocessor import DataPreprocessor

inputDir = "inputs"

if not isdir(inputDir):
    mkdir(inputDir)
    
for inputFile in glob(join( inputDir, "*.inp" )):
    remove(inputFile)
    

data = DataPreprocessor()
nnm = NeuralNetworkManager(data)

dataPerFile = 20

nNeuronsInHiddenLayer = range(  nnm.nOutputs + 1, nnm.nInputs )
runsForOneNN = 10

actualI = 0
actualFile = open( join( inputDir, str(actualI)+".inp" ),'w' )
dataInActualFile = 0

for hiddenNno in nNeuronsInHiddenLayer:
    for afHidden in nnm.activationFunctionHiddenLayer:
        for afOut in nnm.activationFunctionOutputLayer:
            for optim in nnm.optimizers:
                for i in range(runsForOneNN):
                    data = [ str(hiddenNno) , afHidden, afOut, optim ]
                    data = " ".join(data) + "\n"
                    actualFile.write(data)
                    
                    dataInActualFile += 1
                    
                    if dataInActualFile >= dataPerFile:
                        dataInActualFile = 0
                        actualFile.close()
                        actualI += 1
                        
                        actualFile = open( join( inputDir, str(actualI)+".inp" ),'w' )
                        
actualFile.close()