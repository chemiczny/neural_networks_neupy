#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:36:04 2019

@author: michal
"""

import sys
from os.path import isdir
if isdir("/net/people/plgglanow/pythonPackages") and not "/net/people/plgglanow/pythonPackages" in sys.path :
    sys.path.insert(0, "/net/people/plgglanow/pythonPackages" )

from os import mkdir
from neupy import algorithms
import pickle
from time import time        

class SOFMgenerator:
    def __init__(self, dataObject, filesId = "0"):
        self.dataObject = dataObject
        
        self.X = self.dataObject.trainX.values
        self.n_inputs = self.X.shape[1]

        
        if not isdir("savedSOFM"):
            mkdir("savedSOFM")
            
        if not isdir("results"):
            mkdir("results")
            
        self.learningRadius = [ 0, 1, 2]
        self.gridType = [ "rect" , "hexagon" ]
        self.weightInit = [ "init_pca", "sample_from_data", "default" ]
        
        self.bestSOFMfile = "savedSOFM/ANNbest"+filesId+".pickle"
        self.resultsLog = "results/results"+filesId+".dat"
        self.lowestError = 999
        
        self.initResultsFile()
        
        self.fileId = filesId
        
        
    def initResultsFile(self):
        resFile = open(self.resultsLog, 'w')
        
        resFile.write("ID\tOutput rows\tOutput cols\tLearning radius\t")
        resFile.write("Weight init\tGrid type\tBad classified\tPrediction error\n")
        
        resFile.close()
                        
    def runSOFM(self, outputRows, outputCols, learningRadius, weightInit, gridType):

        if weightInit == "default" :
            sofm = algorithms.SOFM( n_inputs = self.n_inputs, features_grid = (outputRows,outputCols) ,
                                   verbose = False, step = 0.5, learning_radius = learningRadius, grid_type = gridType )
            
        else:
            sofm = algorithms.SOFM( n_inputs = self.n_inputs, features_grid = (outputRows,outputCols) ,
                                   verbose = False, step = 0.5, learning_radius = learningRadius, grid_type = gridType,
                                   weight = weightInit)
        try:
            trainingStart = time()
            sofm.train( self.X , epochs = 300 )
            trainingTime = time() - trainingStart
            
            predictionResults, error, errorPercent = self.dataObject.verifyPrediction(  sofm.predict( self.X ) )
            
            if error < self.lowestError:
                self.lowestError = error
                with open( self.bestSOFMfile , 'wb') as f:
                    pickle.dump(sofm, f)
            
            self.logRow( outputRows, outputCols, learningRadius, weightInit, gridType, error, errorPercent )
        except:
            pass
        
                
    def logRow( self, outputRows, outputCols, learningRadius, weightInit, gridType, error, errorPercent):
        logFile = open( self.resultsLog, "a" )
        
        logFile.write( self.fileId + "\t"+str(outputRows)+"\t"+
                      str(outputCols)+"\t"+str(learningRadius)+"\t"+
                      weightInit+"\t"+str(gridType)+
                      "\t"+str(error)+"\t"+str(errorPercent)+"\n")
        
        logFile.close()
