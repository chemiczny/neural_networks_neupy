#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:35:20 2019

@author: michal
"""
import sys
from os.path import isdir
from os import mkdir
if isdir("/net/people/plgglanow/pythonPackages") and not "/net/people/plgglanow/pythonPackages" in sys.path :
    sys.path.insert(0, "/net/people/plgglanow/pythonPackages" )
    
from neupy import layers, algorithms
from neupy.exceptions import StopTraining
import pickle
from time import time
import numpy as np

class ControlTraining(object):
    def __init__(self, epochsNoCheck, maxEpochsValidationErrorIsRising):
        self.epochNoCheck = epochsNoCheck
        self.epochsNo = 0
        
        self.epochsValidationErrorIsRising = 0
        self.maxEpochsValidationErrorIsRising = maxEpochsValidationErrorIsRising
        self.minimumValidationError = 100000000
        self.maximumTime = 60*2
        self.timeStart = time()
        self.status = "Epochs no exceeded"
        self.epochSelected = -1
        
    
    def epoch_end(self, optimizer):
        self.epochsNo += 1
        actualValidationError = optimizer.errors.valid[-1]
        
        if actualValidationError < self.minimumValidationError:
            self.minimumValidationError = actualValidationError
            self.epochSelected = self.epochsNo
        
        
        if self.epochsNo < self.epochNoCheck:
            return
        
        lastValidationError = optimizer.errors.valid[-2]
        
        if actualValidationError > lastValidationError:
            self.epochsValidationErrorIsRising += 1
        else:
            self.epochsValidationErrorIsRising = 0
            
        if self.epochsValidationErrorIsRising > self.maxEpochsValidationErrorIsRising:
            self.status = "Testing error rise"
            raise StopTraining("Training has been interrupted")
            
        last20trainErrors = optimizer.errors.train[-20:]
        
        diff = max(last20trainErrors) - min(last20trainErrors)
        
        if diff < 0.000000001:
            self.status = "Error stabilized"
            raise StopTraining("Training has been interrupted")
            
        timeTaken = time() - self.timeStart
        
        if timeTaken > self.maximumTime:
            self.status = "Time exceeded"
            raise StopTraining("Training has been interrupted")
    
        
class NeuralNetworkManager:
    def __init__(self, dataObject, filesId = "0"):
        self.dataObject = dataObject
        
        self.allX = dataObject.allX.values
        
        self.trainX = dataObject.trainX.values
        self.trainY = dataObject.trainY.values
        
        self.testX = dataObject.testX.values
        self.testY = dataObject.testY.values
        
        self.validationX = dataObject.validationX.values
        self.validationY = dataObject.validationY.values
        
        trainCols = self.trainX.shape[1]
        testCols = self.testX.shape[1]
        validationCols = self.validationX.shape[1]
        
        if trainCols == testCols == validationCols:
            self.nInputs = trainCols
        else:
            print("Different number of columns in train, test and validation set!")
            
        self.nOutputs = 1
        
        self.activationFunctionHiddenLayer = {
#                "linear" : layers.Linear,
                "sigmoid" : layers.Sigmoid,
                "tanh" : layers.Tanh,
                "relu" : layers.Relu
                }
        
        self.activationFunctionOutputLayer =  {
                "linear" : layers.Linear,
#                "sigmoid" : layers.Sigmoid,
#                "tanh" : layers.Tanh
                }
        
        self.optimizers = {
                "gradientDescent" : algorithms.GradientDescent,
                "conjugateGradient" : algorithms.ConjugateGradient,
                "quasiNewton" : algorithms.QuasiNewton,
                "hessian" : algorithms.Hessian,
                "LevenbergMarquardt": algorithms.LevenbergMarquardt,
                "RPROP" : algorithms.RPROP,
                "iRPROPPlus" : algorithms.IRPROPPlus,
                "Adam" : algorithms.Adam
                }
        
        if not isdir("savedNN"):
            mkdir("savedNN")
            
        if not isdir("results"):
            mkdir("results")
        
        self.bestNNfile = "savedNN/ANNbest"+filesId+".pickle"
        self.resultsLog = "results/results"+filesId+".dat"
        self.lowestError = 999
        
        self.initResultsFile()
        
        self.maxEpochsNo = 1000
        self.epochsNoCheck = 50
        self.maxEpochsValidationErrorIsRising = 10
        
        self.fileId = filesId
#        self.noTryOneNN = 3
        
        
    def initResultsFile(self):
        resFile = open(self.resultsLog, 'w')
        
        resFile.write("ID\tHidden neurons\tHidden layer activation function\tOutput layer activation function\t")
        resFile.write("Optimising algorithm\tTrain error\tTest error\tValidation rmsd\tTraining time\tEpochs no eval\tEpoch no selected\tStatus\tR\tR2\n")
        
        resFile.close()
                        
    def runNN(self, hiddenNeurons, afHidden, afOut, optim):
        network = layers.join( layers.Input( self.nInputs ) ,
                                    self.activationFunctionHiddenLayer[afHidden]( hiddenNeurons),
                                    self.activationFunctionOutputLayer[afOut]( self.nOutputs ))
        
        optimizer = self.optimizers[optim]( network, verbose = False,
                        signals = ControlTraining( self.epochsNoCheck, self.maxEpochsValidationErrorIsRising) )
        
        try:
            trainingStart = time()
            optimizer.train( self.trainX, self.trainY, self.testX, self.testY , 
                            epochs = self.maxEpochsNo )
            trainingTime = time() - trainingStart
            
            trainLoss = mse( optimizer.predict(self.trainX) , self.trainY )
            testLoss = mse( optimizer.predict(self.testX), self.testY )
            validationLoss = mse( optimizer.predict(self.validationX), self.validationY )
            
            if validationLoss < self.lowestError:
                self.lowestError = validationLoss
                with open( self.bestNNfile , 'wb') as f:
                    pickle.dump(optimizer, f)
                    
            epochsNo = optimizer.last_epoch
            status = optimizer.signals.status
            selectedEpoch = optimizer.signals.epochSelected
            
            allPredicted = optimizer.predict( self.allX )
            
            a, b, r, r2 = self.dataObject.comparePredictedVsReal(allPredicted)
            
            self.logRow( hiddenNeurons, afHidden, afOut, optim, trainLoss, testLoss, validationLoss, trainingTime, epochsNo, status, selectedEpoch, r, r2 )
        except:
            pass
                
    def logRow( self, hiddenNeurons, afHidden, afOut, optim, trainLoss, testLoss, validationLoss, trainingTime, epochsNo, status, selectedEpoch, r, r2):
        logFile = open( self.resultsLog, "a" )
        
        logFile.write( self.fileId + "\t"+str(hiddenNeurons)+"\t"+
                      afHidden+"\t"+afOut+"\t"+
                      optim+"\t"+str(trainLoss)+
                      "\t"+str(testLoss)+"\t"+str(validationLoss)+
                      "\t"+str(trainingTime)+"\t"+str(epochsNo)+"\t"+str(selectedEpoch)+"\t"+status+"\t"+str(r)+"\t"+str(r2)+"\n")
        
        logFile.close()

        
def mse( A, B):
    B = np.reshape(B, A.shape)
    diff = np.subtract(A,B)
    return   np.mean(  np.square( diff ) ) 