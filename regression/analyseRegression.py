#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:37:50 2019

@author: michal
"""

import pickle
from dataPreprocessor import DataPreprocessor
#from neupy import algorithms
import numpy as np

def mse( A, B):
    B = np.reshape(B, A.shape)
    diff = np.subtract(A,B)
    return   np.mean(  np.square( diff ) ) 

data = DataPreprocessor()

with open('ANNbestinputs20.inp.pickle', 'rb') as f:
    loadedNN = pickle.load(f)
    
print("train")
print(mse( loadedNN.predict(data.trainX), data.trainY ))

print("test")
print(mse( loadedNN.predict(data.testX), data.testY ))

print("validation")
print(mse( loadedNN.predict(data.validationX), data.validationY ))
    
predicted = loadedNN.predict( data.allX.values )

a, b, r,r2 = data.comparePredictedVsReal(predicted)
