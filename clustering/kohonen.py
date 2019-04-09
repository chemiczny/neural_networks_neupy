#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:27:32 2019

@author: michal
"""
from dataPreprocessor import DataPreprocessor
from SOFMgenerator import SOFMgenerator
        
        
data = DataPreprocessor()
sofm = SOFMgenerator(data)

sofm.runSOFM(4, 4, 1, "default", "rect")

#data.analyseCorrelationMatrix()
#X = np.array([
#    [0.1961,  0.9806],
#     [-0.1961,  0.9806],
#     [0.9806,  0.1961],
#    [0.9806, -0.1961],
#     [-0.5812, -0.8137],
#     [-0.8137, -0.5812] ])
#X = data.trainX.values
#kohonen = algorithms.SOFM( n_inputs = X.shape[1], features_grid = (4,3) , verbose = False, step = 0.5, learning_radius = 1 )
#kohonen.train( X, epochs = 100 )
#results = kohonen.predict(X)
#verification, error, erroPercent = data.verifyPrediction(results)
#
#print(verification)
#print(error)
#print(erroPercent)