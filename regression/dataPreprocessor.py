#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 10:33:05 2019

@author: michal
"""
import sys
from os.path import isdir
if isdir("/net/people/plgglanow/pythonPackages") and not "/net/people/plgglanow/pythonPackages" in sys.path :
    sys.path.insert(0, "/net/people/plgglanow/pythonPackages" )

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

class DataPreprocessor:
    def __init__(self):
        self.data = pd.read_csv("ANN_MSzaleniec.txt", sep="\t")
        
        self.inputColumns = []
        self.getInputColumns()
        
        self.column2originalMinValue = {}
        self.column2originalMaxValue = {}
        self.normalize()
        
        self.allX = None
        self.allY = None
        
        self.trainX = None
        self.trainY = None
        self.testX = None
        self.testY = None
        self.validationX = None
        self.validationY = None
        self.splitData()
        
    def getInputColumns(self):
        columnNames = list(self.data.columns.values)
        
        for col in columnNames:
            if "Wejście" in col:
                self.inputColumns.append(col)

    def normalize(self):
        columnNames2normalize = self.inputColumns + [ "rSA Zm.zal" ]
        
        for colName in columnNames2normalize:
            minValue = self.data[colName].min()
            maxValue = self.data[colName].max()
            
            self.data[colName] =  (self.data[colName] - minValue )/( maxValue - minValue )
            self.column2originalMinValue[colName] = minValue
            self.column2originalMaxValue[colName] = maxValue
            
    def splitData(self):
        self.trainX = self.data[  self.data["Próba"] == "Uczenie" ]
        self.trainY = self.trainX["rSA Zm.zal"]
        self.trainX = self.trainX[ self.inputColumns ]
        
        
        self.testX = self.data[  self.data["Próba"] == "Test" ]
        self.testY = self.testX["rSA Zm.zal"]
        self.testX = self.testX[ self.inputColumns ]
        
        self.validationX = self.data[  self.data["Próba"] == "Walidacja" ]
        self.validationY = self.validationX["rSA Zm.zal"]
        self.validationX = self.validationX[ self.inputColumns ]
        
        self.allX =  self.data[ self.inputColumns ]
        self.allY = self.data[ "rSA Zm.zal" ]
        
    def comparePredictedVsReal(self, allPredicted):
        originalMin = self.column2originalMinValue["rSA Zm.zal"]
        originalMax = self.column2originalMaxValue["rSA Zm.zal"]
        
        diff = originalMax - originalMin
        
        realValues = deepcopy(self.allY.values)
        
        realValues *= diff
        allPredicted *= diff
        
        realValues += originalMin
        allPredicted += originalMin
        
        plt.figure()

        plt.scatter( realValues , allPredicted )
        
        print(realValues.shape)
        print(allPredicted.shape)
        xy = np.column_stack( (realValues, allPredicted) )
#        
        print(xy.shape)
        R = np.corrcoef( np.transpose(xy))[0,1]
#        
        print(R, R*R)
#        print(R.shape)
        return realValues, allPredicted
        
    def analyseCorrelationMatrix(self):
        columns =  self.inputColumns + [ "rSA Zm.zal" ]
        
        data2test = self.data[ columns ]
        data2test =  data2test.values 
        
        correlationMatrix = np.corrcoef( data2test , rowvar = False)
        
        n_rows, n_cols = correlationMatrix.shape
        
        for i in range(n_rows):
            for j in range(i):
                if abs(correlationMatrix[i,j]) > 0.7:
                    print( columns[i], columns[j], correlationMatrix[i,j] )
                    plt.figure()
                    plt.scatter(data2test[:,i], data2test[:,j])
                    plt.xlabel(columns[i])
                    plt.ylabel(columns[j])