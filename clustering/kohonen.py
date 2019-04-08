#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:27:32 2019

@author: michal
"""
import sys
from os.path import isdir
if isdir("/net/people/plgglanow/pythonPackages") and not "/net/people/plgglanow/pythonPackages" in sys.path :
    sys.path.insert(0, "/net/people/plgglanow/pythonPackages" )

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neupy import algorithms
#from math import sqrt

class DataPreprocessor:
    def __init__(self):
        self.data = pd.read_csv("WDBC.txt", sep="\t")
        
        self.inputColumns = []
        self.getInputColumns()
        
        self.column2originalMinValue = {}
        self.column2originalMaxValue = {}
        self.normalize()
        
        self.trainX = None
        self.trainY = None
        
        self.splitData()
        
    def getInputColumns(self):
        columnNames = list(self.data.columns.values)
        
        for col in columnNames:
            if not col in [ "ID Pacjenta", "Typ Nowotworu" , "średnia powierzchnia", "największy obwód"]:
                self.inputColumns.append(col)
#            if col in ["średni promień", "średnia powierzchnia", "największy obwód"]:
#                self.inputColumns.append(col)

    def normalize(self):
        self.data = self.data.fillna(0)
        columnNames2normalize = self.inputColumns
        
        for colName in columnNames2normalize:
            minValue = self.data[colName].min()
            maxValue = self.data[colName].max()
            
            self.data[colName] =  (self.data[colName] - minValue )/( maxValue - minValue )
            self.column2originalMinValue[colName] = minValue
            self.column2originalMaxValue[colName] = maxValue
            
    def splitData(self):
#        self.trainX = self.data[ self.inputColumns ]
#        self.trainY = self.data["Typ Nowotworu"]
        
        type1 = self.data[ self.data["Typ Nowotworu"] == "M" ].iloc[0:100]
        type2 = self.data[ self.data["Typ Nowotworu"] == "B" ].iloc[0:100]
        
        concatenated = pd.concat( [type1, type2])
        
        self.trainX = concatenated[ self.inputColumns ]
        self.trainY = concatenated[ "Typ Nowotworu" ]
        
        

        
    def analyseCorrelationMatrix(self):
        columns =  self.inputColumns
        
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
        
        
data = DataPreprocessor()
#data.analyseCorrelationMatrix()
#X = np.array([
#    [0.1961,  0.9806],
#     [-0.1961,  0.9806],
#     [0.9806,  0.1961],
#    [0.9806, -0.1961],
#     [-0.5812, -0.8137],
#     [-0.8137, -0.5812] ])
X = data.trainX.values
kohonen = algorithms.Kohonen( n_inputs = X.shape[1], n_outputs = 8 , verbose = False, step = 0.5)
kohonen.train( X, epochs = 100 )
results = kohonen.predict(X)