#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:34:40 2019

@author: michal
"""
import sys
from os.path import isdir
if isdir("/net/people/plgglanow/pythonPackages") and not "/net/people/plgglanow/pythonPackages" in sys.path :
    sys.path.insert(0, "/net/people/plgglanow/pythonPackages" )

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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
        self.trainX = self.data[ self.inputColumns ]
        self.trainY = self.data["Typ Nowotworu"]
        
#        type1 = self.data[ self.data["Typ Nowotworu"] == "M" ].iloc[0:100]
#        type2 = self.data[ self.data["Typ Nowotworu"] == "B" ].iloc[0:100]
        
#        concatenated = pd.concat( [type1, type2])
        
#        self.trainX = concatenated[ self.inputColumns ]
#        self.trainY = concatenated[ "Typ Nowotworu" ]
        
        
    def analyseCorrelationMatrix(self, threshold = 0.9):
        columns =  self.inputColumns
        
        data2test = self.data[ columns ]
        data2test =  data2test.values 
        
        correlationMatrix = np.corrcoef( data2test , rowvar = False)
        
        n_rows, n_cols = correlationMatrix.shape
        
        correlactionGraph = nx.Graph()
        
        for i in range(n_rows):
            for j in range(i):
                if abs(correlationMatrix[i,j]) > threshold:
                    print( columns[i], columns[j], correlationMatrix[i,j] )
                    correlactionGraph.add_edge( columns[i], columns[j] )
#                    plt.figure()
#                    plt.scatter(data2test[:,i], data2test[:,j])
#                    plt.xlabel(columns[i])
#                    plt.ylabel(columns[j])
                    
#        plt.figure()
#        layout = nx.spring_layout(correlactionGraph)
#        nx.draw_networkx(correlactionGraph, layout)
                    
        columns2delete = []
        
        while True:
            nodes =list(correlactionGraph.nodes)
            
            if not nodes:
                break
            
            firstNode = nodes[0]
            toDelete = list( nx.neighbors( correlactionGraph, firstNode ) )
            columns2delete += toDelete
            
            correlactionGraph.remove_nodes_from( toDelete + [ firstNode] )
                    
        return columns2delete
                    
    def verifyPrediction(self, predictionMap):
        predictionMapList = predictionMap.tolist()
        
        classesNo = predictionMap.shape[1]
        
        results = {}
        
        for i in range(classesNo):
            results[i] = {  "M" : 0 ,  "B" : 0 }
            
        reference = self.trainY.values.tolist()
        
        for row, ref in zip(predictionMapList, reference):
            for i in range(classesNo):
                if row[i] == 1:
                    results[i][ref]+= 1
                    
#        print(results)
        predictionError = 0
        for classid in results:
            predictionError += min( results[classid]["M"], results[classid]["B"] )
            
        predictionErrorPercent = float(predictionError)/len(reference) *100
        return results, predictionError, predictionErrorPercent
    
    def removeFromInputcolumns(self, column2delete):
        inputSet = set(self.inputColumns) - set(column2delete)
        self.inputColumns = list( inputSet )
        self.splitData()
    