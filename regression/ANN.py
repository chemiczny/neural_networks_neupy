#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:27:32 2019

@author: michal
"""

from neuralNetworkGenerator import NeuralNetworkManager
from dataPreprocessor import DataPreprocessor

data = DataPreprocessor()
#data.analyseCorrelaionMatrix()

nnm = NeuralNetworkManager(data)
#nnm.generateNetworks()
#for i in range(4, 10):
#    nnm.tryNN( i, "tanh", "linear", "gradientDescent" )

nnm.runNN( 6, "tanh", "linear", "gradientDescent" )
