#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:24:52 2019

@author: michal
"""
import glob
from os import remove

def mergeLogs( logList  ):
    logFinal = logList[0]
    logs = logList[1]
    final_log = open(logFinal, "a+")
    log_files = glob.glob(logs)
    for log_file in log_files:
        if log_file == logFinal:
            continue
        
        new_log = open(log_file, 'r')
        
        line = new_log.readline()
        line = new_log.readline()
        while line:
            final_log.write(line)
            line = new_log.readline()
        
        new_log.close()
        remove(log_file)
    
    final_log.close()
    

mergeLogs( [ "results/results0.dat" , "results/*inp*dat" ] )