#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 19:24:24 2019

@author: ali
"""
import pandas as pd
import sys
sys.path.insert(0, '../ReadFile')
import InputConstants
class Graph:
    def __init__(self, path):
        
        self.start_file_line = None
        self.end_file_line = None
        self.start_node_line = None
        self.start_link_line = None
        
        input_cons = InputConstants.Inputs()
#        data = pd.read_table(path, header = None)
        with open(path, 'r') as data:
            for cnt, line in enumerate(data):
#                print (line)
                line = line.split(',')
#                print (line)
                if line[0] == input_cons.START_OF_FILE_DELIMETER:
                    self.start_file_line = cnt
                if line[0] == input_cons.END_OF_FILE_DELIMETER:
                    self.end_file_line = cnt
                if line[0] == input_cons.START_OF_NODES_DELIMETER:
                    self.start_node_line = cnt
                if line[0] == input_cons.START_OF_LINK_DELIMETER:
                    self.start_link_line = cnt
                
#        while (all(data.iloc[self.start_file_num]) != input_cons.START_OF_FILE_DELIMETER):
#            self.start_file_num +=1
#            if self.start_file_num == len(data):
#                print ("Your file does not have line that contains 'START OF FILE'")
#                break
#        while (data.iloc[self.end_file_num] != input_cons.END_OF_FILE_DELIMETER):
#            self.end_file_num +=1
#        while (data.iloc[self.start_file_num + self.start_node_num] != input_cons.START_OF_NODES_DELIMETER):
#            self.start_node_num +=1
        
        
        