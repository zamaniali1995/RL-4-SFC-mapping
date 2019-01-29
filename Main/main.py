#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 16:44:41 2019

@author: ali
"""
import sys
sys.path.insert(0, '../ReadFile')
sys.path.insert(0, '../Given')
import InputConstants
from ReadFile import Graph
import tensorflow
input_cons = InputConstants.Inputs()
graph = Graph(input_cons.network_path + input_cons.network_name)

# Learning

#print(graph.start_file_line)

#print (IndentationError.read_path)
#Mf_cal()
            