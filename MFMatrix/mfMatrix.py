#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 11:32:07 2019

@author: ali
"""
import numpy as np
import random
import sys
sys.path.insert(0, '../ReadFile')
import InputConstants
class Mf:
    def __init__(self, graph):
       input_cons = InputConstants.Inputs()
       self.mf_matrix = np.zeros([len(graph.node_list), input_cons.node_features])
       for i in range(len(graph.node_list)):
           self.mf_matrix[i, 0] = graph.node_list[i].cap
           self.mf_matrix[i, 1] = graph.node_list[i].deg
           self.mf_matrix[i, 2] = graph.node_list[i].ban
           self.mf_matrix[i, 3] = graph.node_list[i].dis
    def select_one(self, y, approach):
        if approach == 'roulette_wheel':
            y_one_hot = np.zeros_like(y)
            tmp = []
            for i in range(14):
                tmp.append(y[0][i])
            y_one_hot[0][np.random.choice(y.shape[1], p=tmp)]=1
            return(y_one_hot)
#            y_one_hot = [0] * len(y)
#            _max = np.sum(y)
##            print (_max)
#            pick = random.uniform(0 , _max)
#            curr = 0
#            return(np.random.choice(14, y))
#            for cnt in range(y.shape[1]):
#                curr += y[0][cnt]
##                print(curr)
#                if pick > curr:
#                    y_one_hot[0][cnt] = 1
#                    return (y_one_hot)
#            print (_max)
#       print(graph.node_list[0].name)
                