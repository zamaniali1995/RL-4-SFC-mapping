#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 19:24:24 2019

@author: ali
"""
#import pandas as pd
#import numpy as np
import sys
sys.path.insert(0, '../ReadFile')
import InputConstants
import json
## Node features class
class node:
    def __init__(self, name, cap, deg, bandwidth, dis):
        self.name = name
        self.cap = cap
        self.deg = deg
        self.ban = bandwidth
        self.dis = dis
# link features class
class link:
    def __init__(self, name, cap, bandwidth, length):
        self.cap = cap
        self.bandwidth = bandwidth
        self.length = length
        self.name = name

class Graph:
    def __init__(self, path):
        
#        start_file_line = None
#        end_file_line = None
#        start_node_line = None
#        end_node_line = None
#        start_link_line = None
#        end_link_line = None
#        node_cap_list = []
        link_list = []
        node_len = []
        node_ban = []
        
#        self.node_list = []
        self.link_list = []
        
        input_cons = InputConstants.Inputs()
#        data = pd.read_csv(path, header=None)
#        print(data)
        with open(path, "r") as data_file:
            data = json.load(data_file)
#            node_num = data['networkTopology']['nodes']
            node_name_list = [data['networkTopology']['nodes']
                [node_num][input_cons.network_topology_node_name] 
                    for node_num in range(len(data['networkTopology']['nodes']))]
#            node_cap_list = [data['networkTopology']['nodes'][i][input_cons.network_topology_node_cap] for i in range(len(tmp))]
            link_list = [data['networkTopology']['links'][node_name] 
                for node_name in node_name_list]
# Calculation of average distance of each node from other nodes
            for cnt_node in range(len(node_name_list)):
                len_sum = 0
                for cnt_link in range(len(link_list[cnt_node])):
                    len_sum += link_list[cnt_node][cnt_link][input_cons.network_topology_link_dis]
                node_len.append(len_sum / len(link_list[cnt_node]))
# Calculation of sum of incoming link bandwidth to each node
            for cnt_node in range(len(node_name_list)):
                ban_sum = 0
                for cnt_link in range(len(link_list[cnt_node])):
                    ban_sum += link_list[cnt_node][cnt_link][input_cons.network_topology_link_cap]
                node_ban.append(ban_sum)
            
            self.node_list = [node(node_name_list[cnt],
                              data['networkTopology']['nodes'][cnt][input_cons.network_topology_node_cap],
                              len(link_list[cnt]),
                              node_ban[cnt],
                              node_len[cnt]) 
                              for cnt in range(len(node_name_list))]
                
#            self.link_len_list = [data['networkTopology']['links'][node_name][]]
#            print(len(tmp))
#        with open(path, 'r') as data:
#            for cnt, line in enumerate(data):
#                line = line.split(',')
#                if line[0] == input_cons.START_OF_FILE_DELIMETER:
#                    start_file_line = cnt
#                elif line[0] == input_cons.END_OF_FILE_DELIMETER:
#                    end_file_line = cnt
#                elif line[0] == input_cons.START_OF_NODES_DELIMETER:
#                    start_node_line = cnt
#                elif line[0] == input_cons.END_OF_NODES_DELIMETER:
#                    end_node_line = cnt
#                elif line[0] == input_cons.START_OF_LINK_DELIMETER:
#                    start_link_line = cnt
#                elif line[0] == input_cons.END_OF_LINK_DELIMETER:
#                    end_link_line = cnt
#        if start_file_line == None:
#            print('ReadFile Error: missing "START OF FILE DELIMETER"' )
#        elif end_file_line == None:
#            print('ReadFile Error: missing "END OF FILE DELIMETER"' )
#        elif start_node_line == None:
#            print('ReadFile Error: missing "START OF NODE DELIMETER"' )
#        elif end_node_line == None:
#            print('ReadFile Error: missing "END OF NODE DELIMETER"' )
#        elif start_link_line == None:
#            print('ReadFile Error: missing "START OF LINK DELIMETER"' )
#        elif end_link_line == None:
#            print('ReadFile Error: missing "END OF LINK DELIMETER"' ) 
#        with open(path, 'r') as data:
#            for cnt, line in enumerate(data):
#                if start_node_line < cnt < end_node_line:
#                    tmp = csv.reader(line, delimiter=',')
#                    print(tmp[0])
#                    node_list.append(tmp)
#                if start_link_line < cnt < end_link_line:
#                    link_list.append(line[:])
#        for i in range(len(node_list)):
#            self.node_list.append(node(node_list[i], 2, 4, 4, 10))
#        
#        for i in range(len(link_list)):
#            self.link_list.append(link(link_list[i], 2, 2, 2))
#            

#    def makegragh(self, node):
#        node('v1' ,2, 5, 10, 50)
##        if self.end_file_line == None || self.start_file_line == None :
#            print("Error in reading file. missing delimeters")
#        while (all(data.iloc[self.start_file_num]) != input_cons.START_OF_FILE_DELIMETER):
#            self.start_file_num +=1
#            if self.start_file_num == len(data):
#                print ("Your file does not have line that contains 'START OF FILE'")
#                break
#        while (data.iloc[self.end_file_num] != input_cons.END_OF_FILE_DELIMETER):
#            self.end_file_num +=1
#        while (data.iloc[self.start_file_num + self.start_node_num] != input_cons.START_OF_NODES_DELIMETER):
#            self.start_node_num +=1
        
        
        