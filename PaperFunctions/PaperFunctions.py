#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 19:24:24 2019

@author: ali
"""
###############################################################
# Import packages
###############################################################
import numpy as np
import InputConstants
import json
import random as rd

###############################################################
# Node features class
###############################################################
class _Node:
    def __init__(self, name, cap, deg, bandwidth, dis, function):
        self.name = name
        self.cap = cap
        self.deg = deg
        self.ban = bandwidth
        self.dis = dis
        self.fun = {}
###############################################################
# Link features class
###############################################################
class _Link:
    def __init__(self, name, cap, bandwidth, length):
        self.cap = cap
        self.bandwidth = bandwidth
        self.length = length
        self.name = name

###############################################################
# Chain features class
###############################################################
class _Chain:
    def __init__(self, name, function, bandwidth):
        self.name = name
        self.fun = function
        self.ban = bandwidth
        
###############################################################
# Graph class:|
#             |__>functions:-->
#                           -->  
###############################################################
class Graph:
    def __init__(self, path, funs):
        self.funs = funs
        self.rev_to_cost_val = 0
        self.input_cons = InputConstants.Inputs()
        link_list = []
        node_ban = []
        with open(path, "r") as data_file:
            data = json.load(data_file)
            node_name_list = [data['networkTopology']['nodes']
                [node_num][self.input_cons.network_topology_node_name] 
                    for node_num in range(len(data['networkTopology']['nodes']))]

            self.link_list = data['networkTopology']['links'] 
            link_list = [data['networkTopology']['links'][node_name]
                        for node_name in node_name_list]
 
            for cnt_node in range(len(node_name_list)):
                ban_sum = 0
                for cnt_link in range(len(link_list[cnt_node])):
                    ban_sum += link_list[cnt_node][cnt_link][self.input_cons.network_topology_link_cap]
                node_ban.append(ban_sum)
            self.node_list = [_Node(node_name_list[cnt],
                              rd.randint(self.input_cons.min_node_cap, self.input_cons.max_node_cap),
                              len(link_list[cnt]),
                              node_ban[cnt],
                              0,
                              None) 
                              for cnt in range(len(node_name_list))]
            self.dist, self.hop =  self.__floydWarshall()

    ###############################################################
    # "__function_cpu_usage": returns cpu usage of each nodes
    #               --->input: fun >>> functions name
    #               --->output: CPU usage
    ###############################################################    
    def __function_cpu_usage(self, fun):
        return(self.funs[fun])

    ###############################################################
    # "__function_placement": placement of function "fun" of chain
    #                                               "ser" in "node"  
    #               --->input:  fun >>> functions name
    #                           ser >>> name of chain
    #                           node >> node's number
    #               --->output: none
    ###############################################################        
    def __function_placement(self, node, ser, fun):
        self.node_list[node].fun[ser].append(fun)

    ###############################################################
    # "batch_function_placement": placement batch of function "fun" 
    #                                      of chain "ser" in "node"  
    #               --->input:  ser_list >>> list of service
    #                           node_fun_list >>> list of pair of
    #                                              nodes and funs    
    #               --->output: none
    ###############################################################        
    def batch_function_placement(self, ser_list, node_fun_list):
        for node_fun, ser in zip(node_fun_list, ser_list): 
            for node, fun in node_fun:
                self.__function_placement(node, ser, fun)
    
    ###############################################################
    # "node_is_mapped": checking to find that node_fun_list can  
    #                                      place in nodes or not  
    #               --->input:  node_fun_list >>> list of pair of 
    #                                         nodes and functions 
    #                                       that should be placed.    
    #                           chains >>> objecte of chains class   
    #               --->output: True or False
    ###############################################################        
    def node_is_mapped(self, node_fun_list, chains):
        flag = True
        cpu_used = []
        service_list = [chains[i].name for i in range(len(chains))]
        for node in range(len(self.node_list)):
            _sum = 0
            for ser in service_list:
                funs = self.node_list[node].fun[ser]
                for fun in funs:
                    _sum += self.__function_cpu_usage(fun)
            cpu_used.append(_sum)
        for node_fun in node_fun_list:
            _sum = 0
            cpu_req = [0] * len(self.node_list)
            for node_1, _ in node_fun:
                _sum = 0
                for node_2, fun in node_fun:
                    if node_1 == node_2:
                        _sum += self.__function_cpu_usage(fun)
                cpu_req[node_1] = _sum
            cpu_used = [sum(x) for x in zip(cpu_used, cpu_req)]
        print(cpu_used)
        if _sum > self.node_list[node_1].cap:
            flag = False
        return flag
    
    ###############################################################
    # "rev_to_cost": calculation of revenue to cost ratio  
    #               --->input:  node_fun >>> list of pair of 
    #                                         nodes and functions     
    #                           chains >>> objecte of chains class
    #                           ser_num>>> chains' number
    #               --->output: revenue to cost ratio value
    ###############################################################        
    def rev_to_cost(self, node_fun, ser_num, chains):
        td = self.input_cons.td
        R = self.revenue_measure(node_fun, ser_num, chains,td)
        C = self.cost_measure(node_fun, ser_num, chains, td)
        return (R / C)
    
    ###############################################################
    # "revenue_measure": calculation of revenue  
    #               --->input:  node_fun >>> list of pair of 
    #                                         nodes and functions     
    #                           chains >>> objecte of chains class
    #                           ser_num>>> chains' number
    #               --->output: revenue value
    ###############################################################        
    def revenue_measure(self, node_fun, ser_num, chains):
        td = self.input_cons.td
        cpu_usage = sum([self.__function_cpu_usage(node_fun[i][1])
                    for i in range(len(node_fun))])
        bandwidth_usage = chains[ser_num].ban * (len(node_fun) - 1)
        return td * (cpu_usage + bandwidth_usage)
    
    ###############################################################
    # "cost_measure": calculation of cost  
    #               --->input:  node_fun >>> list of pair of 
    #                                         nodes and functions     
    #                           chains >>> objecte of chains class
    #                           ser_num>>> chains' number
    #               --->output: cost value
    ###############################################################        
    def cost_measure(self, node_fun, ser_num, chains):
        td = self.input_cons.td
        _sum = 0.0001
        for n in range(len(node_fun)-1):
            _sum += self.__hop_count(node_fun[n][0], node_fun[n+1][0])
        return chains[ser_num].ban * _sum * td
    
    ###############################################################
    # "__hop_count": hop count  
    #               --->input:  node_1 >>> |
    #                                      |--> hops between node_1
    #                           node_2 >>> |             and node_2
    #               --->output: hops' number between nodes
    ###############################################################        
    def __hop_count(self, node_1, node_2):
        return self.hop[node_1][node_2]
     
    ###############################################################
    # "link_is_mapped": checking to find that node_fun_list can  
    #                                      place in nodes or not  
    #               --->input:  node_fun_list >>> list of pair of 
    #                                         nodes and functions 
    #                                       that should be placed.    
    #                           chains >>> objecte of chains class   
    #               --->output: True or False
    #   ????????????not completed
    ###############################################################        
    def link_is_mapped(self, node_fun, chains):
        return True
    
    ###############################################################
    # "update_feature_matrix": checking to find that node_fun_list 
    #                                      can place in nodes or not  
    #               --->input:  node_fun_list >>> list of pair of 
    #                                         nodes and functions    
    #               --->output: mf matrix
    ###############################################################         
    def update_feature_matrix(self, node_fun):
        mf_matrix = np.zeros([len(self.node_list),
                                   self.input_cons.node_features])
        for i in range(len(self.node_list)):
            mf_matrix[i, 0] = self.node_list[i].cap
            mf_matrix[i, 1] = self.node_list[i].deg
            mf_matrix[i, 2] = self.node_list[i].ban
            mf_matrix[i, 3] = self.node_list[i].dis
        node = []
        for n in node_fun:
            node.append(n[0])
        node = list(dict.fromkeys(node))
        if  node != []:
            for n_1 in range(len(self.node_list)):
                _sum = 0
                cnt = 0
                for n_2 in node: 
                    if n_1 != n_2:
                        _sum += self.__dis_cal(n_1, n_2)
                        cnt +=1
                if cnt != 0:
                    tmp = _sum / (cnt + 1)
                    self.node_list[n_1].dis = tmp
                    mf_matrix[n_1, 3] = tmp
        else:
            for n_1 in range(len(self.node_list)):
                self.node_list[n_1].dis = 0
                mf_matrix[n_1, 3] = 0
        return mf_matrix
     
    ###############################################################
    # "__dis_cal": distance meaturement  
    #               --->input:  node_1 >>> |
    #                                      |--> hops between node_1
    #                           node_2 >>> |             and node_2
    #               --->output: distance between nodes
    ###############################################################        
    def __dis_cal(self, node_1, node_2):
        return self.dist[node_1][node_2]
    
    ###############################################################
    # "floydWarshall": Solves all pair shortest path via Floyd
    #                                       Warshall Algorithm
    #               --->input: none
    #               --->output: dist matrix >>> distances between 
    #                                            each pair of nodes
    #                           hop matrix >>>> hops between each 
    #                                           pair of nodes
    ###############################################################
    def __floydWarshall(self): 
        node_num = len(self.node_list)
        hop = (np.ones((node_num, node_num)) * np.inf)
        dist = (np.ones((node_num, node_num)) * np.inf)
        for n_1 in range(node_num): 
            node_name = self.node_list[n_1].name
            links = self.link_list[node_name]
            for l in range(len(links)):
                for n_2 in range(node_num):
                    if n_1 == n_2:
                        hop[n_1][n_2] = 0
                        dist[n_1][n_2] = 0
                    elif links[l][self.input_cons.network_topology_link_name] == self.node_list[n_2].name:
                        hop[n_1][n_2] = 1
                        dist[n_1][n_2] = links[l][self.input_cons.network_topology_link_dis]
        for k in range(node_num):       
            # pick all vertices as source one by one 
            for i in range(node_num):       
                # Pick all vertices as destination for the 
                # above picked source 
                for j in range(node_num):       
                    # If vertex k is on the shortest path from  
                    # i to j, then update the value of dist[i][j] 
                    hop[i][j] = min(hop[i][j] , 
                                      hop[i][k]+ hop[k][j] 
                                    )         
                    dist[i][j] = min(dist[i][j] , 
                                      dist[i][k]+ dist[k][j] 
                                    ) 
        return (dist, hop)
     
    ###############################################################
    # "select_one": checking to find that node_fun_list can  
    #                                      place in nodes or not  
    #               --->input:  approach >>> approch for selection  
    #                                         of candidate   
    #               --->output: one hoe vector and candidate
    ###############################################################        
    def select_one(self, y, approach):
        if approach == 'sample':
            y_one_hot = np.zeros_like(y)
            tmp = []
            for i in range(len(self.node_list)):
                tmp.append(y[0][i])
            can = np.random.choice(y.shape[1], p=tmp)
            y_one_hot[0][can]=1
            return(y_one_hot, can)
     
    ###############################################################
    # "make_empty_nodes": this functins remove all functions that were
    #                                      place in the nodes 
    #               --->input:  none   
    #               --->output: none
    ###############################################################        
    def make_empty_nodes(self):
        for i in range(len(self.node_list)):
                for j in range(len(self.data['chains'])):
                    self.node_list[i].fun[self.data['chains'][j]['name']] = []
###############################################################
# Ghains class:|
#             |__>functions:-->
#                           -->  
###############################################################        
class Chains:
    def __init__(self):
        self.input_cons = InputConstants.Inputs()
    
    ###############################################################
    # "read_chains": reading chains 
    #               --->input:  path >>> path of json chain file
    #                           graph >> object of Graph class
    #               --->output: none
    ###############################################################            
    def read_chains(self, path, graph):
        with open(path, "r") as data_file:
            data = json.load(data_file)
            for i in range(len(graph.node_list)):
                for j in range(len(data["chains"])):
                    graph.node_list[i].fun[data["chains"][j]['name']] = []
            return([_Chain(data["chains"][i]['name'],
                                 data["chains"][i]['functions'], 
                                 data["chains"][i]['bandwidth']) 
                                 for i in range(len(data["chains"]))])
    ###############################################################
    # "read_funcions": reading functions 
    #               --->input:  path >>> path of json chain file
    #               --->output: functions list
    ###############################################################            
    def read_funcions(self, path):
         with open(path, "r") as data_file:
            data = json.load(data_file)
         return(data["functions"])

    ###############################################################
    # "creat_chains_functions": reading functions 
    #               --->input:  path >>> path of json chain file
    #                           chain_num >>> number of chains you 
    #                                           want to be generated
    #                           fun_num >>> maximum number of functions
    #                                           of each chain
    #                           ban >>> maximum bandwidth of each chain
    #                           cpu >>> maximum requered cpu core of each
    #                                       chain.
    #               --->output: none
    ###############################################################            
    def creat_chains_functions(self, path, chain_num, fun_num, ban, cpu):
         chains = {}
         chains["chains"] = []
         chains["functions"] = {}
         for f in range(fun_num):
             chains["functions"][str(f)] = rd.randint(1, cpu)
         for c in range(chain_num):
             chain = {}
             rand_fun_num = rd.randint(1, fun_num)         
             chain['name'] = str(c)
             chain['functions'] = [str(f) 
                                for f in range(rand_fun_num)]
             chain['bandwidth'] = rd.randint(1, ban)
             chains["chains"].append(chain)
         with open(path, 'w') as outfile:  
             json.dump(chains, outfile)
