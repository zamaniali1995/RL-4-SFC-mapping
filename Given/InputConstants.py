#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 17:58:40 2019

@author: ali
"""
class Inputs:
    
    network_path = "../Data/"
    network_name = "nsf_14_network.json"
    chains_path = "../Data/"
    chains_name = "nsf_14_network.json"
    chains_random_name = "chain_random.json"
    chains_random_path = "../Data/"
#    START_OF_FILE_DELIMETER = '***START OF FILE***'
#    END_OF_FILE_DELIMETER = '***END OF FILE***'
#    START_OF_NODES_DELIMETER = "***START OF NODE***"
#    END_OF_NODES_DELIMETER = "***END OF NODE***"
#    START_OF_LINK_DELIMETER = "***START OF LINK***"
#    END_OF_LINK_DELIMETER = "***END OF LINK***"
# network topology parameters
    network_topology_node_name = 0
    network_topology_node_cap = 1
    network_topology_link_name = 0
    network_topology_link_dis = 1
    network_topology_link_cap = 2
    function_name = 0
    function_usage = 1
# Learning parameters
    epoch_num = 10000
    batch_Size = 5
    node_features = 4
    learning_rate = 1e-10
# Creat chains parameters
    chains_num = 300
    fun_num = 5
    chain_ban = 10
    cpu_range = 3