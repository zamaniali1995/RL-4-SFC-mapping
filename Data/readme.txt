************Author : Ali Zamani *************
************ zamaniali1995@gmail.com*********
There are three json files in Data folder.
1. nsf_14_network.json :
    In this file nsf_net network topology was defined.
    you can change it with your network topology.
    
    How?
    
    One dictionary was defined in this file which contains "networkTopology"
    and in the "networkTopology" list "Nodes" and "Links" were defined.

    In the "Nodes" list, list of all topology nodes and their capacity were defined. 
    The first element is the name of the node and the second element is capacity.
    You can also change the order of them by two parameters 
        network_topology_node_name = 0
            network_topology_node_cap = 1
    in the InputConstants.py

    In the "Links" dictionary, all network topology links were defined.
    For example:
         "1":[["2", 2100, 100], ["3", 3000, 100], ["8", 4800, 100]]
    determins node "1" conected to node "2" with link that has lenghth 2100 and 
    capacity 100.
    you can change order of node name, length and capacity with three parameters:
        network_topology_link_name = 0
            network_topology_link_dis = 1
            network_topology_link_cap = 2
    in the InputConstants.py
 2. chains.json
    In this file chains were defined which contains two "functions" and "chains" 
    dictionary. You can define your chains in this json file.

    How?
    
    In the "functions" dictionary, functions that used in chains were placed. Each
    the function defined in a list and first element of the list belongs to the function's name and 
    the second element determines the number of CPU core that function needs. You can change the order of
    name and CPU usage of functions by two parameters:
        function_name = 0
            function_usage = 1
    in the InputConstants.py
3. chain_random.json
    This json file creates automatically with
     "generate_chains_functions(creat_chains_functions(path, chain_num, fun_num, ban, CPU)"
    where the path is a path for storing generated file, chain_num is number of chains you want to 
    generate, fun_num is the maximum number of chains' function, the ban is the maximum bandwidth of chains and 
    CPU is the maximum number of cup cores that each function needs.
    
    
    

