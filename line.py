#!/usr/bin/env python
#coding:utf-8

import igraph as ig
import copy as cp
import numpy as np
import itertools as it
from collections import defaultdict

####################################################################
############################## LINE ################################
####################################################################
def run(G,heuristic,percentage):
    """
        @param G: the network tested
        @param heuristic: the heuristic function that will be used to measure the probability that two nodes will eventually become connected
        @param percentage: percentage of edges that will be inserted (for example 3, 5, 10 ...)        
        @return: the enriched network by the edges insertion
    """
    list_ed, list_prob = heuristic_calc(G,heuristic)
    
    len_insert = int((G.ecount()*percentage)/100)
    
    best_edges = edges_insert(list_ed,list_prob,len_insert)
    
    G_m = cp.deepcopy(G)
    
    # Insert the new edges
    for i,j in best_edges:
        G_m.add_edge(i,j)
    
    return G_m
    
def edges_insert(list_ed,list_prob,len_insert):
    """ 
        @param list_prob: 
        @param list_ed: list of filtered edges
        @param len_insert: number of edges to be inserted
        @return: list of edges to be inserted into the network G
    """
    best_edges = []
    
    i_prob = sorted(range(len(list_prob)), key=lambda k: list_prob[k])
    
    chosen_index = i_prob[len_insert - 1]
    value = list_prob[chosen_index]
    L1 =  []
    for i in i_prob:
        value_prob = list_prob[i]
        if value_prob == value:
            L1.append(i)

    L2 = []
    for i in i_prob[:len_insert]:
        if(list_prob[i]!=value):
            L2.append(i)

    n = len_insert - len(L2)
    chosen_indexes = list(np.random.choice(L1,n,replace=False))
    L3 = L2+chosen_indexes
    
    for i in L3:
        best_edges.append(list_ed[i])
        
    return best_edges

def filter_edges(G):
    """ 
        @param G: the network tested
        @return: list of filtered edges (edges not belonging to network G)
    """   
    nodes = list(range(0, G.vcount()))
    edges = list(it.combinations(nodes,2))
    
    list_ed = []
    for i,j in edges:
        if(G.are_connected(i,j) == False):
            list_ed.append((i,j))
            
    return list_ed

########################### HEURISTICS #############################
####################################################################
def heuristic_calc(G,heuristic):
    """ 
        @param G: the network tested
        @param heuristic: the heuristic function that will be used to measure the probability that two nodes will eventually become connected 
        @return: list of edges and their respective insertion probabilities
    """
    if(heuristic == "Deg"):
        list_ed, list_prob = node_degree(G) 
    elif(heuristic == "NbrDeg"):
        list_ed, list_prob,  = node_degree_cn(G)
    elif("Assort" in heuristic):
        metric = heuristic.split("Assort",1)[1]
        list_ed, list_prob  = assortativity(metric,G)    
    elif("S" in heuristic):
        metric = heuristic.split("S",1)[1]
        list_ed, list_prob  = similarity(metric,G)    
    return list_ed, list_prob    

#####################################    
############## Degree ###############

def node_degree(G):
    """ 
        @param G: the network tested
        @return: list of edges and their respective insertion probabilities
    """
    # Deg - Node Degree
    list_nodes = list(range(0, G.vcount()))
    
    d = defaultdict(list)
    sum_degrees = [0]*len(list_nodes)
    
    # Create a dictionary with nodes that are not connected to each of the nodes in the network
    for i in list_nodes:
        for j in list_nodes:
            if(G.are_connected(i,j) == False and i!=j):
                d[i].append(j)
    
    # Create a list with the sum of the degrees of the nodes that can connect to a given node (k)
    for k in d:
        sum_degree_k = 0
        for value in d[k]:
            sum_degree_k = sum_degree_k + G.degree(value)
        sum_degrees[k] = sum_degree_k
    
    # Create a dictionary with the probability of the connection between network nodes (node k to another node)
    prob_dict = defaultdict(list)
    for k in d:
        for value in d[k]:
            prob = G.degree(value) / sum_degrees[k] 
            prob_dict[k].append(prob)

    # And a list with these probabilities
    list_k = []
    for k in d:
        for j, prob in zip(d[k],prob_dict[k]):
            list_k.append([k,j,prob])
    
    # Separate this information into three new groups (i_prob, list_prob, list_ed)
    list_ed = []
    list_prob = []
    for i,j,prob in list_k:
        if((int(j),int(i)) not in list_ed):
            list_ed.append((int(i),int(j)))
            list_prob.append(float(prob))        
    
    return list_ed, list_prob
    
# NbrDeg - Node Degree with Common Neighbors
def node_degree_cn(G):
    """ 
        @param G: the network tested
        @return: list of edges and their respective insertion probabilities
    """
    list_nodes = list(range(0, G.vcount()))
    
    # Create a list of nodes that have a common neighbor, but are not connected
    comb_total = []
    for k in list_nodes:
        edges = []
        neighbors = G.neighbors(k) 
        for neighbor in neighbors:
            neighbors_neighbor = G.neighbors(neighbor)
            for n in neighbors_neighbor:
                edge_a = (k,n)
                edge_b = (n,k)
                if(G.are_connected(k,n) == False and k!=n and edge_a not in edges and edge_b not in edges):
                    edges.append(edge_a)
        comb_total.append(edges)
     
    # Create a list with the nodes' degrees, based on the list generated above (comb_total)
    list_degrees = []
    for edge in comb_total:
        degrees_nei = []
        for initial, neighbor in edge:
            degrees_nei.append(G.degree(neighbor))
        list_degrees.append(degrees_nei)
    
    # Calculate the sum of the neighbors' degrees for each node
    list_sum = []
    for line in list_degrees:
        list_sum.append(np.sum(line))
    
    # Calculate the probabilities of connection between the nodes
    probabilities_list = []
    for neighbors,summatory in zip(list_degrees,list_sum):
        prob_list = []
        for node in neighbors:
            prob_list.append(node/summatory)
        probabilities_list.append(prob_list)

    #  Separate the information in two lists: with the two nodes of each edge and their respective connection probabilities
    list_edge = []
    list_prob = []
    for line1,line2 in zip(comb_total,probabilities_list):
        for edge,probability in zip(line1,line2):
            i,j = edge[0],edge[1]
            if((int(i),int(j)) not in list_edge and (int(j),int(i)) not in list_edge):
                list_edge.append((int(i),int(j)))
                list_prob.append(float(probability))
    
    return list_edge, list_prob 
    
#####################################    
########### Assortativity ###########   
    
def return_assortativity(metric,G):
    """ 
        @param metric: which measured property will be used to calculate the assortativity
        @param G: the network tested
        @return: value of assortativity
    """
    if (metric == "Deg"):
        item = G.degree()
    elif(metric == "Clos"):
        item = G.closeness()
    elif(metric == "Bet"):
        item = G.betweenness()
    elif(metric == "Eigen"):
        item = G.eigenvector_centrality()
    elif(metric == "PageRank"):
        item = G.pagerank()
    elif(metric == "Shell"):
        item = G.shell_index()
    return item    
    
    
def assortativity(metric,G):
    """ 
        @param metric: which measured property will be used to calculate the assortativity
        @param G: the network tested
        @return: list of edges and their respective insertion probabilities
    """
    item = return_assortativity(metric,G)
    assort = G.assortativity(item)

    list_ed = filter_edges(G)
    
    summatory = 0
    difs = []
    list_prob = []
    
    E = 0.1
    
    # Division between positive and negative assortativity
    if(assort > 0.0):
        for i,j in list_ed:
            dif = abs(item[i] - item[j])
            num = 1 / (dif + E)
            difs.append(num)
            summatory += 1 / num
        
    elif(assort <= 0.0):
        for i,j in list_ed:
            dif = abs(item[i] - item[j])
            difs.append(dif)
            summatory += dif
            
    for element in difs:
        list_prob.append(element/summatory)     
        
    return list_ed, list_prob   
    
#####################################    
############ Similarity #############     

def return_similarity(metric,G,i,j,degrees):
    """ 
        @param metric: which measured property will be used to calculate the similarity
        @param G: the network tested
        @param i: the first node
        @param j: the second node
        @param degrees: list with the degrees of all nodes
        @return: the value of a given similarity metric
    """
    if (metric == "CNbr"): # SCNbr - Common Neighbor
        item = common_neighbors(G,i,j)
    elif(metric == "Salt"): # SSalt - Salton
        item = common_neighbors(G,i,j)/np.sqrt(degrees[i]*degrees[j])
    elif(metric == "Jac"): # SJac - Jaccard
        item = common_neighbors(G,i,j)/ len(set(G.neighbors(i)) | set(G.neighbors(j)))
    elif(metric == "Sor"): # SSor - Sorensen
        item = 2*common_neighbors(G,i,j)/(degrees[i]+degrees[j])
    elif(metric == "ResAlloc"): # SAlocRec - Resource Allocation
        z = set(G.neighbors(i)) & set(G.neighbors(j))
        item = np.sum([1/degrees[x] for x in z])
    elif(metric == "HPro"): # SHPro - Hub Promoted
        item = common_neighbors(G,i,j)/max(degrees[i],degrees[j])
    elif(metric == "HDep"): # SHDep - Hub Depressed
        item = common_neighbors(G,i,j)/min(degrees[i],degrees[j])
    elif(metric == "LHN"): # SLHN - Leicht–Holme–Newman
        item = common_neighbors(G,i,j)/(degrees[i]*degrees[j])    
    return item    
    
#####################################

# Common Neighbors
def common_neighbors(G,i,j):
    """ 
        @param G: the network tested
        @param i: the first node
        @param j: the second node
        @return: common neighbor similarity
    """
    return (len(set(G.neighbors(i)) & set(G.neighbors(j))))

#####################################    

def similarity(metric,G):
    """ 
        @param metric: which measured property will be used to calculate the similarity
        @param G: the network tested 
        @return: list of edges and their respective insertion probabilities
    """
    degrees = G.degree()
    
    list_ed = filter_edges(G)
    
    summatory = 0
    similarities = []
    list_prob = []
    for i,j in list_ed:
        value = return_similarity(metric,G,i,j,degrees)
        similarities.append(value)
        summatory += value
        
    for element in similarities:
        list_prob.append(element/summatory)
    
    return list_ed, list_prob
