#!/usr/bin/env python
#coding:utf-8

import igraph as ig
import numpy as np
import csv
import ast


####################################################################
################## COMMUNITY DETECTION ALGORITHMS ##################
####################################################################
# Louvain
def Blondel(G):
    Blondel = G.community_multilevel()
    return Blondel
# InfoMap
def InfoMap(G):
    InfoMap = G.community_infomap()
    return InfoMap
# Leading EigenVector
def EigenVector(G):
    EigenVector = G.community_leading_eigenvector()
    return EigenVector
# WalkTrap
def WalkTrap(G):
    dendrogram = G.community_walktrap()
    WalkTrap = dendrogram.as_clustering()
    return WalkTrap
# FastGreedy
def FastGreedy(G):
    dendrogram = G.community_fastgreedy()
    FastGreedy = dendrogram.as_clustering()
    return FastGreedy
# SpinGlass
def SpinGlass(G):
    SpinGlass = G.community_spinglass()
    return SpinGlass
# Label Propagation Algorithm
def LabelPropagation(G):
    LPA = G.community_label_propagation()
    return LPA

####################################################################
###################### NETWORK CREATION ############################
####################################################################

def add_vertex_with_attrs(graph, attrs):
    n = graph.vcount()
    key, value = attrs.popitem()
    for i in range(n):
        graph.vs[i][key] = value

def membership_toVC(network):
    membership = []
    for v in network.vs():
        attribute = v.attributes()
        membership.append(int(attribute["community"]))
    vertex_clustering = ig.VertexClustering(graph=network, membership=membership)
    return vertex_clustering

def create_net(directory):
    G = ig.read(directory+"network.gml")
    
    add_vertex_with_attrs(G, {"community": 0})
    G.as_undirected()
    f = open(directory+"classLabel.txt", 'r+')
    labels = [line for line in f.readlines()]
    
    for i,v in enumerate(G.vs()):
        v["community"] = int(labels[i])-1
    
    G_VC = membership_toVC(G)
    return G, G_VC

####################################################################
################## CAPTURE NETWORK FEATURES ########################
####################################################################

# Global
def global_features(G):
    dictionary = {}
    degrees = G.degree() # degrees list
    dictionary['k_mean'] = np.mean(degrees) # mean degree
    dictionary['k_min'] = np.min(degrees) # min degree
    dictionary['k_max'] = G.maxdegree() # max degree
    dictionary['radius'] = G.radius() # radius
    dictionary['diameter'] = G.diameter() # diameter
    dictionary['num_nodes'] = G.vcount() # nodes count
    dictionary['num_edges'] = G.ecount() # edges count
    dictionary['r_degree'] = G.assortativity_degree() # degree assortativity
    eigenvector_list, dictionary['eigenvector_value'] = G.eigenvector_centrality(directed = False,return_eigenvalue = True)
    dictionary['num_largest_cliques'] = len(G.largest_cliques())
    dictionary['len_max_clique'] = G.clique_number()
    dictionary['len_maximal_cliques'] = len(G.maximal_cliques())

    return dictionary

# Local
def local_features(G,i,j,feature):
    
    degrees = G.degree() # degrees list
    if (feature == 'degree_vi'):
        return degrees[i]
    elif (feature == 'degree_vj'):
        return degrees[j]
    elif (feature == 'closeness_vi'):
        return G.closeness()[i]
    elif (feature == 'closeness_vj'):
        return G.closeness()[j]
    elif (feature == 'eigenvector_vi'):
        return eigenvector_list[i]
    elif (feature == 'eigenvector_vj'):
        return eigenvector_list[j]
    elif (feature == 'pagerank_vi'):
        return G.pagerank(directed = False)[i]
    elif (feature == 'pagerank_vj'):
        return G.pagerank(directed = False)[j]
    elif (feature == 'num_shortest_paths'):
        return len(G.get_all_shortest_paths(i,j))
    elif (feature == 'len_shortest_paths'):
        return len(G.get_shortest_paths(i,j))
    
####################################################################
######################### NODE SIMILARITY ##########################
####################################################################
    
# Common Neighbor
def common_neighbors(G,i,j):
    return (len(set(G.neighbors(i)) & set(G.neighbors(j))))

# Salton
def salton(G,i,j,degrees):
    return (common_neighbors(G,i,j)/np.sqrt(degrees[i]*degrees[j]))

# Jaccard
def jaccard(G,i,j):
    return (common_neighbors(G,i,j)/ len(set(G.neighbors(i)) | set(G.neighbors(j))))

# Sorensen
def sorensen(G,i,j,degrees):
    return 2*common_neighbors(G,i,j)/(degrees[i]+degrees[j])

# Adamic Adar (common_neighbors)
def adamic_adar(G,i,j):
    return 1/np.log(common_neighbors(G,i,j))

# Resource Allocation
def aloc_recursos(G,i,j,degrees):
    z = set(G.neighbors(i)) & set(G.neighbors(j))
    return np.sum([1/degrees[x] for x in z])

# Hub promoted
def hub_promot(G,i,j,degrees):
    return common_neighbors(G,i,j)/max(degrees[i],degrees[j])

# Hub depressed
def hub_depres(G,i,j,degrees):
    return common_neighbors(G,i,j)/min(degrees[i],degrees[j])

# Leicht-Holme-Newman
def lhn(G,i,j,degrees):
    return common_neighbors(G,i,j)/(degrees[i]*degrees[j])


####################################################################
########################## SAVE AND LOAD ###########################
####################################################################

# Load Networks
def load_realnet_gml(netname):
    directory = "Real Networks GML/"+netname+"/"

    return create_net(directory)

def load_artnet_lfr(netname):
    directory = "Artificial Networks GML/"+netname+"/"

    return create_net(directory)

####################################################################
# Real
def save_real_features(netname,features):
    w = csv.writer(open("Real Networks GML/"+netname+"/"+"global_features.csv", "w"))
    for key, val in features.items():
        w.writerow([key, val])

# Artificial
def save_artificial_features(netname,features):
    w = csv.writer(open("Artificial Networks GML/"+netname+"/"+"global_features.csv", "w"))
    for key, val in features.items():
        w.writerow([key, val])

# Modified
def save_modified_features(path,metric,features):
    w = csv.writer(open("Modified Networks GML/"+path+"_"+metric+"_global_features.csv", "w"))
    for key, val in features.items():
        w.writerow([key, val])
            
# Load information
# Real
def load_real_features(netname):
    reader = csv.reader(open("Real Networks GML/"+netname+"/"+"global_features.csv", 'r'))
    dictionary = {}
    for key,value in reader:
        dictionary[key] = ast.literal_eval(value)
    return dictionary

# Artificial
def load_artificial_features(netname):
    reader = csv.reader(open("Artificial Networks GML/"+netname+"/"+"global_features.csv", 'r'))
    dictionary = {}
    for key,value in reader:
        dictionary[key] = ast.literal_eval(value)
    return dictionary

# Modified
def load_modified_features(path):
    reader = csv.reader(open("Modified Networks GML/"+path+"/"+"global_features.csv", 'r'))
    dictionary = {}
    for key,value in reader:
        dictionary[key] = ast.literal_eval(value)
    return dictionary

# Insert Features
def insert_features(dictionary,key,value):
    if key not in dictionary:
        dictionary[key] = value
    return dictionary

def save_nmis(network,metric,algorithm_r,algorithm_m,algorithm,perc_insert,heuristic):
    with open("Resultados/"+heuristic+"/"+algorithm+"/"+perc_insert+"/"+network+"_"+metric+".txt", 'w') as file_handler:
        for i, j in zip(algorithm_r,algorithm_m):
                file_handler.write("{} {}\n".format(i,j))
    
    

