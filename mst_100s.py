#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:09:39 2020

@author: aaltonen
"""


# Import libraries
import numpy as np
import scipy.io
import networkx as nx
import os

#%% List files

asd_path = 'asd_group/Brainnetome_0mm'
asd_files = os.listdir(asd_path)

nt_path = 'control_group/Brainnetome_0mm'
nt_files = os.listdir(nt_path)

#%% 
n_selection =100 # how many strongest

for file in nt_files:
    
    # First check that we have regressed file
    if 'reg.mat' in file: 
        
        file_path = os.path.join(nt_path, file)
        
        # Load .mat file, note that file contains dictionary -> data under key "Adj"
        mat = scipy.io.loadmat(file_path) 
        
        # Take fishers inverse transform from correlation values
        adj = np.tanh(mat['Adj'])
        
        
        # Read this adjancency matrix into network G
        G = nx.from_numpy_matrix(adj) 
        
        
        # Next we select 100 top edges
        # Or actually can we just remove 30035 weakest edge? Yeah that's easier
        
        # List edges and nodes
        edges = list(G.edges(data=True))
        nodes = list(G.nodes)
        
        # Sort them
        edges_sort = sorted(edges,key=lambda x: x[2]['weight'])
        
        n_edges = len(edges)
        n_nodes = len(nodes)
        
        # Weakest edges
        edges_weak = edges_sort[:n_edges-n_selection]
        
        # Find minimum spanning tree
        mst = nx.minimum_spanning_tree(G)
        mst_edges = list(mst.edges(data=True))

        edges_to_remove = [edge for edge in edges_weak if edge not in mst_edges]

        # Next remove edges
        G.remove_edges_from(edges_to_remove)
        
        fname = file[:-30] + '.weighted.edgelist'
        fpath = os.path.join('MST_100strong', fname)
        nx.write_weighted_edgelist(G, fpath)
        
        
        
        
        
        
        