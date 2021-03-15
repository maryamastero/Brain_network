# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:21:51 2020

@author: arsii

This script reads matlab created adjacency matric files, converts them into 
networkx graphs and saves as weighted edgelists.

"""
import re
import networkx as nx
import scipy.io
from pathlib import Path
from os import listdir
from os.path import isfile, join



#%%
# insert path to adjacency matrices here
# script uses the same folder for edgelist saving.
asd_folder = Path("./asd_group/Brainnetome_0mm/") # ASD subjects 
td_folder = Path("./control_group/Brainnetome_0mm/") # Neurotypical subjecst 

# reads filenames (not subdirectories) into lists
asd_files = [f for f in listdir(asd_folder) if isfile(join(asd_folder, f))]
td_files = [f for f in listdir(td_folder) if isfile(join(td_folder, f))]

#%% Adjacency matrices to weighted edgelists

# process asd subjects
for i in asd_files:
    if re.search('.mat$', i) and re.search('_reg',i): # filters only .mat files in case there is some crap in the folder
        file_to_open = asd_folder / i
        mat = scipy.io.loadmat(file_to_open) # load mat as some dictionary-object-thing
        adj = mat['Adj'] # extract the adjacency matrix
        G = nx.from_numpy_matrix(adj) # create a graph
        file_to_save = asd_folder / re.sub('.mat$', '.gpickle' ,i)  
        nx.write_gpickle(G, file_to_save) # save as weighted edgelist
        
        
# process td subjects    
for i in td_files:
    if re.search('.mat$', i) and re.search('_reg',i):# filters only .mat files in case there is some crap in the folder
        file_to_open = td_folder / i
        mat = scipy.io.loadmat(file_to_open)
        adj = mat['Adj']
        G = nx.from_numpy_matrix(adj)
        file_to_save = td_folder / re.sub('.mat$', '.gpickle' ,i)  
        nx.write_gpickle(G, file_to_save)
        
'''
How to use pickle:
nx.write_gpickle(G, "test.gpickle")
G = nx.read_gpickle("test.gpickle")
'''