# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:21:51 2020

@author: arsii

"""
import re
import networkx as nx
import scipy.io
from pathlib import Path
from os import listdir
from os.path import isfile, join
import numpy as np
#import bct
import seaborn as sns
import matplotlib.pylab as plt
import pandas as pd



#%%
# insert path to adjacency matrices here
# script uses the same folder for edgelist saving.
asd_folder = Path("./asd_group/Brainnetome_0mm/") # ASD subjects 
td_folder = Path("./control_group/Brainnetome_0mm/") # Neurotypical subjecst 

# reads filenames (not subdirectories) into lists
asd_files = [f for f in listdir(asd_folder) if isfile(join(asd_folder, f))]
td_files = [f for f in listdir(td_folder) if isfile(join(td_folder, f))]

#%% Dictionary with brain parcellation regions
df = pd.read_csv("./node_parcellation.txt", sep=',')
#%%
roi_dict = {}

for i in range(df.shape[0]):
    roi_dict[i] = (df['ROI'][i]).strip()

#%% Adjacency matrices to weighted edgelists

# process asd subjects
stack_asd = [] 
sa2 = []
for i in asd_files:
    if re.search('.mat$', i) and re.search('_reg',i): # filters only .mat files in case there is some crap in the folder
        file_to_open = asd_folder / i
        mat = scipy.io.loadmat(file_to_open) # load mat as some dictionary-object-thing
        adj = mat['Adj'] # extract the adjacency matrix
        adj2 = mat['Adj']
        flat = adj.flatten()
        flat = np.sort(flat)
        cutoff = flat[-100] # pick cutoff level for top 100
        adj = np.where(adj <= cutoff,0,1) # set values below cutoff to zero
        adj = adj + adj.T - np.diag(np.diag(adj)) # make it symmetric, this is actually not needed for the test
        np.fill_diagonal(adj,1) 
        
        stack_asd.append(adj) # thresholded
        sa2.append(adj2) # all values
        #G = nx.from_numpy_matrix(adj) # create a graph
        #file_to_save = asd_folder / re.sub('.mat$', '.gpickle' ,i)  
        #nx.write_gpickle(G, file_to_save) # save as weighted edgelist
asd_full = np.dstack(stack_asd) # <- thresholded
sa2_full = np.dstack(sa2) # <- all values

#%%        
stack = []   
# process td subjects    
stack_td = []
st2 = []
for i in td_files:
    if re.search('.mat$', i) and re.search('_reg',i):# filters only .mat files in case there is some crap in the folder
        file_to_open = td_folder / i
        mat = scipy.io.loadmat(file_to_open)
        adj = mat['Adj']
        adj2 = mat['Adj']
        flat = adj.flatten()
        flat = np.sort(flat)
        cutoff = flat[-100]
        adj = np.where(adj <= cutoff,0,1)
        adj = adj + adj.T - np.diag(np.diag(adj))
        np.fill_diagonal(adj,1)
        
        stack_td.append(adj)
        st2.append(adj2)
        #G = nx.from_numpy_matrix(adj)
        #file_to_save = td_folder / re.sub('.mat$', '.gpickle' ,i)  
        #nx.write_gpickle(G, file_to_save)

td_full = np.dstack(stack_td)  
st2_full = np.dstack(st2)      

#%%
# for two sided t-test, t-value 2 refers to p-value < 0.05
pval, adj, null = bct.nbs_bct(asd_full,td_full, thresh= 2, tail="left", k=10000, verbose=True)
#%%
np.save("pvals_NBS_left",pval)
np.save("adj_NBS_left",adj)
np.save("null_NBS_left",null)

#%%
ax = sns.heatmap(adj)
#%%
sns.distplot(null,kde=False)
#%%
np.fill_diagonal(adj,0)
G1 = nx.from_numpy_matrix(adj)
#%%
iso_list1 = nx.isolates(G1)
#%%
G1.remove_nodes_from(list(nx.isolates(G1)))
#%%
nx.draw(G1)
plt.show()
#%%
node_list = []

for e in G1.edges():   
    node_list.append(str(roi_dict[e[0]]+str(' - ')+str(roi_dict[e[1]]))+'\n')
    
#%% write to file
with open("connection_differences_right.txt","w") as f:
    f.writelines(node_list)
    