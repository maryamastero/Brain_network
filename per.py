

import numpy as np
from mlxtend.evaluate import permutation_test

clust_ASD = np.load('clust_ASD.npy')
clust_NT = np.load('clust_NT.npy')
mean_clust_ASD = np.mean(clust_ASD,axis=0)
mean_clust_NT = np.mean(clust_NT,axis=0)

p_value = permutation_test(mean_clust_ASD, mean_clust_NT)
print('P value: %.2f'% p_value)



