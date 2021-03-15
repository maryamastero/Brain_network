import numpy as np
from mlxtend.evaluate import permutation_test
import pandas as pd


ASD = np.load('clust_ASD.npy')
NT = np.load('clust_NT.npy')

p_values_two = []
p_values_ADSgNT = []
p_values_NTgADS = []

#now = datetime.now()

#current_time = now.strftime("%H:%M:%S")
#print("Current Time =", current_time)

for node in range(ASD.shape[0]):
  ASD_node = ASD[node,:]
  NT_node = NT[node,:]

  p_value = permutation_test(ASD_node, NT_node,
                           method='approximate',
                           num_rounds=20000)
  p_values_two.append(p_value)

  p_value = permutation_test(ASD_node, NT_node, func='x_mean > y_mean',
                           method='approximate',
                           num_rounds=20000)
  p_values_ADSgNT.append(p_value)

  p_value = permutation_test(ASD_node, NT_node, func='x_mean < y_mean',
                           method='approximate',
                           num_rounds=20000)
  p_values_NTgADS.append(p_value)

  #print(node)
#now = datetime.now()

#current_time = now.strftime("%H:%M:%S")
#print("Current Time =", current_time)

np.save('p_values_two_side',np.array(p_values_two))
np.save('p_values_ADS_great_NT',np.array(p_values_ADSgNT))
np.save('p_values_NT_great_ASD',np.array(p_values_NTgADS))

def fdr(P_value):
    P_value = p_values_two
    from scipy.stats import rankdata
    ranked_p_values = rankdata(P_value)
    fdr = P_value * len(P_value) / ranked_p_values
    fdr[fdr > 1] = 1
    
    return fdr

p_values_two =  np.load('p_values_two_side.npy')
diff_two = np.where(fdr(p_values_two)<0.05)
print(diff_two)

p_values_ADSgNT = np.load('p_values_ADS_great_NT.npy')
diff_ADSgNT = np.where(fdr(p_values_ADSgNT)<0.05)
print(diff_ADSgNT)

p_values_NTgADS = np.load('p_values_NT_great_ASD.npy')
diff_NTgADS = np.where(fdr(p_values_NTgADS)<0.05)
print(diff_NTgADS)


np.save('diff_two_side',np.array(diff_two))
np.save('diff_ADS_great_NT',np.array(diff_ADSgNT))
np.save('diff_NT_great_ASD',np.array(diff_NTgADS))

roi_dict = {}
df = pd.read_csv("./node_parcellation.txt", sep=',',names=['node', 'ROI'])
for i in range(df.shape[0]):
    roi_dict[i] = (df['ROI'][i]).strip()
    
diff_two =  np.load('diff_two_side.npy')    
node_list_two = []
for d in range(diff_two.shape[1]):
    nodes = roi_dict[d]
    node_list_two.append(nodes)
print(node_list_two)

diff_ADS_great_NT =  np.load('diff_ADS_great_NT.npy')    
node_list_ADS_great_NT = []
for d in range(diff_ADS_great_NT.shape[1]):
    nodes = roi_dict[d]
    node_list_ADS_great_NT.append(nodes)
print(node_list_ADS_great_NT)

diff_NT_great_ASD =  np.load('diff_NT_great_ASD.npy')    
node_list_NT_great_ASD = []
for d in range(diff_NT_great_ASD.shape[1]):
    nodes = roi_dict[d]
    node_list_NT_great_ASD.append(nodes)
print(node_list_NT_great_ASD)






