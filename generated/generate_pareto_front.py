# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 10:41:56 2022

@author: tiago
"""
# internal
from utils.utils import Utils

# external
import pandas as pd
import matplotlib.pyplot as plt
import csv

filepath = "generated_mols"
all_mols = []
all_rges = []
all_usp7 = []

for r in range(1,11):
    
    with open(filepath+ '_run_' + str(r) + '_noise.smi', 'r') as csvFile:
        reader = csv.reader(csvFile)
        
        it = iter(reader)
        next(it, None)  # skip first item.    
        for row in it:
            mol = row[0]
            rges = row[1]
            usp7 = row[2]
            
            if mol not in all_mols and float(usp7)>-0.5:
                all_mols.append(mol)
                all_rges.append(float(rges))
                all_usp7.append(-float(usp7))
    
            
    
# identify non-dominated points
dominated_mols  = []
for idx in range(0,len(all_mols)):
    if (Utils.check(all_rges,all_usp7,all_rges[idx],all_usp7[idx])) == True:
        dominated_mols.append(idx)
        
all_usp7_nd = [element for idx,element in enumerate(all_usp7) if idx not in dominated_mols]
all_rges_nd = [element for idx,element in enumerate(all_rges) if idx not in dominated_mols]
all_mols_nd = [element for idx,element in enumerate(all_mols) if idx not in dominated_mols]

all_usp7_d = [element for idx,element in enumerate(all_usp7) if idx in dominated_mols]
all_rges_d = [element for idx,element in enumerate(all_rges) if idx in dominated_mols]
all_mols_d = [element for idx,element in enumerate(all_mols) if idx in dominated_mols]
 
plt.scatter(all_usp7_d , all_rges_d, color = 'b', label='Dominated', alpha=0.5)
plt.scatter(all_usp7_nd, all_rges_nd, color = 'r', label='Non-dominated', alpha=0.5)

plt.xlabel("Difference USP7 gene expression: conditioned - original")
plt.ylabel("RGES")
plt.title("Sampled molecules")
plt.legend()
plt.ylim(-1.45, -0.1)
plt.xlim(-0.4,0.1)
plt.show()
     
all_data = pd.DataFrame()  
     
all_data['molecules'] = all_mols
all_data['rges'] = all_rges
all_data['usp7'] = all_usp7

dominated = [idx in dominated_mols for idx in range(0,len(all_mols))]
all_data['dominated']  = dominated

all_data.to_pickle('pareto_generation_mols_noise.pkl')