# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:46:29 2021

@author: tiago
"""

# External
import pandas as pd
import numpy as np
import scipy.stats as ss
import random
from utils.utils import Utils
from matplotlib import pyplot as plt

class GEP_manager(object):
    """
    Object that manages the GEP-related data to compare the disease 
    with the obtained GEPs.

    """
    def __init__(self,disease_gep_vae,disease_gep_real,landmark_genes):
        self.disease_gep_vae = disease_gep_vae
        self.disease_gep_real = disease_gep_real
        self.landmark_genes = landmark_genes
        
        # Build the dictionary that associates each gene to an index
        self.d = dict()
        for idx,g in enumerate(self.landmark_genes):
            self.d[g] = idx
            
        self.idxs = [self.d[gene] for gene in self.landmark_genes]
        
        # Load disease signature
        self.ds = pd.read_csv('data/deseq_processed.csv')
        
        self.de_genes = self.ds['Gene']
        self.de_genes_idxs = [self.d[gene] for gene in self.de_genes]
        self.de_genes_up_down = self.ds['up_down']

    def de_genes_differences(self,diffs_memory,query_gep):
        """ Computes and plots the differences between the original and induced
        GEPs for the DE genes

        Args
        ----------
            diffs_memory (list): previously computed differences
            query_gep (array): GEP induced by the sampled compound
        Returns
        -------
            diffs_memory (list): Updated list of the GEP differences
        """
        expression_de_genes_disease = self.disease_gep_real[0,self.de_genes_idxs]
        expression_de_genes_sampled = query_gep[0,self.de_genes_idxs]
        
        if len(diffs_memory)==0:
            
            diffs_memory = [[expression_de_genes_disease[i]-expression_de_genes_sampled[i]] for i in range(0,len(expression_de_genes_disease))]
            
            diffs_memory.append([self.disease_gep_real[0,1988]-query_gep[0,1988]])

        else:
            
            for i in range (0,len(diffs_memory)-1):
                diffs_memory[i].append(expression_de_genes_disease[i]-expression_de_genes_sampled[i])
            
            diffs_memory[len(diffs_memory)-1].append(self.disease_gep_real[0,1988]-query_gep[0,1988])
        
            plt.plot(diffs_memory[0][-75:],label=self.de_genes[0])
            plt.plot(diffs_memory[1][-75:],label=self.de_genes[9])
            plt.plot(diffs_memory[11][-75:],label='USP7')
            plt.legend(loc="upper left")           
            plt.xlabel("Iterations")
            plt.ylabel("Level of expression")
            ax = plt.gca()

            ax.set_ylim([-0.35, 0.35])
            plt.show()
        return diffs_memory
        
        
    def rges_calculator(self,query_gep):   
        """ Computes the reverse gene expression score (RGES)

        Args
        ----------
            query_gep (array): Obtained GEP conditioned by the sampled molecule

        Returns
        -------
            rges (float): Value between -1 and 1 
        """
        
        # Load candidate gene expression profile
        self.query_gep = query_gep

        # self.query_gep = np.random.uniform(low=-5.0, high=5.0, size=(2128,))
        
        ranks = ss.rankdata(self.query_gep)
        
        candidate_gep_ranks = pd.DataFrame()
        
        candidate_gep_ranks['gene_id'] = self.idxs
        candidate_gep_ranks['rank'] = ranks
        
    
        gene_idxs = [self.d[g] for g in self.de_genes]
        
        self.ds['geneId']  = gene_idxs  
        
        ds_up = self.ds[self.ds['up_down']== 'up']
        ds_up = ds_up[["geneId"]]
        
        ds_down = self.ds[self.ds['up_down']== 'down']
        ds_down = ds_down[['geneId']]
        
        rank_up = []
        for g_up in ds_up['geneId']:
            list_idx = candidate_gep_ranks.index[candidate_gep_ranks['gene_id'] == g_up].tolist()
            rank_up.append(candidate_gep_ranks.loc[list_idx[0], 'rank'])   
        
        ds_up['rank'] =  rank_up
        
        rank_down = []
        for g_down in ds_down['geneId']:
            list_idx = candidate_gep_ranks.index[candidate_gep_ranks['gene_id'] == g_down].tolist()
            rank_down.append(candidate_gep_ranks.loc[list_idx[0], 'rank'])   
        
        ds_down['rank'] = rank_down
            
        rank_up.sort()
        rank_down.sort()
        
        num_genes_up = len(rank_up)
        num_genes_down = len(rank_down)
        
        n_genes = len(candidate_gep_ranks)
        
        
        if num_genes_up > 1:
            a_ups = []
            b_ups = []
            
            for j in range(0,num_genes_up):
                a_ups.append(self.aux_func('a',j,num_genes_up,rank_up,n_genes))
                b_ups.append(self.aux_func('b',j,num_genes_up,rank_up,n_genes))
                
            a_up = max(a_ups)
            b_up = max(b_ups)
            
            if a_up > b_up:
                ks_up = a_up
            else:
                ks_up = -b_up
        else:
            ks_up = 0
                
        if num_genes_down > 1:
            
            a_downs = []
            b_downs = []
            
            for j in range(0,num_genes_down):
                a_downs.append(self.aux_func('a',j,num_genes_down,rank_down,n_genes))
                b_downs.append(self.aux_func('b',j,num_genes_down,rank_down,n_genes)) 
            
            a_down = max(a_downs)
            b_down = max(b_downs)
            
            if a_down > b_down:
                ks_down = a_down
            else:
                ks_down = -b_down
        else:
            ks_down = 0
            
        if ks_up == 0 and ks_down != 0:
            rges = -ks_down
        elif ks_up != 0 and ks_down == 0:
            rges = ks_down
        else:
            rges = ks_up - ks_down

        return rges        
    
    def aux_func(self,mode,j,n_genes_class, list_ranks, total_genes):
        """ Computes two parameters that are necessary to obtain the rges

        Args
        ----------
            mode (str): Indication of the kind of parameter to be computed
                        ('a' or 'b')
            j (int): Index of the differentially expressed gene
            n_genes_class (int): Number of differentially expressed genes
            list_ranks (list): list of the genes ranked by the level of 
                               expression in the sampled GEP
            total_genes (int): Total number of analyzed genes

        Returns
        -------
            a/b (float): parameter to compute the rges
        """
        
        if mode == 'a':
            return j/n_genes_class - list_ranks[j]/total_genes
        elif mode == 'b':
            return list_ranks[j]/total_genes - (j-1)/n_genes_class
        
    
    def compare_de_genes(self,query_gep):
        """ Plots the combined bar plot with the level of expression for each 
            differentially expressed gene
        Args
        ----------
            query_gep (array): Obtained GEP conditioned by the sampled molecule
        """
        
        expression_de_genes_disease = self.disease_gep_real[0,self.de_genes_idxs]
        expression_de_genes_sampled = query_gep[0,self.de_genes_idxs]
        
        idxs_sample = random.sample(range(len(self.de_genes.tolist())-1), 6)
       
        de_disease_sample = expression_de_genes_disease[idxs_sample]
        de_query_sample = expression_de_genes_sampled[idxs_sample]
             
        genes = [str(i) +' ' + self.de_genes_up_down[i] for i,gene in enumerate(self.de_genes.tolist()) if i in idxs_sample]
        X_axis = np.arange(len(idxs_sample))
        
        plt.bar(X_axis - 0.2, de_disease_sample, 0.4, label = 'Disease',color = 'b')
        plt.bar(X_axis + 0.2, de_query_sample, 0.4, label = 'Sampled', color = 'r')
        

        plt.xticks(X_axis, genes)
        plt.xlabel("Genes")
        plt.ylabel("Level of expression")
        plt.title("Level of expression of each gene")
        plt.legend()
        plt.show()
        
    def de_genes_visualizer(self,query_gep):
        """ Visualizer of the DE gene expression (all genes or the specified)
        Args
        ----------
            query_gep (array): Obtained GEP conditioned by the sampled molecule
        """
        
        for g in self.de_genes:
            print(g)
        
        looping = True
        while looping:
            input_genes = input("Enter the name of SIX genes (maximum) to be represented (separeted by commas) or type 'all' to compare all DE genes: ")
            
            genes_list = input_genes.split(',')
            try: 
                if genes_list[0] == 'all':
                    
                    
                    genes_list_axis = [gene +' ' + self.de_genes_up_down[i] for i,gene in enumerate(self.de_genes.tolist())]
                    genes_list_axis.append('USP7')
                    
                    de_genes_idxs = self.de_genes_idxs.copy()
                    de_genes_idxs.append(1988)
                    
                    expression_de_genes_disease_real = self.disease_gep_real[0,de_genes_idxs]
                    expression_de_genes_disease_vae = self.disease_gep_vae[0,de_genes_idxs]
                    
                    expression_de_genes_sampled = query_gep[0,de_genes_idxs]


                    Utils.bar_plot_comparison(expression_de_genes_disease_real,expression_de_genes_sampled,genes_list_axis)
                    Utils.bar_plot_comparison(expression_de_genes_disease_vae,expression_de_genes_sampled,genes_list_axis)
                    
                elif len(genes_list) < 2 or len(genes_list) > 6:
                    looping = False
                    
                else:
                    genes_idxs = [idx for idx,gene in enumerate(self.de_genes) if gene in genes_list]
                    
                    genes_list_axis = [gene +' ' + self.de_genes_up_down[i] for i,gene in enumerate(self.de_genes.tolist()) if i in genes_idxs]
                    
                    expression_de_genes_disease = self.disease_gep_vae[0,self.de_genes_idxs]
                    expression_de_genes_sampled = query_gep[0,self.de_genes_idxs]
                    
                    de_disease_sample = expression_de_genes_disease[genes_idxs]
                    de_query_sample = expression_de_genes_sampled[genes_idxs]
                         
                    Utils.bar_plot_comparison(de_disease_sample,de_query_sample,genes_list_axis)
                    # X_axis = np.arange(len(genes_idxs))
                    
            except:
                looping = False