# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:48:26 2021

@author: tiago
"""
# Internal
from model.model_combination import Model_combination
from model.model_gep import Model_gep
from model.generator import Generator  

# External
import numpy as np
from keras.models import Sequential
import pickle
import pandas as pd
import csv
from rdkit import Chem

class DataLoader:
    @staticmethod
    def load_generator(config,generator_type):
        """ Initializes the Generator and loads the weights of the trained 
        models if the generator type is "biased"

        Args
        ----------
            config (bunch): Configuration file
            generator_type (str): type of generator to load (biased or unbiased)

        Returns
        -------
            generator_model (sequential): Molecular generator model 
        """
                
        generator_model = Sequential()
        generator_model=Generator(config,True)
        
        path = ''
        if generator_type == 'biased':
            path = config.generator_biased_path
        elif generator_type == 'unbiased':
            path = config.generator_unbiased_path
        
        generator_model.model.load_weights(path)
        
        return generator_model
    

    @staticmethod
    def load_general_model(config,model_type):
        """ Initializes and loads the model that will combine the two VAE - for 
        molecules and GEPs.

        Args
        ----------
            config (bunch): Configuration file
            generator_type (str): type of VAEs to load (biased or unbiased)

        Returns
        -------
            general_model (Functional): Model containing the molecular 
                                              and gep encoders and the gep 
                                              decoder.
        """
        general_model = Model_combination(config)
        
        path_mols_encoder = config.mols_encoder_path
        path_gep_encoder = config.gep_encoder_path
        path_gep_decoder = config.gep_decoder_path
        
        if model_type == 'biased':
            path_mols_encoder = path_mols_encoder + '_biased'
            path_gep_encoder = path_gep_encoder + '_biased'
            path_gep_decoder = path_gep_decoder + '_biased'
            
        general_model.encoder_mol.load_weights(path_mols_encoder+".h5")
        general_model.decoder_gep.load_weights(path_gep_decoder+".h5")
        general_model.encoder_gep.load_weights(path_gep_encoder+".h5")
        
        return general_model

    @staticmethod
    def load_gep_vae(config):
        """ Initializes and loads the GEP VAE

        Args
        ----------
            config (bunch): Configuration file

        Returns
        -------
            general_model (Functional): Model containing the gep VAE
        """
        general_model = Model_gep(config)
        
        general_model.encoder_model.load_weights(config.gep_encoder_path+".h5")
        general_model.decoder_model.load_weights(config.gep_decoder_path+".h5")
        general_model.autoencoder.load_weights(config.gep_autoencoder_path)
            
        
        return general_model

    @staticmethod
    def load_gep(config):
        """ Loads the GEP data, selects the expression patterns for a specific
        disease and set of genes and computes the mean value.

        Args
        ----------
            config (bunch): Configuration file

        Returns
        -------
            gep_mean (array): Mean value of expression of each analyzed gene 
        """
        # open the landmark genes file
        file = open('data/new_landmark_genes.pkl', 'rb')
        
        # dump information to that file
        landmark_genes = pickle.load(file)
        
        infile = open(config.genes_path,'rb')
        genes = pickle.load(infile)
        infile.close()
        genes_idxs = genes["indexes"].tolist()
        
        #Select the landmark genes that are in the TCGA dataset
        landmark_genes_filtered = [gene for idx,gene in enumerate(landmark_genes) if gene in genes['genes'].tolist()]
        landmark_idxs = [idx for idx,gene in enumerate(landmark_genes) if gene in genes['genes'].tolist()]
        
        if config.disease_gep_type == 'tcga':
            tcga_gep = pd.read_csv('data/disease_gep.csv')
            disease_gep = np.array(tcga_gep['0'])
            disease_gep = disease_gep.reshape((1,1989))
            
        elif config.disease_gep_type == 'gdsc':
            
        
            infile = open(config.gep_path,'rb')
            df = pickle.load(infile)
            infile.close()
            
            data_carcinoma = df[(df["histology"] == 'carcinoma')]
    
            gep = np.zeros([len(data_carcinoma),len(genes_idxs)])
            
            for i in range(0,len(data_carcinoma)):
                gep_vector = (data_carcinoma.iloc[i,-1])
                for j in range(0,len(landmark_idxs)):
                    gep[i,j] = gep_vector[landmark_idxs[j]]
            
            
            disease_gep = np.mean(gep, axis = 0)  
            disease_gep = np.transpose(disease_gep.reshape(-1,1))
            
        idx_usp = len(disease_gep)
        return disease_gep,landmark_genes_filtered

    def load_geps_dataset(self,config):
            """ Loads the GEP vectors
            Args:
                config (json): Path of the configuration file
                gene_set (list): List containing the names of the genes to be 
                                 selected
    
            Returns:
                dataset (list): The list with the training and testing GEPS 
            """
            training = pd.DataFrame()
            testing = pd.DataFrame()

            training = pd.read_pickle(config.path_gep_training)
            testing = pd.read_pickle(config.path_gep_testing)
            
            train_arr = np.array(training)
            testing_arr = np.array(testing)
            
            return train_arr, testing_arr
                
    def load_chembl_dataset(self,config):
        """ Loads the molecular generator dataset
        Args:
            config (json): Path of the configuration file
            
        Returns:
            smiles (list): List of the generator training set 
        """            
        smiles = []
        config.path_chembl    
        with open("data\\train_chembl_22_clean_1576904_sorted_std_final.smi", 'r') as csvFile:
            reader = csv.reader(csvFile)
            
            it = iter(reader)  
            for row in it:
               
                try:
                    smiles.append(row[0])
                except:
                    pass

                
        return smiles
    
    def load_known_drugs(self,config):
        """ Loads the set of known anti-cancer drugs
        Args:
            config (json): Path of the configuration file
            
        Returns:
            raw_smiles (list): List containing the smiles of known anti-cancer
            drugs
        """            
        raw_smiles = []
    
        fp = config.path_known_drugs
        with open(fp, 'r') as csvFile:
            reader = csv.reader(csvFile)
            
            it = iter(reader)
            next(it, None)  # skip first item.    
            for row in it:
            
                row_splited = row[0].split(';')
                
                if fp == 'usp7_chembl.csv' :
                    idx_smiles = 7
                    # idx_pic50 = 5
                else:
                    idx_smiles = 30
                try:
                    
                    sm = row_splited[idx_smiles][1:-1]
                    # print(sm,row_splited[idx_pic50][1:-1])
                    mol = Chem.MolFromSmiles(sm, sanitize=True)
                    s = Chem.MolToSmiles(mol)
 
                    if s not in raw_smiles and len(s)>10:
                        
                        # if fp == 'cancer_drugs.csv':
                        raw_smiles.append(s)
 
                        # elif fp == 'usp7_chembl.csv' and float(row_splited[idx_pic50][1:-1]) > 4.5:
                        #     raw_smiles.append(s)
         
                except:
                    print('Invalid: ',sm)

        return raw_smiles 
    
    def load_paccmann_mols(self,config):
        """ Loads the PaccMannRL generated molecules
        Args:
            config (json): Path of the configuration file
            
        Returns:
            raw_smiles (list): List of the compounds optimized by the PaccMann method
        """            
        raw_smiles = []
    
        fp = config.path_paccmann_drugs
        with open(fp, 'r') as csvFile:
            reader = csv.reader(csvFile)
            
            it = iter(reader)
            next(it, None)  # skip first item.    
            for row in it:
                            
                
                idx_smiles = 2
        
                try:
                    
                    sm = row[idx_smiles]
                    # print(sm,row_splited[idx_pic50][1:-1])
                    mol = Chem.MolFromSmiles(sm, sanitize=True)
                    s = Chem.MolToSmiles(mol)
 
                    if s not in raw_smiles and len(s)>10:

                        raw_smiles.append(s)

         
                except:
                    print('Invalid: ',sm)

        return raw_smiles 

 