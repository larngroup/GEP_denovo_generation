# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 11:49:12 2021

@author: tiago
"""
# external
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem,Lipinski
from rdkit import DataStructs
# import pylev
import math
from scipy.spatial import distance

class Utils:
    """Data Loader class"""
    
    def __init__(self):
        """ Definition of the SMILES vocabulary """
        
        # atoms = [
        #           'H','B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I'
        #         ]
        # special = [
        #         '(', ')', '[', ']', '=', '#', '@', '*', '%', '0', '1', '2',
        #         '3', '4', '5', '6', '7', '8', '9', '.', '/', '\\', '+', '-',
        #           'c', 'n', 'o', 's','p'
        #         ]
        # padding = ['G', 'E','A'] #Go, Padding and End characters
        # self.table = sorted(atoms, key=len, reverse=True) + special + padding
        
        self.table = ['H','Se','se','As','Si','Cl', 'Br','B', 'C', 'N', 'O', 'P', 
                  'S', 'F', 'I', '(', ')', '[', ']', '=', '#', '@', '*', '%', 
                  '0', '1', '2','3', '4', '5', '6', '7', '8', '9', '.', '/',
                  '\\', '+', '-', 'c', 'n', 'o', 's','p','G','E','A']
    
    @staticmethod
    def smilesDict(token_table):
        """ Computes the dictionary that makes the correspondence between 
        each token and the respective integer.

        Args
        ----------
            tokens (list): List of each possible symbol in the SMILES

        Returns
        -------
            tokenDict (dict): Dictionary mapping characters into integers
        """

        tokenDict = dict((token, i) for i, token in enumerate(token_table))
        return tokenDict
    
    @staticmethod
    def pad_seq(smiles,tokens,config):
        """ Performs the padding for each sampled SMILES.

        Args
        ----------
            smiles (str): SMILES string
            tokens (list): List of each possible symbol in the SMILES
            config (json): Configuration file

        Returns
        -------
            smiles (str): Padded SMILES string
        """
        
        if isinstance(smiles, str) == True:
            smiles = [smiles]
            
        maxLength = config.paddSize
    
        for i in range(0,len(smiles)):
            smiles[i] = 'G' + smiles[i] + 'E'
            if len(smiles[i]) < maxLength:
                smiles[i] = smiles[i] + tokens[-1]*(maxLength - len(smiles[i]))
        # print("Padded sequences: ", len(filtered_smiles))
        return smiles
    

    @staticmethod            
    def smiles2idx(smiles,tokenDict):
        """ Transforms each token into the respective integer.

        Args
        ----------
            smiles (str): Sampled SMILES string 
            tokenDict (dict): Dictionary mapping characters to integers 

        Returns
        -------
            newSmiles (str): Transformed smiles, with the characters 
                              replaced by the respective integers. 
        """   
        
        newSmiles =  np.zeros((len(smiles), len(smiles[0])))
        for i in range(0,len(smiles)):
            # print(i, ": ", smiles[i])
            for j in range(0,len(smiles[i])):
                
                try:
                    newSmiles[i,j] = tokenDict[smiles[i][j]]
                except:
                    value = tokenDict[smiles[i][j]]
        return newSmiles
    
    @staticmethod             
    def smiles2mol(smiles_list):
        """
        Function that converts a list of SMILES strings to a list of RDKit molecules 
        Parameters
        ----------
        smiles: List of SMILES strings
        ----------
        Returns list of molecules objects 
        """
        mol_list = []
        if isinstance(smiles_list,str):
            mol = Chem.MolFromSmiles(smiles_list, sanitize=True)
            mol_list.append(mol)
        else:
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi, sanitize=True)
                mol_list.append(mol)
        return mol_list
    
    @staticmethod             
    def tokenize(config,smiles,token_table):
        """ Transforms the SMILES string into a list of tokens.

        Args
        ----------
            config (json): Configuration file
            smiles (str): Sampled SMILES string
            token_table (list): List of each possible symbol in the SMILES

        Returns
        -------
            tokenized (str):  SMILES string with individualized tokens.
        """           

        tokenized = []
        
        for idx,smile in enumerate(smiles):
            N = len(smile)
            i = 0
            j= 0
            tokens = []
            # print(idx,smile)
            while (i < N):
                for j in range(len(token_table)):
                    symbol = token_table[j]
                    if symbol == smile[i:i + len(symbol)]:
                        tokens.append(symbol)
                        i += len(symbol)
                        break
            while (len(tokens) < config.paddSize):
                tokens.append(token_table[-1])
                
                
            tokenized.append(tokens)
    
        return tokenized

    @staticmethod             
    def idx2smi(model_output,tokenDict):
        """ Transforms model's predictions into SMILES

        Args
        ----------
            model_output (array): List with the autoencoder's predictions 
            tokenDict (dict): Dictionary mapping characters into integers

        Returns
        -------
            reconstructed_smiles (array): List with the reconstructed SMILES 
                                          obtained by transforming indexes into
                                          tokens. 
        """           

        key_list = list(tokenDict.keys())
        val_list = list(tokenDict.values())

        reconstructed_smiles =  []
        for i in range(0,len(model_output)):
            smi = []
            for j in range(0,len(model_output[i])):
                
                smi.append(key_list[val_list.index(model_output[i][j])])
                
            reconstructed_smiles.append(smi)
                
        return reconstructed_smiles
    
    @staticmethod
    def remove_padding(trajectory):  
        """ Function that removes the padding characters from the sampled 
            molecule

        Args
        ----------
            trajectory (str): Padded generated molecule

        Returns
        -------
            trajectory (str): SMILES string without the padding character
        """     
        
        if 'A' in trajectory:
        
            firstA = trajectory.find('A')
            trajectory = trajectory[0:firstA]
        return trajectory
    
    @staticmethod
    def reading_csv(config):
        """ This function loads the labels of the biological affinity dataset
        
        Args
        ----------
            config (json): configuration file
        
        Returns
        -------
            raw_labels (list): Returns the respective labels in a numpy array. 
        """
  
        raw_labels = []
            
        with open(config.file_path_a2d, 'r') as csvFile:
            reader = csv.reader(csvFile)
            
            it = iter(reader)
    #        next(it, None)  # skip first item.    
            for row in it:
               
                try:
                    raw_labels.append(float(row[1]))
                except:
                    pass

                
        return raw_labels
    

    @staticmethod 
    def get_reward_MO(rges,sampled_gep,disease_gep,smile,memory_smiles):
        """ This function uses the predictor and the sampled SMILES string to 
        predict the numerical rewards regarding the evaluated properties.

        Args
        ----------
            predictor_affinity (object): Predictive model that accepts a trajectory
                                     and returns the respective prediction 
            smile (str): SMILES string of the sampled molecule
            memory_smiles (list): List of the last 30 generated molecules


        Returns
        -------
            rewards (list): Outputs the list of reward values for the evaluated 
                            properties
        """
        all_rewards = []
        
        index_usp7 = len(sampled_gep[0])-1
        
        initial_level_expression_usp7 = disease_gep[0,index_usp7]  
        level_expression_usp7 = sampled_gep[0,index_usp7]
        diff = initial_level_expression_usp7-level_expression_usp7
        
        if diff < 0.05: #normal 0.15 noise 0.05
            reward_target = 0
        else:
            reward_target = np.exp(diff + 1.3) #np.exp(-diff + 1.3)
            
        all_rewards.append(reward_target)

        reward_gep = np.exp(-rges/3+ 1.15)
        all_rewards.append(reward_gep)
       
        diversity = 1
        if len(memory_smiles)> 30:
            diversity = Utils.external_diversity(smile,memory_smiles)
            
        if diversity < 0.75:
            rew_div = 0.95
            print("\Alert: Similar compounds")
        else:
            rew_div = 1
        all_rewards.append(rew_div)
        return all_rewards,diff

    
    @staticmethod 
    def external_diversity(set_A,set_B):
        """ Computes the Tanimoto external diversity between two sets
        of molecules

        Args
        ----------
            set_A (list): Set of molecules in the form of SMILES notation
            set_B (list): Set of molecules in the form of SMILES notation


        Returns
        -------
            td (float): Outputs a number between 0 and 1 indicating the Tanimoto
                        distance.
        """

        td = 0
        if type(set_A) != list:
            set_A = [set_A]
            
        fps_A = []
        for i, row in enumerate(set_A):
            try:
                mol = Chem.MolFromSmiles(row)
                fps_A.append(AllChem.GetMorganFingerprint(mol, 3))
            except:
                print('ERROR: Invalid SMILES!')
        if set_B == None:
            for ii in range(len(fps_A)):
                for xx in range(len(fps_A)):
                    ts = 1 - DataStructs.TanimotoSimilarity(fps_A[ii], fps_A[xx])
                    td += ts          
          
            td = td/len(fps_A)**2
        else:
            fps_B = []
            for j, row in enumerate(set_B):
                try:
                    mol = Chem.MolFromSmiles(row)
                    fps_B.append(AllChem.GetMorganFingerprint(mol, 3))
                except:
                    print('ERROR: Invalid SMILES!') 
            
            
            for jj in range(len(fps_A)):
                for xx in range(len(fps_B)):
                    ts = 1 - DataStructs.TanimotoSimilarity(fps_A[jj], fps_B[xx]) 
                    td += ts
            
            td = td / (len(fps_A)*len(fps_B))
        print("\nTanimoto distance: " + str(td))  
        return td
                
    @staticmethod 
    def scalarization(rewards,scalarMode,weights,pred_range_rges,pred_range_target):
        """ Transforms the vector of two rewards into a unique reward value.
        
        Args
        ----------
            rewards (list): List of rewards of each property;
            scalarMode (str): String indicating the scalarization type;
            weights (list): List containing the weights indicating the importance 
                            of the each property;
            pred_range (list): List with the prediction max and min values of 
                               the reward to normalize the obtained reward 
                               (between 0 and 1).

        Returns
        -------
            rescaled_reward (float): Scalarized reward
        """
        w_rges = weights[0]
        w_target = weights[1]
        
        rew_rges = rewards[1]
        rew_target = rewards[0]
        rew_div = rewards[2]
        
        max_rges = pred_range_rges[1]
        min_rges = pred_range_rges[0]
    
        max_target = pred_range_target[1]
        min_target = pred_range_target[0]
    
        rescaled_rew_target = (rew_target - min_target )/(max_target - min_target)
        
        if rescaled_rew_target < 0:
            rescaled_rew_target = 0
        elif rescaled_rew_target > 1:
            rescaled_rew_target = 1
    
        rescaled_rew_rges = (rew_rges  - min_rges)/(max_rges -min_rges)
        
        if rescaled_rew_rges < 0:
            rescaled_rew_rges = 0
        elif rescaled_rew_rges > 1:
            rescaled_rew_rges = 1
        
        if scalarMode == 'linear':
            # return rewards[0],rescaled_rew_rges,rescaled_rew_target
            return (w_rges*rescaled_rew_rges + w_target*rescaled_rew_target)*4*rew_div,rescaled_rew_rges,rescaled_rew_target
    
        elif scalarMode == 'chebyshev':
    #        dist_qed = 0
    #        dist_kor = 0
    #        w_a2d = 1-w_a2d
    #        w_bbb = 1 - w_bbb
            dist_rges = abs(rescaled_rew_rges-1)*w_rges
            dist_target = abs(rescaled_rew_target-1)*w_target
            print("distance rges: " + str(dist_rges))
            print("distance target: " + str(dist_target))
            
            if dist_rges > dist_target:
                return rescaled_rew_rges*3
            else:
                return rescaled_rew_target*3
    
    @staticmethod 
    def padding_one_hot(smiles,tokens): 
        """ Performs the padding of the sampled molecule represented in OHE
        Args
        ----------
            smiles (str): Sampled molecule in the form of OHE;
            tokens (list): List of tokens that can constitute the molecules   

        Returns
        -------
            smiles (str): Padded sequence
        """

        smiles = smiles[0,:,:]
        maxlen = 65
        idx = tokens.index('A')
        padding_vector = np.zeros((1,43))
        padding_vector[0,idx] = 1
    
        while len(smiles) < maxlen:
            smiles = np.vstack([smiles,padding_vector])
                
        return smiles
    
    def canonical_smiles(smiles,sanitize=True, throw_warning=False):
        """
        Takes list of generated SMILES strings and returns the list of valid SMILES.
        Parameters
        ----------
        smiles: List of SMILES strings to validate
        sanitize: bool (default True)
            parameter specifying whether to sanitize SMILES or not.
                For definition of sanitized SMILES check
                http://www.rdkit.org/docs/api/rdkit.Chem.rdmolops-module.html#SanitizeMol
        throw_warning: bool (default False)
            parameter specifying whether warnings will be thrown if a SMILES is
            invalid
        Returns
        -------
        new_smiles: list of valid SMILES (if it is valid and has <60 characters)
        and NaNs if SMILES string is invalid
        valid: number of valid smiles, regardless of the its size
            
        """
        new_smiles = []
        valid = 0
        for sm in smiles:
            try:
                mol = Chem.MolFromSmiles(sm, sanitize=sanitize)
                s = Chem.MolToSmiles(mol)
                
                if len(s) <= 65:
                    new_smiles.append(s)       
                else:
                    new_smiles.append('')
                valid = valid + 1 
            except:
                new_smiles.append('')
        return new_smiles,valid
   

    
    def plot_training_progress(training_rewards,losses_generator,losses_vae,training_rges,training_target,rewards_rges,rewards_target):
        """ Plots the evolution of the rewards and loss throughout the 
        training process.
        Args
        ----------
            training_rewards (list): List of the rewards for each sampled 
                                     batch of molecules;
            training_losses (list): List of the rewards for each sampled 
                                     batch of molecules;

        Returns
        -------
            Plot
        """
        plt.plot(training_rewards)
        plt.xlabel('Training iterations')
        plt.ylabel('Average rewards')
        plt.show()
        
        plt.plot(losses_generator)
        plt.xlabel('Training iterations')
        plt.ylabel('Average losses PGA')
        plt.show()
        
        plt.plot(losses_vae)
        plt.xlabel('Training iterations')
        plt.ylabel('Average losses VAE')
        plt.show()
        
        plt.plot(training_rges)
        plt.xlabel('Training iterations')
        plt.ylabel('Average RGES')
        plt.show()
        
        plt.plot(training_target)
        plt.xlabel('Training iterations')
        plt.ylabel('Average difference')
        plt.show()
        
        plt.plot(rewards_rges)
        plt.xlabel('Training iterations')
        plt.ylabel('Average rewards RGES')
        plt.show()
        
        plt.plot(rewards_target)
        plt.xlabel('Training iterations')
        plt.ylabel('Average reward target')
        plt.show()
    
    def plot_individual_rewds(rew_affinity,rew_gep):
        """ This function plots the evolution of rewards for each property to
        optimize
        
        Args
        ----------
            rew_affinity (list): list with previous reward values for the 
                                affinity property
            
            rew_gep (list): list with previous reward values for gep property

        Returns
        -------
            Plot
        """
        
        plt.plot(rew_affinity)
        plt.xlabel('Training iterations')
        plt.ylabel('Average rewards affinity')
        plt.show()
        plt.plot(rew_gep)
        plt.xlabel('Training iterations')
        plt.ylabel('Average rewards bbb')
        plt.show()
    
    def plot_evolution(pred_original,pred_iteration85,property_identifier):
        """ This function plots the comparison between two distributions of 
        some specified property 
        
        Args
        ----------
            pred_original (list): list original model predictions
            pred_iteration85 (list): list model predictions after 85 iterations
            property_identifier (str): string that indicates the desired property
            
        Returns
        -------
            Plot
        """        
        
        pred_original = np.array(pred_original)
        pred_iteration85 = np.array(pred_iteration85)
     
    #    sns.set(font_scale=1)
        legend_0 = "Original" 
        legend_85 = "Biased" 
    
        if property_identifier == 'kor':
            label = 'Predicted pIC50 for KOR'
            plot_title = 'Distribution of predicted pIC50 for generated molecules'
        elif property_identifier == 'a2d':
            label = 'Predicted pIC50 for A2AR'
            plot_title = 'Distribution of predicted pIC50 for generated molecules'
        elif property_identifier == 'logP':
            label = 'Predicted logP'
            plot_title = 'Distribution of predicted logP for generated molecules'
            plt.axvline(x=1.0,color = 'black')
            plt.axvline(x=4.0,color = 'black')
        elif property_identifier == 'qed':
            label = 'Predicted QED'
            plot_title = 'Distribution of predicted QED for generated molecules'
    #    v1 = pd.Series(pred_original, name=legend_0)
    #    v3 = pd.Series(pred_iteration85, name=legend_85)
        v1 = pd.Series(pred_original)
        v3 = pd.Series(pred_iteration85)
     
        ax = sns.kdeplot(v1, shade=True,color='g')
        sns.kdeplot(v3, shade=True,color='r')    
    #    ax.set(xlabel=label,title=plot_title,xlim=(-1, 7))
        
    #    plt.setp(ax.get_legend().get_texts(), fontsize='13') # for legend text
    #    plt.setp(ax.get_legend().get_title(), fontsize='14') # for legend title
    #    sns.set(rc={'figure.figsize':(5.0, 3.0)})
    
        plt.show()

    @staticmethod             
    def reconstruction_evaluation(original_input,model_output):
        """ Evaluates the reconstructed molecules in terms of rate of correctly
            reconstructed molecules, Tanimoto distance and Levenshtein distance.
                    
        Args
        ----------
            original_input (array): List with the original molecules
            model_output (array): List with the autoencoder's predictions 

        Returns
        -------
            metrics (list): List with the evaluated metrics 
        """ 
        metrics = []
        valid = 0 
        tanimoto_distances = []
        levenshtein_distances = []
        
        
        for idx,out_smi in enumerate(model_output):          
            predicted_mol = ''. join(out_smi)
            
            if predicted_mol == original_input[idx]:
                                
                valid +=1
            
            
            try: 
                out_mol = Chem.MolFromSmiles(predicted_mol)
                inp_mol = Chem.MolFromSmiles(original_input[idx])
       
                fp_output = AllChem.GetMorganFingerprint(out_mol, 3)
                fp_original = AllChem.GetMorganFingerprint(inp_mol, 3)
                tanimoto_distances.append(DataStructs.TanimotoSimilarity(fp_output, fp_original))
                
            except:
                print("Invalid: ",predicted_mol)
                tanimoto_distances.append(0)
                
                                
        # levenshtein_distances.append(pylev.levenshtein(predicted_mol, original_input[idx]))
                      
        valid = (valid/len(original_input))*100
        
        metrics.append(valid)
        metrics.append(np.mean(tanimoto_distances))
        # metrics.append(np.mean(levenshtein_distances))
                
        return metrics

    def get_idxs(original_input,model_output):
        """ Performs the transformations to the molecular VAE predictions 
            (gets the indexes of each token, computes mse and removes the 
             initial, final and padding characters.)
                    
        Args
        ----------
            original_input (array): List with the original molecules
            model_output (array): List with the autoencoder's predictions 
    
        Returns
        -------
            model_clean_mols (list): List with the predicted molecules;
            mse: Mean Squared error between the model output and the original
                molecules;
        """    

        mols_new = []
        
        for idx_mol in range(0,len(model_output)):
            mol = np.squeeze(model_output[idx_mol])
            for idx_element in range(0,len(mol)):
                mol[idx_element] = round(mol[idx_element],0)
            mols_new.append(mol)
            
            
        mse = np.mean((np.square(np.array(mols_new) - original_input)).mean(axis=1))
        
        model_clean_mols = []
        
        for idx,inp_mol in enumerate(original_input):
            padd = (inp_mol == 46).sum()
            
            if padd > 0:
                out_mol = np.squeeze(model_output[idx])[0:-padd]
                
            out_mol = out_mol[1:-1]
            
            model_clean_mols.append(out_mol)
        
        return model_clean_mols,mse
                
                
    def moving_average(previous_values, new_value, ma_window_size=10): 
        """
        This function performs a simple moving average between the previous 9 and the
        last one reward value obtained.
        ----------
        previous_values: list with previous values 
        new_value: new value to append, to compute the average with the last ten 
                   elements
        
        Returns
        -------
        Outputs the average of the last 10 elements 
        """
        value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
        value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
        return value_ma
                
            
            
        
    def compute_thresh(rewards,thresh_set):
        """
        Function that computes the thresholds to choose which Generator will be
        used during the generation step, based on the evolution of the reward values.
        Parameters
        ----------
        rewards: Last 3 reward values obtained from the RL method
        thresh_set: Integer that indicates the threshold set to be used
        Returns
        -------
        This function returns a threshold depending on the recent evolution of the
        reward. If the reward is increasing the threshold will be lower and vice versa.
        """
        reward_t_2 = rewards[0]
        reward_t_1 = rewards[1]
        reward_t = rewards[2]
        q_t_1 = reward_t_2/reward_t_1
        q_t = reward_t_1/reward_t
        
        if thresh_set == 1:
            thresholds_set = [0.15,0.3,0.2]
        elif thresh_set == 2:
            thresholds_set = [0.05,0.2,0.1] 
        #        thresholds_set = [0,0,0] 
        
        threshold = 0
        if q_t_1 < 1 and q_t < 1:
            threshold = thresholds_set[0]
        elif q_t_1 > 1 and q_t > 1:
            threshold = thresholds_set[1]
        else:
            threshold = thresholds_set[2]
        
        return threshold

    def serialize_model(generator_biased,combined_model_object,config,pol):
        generator_biased.model.save('models//generator//biased_generator.hdf5')
        
        combined_model_object.encoder_mol.save('models//mols//Encoder_biased.h5')
        combined_model_object.decoder_gep.save('models//gep//Decoder_biased.h5')
        combined_model_object.encoder_gep.save('models//gep//Encoder_biased.h5')
        
        # model_json = generator_biased.model.to_json()
        # with open(config.model_name_biased + "_" +str(pol)+".json", "w") as json_file:
        #     json_file.write(model_json)
            
        # # serialize weights to HDF5
        # generator_biased.model.save_weights(config.model_name_biased + '_' +str(pol)+".h5")
        # print("Updated model saved to disk")
    
    def plot_comparation(rges_unb,rges_b,diffs_unb,diffs_b,n_generated,valid_unb,valid_b):

        rges_unb = np.array(rges_unb)
        rges_b = np.array(rges_b)
        
        diffs_unb = np.array(diffs_unb)
        diffs_b = np.array(diffs_b)

        
        print("Proportion of valid SMILES (UNB,B):", valid_unb/n_generated,valid_b/n_generated )
        
        legend_rges_unb = 'Unbiased model'
        legend_rges_b = 'Biased model'
        print("\n\nMax RGES: (UNB,B)", np.max(rges_unb),np.max(rges_b))
        print("Mean RGES: (UNB,B)", np.mean(rges_unb),np.mean(rges_b))
        print("Min RGES: (UNB,B)", np.min(rges_unb),np.min(rges_b))
    
        label_rges = 'Calculated RGES'
        plot_title_rges = 'Distribution of calculated RGES for the sampled molecules'

        v1 = pd.Series(rges_unb, name=legend_rges_unb)
        v2 = pd.Series(rges_b, name=legend_rges_b)
       
        
        ax = sns.kdeplot(v1, shade=True,color='g',label=legend_rges_unb)
        sns.kdeplot(v2, shade=True,color='r',label =legend_rges_b )
    
        ax.set(xlabel=label_rges, 
               title=plot_title_rges)
      
        plt.legend()
        plt.show()
        
        
        legend_diffs_unb = 'Unbiased model'
        legend_diffs_b = 'Biased model'
        print("\n\nMax diff: (UNB,B)", np.max(diffs_unb),np.max(diffs_b))
        print("Mean diff: (UNB,B)", np.mean(diffs_unb),np.mean(diffs_b))
        print("Min diff: (UNB,B)", np.min(diffs_unb),np.min(diffs_b))
    
        label_diffs = 'Calculated differences'
        plot_title_diffs = 'Distribution of calculated differences for the sampled molecules'
                            

        v1 = pd.Series(diffs_unb, name=legend_diffs_unb)
        v2 = pd.Series(diffs_b, name=legend_diffs_b)
       
        
        ax = sns.kdeplot(v1, shade=True,color='g',label=legend_diffs_unb)
        sns.kdeplot(v2, shade=True,color='r',label =legend_diffs_b)
    
        ax.set(xlabel=label_diffs, 
               title=plot_title_diffs)
      
        plt.legend()
        plt.show()
        
            
    def bar_plot_comparison(disease_gep,query_gep,genes_names):

        X_axis = np.arange(len(query_gep))
        
        plt.bar(X_axis - 0.2, disease_gep, 0.4, label = 'Disease',color = 'b')
        plt.bar(X_axis + 0.2, query_gep, 0.4, label = 'Sampled', color = 'r')
        

        plt.xticks(X_axis, genes_names)
        plt.xlabel("Genes")
        plt.ylabel("Level of expression")
        plt.title("Level of expression of each gene")
        plt.legend()
        plt.rcParams["figure.figsize"] = (20,8)
        plt.show()
        
        
    def euclidean_distance(x,y):
        return distance.euclidean(x,y)
        # return math.sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
 

    def rmse(y, x):
        """
        This function implements the root mean squared error measure
        ----------
        y_true: True label   
        y_pred: Model predictions 
        Returns
        -------
        Returns the rmse metric to evaluate regressions
        """
        
        return  math.sqrt(np.square(np.subtract(x,y)).mean()) 
 
        # from keras import backend
        # return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

    def mse(y_true, y_pred):
        """
        This function implements the mean squared error measure
        ----------
        y_true: True label   
        y_pred: Model predictions 
        Returns
        -------
        Returns the mse metric to evaluate regressions
        """
        from keras import backend
        return backend.mean(backend.square(y_pred - y_true), axis=-1)

    def r_square(y_true, y_pred):
        """
        This function implements the coefficient of determination (R^2) measure
        ----------
        y_true: True label   
        y_pred: Model predictions 
        Returns
        -------
        Returns the R^2 metric to evaluate regressions
        """
        from keras import backend as K
        SS_res =  K.sum(K.square(y_true - y_pred)) 
        SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
        return (1 - SS_res/(SS_tot + K.epsilon()))


    #concordance correlation coeﬃcient (CCC)
    def ccc(y_true,y_pred):
        """
        This function implements the concordance correlation coefficient (ccc)
        ----------
        y_true: True label   
        y_pred: Model predictions 
        Returns
        -------
        Returns the ccc measure that is more suitable to evaluate regressions.
        """
        from keras import backend as K
        num = 2*K.sum((y_true-K.mean(y_true))*(y_pred-K.mean(y_pred)))
        den = K.sum(K.square(y_true-K.mean(y_true))) + K.sum(K.square(y_pred-K.mean(y_pred))) + K.int_shape(y_pred)[-1]*K.square(K.mean(y_true)-K.mean(y_pred))
        return num/den
        
    def update_weights(scaled_rewards_rges,scaled_rewards_target,weights):
        
        mean_rges_previous = np.mean(scaled_rewards_rges[-5:-1])
        mean_rges_current = scaled_rewards_rges[-1:][0]
        
        mean_target_previous = np.mean(scaled_rewards_target[-5:-1])
        mean_target_current = scaled_rewards_target[-1:][0]
        
        growth_rges = (mean_rges_current - mean_rges_previous)/mean_rges_previous
        
        growth_target = (mean_target_current - mean_target_previous)/mean_target_previous
        
        if mean_rges_current*weights[0] > mean_target_current*weights[1] and growth_target < 0.01:
            weights[0] = weights[0]-0.10
            weights[1] = weights[1]+0.10
        elif mean_rges_current*weights[0] < mean_target_current*weights[1] and growth_rges < 0.01:
            weights[0] = weights[0]+0.10
            weights[1] = weights[1]-0.10
            
        print(weights) 
        
        return weights
    
    
    def check(list1,list2, val1,val2):
      
        # traverse in the list
        for idx in range(0,len(list1)):

            if list1[idx] < val1 and list2[idx] < val2:
                return True 
        return False
    
    def check_Lipinski(list_mols):
        
        is_lipinsky = [bool((Lipinski.NumHDonors(mol) <= 5) \
    and (Lipinski.NumHAcceptors(mol) <= 10) \
    and (Lipinski.rdMolDescriptors.CalcExactMolWt(mol) < 500) \
    and(Lipinski.rdMolDescriptors.CalcCrippenDescriptors(mol)[0]) <= 5) for mol in list_mols]
        
        
        return round(sum(is_lipinsky)/len(list_mols),4)*100
        
                       