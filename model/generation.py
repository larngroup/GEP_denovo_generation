    # -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:49:39 2021

@author: tiago
"""

# internal
from .base_model import BaseModel
from dataloader.dataloader import DataLoader
from utils.utils import Utils
from model.generator import Generator  
from model.gep_manager import GEP_manager

# external
import tensorflow as tf
import numpy as np
from rdkit import Chem
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import Crippen
from rdkit.Chem import QED
from utils.sascorer_calculator import SAscore
# from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()

tf.config.experimental_run_functions_eagerly(True)

class generation_process(BaseModel): 
    """Conditional Generation Object"""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.token_table = Utils().table # Load the table of possible tokens
        self.adam = optimizers.Adam(clipvalue=4)
        self.generator_biased = Sequential()
        self.tokenDict = Utils.smilesDict(self.token_table)
        self.scalarMode = 'linear'
        self.weights = [0.5,0.5]
        self.pred_range_rges = [2.9,5] #normal[3,5.4]    noise[2.9,5] 
        self.pred_range_target = [2.8,5.2]  #normal[3.2,5]  noise[2.8,5.2]

    def load_models(self):
        """ Loads the gene expression data and all required models (generator, 
        biological affinity predictor and the VAE combination)
        """      
        self.gep_mean,self.landmark_genes = DataLoader().load_gep(self.config)
        self.generator_unbiased = DataLoader().load_generator(self.config,'unbiased')
        self.general_model_object = DataLoader().load_general_model(self.config,'unbiased')
        # self.general_model_object = DataLoader().load_general_model(self.config,'biased')
        self.gep_vae = DataLoader().load_gep_vae(self.config)
    
    def custom_loss(self,aux_matrix):
        """ Computes the loss function to update the generator through the 
        policy gradient algorithm
        Args
        ----------
            aux_matrix (array): Auxiliary matrix to adjust the dimensions and 
                                padding when performing computations.
        Returns
        -------
            lossfunction (float): Value of the loss 
        """
        def lossfunction(y_true,y_pred):

            y_pred = tf.cast(y_pred, dtype='float64')
            y_true = tf.cast(y_true, dtype='float64')
            y_true = tf.reshape(y_true, (-1,))
           
            return (-1/self.config.batch_size)*K.sum(y_true*K.log(tf.math.reduce_sum(tf.multiply(tf.add(y_pred,10**-7),aux_matrix),[1,2])))
       
        return lossfunction

    def get_biased_model(self,aux_array):
        """ Builds the novel generator model

        Args
        ----------
            aux_matrix (array): Auxiliary matrix to adjust the dimensions when
                                performing loss function computations.

        Returns
        -------
            generator_biased (model): Generator model to be updated during with
                                      the policy-gradient algorithm
        """
        
        self.generator_biased=Generator(self.config,False)
        self.generator_biased.model.compile(
                optimizer=self.adam,
                loss = self.custom_loss(aux_array))
        self.generator_biased.model.load_weights(self.config.generator_unbiased_path)

        
        return self.generator_biased.model
        

    def policy_gradient(self, gamma=1):   
        """ Implements the policy gradient algorithm to bias the Generator """
        
        # Obtain the VAE output representation of the disease GEP
        disease_gep_vae = self.gep_vae.autoencoder.predict(self.gep_mean)
        
        self.gep_calculator = GEP_manager(disease_gep_vae,self.gep_mean,self.landmark_genes)
        
        pol = 1
        cumulative_rewards = []

        # Initialize the variables that will contain the output of each prediction
        dimen = len(self.token_table)
        states = []
        
        pol_reward_target = []
        pol_reward_rges = []
        
        pol_target = []
        pol_rges = []
        
        pol_target_reward_scaled = []
        pol_rges_reward_scaled = []
        
        all_rewards = []
        losses_generator = []
        losses_vae = []
        memory_smiles = []
        diffs_sampled = []
        
        # Re-compile the model to adapt the loss function and optimizer to the RL problem
        self.generator_biased.model = self.get_biased_model(np.arange(47))
        
        # Obtain the disease specific gep
        disease_gep = self.gep_mean
        
        for i in range(self.config.n_iterations):
        
            cur_reward = 0
            cur_rges = 0
            cur_target = 0
            cur_reward_rges = 0 
            cur_reward_target = 0 
            cur_reward_rges_scaled = 0 
            cur_reward_target_scaled = 0 

           
            aux_2 = np.zeros([100,47])
            inputs = np.zeros([100,1])
            
            ii = 0
            
            for m in range(self.config.batch_size):
                # Sampling new trajectory
                correct_mol = False
                
                while correct_mol != True:
                    trajectory_ints,trajectory = self.generator_biased.generate(1)
                             
                    # trajectory = 'GCC'
                    try:                     
                        seq = trajectory[0][1:-1]
                        if 'A' in seq: # A is the padding character
                            seq = Utils.remove_padding(trajectory) 
                            
                        print(seq)
                        mol = Chem.MolFromSmiles(seq)
     
                        trajectory = 'G' + Chem.MolToSmiles(mol) + 'E'
#                                trajectory = 'GCCE'
                    
                        if len(memory_smiles) > 30:
                                memory_smiles.remove(memory_smiles[0])                                    
                        memory_smiles.append(seq)
                        
                        if len(trajectory) < self.config.paddSize:
                            correct_mol = True
                                               
                    except:
                        print("\nInvalid SMILES!")
                          
                # Processing the sampled molecule         
                mol_padded = Utils.pad_seq(seq,self.token_table,self.config)
                tokens = Utils.tokenize(self.config,mol_padded,self.token_table)   
                processed_mol = Utils.smiles2idx(tokens,self.tokenDict)
                
                # Get the GEP induced by the sampled molecule
                sampled_gep = self.general_model_object.model.predict([processed_mol,disease_gep])
                         
                diffs_sampled = self.gep_calculator.de_genes_differences(diffs_sampled,sampled_gep)

                rges = self.gep_calculator.rges_calculator(sampled_gep)
                
                rewards,target = Utils.get_reward_MO(rges,sampled_gep,disease_gep,seq,memory_smiles)
                reward,rescaled_rges,rescaled_target = Utils.scalarization(rewards,self.scalarMode,self.weights,self.pred_range_rges,self.pred_range_target)
                
    
                if m == 0 and i < 30:
                    inputs_vae_mol = processed_mol
                    inputs_vae_gep = disease_gep
                elif m > 0 and i < 30:
                   inputs_vae_mol =  np.vstack([inputs_vae_mol,processed_mol])
                   inputs_vae_gep = np.vstack([inputs_vae_gep,disease_gep])
       
                discounted_reward = reward
                cur_reward += reward
                cur_rges += rges
                cur_target += target
                cur_reward_rges += rewards[1]
                cur_reward_target += rewards[0]
                cur_reward_rges_scaled += rescaled_rges
                cur_reward_target_scaled += rescaled_target
                
                inp_p = np.zeros([100,1])
                for p in range(1,len(trajectory_ints)):
   
                    states.append(discounted_reward)
                    
                    inp_p[p-1,0] = processed_mol[0,p-1]
                                    
                    aux_2_matrix = np.zeros([100,47])
                    aux_2_matrix[p-1,int(processed_mol[0,p])] = 1

                    if ii == 0:
                        aux_2 = aux_2_matrix
                        inputs = np.copy(inp_p)
    
                    else:
                        inputs = np.dstack([inputs,inp_p])
         
                        aux_2 = np.dstack([aux_2,aux_2_matrix])
                    ii += 1
                    
                
            inputs = np.moveaxis(inputs,-1,0)
            new_states = np.array(states)
            
            aux_2 = np.moveaxis(aux_2,-1,0)
               
            self.generator_biased.model.compile(optimizer = self.adam, loss = self.custom_loss(tf.convert_to_tensor(aux_2, dtype=tf.float64, name=None)))
            
            if i < 30:
                loss_vae = self.general_model_object.model.train_on_batch([inputs_vae_mol,inputs_vae_gep],inputs_vae_gep)
            
            sampled_gep_after_train = self.general_model_object.model.predict([processed_mol,disease_gep])
              
            self.gep_calculator.compare_de_genes(sampled_gep_after_train)
            
            #update weights based on the provided collection of samples, without regard to any fixed batch size.
            loss_generator = self.generator_biased.model.train_on_batch(inputs,new_states) # (inputs,targets) update the weights with a batch

            # Clear out variables
            states = []
            inputs = np.empty(0).reshape(0,0,dimen)

            cur_reward = cur_reward / self.config.batch_size
            cur_rges = cur_rges / self.config.batch_size
            cur_target = cur_target / self.config.batch_size
            cur_reward_rges = cur_reward_rges / self.config.batch_size
            cur_reward_target = cur_reward_target / self.config.batch_size
            cur_reward_rges_scaled = cur_reward_rges_scaled  / self.config.batch_size
            cur_reward_target_scaled  = cur_reward_target_scaled  / self.config.batch_size
            
            # serialize model to JSON
            Utils.serialize_model(self.generator_biased,self.general_model_object,self.config,pol)

            all_rewards.append(Utils.moving_average(all_rewards, cur_reward)) 
            
            pol_rges.append(Utils.moving_average(pol_rges, cur_rges))                   
            pol_reward_rges.append(Utils.moving_average(pol_reward_rges, cur_reward_rges))  
            pol_target.append(Utils.moving_average(pol_target, cur_target))  
            pol_rges_reward_scaled.append(Utils.moving_average(pol_rges_reward_scaled, cur_reward_rges_scaled))  
            pol_target_reward_scaled.append(Utils.moving_average(pol_target_reward_scaled, cur_reward_target_scaled))  
            pol_reward_target.append(Utils.moving_average(pol_reward_target, cur_reward_target))  

            losses_generator.append(Utils.moving_average(losses_generator, loss_generator))
            losses_vae.append(Utils.moving_average(losses_vae, 0))
    
            Utils.plot_training_progress(all_rewards,losses_generator,losses_vae,pol_rges,pol_target,pol_reward_rges,pol_reward_target)
            
            if i%5==0 and i > 0:
                self.weights = Utils.update_weights(pol_rges_reward_scaled,pol_target_reward_scaled,self.weights)
            
        cumulative_rewards.append(np.mean(all_rewards[-10:]))
        print("Average Reward RGES: ", np.mean(pol_reward_rges[-10:]))
        print("Average scaled Reward RGES: ", np.mean(pol_rges_reward_scaled[-10:]))
        print("Average Reward USP7: ", np.mean(pol_reward_target[-10:]))
        print("Average scaled Reward USP7: ", np.mean(pol_target_reward_scaled[-10:]))
        
        self.gep_calculator.de_genes_visualizer(sampled_gep_after_train)
        pol+=1

        return cumulative_rewards
    
    def model_validation(self): 
        """ Validation of the model by comparing properties of the molecules 
        generated by the unbiased and optimized Generators
        """   
        
        self.generator_unbiased = DataLoader().load_generator(self.config,'unbiased')
        self.generator_biased = DataLoader().load_generator(self.config,'biased')
        self.general_model_object_biased = DataLoader().load_general_model(self.config,'biased')
        self.general_model_object_unbiased = DataLoader().load_general_model(self.config,'unbiased')
        
        disease_gep_vae = self.gep_vae.autoencoder.predict(self.gep_mean)
        
        self.gep_calculator = GEP_manager(disease_gep_vae,self.gep_mean,self.landmark_genes)
        
        _,trajectory = self.generator_unbiased.generate(self.config.mols_to_generate)
        
        smiles_unbiased = [smile[1:-1] for smile in trajectory]
        
        sanitized_unb,valid_unb = Utils.canonical_smiles(smiles_unbiased, sanitize=True, throw_warning=False) # validar 
        
        sanitized_valid_mols_unb = []
        
        for smi in sanitized_unb:
            if len(smi)>1:
                sanitized_valid_mols_unb.append(smi)
        
        rges_unbiased = []
        diffs_unbiased = []
        
        # disease_gep_vae = self.gep_mean
        for mol in sanitized_valid_mols_unb:
            
            mol_padded = Utils.pad_seq(mol,self.token_table,self.config)
            tokens = Utils.tokenize(self.config,mol_padded,self.token_table)   
            
            processed_mol = Utils.smiles2idx(tokens,self.tokenDict)
            sampled_gep = self.general_model_object_biased.model.predict([processed_mol,self.gep_mean])
            rges = self.gep_calculator.rges_calculator(sampled_gep)
            rges_unbiased.append(rges)
            diffs_unbiased.append(self.gep_mean[0,1988] - sampled_gep[0,1988])

        _,trajectory = self.generator_biased.generate(self.config.mols_to_generate)
        
        smiles_biased = [smile[1:-1] for smile in trajectory]
        
        sanitized_b,valid_b = Utils.canonical_smiles(smiles_biased, sanitize=True, throw_warning=False) # validar 
        
        sanitized_valid_mols_b = []
        
        for smi in sanitized_b:
            if len(smi)>1:
                sanitized_valid_mols_b.append(smi)
        
        rges_biased = []
        sampled_geps = []
        diffs_biased = []
        
        for mol in sanitized_valid_mols_b:
            
            mol_padded = Utils.pad_seq(mol,self.token_table,self.config)
            tokens = Utils.tokenize(self.config,mol_padded,self.token_table)   
            
            processed_mol = Utils.smiles2idx(tokens,self.tokenDict)
            sampled_gep = self.general_model_object_biased.model.predict([processed_mol,self.gep_mean])
            rges = self.gep_calculator.rges_calculator(sampled_gep)
            rges_biased.append(rges)
            sampled_geps.append(sampled_gep)
            diffs_biased.append(self.gep_mean[0,1988] - sampled_gep[0,1988])
            
        Utils.plot_comparation(rges_unbiased,rges_biased,diffs_unbiased,diffs_biased,self.config.mols_to_generate,valid_unb,valid_b)
        
        # Sort the mols by the rges
        sorted_rges_biased = np.argsort(rges_biased)
        
        filtered_mols = [sanitized_valid_mols_b[element] for idx,element in enumerate(sorted_rges_biased) if idx < 75]
        filtered_geps = [sampled_geps[element] for idx,element in enumerate(sorted_rges_biased) if idx < 75]
        filtered_rges = [rges_biased[element] for idx,element in enumerate(sorted_rges_biased) if idx < 75]
        
        for i in range(0,len(filtered_mols)):
            print('\n---- GEP comparison for DE genes ----')
            print("Sampled SMILES: ", filtered_mols[i])
            print("RGES: ", filtered_rges[i])
            print("USP7 expression difference: ",diffs_biased[i])
            print("\n")
            self.gep_calculator.de_genes_visualizer(filtered_geps[i])
            
    def model_evaluation(self): 
        """ Generation of molecules with the optimized generator and evaluation
        of the corresponding properties - saving to the generated folder
        """   
        self.generator_biased = DataLoader().load_generator(self.config,'biased')
        
        disease_gep_vae = self.gep_vae.autoencoder.predict(self.gep_mean)
        
        self.gep_calculator = GEP_manager(disease_gep_vae,self.gep_mean,self.landmark_genes)
        
        _,trajectory = self.generator_biased.generate(self.config.mols_to_generate)
        
        smiles = [smile[1:-1] for smile in trajectory]
        
        sanitized,valid = Utils.canonical_smiles(smiles, sanitize=True, throw_warning=False) # validar 
        
        sanitized_valid = [smi for smi in sanitized if len(smi)>1]
                
        rges_values = []
        diffs_values = []
        qed_values = []
        sas_values = []
        logp_values = []
        mw_values = []
        
        mols_list = Utils.smiles2mol(sanitized_valid)
        
        rate_lipinsky = Utils.check_Lipinski(mols_list)

            
        for idx,smile in enumerate(sanitized_valid):
            
            mol_padded = Utils.pad_seq(smile,self.token_table,self.config)
            tokens = Utils.tokenize(self.config,mol_padded,self.token_table)   
            processed_mol = Utils.smiles2idx(tokens,self.tokenDict)
            
            sampled_gep = self.general_model_object.model.predict([processed_mol,self.gep_mean])
            rges = self.gep_calculator.rges_calculator(sampled_gep)
            rges_values.append(rges)
            diffs_values.append(self.gep_mean[0,1988] - sampled_gep[0,1988])
 
    
            mol = mols_list[idx]
            
            sas_values.append(SAscore([mol])[0]) #list
            qed_values.append(QED.qed(mol))
            logp_values.append(Crippen.MolLogP(mol))
            mw_values.append(Descriptors.MolWt(mol))
        
        
        validity = (valid/self.config.mols_to_generate)*100
        print("\n\n% Valid: ",validity)
        
        unique_smiles = np.unique(sanitized_valid)
        percentage_unique = (len(unique_smiles)/self.config.mols_to_generate)*100
        print("\n%Unique: ",percentage_unique)
        
        print("\nSAS: ", np.mean(sas_values))
        print("SAS std: ", np.std(sas_values))
        
        print("\n\n% obeying Lipinski rule: ",rate_lipinsky)
        
        print("\nlogP: ", np.mean(logp_values))
        print("logP std: ", np.std(logp_values))
        
        print("\nMW: ", np.mean(mw_values))
        print("MW std: ", np.std(mw_values))
        
        print("\nQED: ", np.mean(qed_values))
        print("std QED: ", np.std(qed_values))
        
        print("\nMean diffs: ", np.mean(diffs_values))
        print("Std diffs: ", np.std(diffs_values))

        print("\nMean RGES: ", np.mean(rges_values))
        print("Std RGES: ", np.std(rges_values))
        
        with open( "generated//generated_lead_comparison.smi", 'w') as f:
            f.write("SMILES, RGES, USP7, SAS, QED, logP\n")
            for i,smi in enumerate(sanitized_valid):
                
                data = str(sanitized_valid[i]) + ", " +  str(rges_values[i]) + ", " + str(diffs_values[i]) + ", "  + str(sas_values[i]) + ", " + str(qed_values[i]) + ", " + str(logp_values[i])
                f.write("%s\n" % data)  

        
    def experiments(self,experiment_id): 
        """ Evaluation experiments performed to obtain the results depicted in
        the paper
        Args:
            experiment_id (str): String indicating the experiment to perform
            
        Returns:
            experiment 'a': evaluates properties of a set of generated molecules
                            and compares it with both the Generator original 
                            dataset and PaccMannRl optimized mols in terms of 
                            diversity
            experiment 'b': compares the induced GEPs with the actual GEP vectors
            experiment 'c': generates molecules with the optimized model and 
                            evaluates the induced GEPs 
            experiment 'd': evaluates the desired properties of the generated 
                            compounds and saves it to a file
            experiment 'e': general evaluation of procedure of the biased mols
        """   

        
        if experiment_id == 'a':
            
            chembl_dataset = DataLoader().load_chembl_dataset(self.config)
            
            self.generator_unbiased = DataLoader().load_generator(self.config,'biased')
            
            _,trajectory = self.generator_unbiased.generate(self.config.mols_to_generate)
            
            smiles = [smile[1:-1] for smile in trajectory]
            
            sanitized_b,valid = Utils.canonical_smiles(smiles, sanitize=True, throw_warning=False)
            
            sanitized_valid_mols = []
            
            for smi in sanitized_b:
                if len(smi)>1:
                    sanitized_valid_mols.append(smi)
            
            unique_smiles = list(np.unique(sanitized_valid_mols))
            
            percentage_unique = (len(unique_smiles)/self.config.mols_to_generate)*100
            
                        
            mols_paccmann = DataLoader().load_paccmann_mols(self.config)
            
            novel_list = [e for e in mols_paccmann if e not in chembl_dataset]
            rate_novel = len(novel_list)/len(mols_paccmann)

            loaded_mols = DataLoader().load_known_drugs(self.config)
            external_td_paccmann = Utils.external_diversity(mols_paccmann,loaded_mols)
            print("\n% external diversity PaccMannRL: ",external_td_paccmann)
            
            
            internal_td = Utils.external_diversity(sanitized_valid_mols,None)
            external_td = Utils.external_diversity(unique_smiles,chembl_dataset)
            
            mols_list = Utils.smiles2mol(sanitized_valid_mols)
            
            sas_values = []
            qed_values = []
            for mol in mols_list:
                
                sas_values.append(SAscore([mol])[0]) #list
                qed_values.append(QED.qed(mol))
                
            print("\n\n% Valid: ", (valid/self.config.mols_to_generate)*100)
            print("\n% Unique: ", percentage_unique)
            print("\n% novel: ",rate_novel)
            print("\nTanimoto distance internal: ", internal_td)
            print("Tanimoto distance external dataset: ", external_td)
            print("\nQED: ", np.mean(qed_values))
            print("std QED: ", np.std(qed_values))
            print("\nSAS: ", np.mean(sas_values))
            print("std SAS: ", np.std(sas_values))
      
        elif experiment_id == 'e':
            disease_gep_vae = self.gep_vae.autoencoder.predict(self.gep_mean)
            
            self.gep_calculator = GEP_manager(disease_gep_vae,self.gep_mean,self.landmark_genes)
            
            loaded_mols = DataLoader().load_paccmann_mols(self.config)
            
            # loaded_mols = DataLoader().load_known_drugs(self.config)
            
            sanitized,valid = Utils.canonical_smiles(loaded_mols, sanitize=True, throw_warning=False)
            sanitized_valid = [smi for smi in sanitized if len(smi)>1 and len(smi)<75]
                     
            validity = (valid/len(loaded_mols))*100
            print("\n\n% Valid: ",validity)
            
            unique_smiles = np.unique(sanitized_valid)
            percentage_unique = (len(unique_smiles)/len(loaded_mols))*100
            print("\n\n%Unique: ",percentage_unique)
            
            rges_values = []
            diffs_values = []
            qed_values = []
            sas_values = []
            logp_values = []
            mw_values = []
            
            mols_list = Utils.smiles2mol(sanitized_valid)
            
            rate_lipinsky = Utils.check_Lipinski(mols_list)
            print("\n\n% obeying Lipinski rule: ",rate_lipinsky)
            
            # cleaned_mols = [smi for smi in sanitized_valid if '.' not in smi]
            # del sanitized_valid[6]
            # del sanitized_valid[96]
            # del sanitized_valid[30]
            
            for idx,smile in enumerate(sanitized_valid):
                
                mol_padded = Utils.pad_seq(smile,self.token_table,self.config)
                tokens = Utils.tokenize(self.config,mol_padded,self.token_table)   
                processed_mol = Utils.smiles2idx(tokens,self.tokenDict)
                
                sampled_gep = self.general_model_object.model.predict([processed_mol,self.gep_mean])
                rges = self.gep_calculator.rges_calculator(sampled_gep)
                rges_values.append(rges)
                diffs_values.append(self.gep_mean[0,1988] - sampled_gep[0,1988])
     
        
                mol = mols_list[idx]
                
                sas_values.append(SAscore([mol])[0]) #list
                qed_values.append(QED.qed(mol))
                logp_values.append(Crippen.MolLogP(mol))
                mw_values.append(Descriptors.MolWt(mol))
            
            
            print("\nSAS: ", np.mean(sas_values))
            print("SAS std: ", np.std(sas_values))
            
            print("\nlogP: ", np.mean(logp_values))
            print("logP std: ", np.std(logp_values))
            
            print("\nMW: ", np.mean(mw_values))
            print("MW std: ", np.std(mw_values))
            
            print("\nQED: ", np.mean(qed_values))
            print("QED std: ", np.std(qed_values))
            
            print("\nMean diffs: ", np.mean(diffs_values))
            print("Std diffs: ", np.std(diffs_values))
    
            print("\nMean RGES: ", np.mean(rges_values))
            print("Std RGES: ", np.std(rges_values))
            
            
        else:

            self.generator_biased = DataLoader().load_generator(self.config,'biased')
            self.general_model_object_biased = DataLoader().load_general_model(self.config,'biased')
            
            disease_gep_vae = self.gep_vae.autoencoder.predict(self.gep_mean)
            
            self.gep_calculator = GEP_manager(disease_gep_vae,self.gep_mean,self.landmark_genes)
            
            _,trajectory = self.generator_biased.generate(self.config.mols_to_generate)
            
            smiles_biased = [smile[1:-1] for smile in trajectory]
            
            sanitized_b,valid_b = Utils.canonical_smiles(smiles_biased, sanitize=True, throw_warning=False)
            
            sanitized_valid_mols_b = []
            
            for smi in sanitized_b:
                if len(smi)>1:
                    sanitized_valid_mols_b.append(smi)
        
         
            validity = (valid_b/self.config.mols_to_generate)*100
            print("\n\n% Valid: ",validity)
            
            unique_smiles = np.unique(sanitized_valid_mols_b)
            percentage_unique = (len(unique_smiles)/self.config.mols_to_generate)*100
            print("\n\n%Unique: ",percentage_unique)
                
            if experiment_id == 'b': 
                
                tcga_train,tcga_test = DataLoader().load_geps_dataset(self.config)
            
                geps_vectors = []
                diffs = []
                
                avg_eu_dists_train = []
                avg_eu_dists_test = []
                avg_rmse_dists_train = []
                avg_rmse_dists_test = []
                
                for idx,smile in enumerate(sanitized_valid_mols_b):
                    
                    mol_padded = Utils.pad_seq(smile,self.token_table,self.config)
                    tokens = Utils.tokenize(self.config,mol_padded,self.token_table)   
                    processed_mol = Utils.smiles2idx(tokens,self.tokenDict)
                    
                    sampled_gep = self.general_model_object_biased.model.predict([processed_mol,self.gep_mean])
                    
                    geps_vectors.append(sampled_gep)
                 
                    eu_train = [Utils.euclidean_distance(sampled_gep,np.swapaxes(original_gep[:,np.newaxis],0,1)) for original_gep in tcga_train]
                    eu_test = [Utils.euclidean_distance(sampled_gep,original_gep) for original_gep in tcga_test]
                    
                    rmse_train = [Utils.rmse(sampled_gep,original_gep) for original_gep in tcga_train]
                    rmse_test = [Utils.rmse(sampled_gep,original_gep) for original_gep in tcga_test]
                    
                    avg_eu_dists_train.append(np.mean(eu_train))
                    avg_eu_dists_test.append(np.mean(eu_test))
                    
                    avg_rmse_dists_train.append(np.mean(rmse_train))
                    avg_rmse_dists_test.append(np.mean(rmse_test))
                    
                    diffs.append(self.gep_mean[0,1988] - sampled_gep[0,1988])
               
                  
                print("\n\nMax diffs: ", np.max(diffs))
                print("Mean diffs: ", np.mean(diffs))
                print("Min diffs: ", np.min(diffs))
                print("Std diffs: ", np.std(diffs))
        
                print("\nMean Euclidean distance train: ", np.mean(avg_eu_dists_train))
                print("Std Euclidean distance train: ", np.std(avg_eu_dists_train))
                
                print("\nMean Euclidean distance test: ", np.mean(avg_eu_dists_test))
                print("Std Euclidean distance test: ", np.std(avg_eu_dists_test))
                
                print("\nMean RMSE train: ", np.mean(avg_rmse_dists_train))
                print("Std RMSE train: ", np.std(avg_rmse_dists_train))
                
                print("\nMean RMSE test: ", np.max(avg_rmse_dists_test))
                print("Std RMSE test: ", np.std(avg_rmse_dists_test))
                
            elif experiment_id == 'c':
                
                

                diffs = []
                rges_values = []
                
                for idx,smile in enumerate(sanitized_valid_mols_b):
                    
                    mol_padded = Utils.pad_seq(smile,self.token_table,self.config)
                    tokens = Utils.tokenize(self.config,mol_padded,self.token_table)   
                    processed_mol = Utils.smiles2idx(tokens,self.tokenDict)
                    
                    sampled_gep = self.general_model_object_biased.model.predict([processed_mol,self.gep_mean])
                    rges = self.gep_calculator.rges_calculator(sampled_gep)
                    rges_values.append(rges)
                    
                    diffs.append(self.gep_mean[0,1988] - sampled_gep[0,1988])
                
                print("\n\nMax diffs: ", np.max(diffs))
                print("Mean diffs: ", np.mean(diffs))
                print("Min diffs: ", np.min(diffs))
                print("Std diffs: ", np.std(diffs))
        
                print("\n\nMax RGES: ", np.max(rges_values))
                print("Mean RGES: ", np.mean(rges_values))
                print("Min RGES: ", np.min(rges_values))
                print("Std RGES: ", np.std(rges_values))
            
            elif experiment_id == 'd':
                         
                run = '6'
                unique_smiles = np.unique(sanitized_valid_mols_b)
                
                diffs = []
                rges_values = []
                
                for idx,smile in enumerate(unique_smiles):
                    
                    mol_padded = Utils.pad_seq(smile,self.token_table,self.config)
                    tokens = Utils.tokenize(self.config,mol_padded,self.token_table)   
                    processed_mol = Utils.smiles2idx(tokens,self.tokenDict)
                    
                    sampled_gep = self.general_model_object_biased.model.predict([processed_mol,self.gep_mean])
                    rges = self.gep_calculator.rges_calculator(sampled_gep)
                    rges_values.append(rges)
                    
                    diffs.append(self.gep_mean[0,1988] - sampled_gep[0,1988])
                
                # mols_list = Utils.smiles2mol(sanitized_valid_mols)
                
                print("\n\nMax diffs: ", np.max(diffs))
                print("Mean diffs: ", np.mean(diffs))
                print("Min diffs: ", np.min(diffs))
                print("Std diffs: ", np.std(diffs))
        
                print("\n\nMax RGES: ", np.max(rges_values))
                print("Mean RGES: ", np.mean(rges_values))
                print("Min RGES: ", np.min(rges_values))
                print("Std RGES: ", np.std(rges_values))
                
                with open(self.config.path_generated_mols + '_run_' + run + '.smi', 'w') as f:
                    f.write("SMILES, RGES, USP7\n")
                    for i,smi in enumerate(unique_smiles):
                        
                        data = str(unique_smiles[i]) + " ," +  str(rges_values[i]) + " ," + str(diffs[i])
                        f.write("%s\n" % data)  