	# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 17:05:42 2021

@author: tiago
"""
# internal
from model.generation import generation_process
from configs.configs import configs

# external
import warnings
warnings.filterwarnings('ignore')   


config_file = 'configs\configGeneration.json' # Configuration file 

def run():
    """This function initializes the models and performs the target generation 
    of novel molecules"""
    
    ## load configuration file
    cfg_file=configs.load_cfg(config_file)
     
    ## Implementation of the generation dynamics
    conditional_generation = generation_process(cfg_file)
    conditional_generation.load_models()
    
    ## Validation experiments
    # conditional_generation.model_validation()
    conditional_generation.experiments('a')
    # conditional_generation.policy_gradient()
    # conditional_generation.model_evaluation()
    # gep_model.evaluate()
    # gep_model.save()
    
if __name__ == '__main__':
    run()
