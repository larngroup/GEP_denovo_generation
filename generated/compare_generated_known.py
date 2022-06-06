# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 09:28:28 2021

@author: tiago
"""
# external
import csv
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS

def load_generated_mols(filepath):
    """ Loads the generated mols
    Args
    ----------
        filepath (str): File path containing the generated molecules
    Returns
    -------
        mols_all (list): List of molecules
    """
    if '.smi' in filepath:
        idx_smiles = 0
        
        raw_smiles = []
        
        with open(filepath, 'r') as csvFile:
            reader = csv.reader(csvFile)
            
            it = iter(reader)
            next(it, None)  # skip first item.    
            for row in it:
                try:
                    raw_smiles.append(row[idx_smiles])
                except:
                    pass
        
       
        return list(set(raw_smiles))
    elif '.pkl' in filepath:
        df = pd.read_pickle(filepath)
        
        mols_all = list(df['molecules'])
        mols_d = list(df.loc[df["dominated"] == True, "molecules"])
        mols_nd = list(df.loc[df["dominated"] == False, "molecules"])
        return mols_all
        

def load_known_mols():
    """ 
    Loads the known anti-cancer molecules
    
    Returns
    -------
        raw_smiles (list): List of anti-cancer molecules SMILES
        name_mols (list): List of anti-cancer molecules names
    """
    
    filepaths = ["cancer_drugs.csv","usp7_chembl.csv"]
    # filepaths = ["usp7_chembl.csv"]
    raw_smiles = []
    name_mols = []
    for fp in filepaths:
        
        with open(fp, 'r') as csvFile:
            reader = csv.reader(csvFile)
            
            it = iter(reader)
            next(it, None)  # skip first item.    
            for row in it:
            
                row_splited = row[0].split(';')
                
                if fp == 'usp7_chembl.csv' :
                    idx_smiles = 7
                    idx_pic50 = 5#12
                    idx_name = 0
                else:
                    idx_smiles = 30
                    idx_name = 1
                try:
                    
                    sm = row_splited[idx_smiles][1:-1]
                    # print(sm,row_splited[idx_pic50][1:-1])
                    mol = Chem.MolFromSmiles(sm, sanitize=True)
                    s = Chem.MolToSmiles(mol)
                    name_mol = row_splited[idx_name]
                    if s not in raw_smiles and len(s)>10:
                        
                        if fp == 'cancer_drugs.csv':
                            raw_smiles.append(s)
                            name_mols.append(name_mol)
                        elif fp == 'usp7_chembl.csv' and float(row_splited[idx_pic50][1:-1]) > 4.5:
                            raw_smiles.append(s)
                            name_mols.append(name_mol)
         
                except:
                    print(sm)

    return raw_smiles,name_mols
            
       
def find_similarities(generated,leads,similarity_measure,drug_names):
    """ Searches for similarities between the generated molecules and known 
        drugs using the Tanimoto distance.
    Args
    ----------
        generated (list): Generated molecules
        leads (list): Anti-cancer drugs
        similarity_measure (str): Type of similarity metric (Tanimoto distance
                                    or Tanimoto maximum common substructure)
        drug_names (list): Names of the anti-cancer drugs
    Returns
    -------
        similarities_sorted (dataframe): It contains each generated molecule,
        the most similar drug and the value of the distance.
    """
    similarities = pd.DataFrame()
    similarity = []
    most_similar_mol = []
    name_most_similar_mol = []
    for m in generated:    	
        
        if similarity_measure == 'Tanimoto_s':
            try:
                mol = Chem.MolFromSmiles(m, sanitize=True)
            	
                fp_m = AllChem.GetMorganFingerprint(mol, 3)   
    
                [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(n, sanitize=True), 3) for n in leads]
    
                dists = [DataStructs.TanimotoSimilarity(fp_m, AllChem.GetMorganFingerprint(Chem.MolFromSmiles(n, sanitize=True), 3)) for n in leads]    
                
                most_similar_mol.append(leads[dists.index(max(dists))])
                name_most_similar_mol.append(drug_names[dists.index(max(dists))])
                similarity.append(max(dists))
                
                
            except:
                similarity = ['nan']
                most_similar_mol = ['nan']
                print('Invalid: ' + m)
                
        elif similarity_measure == 'Tanimoto_mcs':
            
         
            ref_mol = Chem.MolFromSmiles(m, sanitize=True)
            numAtomsRefCpd = float(ref_mol.GetNumAtoms())
            dists= []
            for l in leads:
                
                try: 
                    target_mol = Chem.MolFromSmiles(l, sanitize=True)
                    numAtomsTargetCpd = float(target_mol.GetNumAtoms())
    
                    # if numAtomsRefCpd < numAtomsTargetCpd:
                    #     leastNumAtms = int(numAtomsRefCpd)
                    # else:
                    #     leastNumAtms = int(numAtomsTargetCpd)
            
                    pair_of_molecules = [ref_mol, target_mol]
                    numCommonAtoms = rdFMCS.FindMCS(pair_of_molecules, 
                                                    atomCompare=rdFMCS.AtomCompare.CompareElements,
                                                    bondCompare=rdFMCS.BondCompare.CompareOrderExact, matchValences=True).numAtoms
                    dists.append(numCommonAtoms/((numAtomsTargetCpd+numAtomsRefCpd)-numCommonAtoms))
                except:
                    dists.append(-1)
                    print('Invalid: ' + l)
            
            # try:        
            most_similar_mol.append(leads[dists.index(max(dists))])
            name_most_similar_mol.append(drug_names[dists.index(max(dists))])
            similarity.append(max(dists))
            [dists.index(max(dists))]
            # except:
            #     print()
    
        # else:
        #     dists = [pylev.levenshtein(m, l) for l in leads]
        #     most_similar_mol.append(leads[dists.index(min(dists))])
        #     similarity.append(min(dists))
            
            

    similarities['generated'] = generated
    similarities['most similar drug'] = most_similar_mol
    similarities['Tanimoto similarity'] = similarity
    similarities['Name drug'] = name_most_similar_mol
    
    similarities_sorted = similarities.sort_values(by = 'Tanimoto similarity',ascending = False)
    
    return similarities_sorted
    
def draw_mols(smiles_list,title, names=False):
    """ Draws the molecules given as argument and saves it as an image
    Args
    ----------
        smiles_list (list): list of molecules to draw
        title (str): name of the file to save the draw
   
    """
    
    DrawingOptions.atomLabelFontSize = 50
    DrawingOptions.dotsPerAngstrom = 100
    DrawingOptions.bondLineWidth = 3
    
    mols_list = [ Chem.MolFromSmiles(m, sanitize=True) for m in  smiles_list]
    
    if names!= False:
        legend_mols = []
        for i in range(0,len(mols_list)):
            legend_mols.append('Id: '+ names[i])
    
        img = Draw.MolsToGridImage(mols_list, molsPerRow=4, legends=legend_mols, subImgSize=(300,300))
            
    else:
        img = Draw.MolsToGridImage(mols_list, molsPerRow=4, subImgSize=(300,300))
        
    img.show()
    img.save('mols_' + title + '.png')

if __name__ == '__main__':
    
    # Select the input file and the similarity measure 
    file_generated_mols = "pareto_generation_mols_noise.pkl" #generated_lead_comparison.smi or pareto_generation_mols.pkl 
    similarity_measure = 'Tanimoto_mcs' # Tanimoto_s, Tanimoto_mcs, Levenshtein
    
    # Load the set of molecules that interact with USP7 
    known_drugs,name_drugs = load_known_mols()
    
    # Load the set of generated molecules
    generated_mols = load_generated_mols(file_generated_mols)
    
    # Compute the similarity 
    similarities = find_similarities(generated_mols,known_drugs,similarity_measure,name_drugs)
    
    # Select a subset of molecules
    # similarities_filtered = similarities_sorted[similarities_sorted["Tanimoto similarity"] > 0.43]
    similarities_filtered = similarities.head(24)
    
    draw_mols(similarities_filtered['generated'],'generated')
    draw_mols(similarities_filtered['most similar drug'],'known drugs',list(similarities_filtered['Name drug']))
    
    
    
