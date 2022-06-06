# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:12:46 2021

@author: tiago
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import pickle


task = 'disease_gep' #  de_genes or disease_gep
remove_outliers = True

# load the landmark genes
file = open('new_landmark_genes.pkl', 'rb')
landmark_genes = pickle.load(file)

if task == 'de_genes':
    
    filename = "BRCA.uncv2.mRNAseq_raw_counts.txt" 
    # filename = "BRCA.mRNAseq_raw_counts.txt"
    df = pd.read_csv(filename, delimiter = "\t")
    df = df.set_index('HYBRIDIZATION R')
    
    
    # Assign the class to each sample (tumour vs normal tissue)
    samples_type = []
    for i in range(0,len(df.columns)):
         sample = df.columns[i]
         sample_type = sample[13:15]
        
         try:
            sample_int = int(sample_type)
            
            if sample_int < 10:
                samples_type.append('Tumour')
            elif sample_int >= 10 and sample_int < 20:
                samples_type.append('Normal')
            else:
                print('Wrong: ', sample_type)
            
         except:
            print('ERROR: ', sample_type)
    
    
    # Transpose the dataframe to get the format (samples, genes)
    rna_seq = df.T
    
    
    # Transform the columns to have the genes denominations
    gene_name_logical = [len(x[0])>1 for x in rna_seq.columns.str.split('|')]
    sub = rna_seq.loc[:,gene_name_logical]
    sub.columns = [x[0] for x in sub.columns.str.split('|')]
    rnaseq_sub = sub.copy()
    
    
    # Select the subset of landmark genes (according to https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.9b00520)
    tt = 0 
    correct_genes = []
    for c in landmark_genes:
        if c in rnaseq_sub.columns:
            correct_genes.append(c)
            
    rnaseq_sub = rnaseq_sub[correct_genes]
    
    # Convert the expression values into floats
    for j in range(0,len(rnaseq_sub.columns)):
        rnaseq_sub[rnaseq_sub.columns[j]] = rnaseq_sub[rnaseq_sub.columns[j]].astype(float)
    
    # Filter the genes that were not expressed
    exprs_genes = rnaseq_sub[rnaseq_sub>0].dropna(axis=1)
    
    # Plot of the coefficient of variation as the genes are more expressed 
    mean_vec = exprs_genes.mean()
    sd_vec = exprs_genes.std()
    cv_vec = sd_vec / mean_vec
    fig = plt.figure(figsize=(10,8))
    plt.figure(dpi=200)
    plt.scatter(np.log10(mean_vec), np.log10(cv_vec),edgecolor='black',linewidth=.3)
    plt.title('Gene-wise means versus variance')
    plt.xlabel('Mean Gene Expression (log10)')
    plt.ylabel('Coefficient of Variation\nlog10(Std / Mean) Gene expression')
        
    # Plot the distribution of the mean gene expression for each gene
    plt.figure(figsize=(14,6))
    plt.hist(exprs_genes.mean(axis=0), range=(0, 8000), bins=100)
    plt.title('Distribution of Mean Gene Expression')
    plt.xlabel('Mean Gene expression')
    plt.ylabel('No. Genes');
    
    # Adding the information about the type of sample (tumour or normal)
    rnaseq_sub['samples_type'] = samples_type
    rnaseq_sub.index.name = 'bcr_patient_barcode'
    
    # Save the dataframe as csv to be used in R 
    rnaseq_sub.to_csv('gene_expression_data.csv')
    
    # Load results of R processing
    de_tumor_normal = pd.read_csv('brca_DESeq2_100_sampled_genes_samples_type_Tumour_vs_Normal.csv')
    de_tumor_normal = de_tumor_normal.dropna()
    
    # Volcano plot to identify the up and down regulated genes
    # visuz.gene_exp.volcano(df=de_tumor_normal, lfc='log2FoldChange', pv='pvalue',plotlegend=True, legendpos='upper right', 
    #     legendanchor=(1.46,1),valpha=0.5, geneid="Gene", 
    #     genenames=("MAGEA1","ABCA8","RIC3"),
    #     gstyle=2,sign_line=True)
    
    # Transform the results into floats
    for j in range(0,len(de_tumor_normal.columns)):
        if j < len(de_tumor_normal.columns)-1:
            # print(de_tumor_normal.columns[j])
            de_tumor_normal[de_tumor_normal.columns[j]] = de_tumor_normal[de_tumor_normal.columns[j]].astype(float)
    
    # Filter the genes to select the differentially expressed ones that belong to 
    # the set of landmark genes
    tumour_normal_filtered =  de_tumor_normal[(abs(de_tumor_normal['log2FoldChange']) > 2) & (de_tumor_normal['padj'] <0.01) ]
    tumour_normal_filtered = tumour_normal_filtered[tumour_normal_filtered['Gene'].isin(landmark_genes)]
    
    # Assign the class "up" or "down" to each gene according to the direction of 
    # log2FoldChange parameter
    tumour_normal_filtered['up_down'] = 'down'                                                                        
    tumour_normal_filtered.loc[tumour_normal_filtered.log2FoldChange > 0, "up_down"] = "up"
    
    # Save the dataframe as csv to be used in the generation dynamics
    tumour_normal_filtered.to_csv('deseq_processed.csv')
    
    # Plot log2-fold changes between normal and tumour over the mean of normalized counts
    plt.figure(figsize=(16,8))
    plt.scatter(de_tumor_normal['baseMean'], np.log10(de_tumor_normal['log2FoldChange']), c='purple')
    plt.xlim(0,100000);
    plt.xlabel('Mean Gene Expression')
    plt.ylabel('Fold Change (Log 2)')
    plt.title('MA Plot')
    plt.plot(np.linspace(0, 100000, 100), np.linspace(0, 0, 100), c='red')
    plt.plot(np.linspace(0, 100000, 100), np.linspace(2, 2, 100), c='red', ls='-.')
    plt.plot(np.linspace(0, 100000, 100), np.linspace(-2, -2, 100), c='red', ls='-.')
    
elif task == 'disease_gep':
    filename = "BRCA.uncv2.mRNAseq_RSEM_Z_Score.txt" 
    df = pd.read_csv(filename, delimiter = "\t")
    
    resume = df.head()
    
    infile = open("new_selected_genes.pkl",'rb')
    genes = pickle.load(infile)
    infile.close()
    # new_genes = genes
    # new_genes.loc[1988] = ['USP7', 0]
    # new_genes.to_pickle('new_selected_genes.pkl')

    #Select the landmark genes that are in the TCGA dataset
    landmark_genes_filtered = [gene for gene in landmark_genes if gene in genes['genes'].tolist()]
    
    # Assign the class to each sample (tumour vs normal tissue)
    samples_type = []
    for i in range(1,len(df.columns)):
         sample = df.columns[i]
         sample_type = sample[13:15]
    
         try:
            sample_int = int(sample_type)
            
            if sample_int < 10:
                samples_type.append('tumour')
            elif sample_int >= 10 and sample_int < 20:
                samples_type.append('normal')
            else:
                print('Wrong: ', sample_type)
            
         except:
            print('ERROR: ', sample_type)


    # Transpose the dataframe to get the format (samples, genes)
    df = df.T
    
    # Remove unnecessary information from the dataframe
    header_row = 0 
    df.columns = df.iloc[header_row]
    df = df.drop('gene')
    
    # Transform the columns to have the genes denominations
    gene_name_logical = [len(x[0])>1 for x in df.columns.str.split('|')]
    sub = df.loc[:,gene_name_logical]
    sub.columns = [x[0] for x in sub.columns.str.split('|')]
    rnaseq_sub = sub.copy()
    
    # Select the subset of landmark genes (according to https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.9b00520)
    tt = 0 
    correct_genes = []
    for c in landmark_genes_filtered:
        if c in rnaseq_sub.columns:
            correct_genes.append(c)
            
            
    rnaseq_sub = rnaseq_sub[correct_genes]
    
    # Convert the expression values into floats
    for j in range(0,len(rnaseq_sub.columns)):
        rnaseq_sub[rnaseq_sub.columns[j]] = rnaseq_sub[rnaseq_sub.columns[j]].astype(float)
        
    
    # Adding the information about the type of sample (tumour or normal)
    rnaseq_sub['samples_type'] = samples_type
    rnaseq_sub.index.name = 'bcr_patient_barcode'
    
    # Obter df só com os exemplos tumorais
    tumour_rnaseq =  rnaseq_sub[rnaseq_sub['samples_type'] == 'tumour']
    
    if remove_outliers:
        removed_outliers = tumour_rnaseq[(tumour_rnaseq > tumour_rnaseq.quantile(.05)) & (tumour_rnaseq < tumour_rnaseq.quantile(.95))]
    
        mean_gep = removed_outliers.mean(axis=0)
    else:
    # Fazer a média de todos para cada gene
        mean_gep = tumour_rnaseq.mean(axis=0)

    mean_gep = mean_gep.fillna(0)
    mean_gep =  mean_gep[0:-1]
    
    # Save the dataframe as csv to be used in R 
    mean_gep.to_csv('disease_gep.csv')
