# PURPOSE -----------------------------------------------------------------

#' To conduct a differential expression analysis that will be a sample
#' for conducting the analysis using TCGA data
#' The TCGA RNASeq data is illumina hiseq Level 3 RSEM normalized expression data
start_time <- Sys.time()
cat(paste0("start: ",start_time,"\n"))

#' EXAMPLE RUN
#' -----------
#' Rscript TCGA_differential_expression_analysis.R 100 i ii
#'
#'
# install packages/load libraries -----------------------------------------

if (!requireNamespace("BiocManager", quietly = TRUE)){
  suppressWarnings(suppressMessages(install.packages("BiocManager")))}

if(require(DESeq2)){
  suppressWarnings(suppressMessages(library(DESeq2)))
}else{
  suppressWarnings(suppressMessages(BiocManager::install("DESeq2",update=F)))
  suppressWarnings(suppressMessages(library(DESeq2)))
}
if(require(tidyverse)){
  suppressWarnings(suppressMessages(library(tidyverse)))
}else{
  install.packages("tidyverse")
  suppressWarnings(suppressMessages(library(DESeq2)))
}
if(require(BiocParallel)){
  suppressWarnings(suppressMessages(library(BiocParallel)))
}else{
  BiocManager::install("BiocParallel",update=F)
  suppressWarnings(suppressMessages(library(BiocParallel)))
}
register(MulticoreParam(4))

# set variables -----------------------------------------------------------

set.seed(42)

args = commandArgs(trailingOnly=TRUE)

data_dir <- ""
rnaseq_file <- "gene_expression_data.csv.gz"
clinical_file <- "y.csv"
#gene_size <- args[1]
gene_size <- 1e2

cancer_type <- 'lihc'

outfile <- paste0(cancer_type,"_DESeq2_")

response_name <- "samples_type"
#t <- args[2]
t <- "Tumour"
#c <- args[3]
c <- "Normal"

# load data ---------------------------------------------------------------

#clinical_data <- read_csv(paste0(data_dir,clinical_file)) %>%
 # as_tibble() %>%
#  rename(
#    "bcr_patient_barcode" = "X1"
#  )


rnaseq_data <-
  paste0(data_dir,rnaseq_file) %>%
  gzfile() %>%
  read.table(sep = ",",header = T) %>%
  as_tibble()  

# clean rnaseq data for input ----------------------------------------------------

#rnaseq_data_clean <- inner_join(rnaseq_data,clinical_data)

# DE analysis -------------------------------------------------------------

rnaseq_data$bcr_patient_barcode <-
  unlist(
    lapply(
      strsplit(
        str_to_lower(rnaseq_data$bcr_patient_barcode),"-"),
      function(x)paste0(x[1:3],collapse="-")
    )
  )

colDat <- rnaseq_data %>% select(bcr_patient_barcode, response_name)


rows.to.keep <-
  rnaseq_data$bcr_patient_barcode %in% colDat$bcr_patient_barcode

rsem.in <- data.frame(
  t(
    rnaseq_data[rows.to.keep,colnames(rnaseq_data)!="bcr_patient_barcode"]
  )
)

if(!is.na(as.integer(gene_size))){
  rsem.in.sub <- rsem.in[sample(1:nrow(rsem.in),gene_size),]
  rows <- rownames(rsem.in.sub)
  rsem.in.sub <- apply(rsem.in.sub,2,as.integer)
  rownames(rsem.in.sub) <- rows
}
if(gene_size=="full"){
  rsem.in.sub <- rsem.in
  rows <- rownames(rsem.in.sub)
  rsem.in.sub <- apply(rsem.in.sub,2,as.integer)
  rownames(rsem.in.sub) <- rows
  rsem.in.sub <- na.omit(rsem.in.sub)
}

formula <- as.formula(paste0("~",response_name))

suppressMessages(dds <-
                   DESeqDataSetFromMatrix(rsem.in.sub,
                                          colDat,formula))


res <- DESeq(dds)
results <- results(res,contrast=c(response_name,t,c))
results_df <- data.frame(results)
rownames(results_df) <- 1:nrow(results_df)
results_df$Gene <- row.names(dds)


outfile2 <- paste0(outfile,gene_size,
                   "_sampled_genes_",response_name,"_",
                   t,"_vs_",c,".csv")

results_df %>%
  write_csv(paste0(data_dir,outfile2))

end_time <- Sys.time()
cat(paste0("end: ",end_time,"\n"))
cat(
  paste0("duration: ",
         round(as.numeric(end_time-start_time,units="mins"),2),
         " minutes\n")
)

