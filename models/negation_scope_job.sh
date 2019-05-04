#!/bin/bash
#
#SBATCH --job-name=sentiment_neg_MTL --account=nn9447k
#SBATCH --time=24:00:00
#
#
# Max memory usage:
#SBATCH --mem-per-cpu=15G
#
# Number of threads per task:
#SBATCH --cpus-per-task=1

## Set up job environtment:
source /cluster/bin/jobsetup
module purge   # clear any inherited modules
set -o errexit # exit on errors

# Load models
module use -a /proj*/nlpl/software/modulefiles/
module load nlpl-pytorch/1.0.0
module load nlpl-nltk

# Set up input/output files
cd ../..
cp -r inductive_biases/models $SCRATCH
cp -r inductive_biases/data $SCRATCH
cp -r embeddings $SCRATCH

# make sure the results are copied back to submit directory
chkfile models/saved_models/*
chkfile embeddings

# Run command
cd $SCRATCH
cd models

# Baseline classifier with no multi-tasking
python3 hard_parameter_bilstm_crf.py -aux none -emb embeddings/google.txt

# Multi-task classifier
python3 hard_parameter_bilstm_crf.py -aux negation_scope -emb embeddings/google.txt

