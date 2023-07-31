#!/bin/bash
#
#SBATCH --job-name=$1
#
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-type='FAIL'
#SBATCH --mail-user='manon.dausort@uclouvain.be'

#SBATCH --output='/CECI/proj/pilab/PermeableAccess/alcooliques_As2Z4vF8GNv/TractAnalysis/slurm/slurmJob_%j.out'
#SBATCH --error='/CECI/proj/pilab/PermeableAccess/alcooliques_As2Z4vF8GNv/TractAnalysis/slurm/slurmJob_%j.err'
        
python $1 $2 $3 $4