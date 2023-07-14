#!/bin/bash
#
#SBATCH --job-name=$1
#
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-type='FAIL'
#SBATCH --mail-user='nicolas.delinte@uclouvain.be'

#SBATCH --output='/home/users/n/d/ndelinte/Python/slurmJob.out'
#SBATCH --error='/home/users/n/d/ndelinte/Python/slurmJob.err'
        
python $1 $2 $3