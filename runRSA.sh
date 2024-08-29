#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH --partition=batch,cb3
#SBATCH --job-name=rsa-sl
#SBATCH --error=/work/cb3/ahumphries/RSA-SL/errors/job.rsa-sl.%J.err
#SBATCH --output=/work/cb3/ahumphries/RSA-SL/output/job.rsa-sl.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahumphries2@unl.edu
#SBATCH --licenses=common


cd $WORK/RSA-SL

# Load Conda module and activate environment
module load anaconda
conda activate /common/cb3/ahumphries/rsaenv


# Run the main analysis script
python rsa-sl.py