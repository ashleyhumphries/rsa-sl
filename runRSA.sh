#!/bin/tcsh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1600
#SBATCH --partition=batch,cb3
#SBATCH --job-name=rsa-sl
#SBATCH --error=/work/cb3/ahumphries/RSA-SL/errors/job.rsa-sl.%J.err
#SBATCH --output=/work/cb3/ahumphries/RSA-SL/output/job.rsa-sl.%J.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahumphries2@unl.edu

module load python/3.8
cd $WORK/RSA-SL


# Run the main analysis script
python rsa-sl.py