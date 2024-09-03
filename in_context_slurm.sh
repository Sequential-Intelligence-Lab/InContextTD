#!/bin/bash
#SBATCH --job-name="in_context td non-linear"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehb2bf@virginia.edu
#SBATCH --exclude=adriatic[01-04],cheetah01,cheetah02,cheetah03,cheetah04,jaguar01,jaguar02,jaguar04,jaguar05,jaguar06,lotus,lynx[01-02]
#SBATCH --partition=main
#SBATCH --error="incontext_nonlinear.err"
#SBATCH --output="incontext_nonlinear.output"
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=2
cd /u/ehb2bf/InContextTD
module load anaconda3
conda activate incontextenv
srun python main.py --suffix=standard -v # standard
#srun python main.py --suffix=representable --sample_weight -v & # representable value function
#srun python main.py --suffix=sequential --mode=sequential -v & # standard sequential
#srun python main.py --suffix=large -d=8 -s=20 -n=60 -v & # large scale
#srun python main.py --suffix=relu --activation=relu -v & # relu activation
#wait # wait for all background jobs to finish
