#!/bin/bash
#SBATCH -p debug
#SBATCH --account=project_465001125
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -e log_analysis.txt
#SBATCH -o log_analysis.txt


# use a custom Psiflow hack
PATH="/pfs/lustrep2/scratch/project_465001125/water_phase_diagram/psiflow_hacking/psiflow_env/bin:$PATH"

pwd
date

#python generate_workflow.py
python -u analysis.py

date
echo "Done"
