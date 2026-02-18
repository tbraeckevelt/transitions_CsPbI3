#!/bin/bash
#SBATCH -p standard
#SBATCH --account=project_465001125
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH -e log.txt
#SBATCH -o log.txt


# use a custom Psiflow hack
PATH="/pfs/lustrep2/scratch/project_465001125/water_phase_diagram/psiflow_hacking/psiflow_env/bin:$PATH"

pwd
date

#python generate_workflow.py
python -u execute_workflow.py config_psiflow.yaml

date
echo "Done"
