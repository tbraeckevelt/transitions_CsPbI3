This folder contains the files used to extract MD sampled structures from the TI workflow discussed in the `Free_energy_determination` folder, to subsequently recalculate them with VASP, and finally to train a MACE MLIP on the resulting VASP data set.

## Explanation workflow to generate MLIP

1. First of all the TI workflow dicussed in the `Free_energy_determination` folder is performed for all structures of the `StructureGeneration` folder using the universal MLIP: MACE-MP-0. Not the whole workflow was performed, only the initialization, sampling, and nvt or npt REX steps to generate structures.

2. From these MD simulations we extract structures via the `GenerateInputFilesVASP.py` Python scripts, which also specifies how many structures are taken from each MD simulation. Moreover it also creates the folder to perform the VASP calculation of each structure.

3. `CollectData.py` investigates if the VASP calculation is succesful and if so, adds the structure with the VASP energy and forces to the vaspdata.xyz (we tarred this file to reduce the size) data set.

4. `only_train.py` performs the MLIP training using psiflow, the computational resources needed for this training is reported in `config_psiflow.yaml` and `log_training.txt` is the output log with the final RMSEs of the model. 

After the training the hamiltonian is present in the context dir and can be loaded via:
`hamiltonian = load_hamiltonian(root / 'psiflow_internal_training/context_dir/model_000001.pth')`

The model that we used to determine the free energies is present in the `Free_energy_determination` folder (`modelvasp.tar.gz`)





This folder contains the files used to extract MD sampled structures from the TI workflow discussed in the `Free_energy_determination` folder, to subsequently recalculate them with VASP, and finally to train a MACE MLIP on the resulting VASP data set.

## Explanation workflow to generate MLIP

1. First of all the TI workflow dicussed in the `Free_energy_determination` folder is performed for all structures of the `StructureGeneration` folder using the universal MLIP: MACE-MP-0. Not the whole workflow was performed, only the initialization, sampling, and nvt or npt REX steps to generate structures.

2. From these MD simulations we extract structures via the `GenerateInputFilesVASP.py` Python scripts, which also specifies how many structures are taken from each MD simulation. Moreover it also creates the folder to perform the VASP calculation of each structure.

3. `CollectData.py` investigates if the VASP calculation is succesful and if so, adds the structure with the VASP energy and forces to the `vaspdata.xyz` (we tarred this file to reduce the size) data set.

4. `only_train.py` performs the MLIP training using psiflow, the computational resources needed for this training is reported in `config_psiflow.yaml` and `log_training.txt` is the output log with the final RMSEs of the model. 

After the training the hamiltonian is present in the context dir and can be loaded via:
`hamiltonian = load_hamiltonian(root / 'psiflow_internal_training/context_dir/model_000001.pth')`

The model that we used to determine the free energies is present in the `Free_energy_determination` folder (`modelvasp.tar.gz`)