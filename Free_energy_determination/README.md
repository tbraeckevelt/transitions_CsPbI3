
This folder contains the `thermoflow` directory, which is a collection of Python scripts that use **psiflow** to perform the TI workflow discussed in our A-site paper.  
Note that this workflow is implemented only for CsPbI3, and not for the organic metal halide perovskites (MHPs), since the bias on the rotational degrees of freedom of the organic molecules was not implemented.

The workflow is executed using the MACE model trained in the `MLPtraining` folder (see the archived file `modelvasp.tar.gz`). It is applied to all structures provided in the `StructureGeneration` folder.

The output files of these simulations are provided in the corresponding subfolders within this directory. The trajectory files, optimized structures, and Hessians are not included. In addition, the post-analysis outputfiles with the Helmholtz free energies from the simulations are included here.

Note that the folder structure in this directory does not directly mirror the structure of the `StructureGeneration` folder. The file `explanation_path_names.txt` specifies which generated structures correspond to which subfolders.

The free energy and interface energy plots shown in the manuscript and Supporting Information were generated using the post-analysis script `Compare_free_energies.py`.
