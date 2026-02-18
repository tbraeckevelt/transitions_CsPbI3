# Asitepaper
A repository for the paper about the phase transitions mechanisms in CsPbI3

## Repository Structure

- **StructureGeneration**: Contains all structure discussed in the paper (and some extra unstable structures) + the Python scripts to generate those structure.
- **MLPtraining**: Contains the files used to extract MD sampled structures from the TI workflow discussed in the `Free_energy_determination` folder, to subsequently recalculate them with VASP, and finally to train a MACE MLIP on the resulting VASP data set.
- **Free_energy_determination**: Contains the in- and outputfiles of the workflow to determine the Helmholtz free energy as a function of temperature for the structures given in `StructureGeneration` folder.

For detailed instructions, refer to the README file in each folder.
