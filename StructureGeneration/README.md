
This folder contains all structures discussed in the paper (as well as several additional unstable structures), together with the Python scripts used to generate them.  
The folder names correspond to the relevant (sub)sections in the manuscript or Supporting Information (SI).

## Folder Structure

- **layer-by-layer-transformation**  
  Structures discussed in Sec. 4 of the manuscript  
  - **gamma-to-zeta**  
    Structures discussed in Sec. 4.1 of the manuscript  
  - **gamma-to-deltaFA**  
    Structures discussed in Sec. 4.2 of the manuscript  
  - **gamma-to-deltaCs**  
    Structures discussed in Sec. 4.3 of the manuscript  

- **row-by-row-nucleation**  
  Structures discussed in Sec. 4 of the SI  
  - **zeta-in-gamma**  
    Structures discussed in Sec. 4.1 of the SI  
  - **deltaFA-to-gamma**  
    Structures discussed in Sec. 4.2 of the SI  
  - **deltaCs-to-gamma**  
    Structures discussed in Sec. 4.3 of the SI  

- **row-by-row-transformation**  
  Structures discussed in Sec. 5 of the SI  
  - **zeta-to-deltaFA**  
    Structures discussed in Sec. 5.1 of the SI  
  - **zeta-to-deltaCs**  
    Structures discussed in Sec. 5.2 of the SI  


Note that some subfolders contain a **notreported** directory. 
This directory includes structures that are not discussed in the manuscript or SI because they were found to be unstable. 
Nevertheless, a subset of these structures was included in the training dataset used to generate the MLIP. 
Interested readers may also inspect these structures to identify transformation pathways that were deemed unpromising and therefore not pursued further.
One exception is the `int_line_Csopt1.xyz` and `int_line_Csopt2.xyz` structures in the `row-by-row-nucleation/zeta-in-gamma/not-reported/0-dim` folder, which is shown in figure 9 of the manuscript. However, as mentioned these structures are unstable.

## Files
Each subfolder contains, in addition to the `.xyz` files, a Python script to generate the structures.  
All scripts import `buildingblocks.py`, which defines the `CsPbI3` class: a utility class providing convenient functions to construct the structures.  

As multiple construction approaches are possible, some subfolders also include an `Oldscripts` directory containing alternative Python scripts for generating the same structures.  
These scripts may serve as inspiration for readers who wish to construct their own structures.

## General principle of the Python scripts used to generate structures

The `buildingblocks.py` script defines the `CsPbI3` class, which initializes an empty ASE `Atoms` object and sets the Pb–I bond length of the octahedra (denoted as `l`). For all structures, `l = 3.2`.  
This `Atoms` object can subsequently be extended with `Cs8PbI6` building blocks and an appropriate simulation cell. Depending on the target structure, the building blocks may need to be rotated.

Since these building blocks do not satisfy the CsPbI3 stoichiometry, excess Cs and I atoms are removed at a later stage, typically where overlaps occur between neighboring building blocks.  
A `print_charge()` class method is provided to verify the validity of the generated structure.

The structure-specific Python scripts generally define parameters controlling the length, width, and depth of the simulation cell, as well as two lattice vectors (`b` and `c`) that indicate the directions in which neighboring octahedra are added. In addition, an array (depending on `b` and/or `c`) specifies rotations or shifts of neighboring octahedra along the `a` direction. These parameters can be modified to generate alternative structures. However, not all parameter combinations lead to valid structures. While testing and modifying the existing scripts can be instructive, constructing new structures from scratch (using these scripts as inspiration) is often more robust.

The most challenging aspect is ensuring that octahedra in different `(b,c)` rows connect properly without overlapping, while maintaining the correct CsPbI3 stoichiometry. The placement of Cs atoms can also be nontrivial. For rotated octahedra, a practical strategy is to remove the Cs atoms from the building blocks and reinsert them at carefully chosen positions. Suitable positions were typically identified through trial and error.

