from ase.io import read, write
from ase import Atom, Atoms
import numpy as np
from ase.build import niggli_reduce
from ase.build.supercells import make_supercell
import sys
sys.path.insert(0, '/home/tom/Documents/Doctoraat/StructureFiles/MakeCsPbI3IntermediateStruc')
from buildingblocks import CsPbI3
from scipy.spatial.transform import Rotation as R


def add_extra_shifts(j, k, double = False):
    extra_shifts[(0, j, k)] = [ 2, 1, 3]
    extra_shifts[(0, j, (k+1)%depth)] = [ 0, 1, 1]
    if double:
        extra_shifts[(0, (j-1)%height, (k+2)%depth)] = [ (-1)**k, 0, 2]


#Create the CsPbI3 perovskite structure
cspbi3 = CsPbI3()

length = 1
height = 4 #height of the RP phase
depth = 8
a_vec = np.array([2*cspbi3.l*np.sqrt(2), 0.0, 0.0])
b_vec =  np.array([0.0, -2*cspbi3.l, 2*np.sqrt(2)*cspbi3.l])
c_vec = np.array([0.0, 0.0, np.sqrt(2)*cspbi3.l])


extra_shifts = {}
double = True
add_extra_shifts(0, 0, double = double)
add_extra_shifts(0, 4, double = double)
add_extra_shifts(1, 0, double = double)
#add_extra_shifts(1, 4, double = double)
#add_extra_shifts(2, 0, double = double)
#add_extra_shifts(2, 4, double = double)
#add_extra_shifts(3, 0, double = double)
#add_extra_shifts(3, 4, double = double)


for k in range(depth):
    for j in range(height):
        for i in range(length):
            trans = i*a_vec + j*b_vec + k*c_vec
            if (i, j, k) in extra_shifts:
                extra_shift = extra_shifts[(i, j, k)]*np.array([cspbi3.l/np.sqrt(2), cspbi3.l, cspbi3.l/np.sqrt(2)])
                print(extra_shift)
            else:
                extra_shift = np.array([(-1)**k*cspbi3.l/np.sqrt(2), 0.0, 0.0])
            new_oct = cspbi3.get_edge_octahedra()
            for at in new_oct:
                at.position += trans+extra_shift
                cspbi3.atoms.append(at)
    print(k)
cspbi3.add_cell(np.array([a_vec*length, b_vec*height, c_vec*depth]))
cspbi3.print_charge()

cspbi3.atoms.cell[1,2] = 0.0
cspbi3.atoms = make_supercell(cspbi3.atoms, np.diag([2, 1, 1]))
#niggli_reduce(sup_atoms)
outname = "teststruc.cif"
write(outname, cspbi3.atoms, format="cif")
write('int_3a.xyz', cspbi3.atoms, format="extxyz")
