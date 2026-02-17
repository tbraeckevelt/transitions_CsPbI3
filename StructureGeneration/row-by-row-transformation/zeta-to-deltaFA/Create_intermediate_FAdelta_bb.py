from ase.io import read, write
from ase import Atom, Atoms
import numpy as np
from ase.build import niggli_reduce
from ase.build.supercells import make_supercell
import sys
sys.path.insert(0, '/home/tom/Documents/Doctoraat/StructureFiles/MakeCsPbI3IntermediateStruc')
from buildingblocks import CsPbI3

#Create the CsPbI3 perovskite structure
cspbi3 = CsPbI3()

height = 1 #height of the RP phase
depth = 2
b_vec =  np.array([np.sqrt(2)*cspbi3.l, 2*cspbi3.l, 0.0])
c_vec = np.array([0.0, 0.0, 2*np.sqrt(2)*cspbi3.l])

shifts ={
    'default' : [1]*8,
}

faces ={
    'default' : [1]*1 + [1]*6 + [1]*1,
    (0, 1)    : [1]*1 + [-1]*6 + [1]*1,
}

rot = 1
check_pbc = None
for k in range(depth):
    for j in range(height):
        trans = j*b_vec + k*c_vec
        for z_shift, face in zip(shifts.get((j, k), shifts['default']), faces.get((j, k), faces['default'])):
            trans, rot = cspbi3.add_octahedra(trans = trans, z_shift = z_shift, rot = rot*face, Cs_scheme = face)
            check_trans = trans-j*b_vec - k*c_vec
            if np.abs(check_trans[1] + 4*cspbi3.l) < 0.01:
                trans += 2*b_vec #-2*np.array([0.0, 0.0, np.sqrt(2)*cspbi3.l])
        to_check_pbc = trans-j*b_vec - k*c_vec
        #if to_check_pbc[1] != 0.0 and np.abs(to_check_pbc[1]/(2*cspbi3.l) - int(to_check_pbc[1]/(2*cspbi3.l))) < 0.01:
        #    to_check_pbc -= int(to_check_pbc[1]/(2*cspbi3.l))*b_vec +c_vec
        if check_pbc is not None:
            assert np.allclose(to_check_pbc, check_pbc), 'for all j and k the final translation vector should be the same'
        check_pbc = to_check_pbc
    print(k)
cspbi3.add_cell(np.array([check_pbc-2*b_vec, b_vec*height, c_vec*depth]))
cspbi3.print_charge()

cspbi3.atoms = make_supercell(cspbi3.atoms, np.diag([1, 4, 1]))
#niggli_reduce(sup_atoms)
outname = "teststruc.cif"
write(outname, cspbi3.atoms, format="cif")
write('int_layer.xyz', cspbi3.atoms, format="extxyz")
