from ase.io import read, write
from ase import Atom, Atoms
import numpy as np
from ase.build import niggli_reduce
from ase.build.supercells import make_supercell
import sys
sys.path.insert(0, '/home/tom/Documents/Doctoraat/StructureFiles/MakeCsPbI3IntermediateStruc')
from buildingblocks import CsPbI3
from scipy.spatial.transform import Rotation as R



def add_nucleation(i, j, k, defect_height = 3, defect_depth = 3):
    for h in range(defect_height):
        for l in range(defect_depth):
            extra_shifts[(i, (j+h)%height, (k+l)%depth)] = [ 3*(-1)**(k+l), 0, 0]


#Create the CsPbI3 perovskite structure
cspbi3 = CsPbI3()
a_vec = np.array([2*cspbi3.l*np.sqrt(2), 0.0, 0.0])
b_vec =  np.array([0.0, 2*cspbi3.l, 0.0])
c_vec = np.array([0.0, 0.0, np.sqrt(2)*cspbi3.l])

length = 3
height = 6 
depth = 6

faces = {}
extra_shifts = {}
#add_nucleation(1, 1, 1, defect_height = 2, defect_depth = depth)
#add_nucleation(1, 1, 1, defect_height = 4, defect_depth = 2)
#add_nucleation(1, 2, 3, defect_height = 2, defect_depth = 2)
Cs_option = 2


remove_cs_atoms = []
for k in range(depth):
    for j in range(height):
        for i in range(length):
            if (i, j, (k+1)%depth) in extra_shifts or (i, j, k) in extra_shifts or (i, j, (k-1)%depth) in extra_shifts or ((i+1)%length, j, k) in extra_shifts or ((i-1)%length, j, k) in extra_shifts:
                new_oct = cspbi3.get_edge_octahedra_full()
            else:
                new_oct = cspbi3.get_edge_octahedra_reduced()

            trans = i*a_vec + j*b_vec + k*c_vec
            if (i, j, k) in extra_shifts:
                extra_shift = extra_shifts[(i, j, k)]*np.array([cspbi3.l/np.sqrt(2), cspbi3.l, cspbi3.l/np.sqrt(2)])
            else:
                extra_shift = np.array([(-1)**k*cspbi3.l/np.sqrt(2), 0.0, 0.0])

            for at in new_oct:
                #Add extra Cs atoms
                if at.symbol == 'Cs':
                    if (i, j, k) in extra_shifts:
                        for sgn in [-1 , 1]:
                            if (i, (j+sgn)%height, k) not in extra_shifts:
                                if at.position[1]*sgn > 0.0 and at.position[2] == 0.0:
                                    if at.position[0] == -cspbi3.l*np.sqrt(2)*np.sign(extra_shifts[(i, j, k)][0]):
                                        if Cs_option == 1:
                                            at.position +=np.array([ np.sign(extra_shifts[(i, j, k)][0])*cspbi3.l*np.sqrt(2), sgn*cspbi3.l, 0.0])
                                        elif Cs_option == 2:
                                            at.position -=np.array([ 0.0, sgn*cspbi3.l, 0.0])
                                        else:
                                            raise NotImplementedError

                #Place the octahedra at the right place
                at.position += trans+extra_shift
                cspbi3.atoms.append(at)
    print(k)

cspbi3.add_cell(np.array([a_vec*length, b_vec*height, c_vec*depth]))
cspbi3.print_charge()

#cspbi3.atoms = make_supercell(cspbi3.atoms, np.diag([1, 1, 1]))
#niggli_reduce(sup_atoms)
outname = "teststruc.cif"
write(outname, cspbi3.atoms, format="cif")
#write('gamma.xyz', cspbi3.atoms, format="extxyz")
