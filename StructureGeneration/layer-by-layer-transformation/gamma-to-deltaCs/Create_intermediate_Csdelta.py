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

height = 4 #height of the RP phase
depth = 8
a_vec = np.array([cspbi3.l*np.sqrt(2), 0.0, 0.0])
b_vec =  np.array([0.0, 2*cspbi3.l, 0.0])
c_vec = np.array([0.0, 0.0, 2*np.sqrt(2)*cspbi3.l])

shifts ={
    'default' : [-1, 1],
}

length = len(shifts['default'])
for k, v in shifts.items():
    assert length == len(v), "all arrays of shifts should be the same length"
assert height % 2 == 0, "we need an even number of octahedras in the y direction"

Csdelta_layers = [1,2,3,4,5,6,7] #

for k in range(depth):
    if k in Csdelta_layers:
        assert (k-1)%depth in Csdelta_layers or (k+1)%depth in Csdelta_layers, "a Csdelta layerthickness of one is not implemented"
        if (k+1)%depth not in Csdelta_layers and k%2 == 1:
            extra_trans = np.array([cspbi3.l/np.sqrt(2), -1.0625, 1.67839 - 0.92146])
        elif (k-1)%depth not in Csdelta_layers and k%2 == 0:
            extra_trans = np.array([cspbi3.l/np.sqrt(2), -1.0625, -1.67839 + 0.92146])
        elif (k-1)%depth in Csdelta_layers and (k+1)%depth in Csdelta_layers:
            extra_trans = np.array([0.0, -1.0625/2, (-1)**k*(-1.67839 + 0.92146)/2])
        else:
            extra_trans = np.array([0.0, 0.0, 0.0])
    for j in range(height):
        mirror_z = False
        if j%2 == 1:
            mirror_z = True
        trans = j*b_vec + k*c_vec
        for i,z_shift in enumerate(shifts.get((j, k), shifts['default'])):
            if k in Csdelta_layers and (k-j)%2 == 0:
                for at in cspbi3.get_Csdelta_octahedras(center = np.array([0.0, 2*cspbi3.l, -(-1)**j*cspbi3.l/np.sqrt(2)]), rot_degree=(-1)**j*70.5, mirror_z=mirror_z):
                    if at.symbol == 'Cs':
                        cspbi3.atoms.append(Atom('Cs', np.array([cspbi3.l*np.sqrt(2)*i, cspbi3.l*(2*j+1), 2*cspbi3.l*(k+0.25)*np.sqrt(2)])))
                        cspbi3.atoms.append(Atom('Cs', np.array([cspbi3.l*np.sqrt(2)*i, cspbi3.l*(2*j+1), 2*cspbi3.l*(k-0.25)*np.sqrt(2)])))
                    else:
                        at.position += trans+np.array([cspbi3.l/np.sqrt(2), 0.0, (-1)**j*cspbi3.l*np.sqrt(2)]) + extra_trans
                        cspbi3.atoms.append(at)
                trans += np.array([cspbi3.l*np.sqrt(2), 0.0, 0.0])
            elif k not in Csdelta_layers:
                trans, rot = cspbi3.add_octahedra(trans = trans, z_shift = z_shift)
    print(k)
cspbi3.add_cell(np.array([a_vec*length, b_vec*height, c_vec*depth]))

#Remove some Cs atoms coming from the gamma phase
for at in cspbi3.atoms:
    if at.position[1] < 0.0:
        at.position += b_vec*height
    if at.position[2] < 0.0:
        at.position += c_vec*depth

for k in Csdelta_layers:
    for j in range(height):
        for i in range(length):
            for at in cspbi3.atoms:
                if at.symbol == 'Cs':
                    if np.allclose(at.position, np.array([cspbi3.l*(2*i+1)/np.sqrt(2), cspbi3.l*(2*j+1), 2*cspbi3.l*(k+0.25)*np.sqrt(2)])):
                        cspbi3.atoms.pop(i=at.index)
                        break
                    elif np.allclose(at.position, np.array([cspbi3.l*(2*i+1)/np.sqrt(2), cspbi3.l*(2*j+1), 2*cspbi3.l*(k-0.25)*np.sqrt(2)])):
                        cspbi3.atoms.pop(i=at.index)
                        break

cspbi3.print_charge()

cspbi3.atoms = make_supercell(cspbi3.atoms, np.diag([2, 1, 1]))
#niggli_reduce(sup_atoms)
outname = "teststruc.cif"
write(outname, cspbi3.atoms, format="cif")
write('int_6_double.xyz', cspbi3.atoms, format="extxyz")
