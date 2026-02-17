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
    extra_shifts[(0, (j-1)%height, (k+3)%depth)] = [-(-1)**k, 0, 0] # Default behavior
    if double:
        extra_shifts[(0, (j-1)%height, (k+2)%depth)] = [ (-1)**k, 0, 2]
    else:
        extra_shifts[(0, (j-1)%height, (k+2)%depth)] = [ (-1)**k, 0, 0] # Default behavior


def add_faces(j, k, double = False):
    add_extra_shifts((j-1)%height,(k+4)%depth, double = double)  # Due to overlapping atoms and excess I atoms otherwise
    faces[(0, j, k)] = 'central'
    extra_shifts[(0, j, k)] = [ 2, 1, 3]
    faces[(0, j, (k+1)%depth)] = 'central'
    extra_shifts[(0, j, (k+1)%depth)] = [ 0, 1, 1]
    faces[(0, j, (k+2)%depth)] = 'right'
    extra_shifts[(0, j, (k+2)%depth)] = [ (-1)**k, 0, 0]  # Default behavior
    faces[(0, (j-1)%height, (k+3)%depth)] = 'left'
    extra_shifts[(0, (j-1)%height, (k+3)%depth)] = [-(-1)**k, 0, 0] # Default behavior
    if double:
        faces[(0, (j-1)%height, (k+3)%depth)] = 'alsoright'
        extra_shifts[(0, (j-1)%height, (k+3)%depth)] = [-(-1)**k, 0,-2]  
    else:
        faces[(0, (j-1)%height, (k+3)%depth)] = 'left'
        extra_shifts[(0, (j-1)%height, (k+3)%depth)] = [-(-1)**k, 0, 0] # Default behavior


#Create the CsPbI3 perovskite structure
cspbi3 = CsPbI3()
Rotation = R.from_euler('x', 70.5287794, degrees=True)
a_vec = np.array([2*cspbi3.l*np.sqrt(2), 0.0, 0.0])
b_vec =  np.array([0.0, -2*cspbi3.l, 2*np.sqrt(2)*cspbi3.l])
c_vec = np.array([0.0, 0.0, np.sqrt(2)*cspbi3.l])

length = 1
height = 8 
depth = 8

faces = {}
extra_shifts = {}
faces_to_add =[
    (0, 0, False), 
    (1, 0, False),
    (2, 0, False), 
    (3, 0, False),
]
for (j,k, double) in faces_to_add:
    add_faces(j,k, double)
    if double:
        assert ((j-1)%height, k, False) or ((j-1)%height, k, True) in faces_to_add, "The neighbouring octahedra need to be rotated"

add_extra_shifts(0, 12, double = False)
add_extra_shifts(1, 12, double = False)
add_extra_shifts(2, 12, double = False)
add_extra_shifts(3, 12, double = False)


for k in range(depth):
    for j in range(height):
        for i in range(length):
            new_oct = cspbi3.get_edge_octahedra()
            
            trans = i*a_vec + j*b_vec + k*c_vec
            if (i, j, k) in extra_shifts:
                extra_shift = extra_shifts[(i, j, k)]*np.array([cspbi3.l/np.sqrt(2), cspbi3.l, cspbi3.l/np.sqrt(2)])
            else:
                extra_shift = np.array([(-1)**k*cspbi3.l/np.sqrt(2), 0.0, 0.0])
            
            rot= False
            if (i, j, k) in faces:
                rot = True
                if faces[(i, j, k)] == 'right' or  faces[(i, j, k)] == 'alsoright':
                    pre_rot = np.array([ 0.0, 0.0,-cspbi3.l/np.sqrt(2)])
                    post_rot =  np.array([ cspbi3.l*np.sqrt(2), 0.0, cspbi3.l/np.sqrt(2)])
                elif faces[(i, j, k)] == 'left':
                    pre_rot = np.array([ 0.0, 0.0, cspbi3.l/np.sqrt(2)])
                    post_rot =  np.array([-cspbi3.l*np.sqrt(2), 0.0,-cspbi3.l/np.sqrt(2)])
                else:
                    pre_rot = np.array([0.0, 0.0, 0.0])
                    post_rot = np.array([0.0, 0.0, 0.0])
            
            for at in new_oct:
                if rot:
                    #Remove all Cs atoms of the rotated octahedra, except two
                    if at.symbol == 'Cs':
                        if faces[(i, j, k)] == 'central':
                            if (i, (j+1)%height, (k-4)%depth) not in extra_shifts and at.position[1]< 0.0 and at.position[2]> 0.0:
                                at.position-= np.array([0.0, 0.8*cspbi3.l, cspbi3.l/np.sqrt(2)])
                            elif at.position[1]< 0.0 and at.position[2]< 0.0: #(i, (j-1)%height, k) in faces and
                                at.position += np.array([cspbi3.l/np.sqrt(2), cspbi3.l,-cspbi3.l/np.sqrt(2)])
                            else:
                                continue
                        elif faces[(i, j, k)] != 'alsoright' or at.position[1]> 0.0 or at.position[2]< 0.0:
                            continue
                    #Perform rotation
                    at.position += pre_rot
                    at.position = Rotation.apply(at.position)
                    at.position += post_rot

                #Extra conditions to put the Cs atoms in the right places
                if at.symbol == 'Cs':
                    if (i, j, (k+1)%depth) in faces:
                        if faces[(i, j, (k+1)%depth)] != 'central':
                            if at.position[1]< 0.0 and at.position[2]> 0.0:
                                continue
                    if (i, j, (k-1)%depth) in faces:
                        if faces[(i, j, (k-1)%depth)] != 'central':
                            if at.position[1]> 0.0 and at.position[2]< 0.0:
                                continue
                    if (i,  j, k) in extra_shifts:
                        if extra_shifts[(i, j, k)] == [ 2, 1, 3] or extra_shifts[(i, j, k)] == [ 0, 1, 1]:
                            if (i, (j+1)%height, (k-4)%depth) in faces or (i, (j+1)%height, (k-5)%depth) in faces:
                                if at.position[1]< 0.0 and at.position[2]< 0.0: 
                                    at.position += np.array([0.0, 0.2*cspbi3.l, np.sqrt(2)*0.2*cspbi3.l])
                            if (i, (j-1)%height, (k+4)%depth) in faces or (i, (j-1)%height, (k+3)%depth) in faces:
                                if at.position[1]> 0.0 and at.position[2]> 0.0: 
                                    at.position -= np.array([0.0, 0.2*cspbi3.l, np.sqrt(2)*0.2*cspbi3.l])

                #Place the octahedra at the right place
                at.position += trans+extra_shift
                cspbi3.atoms.append(at)
    print(k)
cspbi3.add_cell(np.array([a_vec*length, b_vec*height, c_vec*depth]))
cspbi3.print_charge()


cspbi3.atoms.cell[1,:] -= np.floor(2*height/depth)*cspbi3.atoms.cell[2,:]  # try to make it more orthorhombic, ideally height = n*depth/2
cspbi3.atoms = make_supercell(cspbi3.atoms, np.diag([2, 1, 1]))
#niggli_reduce(sup_atoms)
outname = "teststruc.cif"
write(outname, cspbi3.atoms, format="cif")
#write('int_layer_2c_double_connected.xyz', cspbi3.atoms, format="extxyz")
