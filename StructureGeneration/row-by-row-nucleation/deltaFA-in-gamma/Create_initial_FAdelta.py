from ase.io import read, write
from ase import Atom, Atoms
import numpy as np
from ase.build import niggli_reduce
from ase.build.supercells import make_supercell
import sys
sys.path.insert(0, '/home/tom/Documents/Doctoraat/StructureFiles/MakeCsPbI3IntermediateStruc')
from buildingblocks import CsPbI3
from scipy.spatial.transform import Rotation as R


def add_faces(j, shift_one = 'down', shift_two = 'up'):

    if shift_one == 'down':
        faces[(0, (j+1)%height, (j-1)%depth)] = 'left'
        extra_shifts[(0, (j+1)%height, (j-1)%depth)] = [-(-1)**j, 0, 0] # Default behavior
        faces[(0, (j+1)%height, j)] = 'shift_down'
        extra_shifts[(0, (j+1)%height, j)] = [ (-1)**j,-1,-1] 
        if (0, j, (j-2)%depth) not in faces:
            extra_shifts[(0, j, (j-2)%depth)] = [ 0,-1,-1]
        if (0, j, (j-1)%depth) not in faces:
            extra_shifts[(0, j, (j-1)%depth)] = [ 2,-1,-3]
    elif shift_one == 'up':
        faces[(0, j, (j-2)%depth)] = 'left'
        extra_shifts[(0, j, (j-2)%depth)] = [-(-1)**j, 2, 2] 
        faces[(0, j, (j-1)%depth)] = 'shift_down'
        extra_shifts[(0, j, (j-1)%depth)] = [ (-1)**j, 1, 1] 
        if (0, (j+1)%height, (j-1)%depth) not in faces:
            extra_shifts[(0, (j+1)%height, (j-1)%depth)] = [ 0, 1, 3]
        if (0, (j+1)%height, j) not in faces:
            extra_shifts[(0, (j+1)%height, j)] = [ 2, 1, 1]
    else:
        raise AssertionError

    faces[(0, j, j)] = 'central'
    extra_shifts[(0, j, j)] = [ (-1)**j, 0, 0] # Default behavior

    if shift_two == 'up':
        faces[(0, (j-1)%height, j)] = 'shift_up'
        extra_shifts[(0, (j-1)%height, j)] = [ (-1)**j, 1, 1]
        faces[(0, (j-1)%height, (j+1)%depth)] = 'right'
        extra_shifts[(0, (j-1)%height, (j+1)%depth)] = [-(-1)**j, 0, 0] # Default behavior
        if (0, j, (j+1)%depth) not in faces:
            extra_shifts[(0, j, (j+1)%depth)] = [ 0, 1, 3]
        if (0, j, (j+2)%depth) not in faces:
            extra_shifts[(0, j, (j+2)%depth)] = [ 2, 1, 1]
    elif shift_two == 'down':
        faces[(0, j, (j+1)%depth)] = 'shift_up'
        extra_shifts[(0, j, (j+1)%depth)] = [ (-1)**j,-1,-1]
        faces[(0, j, (j+2)%depth)] = 'right'
        extra_shifts[(0, j, (j+2)%depth)] = [-(-1)**j,-2,-2] # Default behavior
        if (0, (j-1)%height, j) not in faces:
            extra_shifts[(0, (j-1)%height, j)] = [ 2,-1,-1]
        if (0, (j-1)%height, (j+1)%depth) not in faces:
            extra_shifts[(0, (j-1)%height, (j+1)%depth)] = [ 0,-1,-3]
    else:
        raise AssertionError



#Create the CsPbI3 perovskite structure
cspbi3 = CsPbI3()
alpha = 70.5287794*np.pi/180  # numpy works with radians
Rotation = R.from_euler('x', alpha*180/np.pi, degrees=True)
b_shift = -cspbi3.l* np.cos(alpha/2)*np.sqrt(2)*np.cos(3*alpha/2)/np.sin(alpha)  # shifts needed for the placement of atoms
c_shift = -cspbi3.l* np.sin(alpha/2)*np.sqrt(2)*np.cos(3*alpha/2)/np.sin(alpha)  # shifts needed for the placement of atoms
a_vec = np.array([2*cspbi3.l*np.sqrt(2), 0.0, 0.0])
b_vec =  np.array([0.0, 2*cspbi3.l, 0.0])
c_vec = np.array([0.0, 0.0, np.sqrt(2)*cspbi3.l])

length = 1
height = 8  # Make sure pbc in y or z directions are not relevant
depth = 8

faces = {}
extra_shifts = {}
shift_one = 'down'  # needs to be the same for all added faces
shift_two = 'up'    # needs to be the same for all added faces
add_faces(3, shift_one = shift_one, shift_two = shift_two)
add_faces(4, shift_one = shift_one, shift_two = shift_two)
add_faces(5, shift_one = shift_one, shift_two = shift_two)
add_faces(6, shift_one = shift_one, shift_two = shift_two)


for k in range(depth):
    for j in range(height):
        for i in range(length):
            if (i, j, (k+1)%depth) in extra_shifts or (i, j, (k-1)%depth) in extra_shifts:
                new_oct = cspbi3.get_edge_octahedra_full()
            else:
                new_oct = cspbi3.get_edge_octahedra_reduced()
                for at in new_oct:
                    if at.symbol == 'Cs':
                        at.position = np.array([ np.sqrt(2)*cspbi3.l, cspbi3.l, 0])
            
            trans = i*a_vec + j*b_vec + k*c_vec
            if (i, j, k) in extra_shifts:
                extra_shift = extra_shifts[(i, j, k)]*np.array([cspbi3.l/np.sqrt(2), cspbi3.l, cspbi3.l/np.sqrt(2)])
            else:
                extra_shift = np.array([(-1)**k*cspbi3.l/np.sqrt(2), 0.0, 0.0])
            
            rot= False
            if (i, j, k) in faces:
                
                rot = True
                if faces[(i, j, k)] == 'right':
                    pre_rot = np.array([ 0.0, 0.0,-cspbi3.l/np.sqrt(2)])
                    post_rot =  np.array([ cspbi3.l*np.sqrt(2), 0.0, cspbi3.l/np.sqrt(2)])
                elif faces[(i, j, k)] == 'left':
                    pre_rot = np.array([ 0.0, 0.0, cspbi3.l/np.sqrt(2)])
                    post_rot =  np.array([-cspbi3.l*np.sqrt(2), 0.0,-cspbi3.l/np.sqrt(2)])
                elif faces[(i, j, k)] == 'shift_up':
                    rot=False
                    extra_shift += np.array([0.0, b_shift, c_shift])
                    print(b_shift, c_shift)
                    print(-cspbi3.l*np.sqrt(2)*np.cos(3*alpha/2)/np.sin(alpha))
                elif faces[(i, j, k)] == 'shift_down':
                    rot=False
                    extra_shift -= np.array([0.0, b_shift, c_shift])
                else:
                    pre_rot = np.array([0.0, 0.0, 0.0])
                    post_rot = np.array([0.0, 0.0, 0.0])
            
            for at in new_oct:
                if rot:
                    #Remove all Cs atoms of the rotated octahedra
                    if at.symbol == 'Cs':
                        continue
                    #Perform rotation
                    at.position += pre_rot
                    at.position = Rotation.apply(at.position)
                    at.position += post_rot

                #Extra conditions to put the Cs atoms in the right places
                if at.symbol == 'Cs':
                    if (i, j, (k+1)%depth) in faces:
                        if faces[(i, j, (k+1)%depth)] == 'left' and at.position[1]< 0.0 and at.position[2]> 0.0:
                            continue
                    elif (i, (j-1)%height, k) in faces:
                        if faces[(i, (j-1)%height, k)] == 'left' and at.position[1]< 0.0 and at.position[2]> 0.0:
                            continue
                    if (i, j, (k-1)%depth) in faces:
                        if faces[(i, j, (k-1)%depth)] == 'right' and at.position[1]> 0.0 and at.position[2]< 0.0:
                            continue
                    elif (i, (j+1)%height, k) in faces:
                        if faces[(i, (j+1)%height, k)] == 'right' and at.position[1]> 0.0 and at.position[2]< 0.0:
                            continue
                    if (i, j, k) in faces:
                        if faces[(i, j, k)][:5] == 'shift':
                            if  at.position[2] != 0.0: # (at.position[1]< 0.0 and at.position[2]> 0.0) or (at.position[1]> 0.0 and at.position[2]< 0.0):
                                continue
                            elif faces[(i, j, k)] == 'shift_down' and (i, (j-1)%height, (k-1)%depth) not in faces:
                                if at.position[1] < 0.0 and at.position[2] == 0.0:
                                    at.position -= np.array([0.0, cspbi3.l-b_shift, cspbi3.l/np.sqrt(2)-c_shift])
                                    at.position += np.array([ 0.0,-cspbi3.l*0.3, cspbi3.l*1.1])
                            elif faces[(i, j, k)] == 'shift_up' and (i, (j+1)%height, (k+1)%depth) not in faces:
                                if at.position[1] > 0.0 and at.position[2] == 0.0:
                                    at.position += np.array([0.0, cspbi3.l-b_shift, cspbi3.l/np.sqrt(2)-c_shift])
                                    at.position -= np.array([ 0.0,-cspbi3.l*0.3, cspbi3.l*1.1])
                    elif (i, j, k) in extra_shifts:
                        if (i, (j-1)%height, (k-1)%depth) not in faces and at.position[1] > 0.0 and at.position[2] > 0.0:
                            at.position -= np.array([ 0.0, cspbi3.l*0.2, cspbi3.l*0.3])
                        if (i, (j+1)%height, (k+1)%depth) not in faces and at.position[1] < 0.0 and at.position[2] < 0.0:
                            at.position += np.array([ 0.0, cspbi3.l*0.2, cspbi3.l*0.3])
                    if (i, (j+1)%height, (k+1)%depth) in faces:
                        if faces[(i, (j+1)%height, (k+1)%depth)] == 'central':
                            if at.position[1]> 0.0 and at.position[2]> 0.0:
                                continue
                            if at.position[1] > 0.0 and at.position[2] == 0.0:
                                at.position += np.array([ 0.0,-cspbi3.l*0.3, cspbi3.l*1.1])
                    if (i, (j-1)%height, (k-1)%depth) in faces:
                        if faces[(i, (j-1)%height, (k-1)%depth)] == 'central':
                            if at.position[1]< 0.0 and at.position[2]< 0.0:
                                continue
                            if at.position[1] < 0.0 and at.position[2] == 0.0:
                                at.position -= np.array([ 0.0,-cspbi3.l*0.3, cspbi3.l*1.1])

                #Place the octahedra at the right place
                at.position += trans+extra_shift
                cspbi3.atoms.append(at)
    print(k)
cspbi3.add_cell(np.array([a_vec*length, b_vec*height, c_vec*depth]))
cspbi3.print_charge()

cspbi3.atoms = make_supercell(cspbi3.atoms, np.diag([2, 1, 1]))
#niggli_reduce(sup_atoms)
outname = "teststruc.cif"
write(outname, cspbi3.atoms, format="cif")
#write('int_4a_double.xyz', cspbi3.atoms, format="extxyz")
