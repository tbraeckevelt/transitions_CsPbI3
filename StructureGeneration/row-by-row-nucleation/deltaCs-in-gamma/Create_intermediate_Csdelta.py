from ase.io import read, write
from ase import Atom, Atoms
import numpy as np
from ase.build import niggli_reduce
from ase.build.supercells import make_supercell
import sys
sys.path.insert(0, '/home/tom/Documents/Doctoraat/StructureFiles/MakeCsPbI3IntermediateStruc')
from buildingblocks import CsPbI3
from scipy.spatial.transform import Rotation as R

def add_delta_right(j,k):
    faces[(0, (j-1)%height, (k+2)%depth)] = '+right'
    extra_shifts[(0, (j-1)%height, (k+2)%depth)] = [2, 1,-1]
    faces[(0, j, (k+2)%depth)] = '+right'
    extra_shifts[(0, j, (k+2)%depth)] = [0,-1,-1]
    faces[(0, j, (k+1)%depth)] = '+right'
    extra_shifts[(0, j, (k+1)%depth)] = [-(-1)**k,-2, 0]
    faces[(0, j, k)] = '+right'
    extra_shifts[(0, j, k)] = [ (-1)**k,-2, 2]

def add_delta_left(j,k):
    faces[(0, (j-1)%height, (k-1)%depth)] = '-left'
    extra_shifts[(0, (j-1)%height, (k-1)%depth)] = [2, 1, 1]
    faces[(0, j, (k-1)%depth)] = '-left'
    extra_shifts[(0, j, (k-1)%depth)] = [0,-1, 1]
    faces[(0, j, k)] = '-left'
    extra_shifts[(0, j, k)] = [ (-1)**k,-2, 0]
    faces[(0, j, (k+1)%depth)] = '-left'
    extra_shifts[(0, j, (k+1)%depth)] = [-(-1)**k,-2,-2]

def add_deltas(j, k, amount = 2):

    if amount >= 2:
        #first delta struc
        faces[(0, (j+1)%height, (k-1)%depth)] = '-left'
        extra_shifts[(0, (j+1)%height, (k-1)%depth)] = [0,-1, 1]
        faces[(0, (j+1)%height, k)] = '-left'
        extra_shifts[(0, (j+1)%height, k)] = [2,-1,-1]
        faces[(0, j, k)] = '-left'
        faces[(0, (j+1)%height, (k+1)%depth)] = '-left'
        extra_shifts[(0,  (j+1)%height, (k+1)%depth)] = [-(-1)**k, -2,-2]

        #second delta struc
        faces[(0, j, (k+1)%depth)] = '+right'
        extra_shifts[(0, j, (k+1)%depth)] = [2,-1, 1]
        faces[(0, (j+0)%height, (k+2)%depth)] = '+right'
        extra_shifts[(0, (j+0)%height, (k+2)%depth)] = [0,-1,-1]
        faces[(0, (j-1)%height, (k+1)%depth)] = '+right'
        faces[(0, (j-1)%height, (k+0)%depth)] = '+right'
        extra_shifts[(0, (j-1)%height, (k+0)%depth)] = [ (-1)**k, 0, 2]

        for i in range(2, amount):
            if i%2 == 0:
                add_delta_right(j+i,k)
            else:
                add_delta_left(j+i,k)

    else:
        raise NotImplementedError


#Create the CsPbI3 perovskite structure
cspbi3 = CsPbI3()
alpha = 70.5287794*np.pi/180  # numpy works with radians
a_vec = np.array([2*cspbi3.l*np.sqrt(2), 0.0, 0.0])
b_vec =  np.array([0.0, 2*cspbi3.l, 0.0])
c_vec = np.array([0.0, 0.0, np.sqrt(2)*cspbi3.l])

length = 1
height = 8  # Make sure pbc in y or z directions are not relevant
depth = 8

faces = {}
extra_shifts = {}
#add_deltas(2,3, amount = 6)


remove_cs_atoms = []
for k in range(depth):
    for j in range(height):
        for i in range(length):
            new_oct = cspbi3.get_edge_octahedra_full()

            trans = i*a_vec + j*b_vec + k*c_vec
            if (i, j, k) in extra_shifts:
                extra_shift = extra_shifts[(i, j, k)]*np.array([cspbi3.l/np.sqrt(2), cspbi3.l, cspbi3.l/np.sqrt(2)])
            else:
                extra_shift = np.array([(-1)**k*cspbi3.l/np.sqrt(2), 0.0, 0.0])
            
            rot= False
            if (i, j, k) in faces:
                if faces[(i, j, k)][0] == '-':
                    rot = -1
                elif faces[(i, j, k)][0] == '+':
                    rot = 1
                
                if faces[(i, j, k)][1:] == 'right':
                    pre_rot = np.array([ 0.0, 0.0,-cspbi3.l/np.sqrt(2)])
                    post_rot =  np.array([ cspbi3.l*np.sqrt(2), 0.0, cspbi3.l/np.sqrt(2)])
                elif faces[(i, j, k)][1:] == 'left':
                    pre_rot = np.array([ 0.0, 0.0, cspbi3.l/np.sqrt(2)])
                    post_rot =  np.array([-cspbi3.l*np.sqrt(2), 0.0,-cspbi3.l/np.sqrt(2)])
                else:
                    pre_rot = np.array([0.0, 0.0, 0.0])
                    post_rot = np.array([0.0, 0.0, 0.0])
            
            for at in new_oct:
                if rot is not False:
                    Rotation = R.from_euler('x', rot * alpha*180/np.pi, degrees=True)

                    #Remove all Cs atoms of the rotated octahedra
                    if at.symbol == 'Cs':
                        
                        if (i, j, (k-1)%depth) not in faces and (i, (j-1)%height, k) not in faces and (i, (j+1)%height, k) in faces:
                            if faces[(i, j, k)] == '+right' and at.position[1] > 0.0 and at.position[2] == 0.0:
                                at.position += trans + np.array([(-1)**k*cspbi3.l/np.sqrt(2), 0.0, 0.0])
                                cspbi3.atoms.append(at)
                                at_cs = Atom('Cs', at.position + np.array([cspbi3.l*np.sqrt(2), 0.0, 0.0]))
                                cspbi3.atoms.append(at_cs)
                        
                        if faces[(i, j, k)] == '-left' and (i, j, k) in extra_shifts:
                            if extra_shifts[(i,j,k)][1] == -2 and  extra_shifts[(i,j,k)][2] == -2:
                                if at.position[1] < 0.0 and at.position[2] == 0.0:
                                    at.position += trans + np.array([(-1)**k*cspbi3.l/np.sqrt(2), 0.0, 0.0])
                                    cspbi3.atoms.append(at)
                                    at_cs = Atom('Cs', at.position + np.array([cspbi3.l*np.sqrt(2), 0.0, 0.0]))
                                    cspbi3.atoms.append(at_cs)
                                    if (i,(j+1)%height,k) not in faces:
                                        at_cs = Atom('Cs', at.position + np.array([-cspbi3.l/np.sqrt(2), cspbi3.l, -cspbi3.l/np.sqrt(2)]))
                                        cspbi3.atoms.append(at_cs)
                                        at_cs = Atom('Cs', at.position + np.array([ cspbi3.l/np.sqrt(2), cspbi3.l, -cspbi3.l/np.sqrt(2)]))
                                        cspbi3.atoms.append(at_cs)
                            if (i, j, (k-1)%depth) not in faces:
                                if (i, (j+1)%height, k) not in faces:
                                    if at.position[0] > 0.0 and at.position[1] > 0.0 and at.position[2] == 0.0:
                                        at.position += trans + np.array([(-1)**k*cspbi3.l/np.sqrt(2), 0.0, 0.0])
                                        remove_cs_atoms.append(at.position.copy())
                                        at.position += np.array([0.0, 0.0,-cspbi3.l/np.sqrt(8)])
                                        cspbi3.atoms.append(at)
                                elif (i, j, (k+1)%depth) in faces:
                                    if faces[(i, j, (k+1)%depth)] == '-left':
                                        if at.position[0] > 0.0 and at.position[1] > 0.0 and at.position[2] == 0.0:
                                            at.position += trans 
                                            cspbi3.atoms.append(at)
                                            at_cs = Atom('Cs', at.position + np.array([cspbi3.l*np.sqrt(2), 0.0, 0.0]))
                                            cspbi3.atoms.append(at_cs)
                        
                        if faces[(i, j, k)] == '+right' and (i, j, k) in extra_shifts:
                            if extra_shifts[(i,j,k)][1] == -2 and  extra_shifts[(i,j,k)][2] == 2:
                                if at.position[1] < 0.0 and at.position[2] == 0.0:
                                    at.position += trans + np.array([(-1)**k*cspbi3.l/np.sqrt(2), 0.0, 0.0])
                                    cspbi3.atoms.append(at)
                                    at_cs = Atom('Cs', at.position + np.array([cspbi3.l*np.sqrt(2), 0.0, 0.0]))
                                    cspbi3.atoms.append(at_cs)
                                    if (i,(j+1)%height,k) not in faces:
                                        at_cs = Atom('Cs', at.position + np.array([-cspbi3.l/np.sqrt(2), cspbi3.l, cspbi3.l/np.sqrt(2)]))
                                        cspbi3.atoms.append(at_cs)
                                        at_cs = Atom('Cs', at.position + np.array([ cspbi3.l/np.sqrt(2), cspbi3.l, cspbi3.l/np.sqrt(2)]))
                                        cspbi3.atoms.append(at_cs)
                            if (i, j, (k+1)%depth) not in faces:
                                if (i, (j+1)%height, k) not in faces:
                                    if at.position[0] > 0.0 and at.position[1] > 0.0 and at.position[2] == 0.0:
                                        at.position += trans + np.array([(-1)**k*cspbi3.l/np.sqrt(2), 0.0, 0.0])
                                        remove_cs_atoms.append(at.position.copy())
                                        at.position += np.array([0.0, 0.0, cspbi3.l/np.sqrt(8)])
                                        cspbi3.atoms.append(at)
                                elif (i, j, (k-1)%depth) in faces:
                                    if faces[(i, j, (k-1)%depth)] == '+right':
                                        if at.position[0] > 0.0 and at.position[1] > 0.0 and at.position[2] == 0.0:
                                            at.position += trans 
                                            cspbi3.atoms.append(at)
                                            at_cs = Atom('Cs', at.position + np.array([cspbi3.l*np.sqrt(2), 0.0, 0.0]))
                                            cspbi3.atoms.append(at_cs)
                                            
                        continue
                    #Perform rotation
                    at.position += pre_rot
                    at.position = Rotation.apply(at.position)
                    at.position += post_rot

                #Extra conditions to put the Cs atoms in the right places
                if at.symbol == 'Cs':
                    if (i, j, (k+1)%depth) in faces and at.position[2] > 0.0:
                        continue
                    if (i, j, (k-1)%depth) in faces and at.position[2] < 0.0:
                        continue

                #Place the octahedra at the right place
                at.position += trans+extra_shift
                cspbi3.atoms.append(at)
    print(k)
cspbi3.add_cell(np.array([a_vec*length, b_vec*height, c_vec*depth]))
#Remove excess Cs atoms
for Cs_pos in remove_cs_atoms:
    cspbi3.atoms.append(Atom('Cs', Cs_pos))
    for at in cspbi3.atoms:
        dist = cspbi3.atoms.get_distance(at.index,-1, mic = True)
        assert at.index != len(cspbi3.atoms) -1, "No atoms found to remove"
        if dist  < 0.01:
            cspbi3.atoms.pop(i=-1)
            cspbi3.atoms.pop(i=at.index)
            break
cspbi3.print_charge()

cspbi3.atoms = make_supercell(cspbi3.atoms, np.diag([2, 1, 1]))
#niggli_reduce(sup_atoms)
outname = "teststruc.cif"
write(outname, cspbi3.atoms, format="cif")
write('gamma_double.xyz', cspbi3.atoms, format="extxyz")
