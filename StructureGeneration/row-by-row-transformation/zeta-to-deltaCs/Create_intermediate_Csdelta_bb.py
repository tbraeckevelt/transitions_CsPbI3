from ase.io import read, write
from ase import Atom, Atoms
import numpy as np
from ase.build.supercells import make_supercell
import sys
sys.path.insert(0, '/home/tom/Documents/Doctoraat/StructureFiles/MakeCsPbI3IntermediateStruc')
from buildingblocks import CsPbI3

#Create the CsPbI3 perovskite structure
cspbi3 = CsPbI3()

total_length =  4
coords = [[1,2], [2,0], [2,2]] #[0,0], [0,2], [1,0], [1,2], [2,0], [2,2], [3,0], [3,2]] # coordinates of the Pb atoms in the eta phase
height = 4 #height of the RP phase
depth = 4

assert depth % 2 == 0, "Depth must be even"

#Create the CsPbI3 perovskite structure
Pos_start = np.array([0.0, 0.0, 0.0])
for i in range(depth):
    if i % 2 == 0:
        Pos_start = np.array([0.0, 0.0, 2*np.sqrt(2)*cspbi3.l*i])
    else:
        Pos_start = np.array([cspbi3.l/np.sqrt(2), 0.0, 2*np.sqrt(2)*cspbi3.l*i])
    for j in range(height):
        if [j,i] in coords:
            atoms_fu = cspbi3.get_Csdelta_part(rot_degree=66.51*(-1)**j, rot_axis='x')
            atoms_fu.positions += np.array([0.0, -cspbi3.l, 0.0])
        elif [j,(i-1*(-1)**j)%depth] in coords:
            atoms_fu = cspbi3.get_Csdelta_part(rot_degree=66.51*(-1)**j, rot_axis='x', mirror = True)
            atoms_fu.positions += np.array([0.0, cspbi3.l, 0.0])
        else:
            atoms_fu = cspbi3.get_edge_octahedra_reduced()
        for at in atoms_fu:
            at.position += Pos_start
            # Place the Cs in different positions depending on its environment
            if at.symbol == "Cs":
                at.position = np.array([ cspbi3.l/np.sqrt(8) * (-1)**(i+j), cspbi3.l, np.sqrt(2)*cspbi3.l]) + Pos_start
                if j % 2 == 0:
                    if [(j+1)%height,(i+1)%depth] in coords:
                        at.position = np.array([cspbi3.l/np.sqrt(2), cspbi3.l, cspbi3.l/np.sqrt(2)]) + Pos_start
                        if [j,(i-1)%depth] in coords:
                            at.position -= np.array([0.0, cspbi3.l/np.sqrt(2), 0.0])
                    elif [j,i] in coords:
                        at.position = np.array([cspbi3.l/np.sqrt(2), cspbi3.l, cspbi3.l/np.sqrt(2)]) + Pos_start
                        if [(j+1)%height,i] in coords:
                            at.position += np.array([0.0, cspbi3.l/np.sqrt(2), 0.0])
                else:
                    if [j,(i+1)%depth] in coords:
                        at.position = np.array([0.0, cspbi3.l,  2*np.sqrt(2)*cspbi3.l - cspbi3.l/np.sqrt(2)]) + Pos_start
                        if [(j+1)%height,(i+1)%depth] in coords:
                            at.position += np.array([0.0, cspbi3.l/np.sqrt(2), 0.0])                            
                    elif [(j+1)%height,i] in coords:
                        at.position = np.array([0.0, cspbi3.l,  2*np.sqrt(2)*cspbi3.l - cspbi3.l/np.sqrt(2)]) + Pos_start
                        if [j,(i+2)%depth] in coords:
                            at.position -= np.array([0.0, cspbi3.l/np.sqrt(2), 0.0])
            cspbi3.atoms.append(at)
        Pos_start += np.array([0.0, cspbi3.l*2, 0.0])

cspbi3.add_cell(np.array([[cspbi3.l*np.sqrt(2), 0.0, 0.0], [0.0, 2*cspbi3.l*height, 0.0], [0.0, 0.0, 2*np.sqrt(2)*cspbi3.l*depth]]))
cspbi3.print_charge()

cspbi3.atoms = make_supercell(cspbi3.atoms, np.diag([total_length, 1, 1]))

outname = "teststruc.cif"
write(outname, cspbi3.atoms, format="cif")

#outname = 'eta_Csdel'
#for coord in coords:
#    outname += "_" + str(coord[0]) + str(coord[1])
#write(outname + '.xyz', sup_atoms, format="extxyz")
#write('Csdel_4_4_4.xyz', sup_atoms, format="extxyz")



























'''

def rotate_atoms(atoms, pos,  angle, axis='z'):
    """Rotate atoms by a given angle around a specified axis and a specific position."""
    rotation = R.from_euler(axis, angle, degrees=True)
    for at in atoms:
        at.position -= pos
        at.position = rotation.apply(at.position)
        at.position += pos



atoms = Atoms()
atoms_dup = Atoms()

l = 3.2 # Pb-I distance
total_length =  4
coords = [[0,0], [0,2], [1,0], [1,2], [2,0], [2,2], [3,0], [3,2]] #[0,0], [0,2], [1,0], [1,2], [2,0], [2,2], [3,0], [3,2]] # coordinates of the Pb atoms in the eta phase
height = 4 #height of the RP phase
depth = 4

assert depth % 2 == 0, "Depth must be even"

#CsPbI3 building block
atoms_fu_eta = Atoms(symbols=["Cs", "Pb", "I", "I", "I", "I"], 
                    positions=[
                        np.array([ 0, l, np.sqrt(2)*l]), 
                        np.array([0.0, 0.0, 0.0]), 
                        np.array([ 0.0, l, 0.0]), 
                        np.array([ 0.0,-l, 0.0]), 
                        np.array([-l/np.sqrt(2), 0.0,  l/np.sqrt(2)]), 
                        np.array([-l/np.sqrt(2), 0.0, -l/np.sqrt(2)])]) 

atoms_fu_del = Atoms(symbols=["Cs", "Pb", "I", "I", "I", "I"], 
                    positions=[
                        np.array([ 0, 2*l, np.sqrt(2)*l]), 
                        np.array([0.0, 1.15*l, 0.0]), 
                        np.array([ 0.0, 2.34*l, 0.0]), 
                        np.array([ 0.0,0, 0.0]), 
                        np.array([-l/np.sqrt(2), 1.05*l,  l/np.sqrt(2)]), 
                        np.array([-l/np.sqrt(2), 1.05*l, -l/np.sqrt(2)])]) 

#Create the CsPbI3 perovskite structure
Pos_start = np.array([0.0, 0.0, 0.0])
for i in range(depth):
    if i % 2 == 0:
        Pos_start = np.array([0.0, 0.0, 2*np.sqrt(2)*l*i])
    else:
        Pos_start = np.array([l/np.sqrt(2), 0.0, 2*np.sqrt(2)*l*i])
    for j in range(height):
        if [j,i] in coords:
            atoms_fu = atoms_fu_del.copy()
            atoms_fu.rotate(66.51*(-1)**j, 'x')
            atoms_fu.positions += np.array([0.0, -l, 0.0])
        elif [j,(i-1*(-1)**j)%depth] in coords:
            atoms_fu = atoms_fu_del.copy()
            atoms_fu.positions *= -1
            atoms_fu.rotate(66.51*(-1)**j, 'x')
            atoms_fu.positions += np.array([0.0, +l, 0.0])
        else:
            atoms_fu = atoms_fu_eta.copy()
        for at in atoms_fu:
            at.position += Pos_start
            if at.symbol == "Cs":
                at.position = np.array([ l/np.sqrt(8) * (-1)**(i+j), l, np.sqrt(2)*l]) + Pos_start
                if j % 2 == 0:
                    if [(j+1)%height,(i+1)%depth] in coords:
                        at.position = np.array([l/np.sqrt(2), l, l/np.sqrt(2)]) + Pos_start
                        if [j,(i-1)%depth] in coords:
                            at.position -= np.array([0.0, l/np.sqrt(2), 0.0])
                    elif [j,i] in coords:
                        at.position = np.array([l/np.sqrt(2), l, l/np.sqrt(2)]) + Pos_start
                        if [(j+1)%height,i] in coords:
                            at.position += np.array([0.0, l/np.sqrt(2), 0.0])
                else:
                    if [j,(i+1)%depth] in coords:
                        at.position = np.array([0.0, l,  2*np.sqrt(2)*l - l/np.sqrt(2)]) + Pos_start
                        if [(j+1)%height,(i+1)%depth] in coords:
                            at.position += np.array([0.0, l/np.sqrt(2), 0.0])                            
                    elif [(j+1)%height,i] in coords:
                        at.position = np.array([0.0, l,  2*np.sqrt(2)*l - l/np.sqrt(2)]) + Pos_start
                        if [j,(i+2)%depth] in coords:
                            at.position -= np.array([0.0, l/np.sqrt(2), 0.0])    
            atoms_dup.append(at)
        Pos_start += np.array([0.0, l*2, 0.0])

cell= np.array([[l*np.sqrt(2), 0.0, 0.0], [0.0, 2*l*height, 0.0], [0.0, 0.0, 2*np.sqrt(2)*l*depth]])

#Remove duplicate atoms
for at in atoms_dup:
    dir_pos = np.dot(at.position, np.linalg.inv(cell))
    dir_pos -= np.floor(dir_pos+0.0001)
    at.position = np.dot(dir_pos, cell)
    flag = True
    for at2 in atoms:
        dist = np.linalg.norm(at.position - at2.position)
        if dist < 0.1:
            flag = False
            #assert at.symbol == at2.symbol
            if at.symbol != at2.symbol:
                print("Different symbol at same position("+str(at.position)+"):" + at.symbol +" and " + at2.symbol)
                at2.symbol = "I"
    if flag:
        atoms.append(at)

at_numb_lst = []
pos_lst = []
num_at_dct = {}
for el in ["Cs", "Pb", "I"]:
    count = 0
    for at in atoms:
        if at.symbol == el:
            count +=1
            at_numb_lst.append(at.number)
            pos_lst.append(at.position.copy())
    num_at_dct[el] = count
print(num_at_dct)   
print("net charge: ", num_at_dct["Cs"] + 2*num_at_dct["Pb"] - num_at_dct["I"])
if num_at_dct["Cs"] + 2*num_at_dct["Pb"] - num_at_dct["I"] != 0:
    print("Warning: net charge is not zero, some atoms are overlapping that should not overlap")

new_atoms = Atoms(numbers=at_numb_lst, positions=pos_lst, pbc=True, cell=cell)

sup_atoms = make_supercell(new_atoms, np.diag([total_length, 1, 1]))
#niggli_reduce(sup_atoms)

outname = "teststruc.cif"
write(outname, sup_atoms, format="cif")

#outname = 'eta_Csdel'
#for coord in coords:
#    outname += "_" + str(coord[0]) + str(coord[1])
#write(outname + '.xyz', sup_atoms, format="extxyz")
write('Csdel_4_4_4.xyz', sup_atoms, format="extxyz")

'''