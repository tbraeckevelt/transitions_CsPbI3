'''
This script works, but is not used to create the structures, Create_intermediate_zeta_ob.py is used instead
'''


from ase.io import read, write
from ase import Atom, Atoms
import numpy as np
from ase.build import niggli_reduce
from ase.build.supercells import make_supercell

atoms = Atoms()
atoms_dup = Atoms()

l = 3.2 # Pb-I distance
total_length =  4
eta_layers = [0,1,2,3] # [0,1,2,3,4,5]
height = 4 #height of the RP phase
depth = 4

#CsPbI3 building block
atoms_fu_gam = Atoms(symbols=["Cs", "Cs", "Cs", "Cs", "Cs", "Cs", "Cs", "Cs", "Pb",  "I", "I", "I", "I", "I", "I"], 
                    positions=[np.array([ l, l, l]), 
                            np.array([-l, l, l]), 
                            np.array([ l,-l, l]), 
                            np.array([-l,-l, l]), 
                            np.array([ l, l,-l]), 
                            np.array([-l, l,-l]), 
                            np.array([ l,-l,-l]), 
                            np.array([-l,-l,-l]), 
                            np.array([0.0, 0.0, 0.0]), 
                            np.array([  l, 0.0, 0.0]), 
                            np.array([ -l, 0.0, 0.0]), 
                            np.array([0.0,   l, 0.0]), 
                            np.array([0.0,  -l, 0.0]), 
                            np.array([0.0, 0.0,   l]), 
                            np.array([0.0, 0.0,  -l])])

#Rotate atoms_fu 45 degrees around the z-axis
#atoms_fu_eta = atoms_fu_gam.copy()
#atoms_fu_eta.rotate(45, 'z')

#Create the CsPbI3 perovskite structure
Pos_start = np.array([0.0, 0.0, 0.0])
for i in range(total_length):

    #if i in eta_layers:
    #    atoms_fu = atoms_fu_eta
    #else:
    #    atoms_fu = atoms_fu_gam
    
    atoms_fu = atoms_fu_gam
    for at in atoms_fu.copy():
        at.position += Pos_start
        atoms_dup.append(at)

    if i in eta_layers:
        Pos_start += np.array([l, l, 0.0])
    else:
        Pos_start += np.array([l*2, 0.0, 0.0])

cell= np.array([Pos_start, [-2*l, 2*l, 0.0], [0.0, 0.0, 2*l]])

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

sup_atoms = make_supercell(new_atoms, np.diag([1, height, depth]))
niggli_reduce(sup_atoms)

outname = "teststruc.cif"
write(outname, sup_atoms, format="cif")

#outname = 'gam_eta'
#for i in eta_layers:
#    outname += "_" + str(i)
#write(outname + '.xyz', sup_atoms, format="extxyz")
#write('eta_4_4_4.xyz', sup_atoms, format="extxyz")