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

total_length =  4
height = 1 #height of the RP phase
depth = 1
b_vec =  np.array([0.0, 2*cspbi3.l, 0.0])
c_vec = np.array([0.0, 0.0, 2*np.sqrt(2)*cspbi3.l])

shifts ={
    'default' : [-1, 1,-1, 1,-1, 1,-1, 1],
}

check_pbc = None
for k in range(depth):
    for j in range(height):
        trans = j*b_vec + k*c_vec
        for z_shift in shifts.get((j, k), shifts['default']):
            trans, rot = cspbi3.add_octahedra(trans = trans, z_shift = z_shift)
        to_check_pbc = trans-j*b_vec - k*c_vec
        if check_pbc is not None:
            assert np.allclose(to_check_pbc, check_pbc), 'for all j and k the final translation vector should be the same'
        check_pbc = to_check_pbc
    print(k)
cspbi3.add_cell(np.array([check_pbc, b_vec*height, c_vec*depth]))
cspbi3.print_charge()

cspbi3.atoms = make_supercell(cspbi3.atoms, np.diag([1, 4, 2]))
#niggli_reduce(sup_atoms)
outname = "teststruc.cif"
write(outname, cspbi3.atoms, format="cif")

#outname = "int_1.xyz"
write(outname, cspbi3.atoms, format="extxyz")



































'''
atoms = Atoms()
atoms_dup = Atoms()

l = 3.2 # Pb-I distance
total_length =  8
eta_layers = [0,1,2,3,4,5,6,7] # [0,1,2,3,4,5]
height = 4 #height of the RP phase
depth = 2

#CsPbI3 building block
atoms_fu_eta = Atoms(symbols=["Cs", "Cs", "Cs", "Cs", "Pb", "I", "I", "I", "I", "I", "I"], 
                    positions=[
                        np.array([ l/np.sqrt(2), l, np.sqrt(2)*l]), 
                        np.array([ l/np.sqrt(2),-l, np.sqrt(2)*l]), 
                        np.array([ l/np.sqrt(2), l,-np.sqrt(2)*l]), 
                        np.array([ l/np.sqrt(2),-l,-np.sqrt(2)*l]),  
                        np.array([ l/np.sqrt(2), 0.0, 0.0]), 
                        np.array([ l*np.sqrt(2), 0.0,  l/np.sqrt(2)]),
                        np.array([ l*np.sqrt(2), 0.0, -l/np.sqrt(2)]), 
                        np.array([ l/np.sqrt(2), l, 0.0]), 
                        np.array([ l/np.sqrt(2),-l, 0.0]), 
                        np.array([ 0.0, 0.0,  l/np.sqrt(2)]), 
                        np.array([ 0.0, 0.0, -l/np.sqrt(2)])]) 

#Create the CsPbI3 perovskite structure
Pos_start = np.array([0.0, 0.0, 0.0])
for i in range(total_length):
    
    atoms_fu = atoms_fu_eta
    for at in atoms_fu.copy():
        at.position += Pos_start
        atoms_dup.append(at)

    if i in eta_layers:
        Pos_start += np.array([np.sqrt(2)*l, 0.0, 0.0])
    else:
        if Pos_start[2] > 0.0:
            Pos_start += np.array([np.sqrt(2)*l, 0.0 , -np.sqrt(2)*l])
        else:
            Pos_start += np.array([np.sqrt(2)*l, 0.0, np.sqrt(2)*l])

cell= np.array([Pos_start, [0.0, 2*l, 0.0], [0.0, 0.0, 2*np.sqrt(2)*l]])

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
#niggli_reduce(sup_atoms)

outname = "teststruc.cif"
write(outname, sup_atoms, format="cif")

#outname = 'gam_eta'
#for i in eta_layers:
#    outname += "_" + str(i)
#write(outname + '.xyz', sup_atoms, format="extxyz")
#write('eta_8_2_4.xyz', sup_atoms, format="extxyz")
'''