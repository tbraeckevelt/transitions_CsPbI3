'''
This script can construct the intermediate structures, but is not used anymore, Create_intermediate_FAdelta_bb.py is used instead
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
FA_layers = [0,1,2,3] # [0,1,2,3,4,5]
height = 4 #height of the RP phase
depth = 4

assert len(FA_layers) % 2 == 0, 'PBC restric to even number of FA layers'
assert height % 2 == 0, 'this is a specific implementation for even heights to create more ortorhombic cells'

gam = True

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

#Rotate atoms_fu -70.53 degrees around the z-axis
atoms_fu_eta_tilt = atoms_fu_eta.copy()
atoms_fu_eta_tilt.rotate(-70.53, 'z')

#Create the CsPbI3 perovskite structure
for j in range(2):
    if j == 0:
        Pos_start = np.array([0.0, 0.0, 0.0])
    elif j == 1:
        if gam:
            Pos_start = np.array([np.sqrt(2)*l, 2*l, np.sqrt(2)*l])
        else:
            Pos_start = np.array([np.sqrt(2)*l, 2*l, 0.0])
    tilt = -1
    for i in range(total_length):
        if i in FA_layers:
            tilt*=-1
        
        if tilt == -1:
            atoms_fu = atoms_fu_eta
        else:
            atoms_fu = atoms_fu_eta_tilt
        
        for at in atoms_fu.copy():
            at.position += Pos_start
            atoms_dup.append(at)

        if tilt == -1:
            Pos_start += np.array([np.sqrt(2)*l, 0.0, 0.0])
        else:
            Pos_start += np.array([np.sqrt(2)*l*np.cos(-70.53*np.pi/180), np.sqrt(2)*l*np.sin(-70.53*np.pi/180), 0.0])

        if gam and (i+1)%total_length not in FA_layers:
            if Pos_start[2] > 0.0:
                Pos_start -= np.array([0.0, 0.0, np.sqrt(2)*l])
            else:
                Pos_start += np.array([0.0, 0.0, np.sqrt(2)*l])


dotprod = 1
b_vec =  np.array([2*np.sqrt(2)*l, 4*l,  0.0])
c_vec = [0.0, 0.0, 2*np.sqrt(2)*l]

for shift_b in range(-total_length, total_length + 1):
    a_vec = Pos_start - shift_b*b_vec
    new_dotprod = np.abs(np.dot(a_vec/np.linalg.norm(a_vec), b_vec/np.linalg.norm(b_vec)))
    if new_dotprod < dotprod:
        dotprod = new_dotprod
        a_vec_min = a_vec
    
    if gam:
        a_vec = Pos_start - shift_b*b_vec - np.array([np.sqrt(2)*l, 2*l,  np.sqrt(2)*l])
    else:
        a_vec = Pos_start - shift_b*b_vec + np.array([np.sqrt(2)*l, 2*l,  0.0])
    new_dotprod = np.abs(np.dot(a_vec/np.linalg.norm(a_vec), b_vec/np.linalg.norm(b_vec)))
    c_dotprod = np.abs(np.dot(a_vec/np.linalg.norm(a_vec), c_vec/np.linalg.norm(c_vec)))
    if new_dotprod < dotprod and c_dotprod < dotprod:
        dotprod = new_dotprod
        a_vec_min = a_vec

#a_vec_min = Pos_start
cell= np.array([a_vec_min, b_vec, [0.0, 0.0, 2*np.sqrt(2)*l]])


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

sup_atoms = make_supercell(new_atoms, np.diag([1, height/2, depth]))
#niggli_reduce(sup_atoms)

outname = "teststruc.cif"
write(outname, sup_atoms, format="cif")

#outname = 'gam_eta'
#for i in eta_layers:
#    outname += "_" + str(i)
#write(outname + '.xyz', sup_atoms, format="extxyz")
#write('FAdel_4_4_4_gam.xyz', sup_atoms, format="extxyz")