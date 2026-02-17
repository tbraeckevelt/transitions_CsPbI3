'''
This script can construct the intermediate structures, but is not used anymore, Create_intermediate_Csdelta_bb.py is used instead
'''

from ase.io import read, write
from ase import Atom, Atoms
import numpy as np
from ase.build import niggli_reduce
from ase.build.supercells import make_supercell
from scipy.spatial.transform import Rotation as R

atoms = Atoms()
atoms_dup = Atoms()

atoms_eta = read("zeta.xyz")  # Should be a ? x 4 x 4 zeta supercell with the layers not shifted, that shift is performed in this script
l = 3.2 # Pb-I distance
coords = [] #[0,0], [0,2], [1,0], [1,2], [2,0], [2,2], [3,0], [3,2]] # coordinates of the Pb atoms in the eta phase

yco = []
zco = []
for at in atoms_eta:
    if at.symbol == "Pb":
        yco.append(at.position[1])
        zco.append(at.position[2])

unique_yco = np.unique(yco)
unique_zco = np.unique(zco)
odd_zco = np.sort(unique_zco)[1::2]
print("Odd z-coordinates: ", odd_zco)

assert len(unique_yco) == 4
assert len(unique_zco) == 4
print("Unique y-coordinates: ", unique_yco)
print("Unique z-coordinates: ", unique_zco)



#shift the odd layers in x direction with l
shifted_at = []
for at in atoms_eta:
    if at.symbol == "Pb":
        if at.position[2] in odd_zco:
            for at2 in atoms_eta:
                if at2.symbol == "I":
                    dist = atoms_eta.get_distance(at.index, at2.index)
                    if dist <= l+0.1 and at2.index not in shifted_at:
                        at2.position += np.array([l/np.sqrt(2), 0.0, 0.0])
                        shifted_at.append(at2.index)
            at.position += np.array([l/np.sqrt(2), 0.0, 0.0])
            shifted_at.append(at.index)

for at in atoms_eta:
    if at.symbol == "Cs":
        at.position += np.array([l/np.sqrt(8) * (-1)**int(at.position[1]/(2*l)), 0.0, 0.0])

#Make sure all I atoms are in the same cell as the Pb atoms
for at in atoms_eta:
    if at.symbol == "I":
        for at2 in atoms_eta:
            if at2.symbol == "Pb":
                dist = atoms_eta.get_distance(at.index, at2.index)
                dist_mic = atoms_eta.get_distance(at.index, at2.index, mic=True)
                if dist_mic <= l+0.1 and dist > l+0.1:
                    dist_mic_vec = atoms_eta.get_distance(at2.index, at.index, mic=True, vector=True)
                    at.position = at2.position + dist_mic_vec

for coord in coords:
    coord_Pb1 = np.array([unique_yco[coord[0]], unique_zco[coord[1]]])
    coord_Pb2 = np.array([unique_yco[coord[0]], unique_zco[(coord[1]+(-1)**coord[0])%len(unique_zco)]])
    print("coord_Pb1: ", coord_Pb1)
    print("coord_Pb2: ", coord_Pb2)

    Pb1_atoms = []
    Pb2_atoms = []
    Pb1_indices = []
    Pb2_indices = []
    for at in atoms_eta:
        if at.symbol == "Pb":
            if at.position[1] == coord_Pb1[0]:
                if at.position[2] == coord_Pb1[1]:
                    Pb1_atoms.append(at.index)
                    Pb1_indices.append(at.index)
                    for at2 in atoms_eta:
                        if at2.symbol == "I" and np.abs(at2.position[1] - coord_Pb1[0]) < 0.1:
                            dist = atoms_eta.get_distance(at.index, at2.index)
                            if dist <= l+0.1 and at2.index not in Pb1_atoms:
                                Pb1_atoms.append(at2.index)
                elif at.position[2] == coord_Pb2[1]:
                    Pb2_atoms.append(at.index)
                    Pb2_indices.append(at.index)
                    for at2 in atoms_eta:
                        if at2.symbol == "I" and np.abs(at2.position[1] - coord_Pb1[0]) < 0.1:
                            dist = atoms_eta.get_distance(at.index, at2.index)
                            if dist <= l+0.1 and at2.index not in Pb2_atoms:
                                Pb2_atoms.append(at2.index)


    Rotation = R.from_euler('x', 66.51*(-1)**coord[0], degrees=True)

    for at in atoms_eta:
        if at.index in Pb1_atoms:
            #translate 
            if at.symbol == "Pb":
                at.position -= np.array([0.0, coord_Pb1[0]-l*1.15, coord_Pb1[1]])
            else:
                at.position -= np.array([0.0, coord_Pb1[0]-l*1.05, coord_Pb1[1]])
            #rotate around x axis 
            at.position = Rotation.apply(at.position)
            #translate back
            at.position += np.array([0.0, coord_Pb1[0]-l, coord_Pb1[1]])
        elif at.index in Pb2_atoms:
            #translate 
            if at.symbol == "Pb":
                at.position -= np.array([0.0, coord_Pb2[0]+l*1.15, coord_Pb2[1]])
            else:
                at.position -= np.array([0.0, coord_Pb2[0]+l*1.05, coord_Pb2[1]])
            #rotate around x axis 
            at.position = Rotation.apply(at.position)
            #translate back
            at.position += np.array([0.0, coord_Pb2[0]+l, coord_Pb2[1]])

    for at in atoms_eta:
        if at.symbol == "Cs":
            for Pb1_index in Pb1_indices:
                dist = atoms_eta.get_distance(at.index, Pb1_index, mic=True)
                if dist <= l:
                    at.position += np.array([l/np.sqrt(8), 0.0, 0.8*l/np.sqrt(2)])*(-1)**coord[0]
            for Pb2_index in Pb2_indices:
                dist = atoms_eta.get_distance(at.index, Pb2_index, mic=True)
                if dist <= l:
                    at.position += np.array([l/np.sqrt(8), 0.0, -0.8*l/np.sqrt(2)])*(-1)**coord[0]

for at in atoms_eta:
    if at.symbol == "Cs":
        flag = True
        for at2 in atoms_eta:
            if at2.symbol == "I":
                dist = atoms_eta.get_distance(at.index, at2.index, mic=True)
                if dist < l:
                    flag = False
                    break
        if not flag:
            at.position += np.array([0.0, l/np.sqrt(2), 0.0])
            for at2 in atoms_eta:
                if at2.symbol == "I":
                    dist = atoms_eta.get_distance(at.index, at2.index, mic=True)
                    if dist < 0.6*l:
                        flag = True
                        break
            if flag:
                at.position -= np.array([0.0, np.sqrt(2)*l, 0.0])
                for at2 in atoms_eta:
                    if at2.symbol == "I":
                        dist = atoms_eta.get_distance(at.index, at2.index, mic=True)
                        if dist < 0.6*l:
                            at.position += np.array([0.0, l/np.sqrt(2), 0.0])
                            break

outname = "teststruc.cif"
write(outname, atoms_eta, format="cif")


#outname = 'eta_Csdel'
#for coord in coords:
#    outname += "_" + str(coord[0]) + str(coord[1])
#write(outname + '.xyz', atoms_eta, format="extxyz")
#write('Csdel_4_4_4.xyz', atoms_eta, format="extxyz")

