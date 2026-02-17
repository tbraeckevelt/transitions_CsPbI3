from ase.io import read, write
from ase import Atom, Atoms
import numpy as np
from scipy.spatial.transform import Rotation as R


def shift_and_rotate(atoms, center= np.array([0.0, 0.0, 0.0]), rot_degree = None, rot_axis = 'x'):
    for at in atoms:
            at.position -= np.array(center)
    if rot_degree is not None:
        atoms.rotate(rot_degree, rot_axis)
    return atoms


class CsPbI3:

    def __init__(self, l = 3.2):
        """
        l : float, optional
            The distance from the center to the corners/axes (half the edge length 
            of the octahedron). Default is 3.2.
        """
        self.atoms = Atoms()
        self.l = l

    def get_corner_octahedra(self, center= np.array([0.0, 0.0, 0.0]), rot_degree = None, rot_axis = 'x'):
        """
        Generate a corner-sharing CsPbI3 octahedral building block as an Atoms object.
        This method creates a single CsPbI3 octahedron with the Pb atom at the center, 
        Cs atoms at the corners, and I atoms along the axes. The octahedron can be 
        shifted to a given center position and rotated around a specific axis.
        Parameters
        ----------
        center : np.array, optional
            The coordinates [x, y, z] to which the octahedron will be shifted. 
            Default is np.array([0.0, 0.0, 0.0]).
        rot_degree : float, optional
            The angle in degrees to rotate the octahedron. Default is 0.
        rot_axis : str, optional
            The axis ('x', 'y', or 'z') around which to rotate the octahedron. 
            Default is 'x'.
        Returns
        -------
        atoms__corner : Atoms
            An Atoms object representing the corner-sharing CsPbI3 octahedral building block.
        """

        # Returns a octahedra CsPbI3 building block as an Atoms object with I atoms along the axis 
        atoms_corner = Atoms(
            symbols=["Cs"] * 8 + ["Pb"] * 1 + ["I"] * 6, 
            positions=[
                np.array([ self.l, self.l, self.l]), 
                np.array([-self.l, self.l, self.l]), 
                np.array([ self.l,-self.l, self.l]), 
                np.array([-self.l,-self.l, self.l]), 
                np.array([ self.l, self.l,-self.l]), 
                np.array([-self.l, self.l,-self.l]), 
                np.array([ self.l,-self.l,-self.l]), 
                np.array([-self.l,-self.l,-self.l]), 
                np.array([0.0, 0.0, 0.0]), 
                np.array([self.l, 0.0, 0.0]), 
                np.array([-self.l, 0.0, 0.0]), 
                np.array([0.0, self.l, 0.0]), 
                np.array([0.0,-self.l, 0.0]), 
                np.array([0.0, 0.0, self.l]), 
                np.array([0.0, 0.0,-self.l])])
        return shift_and_rotate(atoms_corner, center=center, rot_degree=rot_degree, rot_axis=rot_axis)

        
    def get_edge_octahedra(self, center= np.array([0.0, 0.0, 0.0]), rot_degree=None, rot_axis='x'):
        """
        Same as corner octahedra but rotated around y axis such that edges could connect along x or z-axis
        """
        
        #CsPbI3 building block
        atoms_edge = Atoms(
            symbols=["Cs"] * 4 + ["Pb"] * 1 + ["I"] * 6, 
            positions=[
                #np.array([ np.sqrt(2)*self.l, self.l, 0]), 
                #np.array([ np.sqrt(2)*self.l,-self.l, 0]), 
                #np.array([-np.sqrt(2)*self.l, self.l, 0]), 
                #np.array([-np.sqrt(2)*self.l,-self.l, 0]), 
                np.array([ 0, self.l, np.sqrt(2)*self.l]), 
                np.array([ 0,-self.l, np.sqrt(2)*self.l]), 
                np.array([ 0, self.l,-np.sqrt(2)*self.l]), 
                np.array([ 0,-self.l,-np.sqrt(2)*self.l]), 
                np.array([0.0, 0.0, 0.0]), 
                np.array([ self.l/np.sqrt(2), 0.0,  self.l/np.sqrt(2)]),
                np.array([ self.l/np.sqrt(2), 0.0, -self.l/np.sqrt(2)]), 
                np.array([ 0.0, self.l, 0.0]), 
                np.array([ 0.0,-self.l, 0.0]), 
                np.array([-self.l/np.sqrt(2), 0.0,  self.l/np.sqrt(2)]), 
                np.array([-self.l/np.sqrt(2), 0.0, -self.l/np.sqrt(2)])])  
        return shift_and_rotate(atoms_edge, center=center, rot_degree=rot_degree, rot_axis=rot_axis)
    
    def get_edge_octahedra_full(self, center= np.array([0.0, 0.0, 0.0]), rot_degree=None, rot_axis='x'):
        """
        Same as corner octahedra but rotated around y axis such that edges could connect along x or z-axis
        """
        
        #CsPbI3 building block
        atoms_edge = Atoms(
            symbols=["Cs"] * 8 + ["Pb"] * 1 + ["I"] * 6, 
            positions=[
                np.array([ np.sqrt(2)*self.l, self.l, 0]), 
                np.array([ np.sqrt(2)*self.l,-self.l, 0]), 
                np.array([-np.sqrt(2)*self.l, self.l, 0]), 
                np.array([-np.sqrt(2)*self.l,-self.l, 0]), 
                np.array([ 0, self.l, np.sqrt(2)*self.l]), 
                np.array([ 0,-self.l, np.sqrt(2)*self.l]), 
                np.array([ 0, self.l,-np.sqrt(2)*self.l]), 
                np.array([ 0,-self.l,-np.sqrt(2)*self.l]), 
                np.array([0.0, 0.0, 0.0]), 
                np.array([ self.l/np.sqrt(2), 0.0,  self.l/np.sqrt(2)]),
                np.array([ self.l/np.sqrt(2), 0.0, -self.l/np.sqrt(2)]), 
                np.array([ 0.0, self.l, 0.0]), 
                np.array([ 0.0,-self.l, 0.0]), 
                np.array([-self.l/np.sqrt(2), 0.0,  self.l/np.sqrt(2)]), 
                np.array([-self.l/np.sqrt(2), 0.0, -self.l/np.sqrt(2)])])  
        return shift_and_rotate(atoms_edge, center=center, rot_degree=rot_degree, rot_axis=rot_axis)
    
    def get_edge_octahedra_reduced(self, center= np.array([0.0, 0.0, 0.0]), rot_degree=None, rot_axis='x'):
        """
        Same as corner octahedra but rotated around y axis such that edges could connect along x or z-axis
        """
        
        #CsPbI3 building block
        atoms_edge = Atoms(
            symbols=["Cs", "Pb", "I", "I", "I", "I"], 
            positions=[
                np.array([ 0, -self.l, -np.sqrt(2)*self.l]), 
                np.array([0.0, 0.0, 0.0]), 
                np.array([ 0.0, self.l, 0.0]), 
                np.array([ 0.0,-self.l, 0.0]), 
                np.array([-self.l/np.sqrt(2), 0.0,  self.l/np.sqrt(2)]), 
                np.array([-self.l/np.sqrt(2), 0.0, -self.l/np.sqrt(2)])])  
        return shift_and_rotate(atoms_edge, center=center, rot_degree=rot_degree, rot_axis=rot_axis)
    
    def get_face_octahedra(self, center= np.array([0.0, 0.0, 0.0]), rot_degree=None, rot_axis='x'):
        """
        Same as edge octahedra but rotated around z axis such that face could connect along x-axis
        """

        atoms_face = self.get_edge_octahedra()
        atoms_face.rotate(-70.5287794, 'z')
        return shift_and_rotate(atoms_face, center=center, rot_degree=rot_degree, rot_axis=rot_axis)
    
    def get_Csdelta_part(self, center= np.array([0.0, 0.0, 0.0]), rot_degree=None, rot_axis='x', mirror = False):
        """
        Same as other get_octahedra functions but for a specific Csdelta structure
        """

        atoms_del = Atoms(
            symbols=["Cs"] * 1 + ["Pb"] * 1 + ["I"] * 4, 
            positions=[
                np.array([ 0, 2*self.l, np.sqrt(2)*self.l]), 
                np.array([0.0, 1.15*self.l, 0.0]), 
                np.array([ 0.0, 2.34*self.l, 0.0]), 
                np.array([ 0.0,0, 0.0]), 
                np.array([-self.l/np.sqrt(2), 1.05*self.l,  self.l/np.sqrt(2)]), 
                np.array([-self.l/np.sqrt(2), 1.05*self.l, -self.l/np.sqrt(2)])]) 
        if mirror:
            atoms_del.positions *= -1
        return shift_and_rotate(atoms_del, center=center, rot_degree=rot_degree, rot_axis=rot_axis)
    
    def get_Csdelta_octahedras(self, center= np.array([0.0, 0.0, 0.0]), rot_degree=None, rot_axis='x', mirror = False, mirror_z = False):
        """
        Same as other get_octahedra functions but for a specific Csdelta structure
        """

        atoms_del1 = self.get_edge_octahedra_reduced()
        atoms_del2 = self.get_edge_octahedra_reduced(center= np.array([self.l/np.sqrt(2), -self.l, self.l/np.sqrt(2)]))
        cspbI3 = CsPbI3()
        cspbI3.atoms = atoms_del1
        for at in atoms_del2:
            cspbI3.atoms.append(at)
        cspbI3.remove_duplicate_atoms()
        atoms_del = cspbI3.atoms
        if mirror:
            atoms_del.positions *= -1
        if mirror_z:
            for at in atoms_del:
                at.position[2] *= -1
        return shift_and_rotate(atoms_del, center=center, rot_degree=rot_degree, rot_axis=rot_axis)


    def add_octahedra(self, trans = np.array([0.0, 0.0, 0.0]), z_shift = 1, rot = False, Cs_scheme = 0, ontop = False):
        """
        Adds an octahedral building block to the current structure.
        Parameters:
            trans (np.ndarray, optional):   Translation vector to apply to the new octahedra's atomic positions. 
                                            Defaults to np.array([0.0, 0.0, 0.0]).
            z_shift (float, optional): Factor to shift the octahedra along the z-axis. Defaults to 1.
            rot (int or bool, optional): If -1, applies a specific rotation for face octahedra placement; otherwise, no rotation. Defaults to False.
            Cs_scheme (int, optional): If -1, omits Cs atoms when adding the octahedra. Defaults to 0.
        Returns:
            np.ndarray: The translation vector needed to add a new octahedra.
        """
        Rotation = R.from_euler('z', -70.5287794, degrees=True)
        if ontop:
            trans += Rotation.apply([-np.sqrt(2)*self.l, 2*self.l, 0.0])
        if rot == -1:
            trans_rot = Rotation.apply([-self.l/np.sqrt(2), 0.0,  0.0])
            new_oct = self.get_face_octahedra(center = trans_rot+ np.array([ 0.0, 0.0, -z_shift*self.l/np.sqrt(2)]))
            new_trans = trans-2*(trans_rot)
        else:
            new_oct = self.get_edge_octahedra(center = np.array([-self.l/np.sqrt(2), 0.0, -z_shift*self.l/np.sqrt(2)]))
            new_trans = trans+np.array([self.l*np.sqrt(2), 0.0,  0.0])
        for at in new_oct:
            if Cs_scheme == -1 and at.symbol == 'Cs':
                continue 
            at.position += trans
            self.atoms.append(at)
        #self.remove_duplicate_atoms()
        return new_trans, rot
    
    def add_cell(self, cell):
        self.atoms = Atoms(numbers=self.atoms.numbers, positions=self.atoms.positions, pbc=True, cell=cell)
        self.remove_duplicate_atoms()

    def remove_duplicate_atoms(self):
        new_atoms = Atoms(pbc = self.atoms.pbc, cell = self.atoms.cell)
        for at in self.atoms:
            new_atoms.append(at)
            for j, at2 in enumerate(new_atoms):
                if j == len(new_atoms) -1:
                    break
                dist = new_atoms.get_distance(j,-1, mic= True)
                if dist < 0.1:
                    if at.symbol != at2.symbol:
                        print("Different symbol at same position("+str(at.position)+"):" + at.symbol +" and " + at2.symbol)
                        at2.symbol = "I"
                    new_atoms.pop(i=-1)
                    break
        self.atoms = new_atoms

    def print_charge(self):
        symbols = [at.symbol for at in self.atoms]
        unique_symbols = set(symbols)
        assert unique_symbols.issubset({"Cs", "Pb", "I"}), f"Unexpected atom symbols found: {unique_symbols - {'Cs', 'Pb', 'I'}}"
        count_dict = {}
        for sym in unique_symbols:
            count_dict[sym] = symbols.count(sym)
            print(f"{sym}: {count_dict[sym]}")
        # Calculate net charge using Cs = +1, Pb = +2, I = -1
        charge_dict = {"Cs": 1, "Pb": 2, "I": -1}
        net_charge = sum(count_dict[sym] * charge_dict[sym] for sym in unique_symbols)
        print("net charge (Cs=+1, Pb=+2, I=-1):", net_charge)

