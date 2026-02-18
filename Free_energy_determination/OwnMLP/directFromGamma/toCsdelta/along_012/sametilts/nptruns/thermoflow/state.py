from typing import Optional, Union, Any
from dataclasses import dataclass
import enum
import copy

import numpy as np
import typeguard
from ase.units import bar, kB
from psiflow.sampling import Walker, SimulationOutput
from psiflow.hamiltonians import Hamiltonian


from psiflow.geometry import (
    transform_lower_triangular,
    reduce_box_vectors,
)


def make_hamiltonian_name(hamiltonian: Hamiltonian):
    mixture = 1 * hamiltonian
    desc = []
    for c, h in zip(mixture.coefficients, mixture.hamiltonians):
        desc.append(f'{c:.4f} x {h.__class__.__name__}')
    return ' + '.join(desc)


def get_hamiltonian_names(hamiltonian: str) -> list[str]:
    """"""
    hamilnames = []
    for s in hamiltonian.split(' '):
        if len(s) > 3 and s.isalpha():
            count=0
            while s+str(count) in hamilnames:
                count+=1
            hamilnames.append(s+str(count))
    return hamilnames


def recover_lambda(lmd_inter, lmd_begin, lmd_end):
    lmd_diffs = lmd_end - lmd_begin
    mask = lmd_diffs != 0  # Only consider positions where list1 != list2
    lambdas = (lmd_inter[mask] - lmd_begin[mask]) / lmd_diffs[mask]
    assert np.allclose(lambdas, lambdas[0]), 'all interpolations should be the same'
    assert len(lambdas) > 0, 'atleas one lmd values should differ between begin and end'
    return lambdas[0]


@dataclass
class Metric:
    """"""
    data: np.ndarray = None
    mean: float = 0
    var: float = 0
    error: float = 0

    def __post_init__(self):
        if self.data is not None:
            self.update()

    @property
    def num_samples(self) -> int: return self.data.shape[0]

    def append(self, values: np.ndarray) -> None:
        assert len(values.shape) == 1
        if self.data is None:
            self.data = values
        else:
            self.data = np.concatenate([self.data, values])
        self.update()

    def update(self) -> None:
        """"""
        if len(self.data) == 0:
            self.mean, self.var = 0, 0
            self.error = 0
        else:
            self.mean, self.var = self.data.mean(), self.data.var()
            self.error = (self.var ** .5) / (self.num_samples ** .5)


class Ensemble(enum.Enum):
    NVT = 0
    NPT = 1


@dataclass(frozen=True)
class WalkerState:
    hamiltonian: str
    temperature: float
    natoms: int
    nbeads: int = 1
    masses: Optional[tuple] = None
    mass_scaling: Optional[float] = None
    pressure: Optional[float] = None
    cell: Optional[tuple] = None      # automatic change to lower triangular form and take the three distances and angles !!!

    def __post_init__(self):
        assert (self.pressure is None) != (self.cell is None)       # xor

    @property
    def lambda_params(self) -> list[float]:
        return [float(s) for s in self.hamiltonian.split(' ') if not s.isalnum() and s != '+']

    @property
    def ensemble(self) -> Ensemble:
        if self.pressure is not None:
            return Ensemble.NPT
        return Ensemble.NVT

    @classmethod
    def from_walker(cls, walker: Walker):
        mass_scaling = None
        if walker.masses is not None:
            ratios = walker.masses / walker.start.atomic_masses
            if np.allclose(ratios, ratios[0]):
                mass_scaling = ratios[0]
        
        state = cls(
            hamiltonian=make_hamiltonian_name(walker.hamiltonian),
            temperature=walker.temperature,
            pressure=walker.pressure,
            natoms=len(walker.start),
            nbeads=walker.nbeads,
            masses=tuple(walker.masses),
            mass_scaling = mass_scaling,
            cell=walker.start.cell if walker.pressure is None else None,
        )
        return state
    
    def __eq__(self, other: 'WalkerState') -> bool:
        if self.hamiltonian != other.hamiltonian:  # what about 0.0 hamiltonian parts? See also equal_hamiltonian method
            return False
        if (self.temperature, self.pressure) != (other.temperature, other.pressure):
            return False
        if self.natoms != other.natoms:
            return False
        if self.cell is not None and other.cell is not None and not np.allclose(self.cell, other.cell):
            return False
        return True
    
    def __hash__(self):
        items = []
        for v in self.__dict__.values():
            if isinstance(v, np.ndarray):
                items.append(tuple(v.flatten()))
            else:
                items.append(v)
        return hash(tuple(items))


@typeguard.typechecked
class ThermodynamicState:
    """"""
    def __init__(
        self,
        state: WalkerState
    ):
        self.state = state

        self.properties: dict[str, Metric] = {}
        self.gradients: dict[str, float] = {}
        self.gradients_error: dict[str, float] = {}

        self.properties['energy'] = Metric()
        if state.pressure is not None:
            self.properties['volume'] = Metric()
            self.properties['cell'] = {}
            for cell_comp in ['ax', 'by', 'cz', 'bx', 'cx', 'cy']:
                self.properties['cell'][cell_comp] = Metric()
        for contrib_name in self._get_hamiltonian_names():
            self.properties[contrib_name] = Metric()
        if state.masses is not None:
            self.properties['kinetic_cv'] = Metric()

        self.free_energy = None  #devided by kB*T for easy addition of the temperature TI contributions
        self.free_error = 0.0  #devided by kB*T for easy addition of the temperature TI contributions
    
    def __getattr__(self, item: str):
        state = object.__getattribute__(self, "state")
        return getattr(state, item)

    def _get_hamiltonian_names(self) -> list[str]:
        return get_hamiltonian_names(self.hamiltonian)

    def add_output(self, output: dict[str, np.ndarray]) -> None:
        """"""

        # TODO: why would this be a future?
        self.properties['energy'].append(output['potential{electronvolt}'])

        if self.pressure is not None:  # use enthalpy
            self.properties['volume'].append(output['volume{angstrom3}'])
            for i, cell_comp in enumerate(['ax', 'by', 'cz', 'bx', 'cx', 'cy']):
                complst = [cell[i] for cell in output['cell_h{angstrom}']]
                self.properties['cell'][cell_comp].append(np.array(complst))
        else:
            # check cell shapes
            cell_h_ref = self.cell[(0, 1, 2, 1, 2, 2), (0, 1, 2, 0, 0, 1)]      # xx yy zz xy xz yz
            assert all(np.allclose(cell_h, cell_h_ref) for cell_h in output['cell_h{angstrom}'])

        for idx, contrib_name in enumerate(self._get_hamiltonian_names()):
            self.properties[contrib_name].append(output[contrib_name])

        if self.mass_scaling is not None:
            self.properties['kinetic_cv'].append(output['kinetic_cv{electronvolt}'])

    def calc_lambda_gradient(self, coefficients: dict[str, float]) -> tuple[float, float]:
        """"""
        assert len(coefficients) == len(self.lambda_params)
        delta_e = np.zeros(self.properties['energy'].num_samples)
        for contrib_name, coefficient in coefficients.items():
            delta_e += coefficient * self.properties[contrib_name].data
        delta_e = Metric(data=delta_e)
        self.gradients['lambda'] = delta_e.mean / (kB * self.temperature)
        self.gradients_error['lambda'] = delta_e.error / (kB * self.temperature)
        return self.gradients['lambda'], self.gradients_error['lambda']

    def calc_temperature_gradient(self, dy_tem: bool = True)-> tuple[float, float, float]:
        """
        See DOI: 10.1103/PhysRevB.97.054102, if dy_tem = True, we calculate the gradient in y_tem = ln(temperature) (wtih T_0 = 0)"""
        u_mean = self.properties['energy'].mean
        u_error = self.properties['energy'].error
        if self.pressure is not None:       # use enthalpy
            u_mean += self.properties['volume'].mean * self.pressure * 10 * bar
            u_error += self.properties['volume'].error * self.pressure * 10 * bar

        # grad_u = < - u / kBT**2 >
        grad_u = - u_mean / (kB * self.temperature**2)
        # TODO: this is a theoretical contribution?
        # grad_k = < - E_kin > / kBT**2 >
        grad_k = (-1.0) * (3 * self.natoms - 3) / (2 * self.temperature)

        if dy_tem:
            y_tem = np.log(self.temperature)
            self.gradients['y_tem'] = (grad_u + grad_k) * self.temperature
            self.gradients_error['y_tem'] = u_error / (kB * self.temperature)
            return y_tem, self.gradients['y_tem'], self.gradients_error['y_tem']
        else:
            self.gradients['temperature'] = grad_u + grad_k
            self.gradients_error['temperature'] = u_error / (kB * self.temperature**2)
            return self.temperature, self.gradients['temperature'], self.gradients_error['temperature']
    
    def calc_pressure_gradient(self) -> tuple[float, float, float]:
        """"""
        assert self.pressure is not None, "Pressure must be given"
        v_mean = self.properties['volume'].mean
        v_error = self.properties['volume'].error
        self.gradients['pressure'] = v_mean  / (kB * self.temperature)
        self.gradients_error['pressure'] = v_error / (kB * self.temperature)
        return self.pressure * 10 * bar, self.gradients['pressure'], self.gradients_error['pressure']

    def calc_gibbs_correction(
        self,
        cell: np.ndarray,
        bin_size: float = 0.02,  # bin size in Angstrom^3 per atom
        window: int = 9,
    ) -> tuple[float, float]:
        from .psiflow_utils import running_average, calculate_hist
        """
        Calculates the normalized Gibbs correction term.
        G(P,T)/(k_B*T) = F(V,T)/(k_B*T) + P*V/(k_B*T) + log(prob(V|P,T))
        """
        # TODO: does this need the average cell or should it compute the average cell?
        assert self.pressure is not None, "Pressure must be given"
        # transform the cell to an equivalent unique form
        transform_lower_triangular(np.zeros((3, 3)), cell, reorder=True)
        reduce_box_vectors(cell)

        volumes = (self.properties['volume'].data)/self.natoms    # Should we normalize per atom?
        hist, center_bin = calculate_hist(volumes, bin_size)
        run_hist = running_average(hist, window)
        run_bin = running_average(center_bin, window)
        vol = np.linalg.det(cell)/self.natoms                     # Should we normalize per atom?
        min_diff = np.abs(vol - run_bin[0])
        hist_opt = run_hist[0]
        for hist_val, bin_val in zip(run_hist, run_bin):
            diff_vol = np.abs(vol - bin_val)
            if diff_vol <= min_diff:
                hist_opt = hist_val
                min_diff = diff_vol

        # TODO: why no error?
        self.gradients['cell'] = self.pressure * 10 * bar * vol * self.natoms / (kB * self.temperature) + np.log(hist_opt)
        self.gradients_error['cell'] = 0
        return self.gradients['cell'], self.gradients_error['cell']
    
    def calc_masses_gradient(self, dg_mass: bool = True)-> tuple[float, float, float]:
        """
        See SI of DOI: 10.1021/acs.jpclett.6b00777, if dg_mass = True, we calculate the gradient in 
        g_mass = sqrt(m0/m) = 1/sqrt(self.mass_scaling) and ignore the clasical term as that is equal for all phases
        """
        assert self.mass_scaling is not None, "masses must be given"
        kincv_mean = self.properties['kinetic_cv'].mean
        kincv_error = self.properties['kinetic_cv'].error
        if dg_mass:
            g_mass = 1/np.sqrt(self.mass_scaling)
            self.gradients['g_mass'] = 2* kincv_mean / (g_mass* kB * self.temperature)             # check sign of the correction!
            self.gradients_error['g_mass'] = 2* kincv_error / (g_mass* kB * self.temperature) 
            return g_mass, self.gradients['g_mass'], self.gradients_error['g_mass']
        else:
            self.gradients['mass_scaling'] = kincv_mean / (self.mass_scaling* kB * self.temperature)             # check sign of the correction!
            self.gradients_error['mass_scaling'] = kincv_error / (self.mass_scaling * kB * self.temperature) 
            return self.mass_scaling, self.gradients['mass_scaling'], self.gradients_error['mass_scaling']

    def equal_hamiltonian(self, other: 'ThermodynamicState') -> bool:
        hamiltonian_coeff = {}
        for hamilname, lmd_param  in zip(self._get_hamiltonian_names(), self.lambda_params):
            if lmd_param != 0.0:
                hamiltonian_coeff[hamilname] = lmd_param
        tel = 0
        for hamilname, lmd_param in zip(other._get_hamiltonian_names(), other.lambda_params):
            if lmd_param != 0.0:
                if lmd_param != hamiltonian_coeff[hamilname]:
                    return False
                tel +=1
        if tel == len(hamiltonian_coeff):
            return True
        else:
            return False

    def __eq__(self, other: 'ThermodynamicState') -> bool:
        if self.hamiltonian != other.hamiltonian:  # what about 0.0 hamiltonian parts? See also equal_hamiltonian method
            return False
        if (self.temperature, self.pressure) != (other.temperature, other.pressure):
            return False
        if self.natoms != other.natoms:
            return False
        if self.cell is not None and other.cell is not None and not np.allclose(self.cell, other.cell):
            return False
        return True


def combine_outputs(outputs: list[dict]) -> list[ThermodynamicState]:
    """"""
    sorted_states = {}
    for output in outputs:
        state = output.pop('state')
        if state not in sorted_states:
            sorted_states[state] = ThermodynamicState(state)
        sorted_states[state].add_output(output)

    return list(sorted_states.values())


def calc_ti_lambda(states: list[ThermodynamicState], state_begin: ThermodynamicState, state_end: ThermodynamicState) -> ThermodynamicState:

    temperature = state_begin.temperature
    assert temperature == state_end.temperature, "temperature of all states should be the same"
    pressure = state_begin.pressure
    assert pressure == state_end.pressure, "pressure of all states should be the same"
    cell = state_begin.cell
    if pressure is None:
        assert np.allclose(cell, state_end.cell), "cell of all states should be the same"   
    natoms = state_begin.natoms
    assert state_end.natoms == natoms, "the number of atoms of all states should be the same"
    hamilnames = state_begin._get_hamiltonian_names()
    assert state_end._get_hamiltonian_names() == hamilnames, "the mixed hamiltonians of all states should be the same"
    lmd_begin = np.array(state_begin.lambda_params)
    lmd_end = np.array(state_end.lambda_params)
    lmd_diffs = lmd_end - lmd_begin
    coeffs = {}
    for hamilname, lmd in zip(hamilnames, lmd_diffs):
        coeffs[hamilname] = lmd    
    #mass_scaling = ref_state.mass_scaling

    sorted_states = sorted(states, key=lambda state: recover_lambda(np.array(state.lambda_params), lmd_begin, lmd_end))
    p_state = copy.deepcopy(state_begin)
    for i, state in enumerate(sorted_states):
        assert state.temperature == temperature, "temperature of all states should be the same"
        assert state.pressure == pressure, "pressure of all states should be the same"
        if pressure is None:
            assert np.allclose(cell, state.cell), "cell of all states should be the same"    
        assert state._get_hamiltonian_names() == hamilnames, "the mixed hamiltonians of all states should be the same"
        assert state.natoms == natoms, "the number of atoms of all states should be the same"
        #assert mass_scaling == state.mass_scaling, "the mass rescaling for pimd of all states should be the same"
        
        val = recover_lambda(np.array(state.lambda_params), lmd_begin, lmd_end)
        grad, grad_error = state.calc_lambda_gradient(coefficients= coeffs)

        if i == 0:
            weight = copy.copy(val) # should be 0.0
            state.free_energy = p_state.free_energy + weight * grad
            state.free_error = np.sqrt(p_state.free_error**2 + weight**2 * grad_error**2)
        else:
            weight = ( val - p_val ) / 2
            state.free_energy = p_state.free_energy + weight * (p_grad + grad) 
            p_error2 = p_state.free_error**2 - p_weight**2 * p_grad_error**2
            state.free_error = np.sqrt(p_error2 + (weight+p_weight)**2 * p_grad_error**2 + weight**2 * grad_error**2)

        p_weight = copy.copy(weight)
        p_val = copy.copy(val)
        p_state = copy.deepcopy(state)
        p_grad = copy.copy(grad)
        p_grad_error = copy.copy(grad_error)

    if p_val < 1.0: # if lambda = 1 is not included, extrapolate to 1, due to assert statements in ti.py, this should not occur
        print('WARNING the last lambda value is not equal to 1, extrapolate to 1, but this will probably lead to wrong results!')
        walkerstate_end = WalkerState(
            hamiltonian = state_end.hamiltonian,  # This will probably not be correct!!!!!
            temperature = temperature, 
            natoms = natoms, 
            cell = cell
        )
        state_end = ThermodynamicState(state = walkerstate_end)
        weight = 1.0 - p_val
        state_end.free_energy = p_state.free_energy + weight * grad
        p_error2 = p_state.free_error**2 - p_weight**2 * grad_error**2
        state_end.free_error = np.sqrt(p_error2 + (weight+p_weight)**2 * grad_error**2)
        return state_end
    else:
        return state 


def calc_ti_md(states: list[ThermodynamicState], ref_state: ThermodynamicState, E_opt: Optional[float] = None) -> None:
    
    #decide if temperature path or pressure path
    path = None
    if all(state.temperature == ref_state.temperature for state in states):
        path= 'pressure'
        for state in states:
            assert state.pressure is not None
    elif ref_state.pressure is None:
        if all(np.allclose(state.cell, ref_state.cell) for state in states):
            path = 'temperature'
    else:
        if all(state.pressure == ref_state.pressure for state in states):
            path = 'temperature'
            if E_opt is not None:
                E_opt += ref_state.pressure * 10 * bar * np.linalg.det(ref_state.cell)
    assert path is not None, "No temperature or pressure path is found check states and ref_state"

    natoms = ref_state.natoms
    #mass_scaling = ref_state.mass_scaling

    sorted_states = sorted(states, key=lambda state: getattr(state, path))
    below_ref = -1
    ref_val, ref_grad, ref_grad_error = getattr(ref_state, f"calc_{path}_gradient")()
    if path == 'temperature' and E_opt is not None:   # Use analytical formula to correct the discretization error for ground state energy
        ref_grad += E_opt/(kB*ref_state.temperature )
    p_state = None
    for i, state in enumerate(sorted_states):
        assert state.equal_hamiltonian(ref_state), "the hamiltonians of all states should be the same"
        assert state.natoms == natoms, "the number of atoms of all states should be the same"
        #assert mass_scaling == state.mass_scaling, "the mass rescaling for pimd of all states should be the same"

        val, grad, grad_error = getattr(state, f"calc_{path}_gradient")()
        if path == 'temperature' and E_opt is not None:   # Use analytical formula to correct the discretization error for ground state energy
            grad += E_opt/(kB*state.temperature )
        if p_state is None:
            if ref_val == val: # we assume both states are the same
                weight = 0.0
                state.free_energy = copy.copy(ref_state.free_energy)
                state.free_error = copy.copy(ref_state.free_error)
            elif ref_val < val:
                weight = (val - ref_val)/2
                state.free_energy = ref_state.free_energy + weight * (ref_grad + grad) 
                if path == 'temperature' and E_opt is not None:   # Use analytical formula to correct the discretization error for ground state energy
                    state.free_energy += E_opt/(kB*state.temperature ) - E_opt/(kB*ref_state.temperature )
                state.free_error = np.sqrt(ref_state.free_error**2 + weight**2 * (ref_grad_error**2 + grad_error**2))
            else:
                below_ref = i
                continue
        else:
            weight = (val - p_val)/2
            state.free_energy = p_state.free_energy + weight * (p_grad + grad) 
            if path == 'temperature' and E_opt is not None:   # Use analytical formula to correct the discretization error for ground state energy
                state.free_energy += E_opt/(kB*state.temperature ) - E_opt/(kB*p_state.temperature )
            p_error2 = p_state.free_error**2 - p_weight**2 * p_grad_error**2
            state.free_error = np.sqrt(p_error2 + (weight+p_weight)**2 * p_grad_error**2 + weight**2 * grad_error**2)

        p_weight = copy.copy(weight)
        p_val = copy.copy(val)
        p_state = copy.deepcopy(state)
        p_grad = copy.copy(grad)
        p_grad_error = copy.copy(grad_error)

    # Go backwards to fill in the free energies for the values below the reference values
    p_weight = 0
    p_val = copy.copy(ref_val)
    p_state = copy.deepcopy(ref_state)
    p_grad = copy.copy(ref_grad)
    p_grad_error = copy.copy(ref_grad_error)
    for i in range(below_ref, -1, -1):
        state = sorted_states[i]
        val, grad, grad_error = getattr(state, f"calc_{path}_gradient")()
        if path == 'temperature' and E_opt is not None:   # Use analytical formula to correct the discretization error for ground state energy
            grad += E_opt/(kB*state.temperature )
        weight = (val - p_val)/2
        state.free_energy = p_state.free_energy + weight * (p_grad + grad)
        if path == 'temperature' and E_opt is not None:   # Use analytical formula to correct the discretization error for ground state energy
            state.free_energy += E_opt/(kB*state.temperature ) - E_opt/(kB*p_state.temperature )
        p_error2 = p_state.free_error**2 - p_weight**2 * p_grad_error**2
        state.free_error = np.sqrt(p_error2 + (weight+p_weight)**2 * p_grad_error**2 + weight**2 * grad_error**2)

        p_weight = copy.copy(weight)
        p_val = copy.copy(val)
        p_state = copy.deepcopy(state)
        p_grad = copy.copy(grad)
        p_grad_error = copy.copy(grad_error)



def calc_ti_pimd(states: list[ThermodynamicState], ref_state: ThermodynamicState) -> ThermodynamicState:
    temperature = ref_state.temperature
    pressure = ref_state.pressure
    cell = ref_state.cell
    natoms = ref_state.natoms
    hamiltonian = ref_state.hamiltonian

    sorted_states = sorted(states, key=lambda state: state.mass_scaling, reverse=True)
    p_state = copy.deepcopy(ref_state)
    for i, state in enumerate(sorted_states):
        assert temperature == state.temperature, "temperature of all states should be the same"
        assert state.pressure == pressure, "pressure of all states should be the same"
        if pressure is None:
            assert np.allclose(cell, state.cell), "cell of all states should be the same"    
        assert state.hamiltonian == hamiltonian, "the hamiltonians of all states should be the same"
        assert state.natoms == natoms, "the number of atoms of all states should be the same"

        val, grad, grad_error = state.calc_masses_gradient()
        if i == 0:
            weight = copy.copy(val)
            state.free_energy = p_state.free_energy + weight * grad
            state.free_error = np.sqrt(p_state.free_error**2 + weight**2 * grad_error**2)
        else:
            weight = ( val - p_val ) / 2
            state.free_energy = p_state.free_energy + weight * (p_grad + grad) 
            p_error2 = p_state.free_error**2 - p_weight**2 * p_grad_error**2
            state.free_error = np.sqrt(p_error2 + (weight+p_weight)**2 * p_grad_error**2 + weight**2 * grad_error**2)

        p_weight = copy.copy(weight)
        p_val = copy.copy(val)
        p_state = copy.deepcopy(state)
        p_grad = copy.copy(grad)
        p_grad_error = copy.copy(grad_error)


    if p_val < 1.0: # if g_mass = 1 is not included, extrapolate to 1
        walkerstate_end = WalkerState(
            hamiltonian = hamiltonian, 
            temperature = temperature, 
            pressure = pressure,
            natoms = natoms, 
            cell = cell
        )
        state_end = ThermodynamicState(state = walkerstate_end)
        weight = 1.0 - p_val
        state_end.free_energy = p_state.free_energy + weight * grad
        p_error2 = p_state.free_error**2 - p_weight**2 * grad_error**2
        state_end.free_error = np.sqrt(p_error2 + (weight+p_weight)**2 * grad_error**2)
        return state_end
    else:
        return state 