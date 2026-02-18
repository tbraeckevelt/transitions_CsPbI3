import matplotlib.pyplot as plt
import ase
import typeguard
from typing import Any, Union, Tuple
import numpy as np
import sys
from pathlib import Path
import psiflow
import yaml
from ase.io import read
from ase.units import bar, kB
import pickle
from psiflow.free_energy.phonons import _compute_frequencies, second
from .state import ThermodynamicState, recover_lambda
from .psiflow_utils import PATHLIKE, running_average, calculate_hist
from .workflow import TaskEntry


@typeguard.typechecked
def print_hist_npt(state: ThermodynamicState, pathdir: PATHLIKE, bin_size: float = 0.01, window: int = 9):
    cells = []
    cell_dct = state.properties['cell']
    for i in range(len(cell_dct['ax'].data)):
        (ax, by, cz, bx, cx, cy) = (cell_dct[cell_comp].data[i] for cell_comp in ['ax', 'by', 'cz', 'bx', 'cx', 'cy'])
        cells.append(np.array([[ax, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz]])/(state.natoms)**(1.0/3.0))
    ave_cell = np.mean(cells, axis=0)

    vol_lst = [np.linalg.det(cell) for cell in cells]
    hist, center_bin = calculate_hist(vol_lst, bin_size)
    run_hist  = running_average(hist, window)
    run_bin   = running_average(center_bin, window)
    ave_vol = np.linalg.det(ave_cell)
    plt.figure()
    plt.plot(center_bin, hist, label = str(bin_size))
    plt.plot(run_bin, run_hist, label = "window="+str(window))
    plt.axvline(x=ave_vol, color='k', linestyle='dashed')
    plt.xlabel("volume per atom [A^3]")
    plt.ylabel("density")
    plt.legend()
    plt.title(str(np.round(state.temperature)) + " K, " + str(np.round(state.pressure, 1)) + " MPa")
    plt.savefig(pathdir / f'hist_vol_T{round(state.temperature)}_P{round(state.pressure, 1)}.pdf', bbox_inches='tight')
    plt.close()

    lengths_lst = [ase.cell.Cell(cell).lengths() for cell in cells]
    lengths_lst = np.array(lengths_lst).T  # Transpose to get lengths for each axis
    labels = ['a', 'b', 'c']
    ave_length = ase.cell.Cell(ave_cell).lengths()
    
    colors = ["tab:blue", "tab:orange", "tab:green"]
    plt.figure()
    for i, lengths in enumerate(lengths_lst):
        hist, center_bin = calculate_hist(lengths, bin_size/5)
        run_hist = running_average(hist, window)
        run_bin = running_average(center_bin, window)
        
        plt.plot(center_bin, hist, color=colors[i], label=labels[i]+"_"+str(bin_size/5))
        plt.plot(run_bin, run_hist, color=colors[i], linestyle='dashed', label=labels[i]+"_"+"window=" + str(window))
        plt.axvline(x=ave_length[i], color=colors[i])
    plt.xlabel("normalized length [A]")
    plt.ylabel("density")
    plt.legend()
    plt.title(str(np.round(state.temperature)) + " K, " + str(np.round(state.pressure, 1)) + " MPa")
    plt.savefig(pathdir / f"hist_length_T{round(state.temperature)}_P{round(state.pressure, 1)}.pdf", bbox_inches='tight')
    plt.close()

    angles_lst = [ase.cell.Cell(cell).angles() for cell in cells]
    angles_lst = np.array(angles_lst).T  # Transpose to get angles for each axis
    labels = ['alpha', 'beta', 'gamma']
    ave_angle = ase.cell.Cell(ave_cell).angles()
    
    colors = ["tab:blue", "tab:orange", "tab:green"]
    plt.figure()
    for i, angles in enumerate(angles_lst):
        hist, center_bin = calculate_hist(angles, bin_size*5)
        run_hist = running_average(hist, window)
        run_bin = running_average(center_bin, window)
        
        plt.plot(center_bin, hist, color=colors[i], label=labels[i]+"_"+str(bin_size*5))
        plt.plot(run_bin, run_hist, color=colors[i], linestyle='dashed', label=labels[i]+"_"+"window=" + str(window))
        plt.axvline(x=ave_angle[i], color=colors[i])
    plt.xlabel("angles [degree]")
    plt.ylabel("density")
    plt.legend()
    plt.title(str(np.round(state.temperature)) + " K, " + str(np.round(state.pressure, 1)) + " MPa")
    plt.savefig(pathdir / f"hist_angles_T{np.round(state.temperature)}_P{np.round(state.pressure, 1)}.pdf", bbox_inches='tight')
    plt.close()


def print_hist_opt(entry: TaskEntry, pathdir: PATHLIKE, bin_size: float = 0.001):
    traj = read(str(entry.outputs[0]), index=':')
    energies = [atoms.get_potential_energy()/len(atoms) for atoms in traj]
    hist, center_bin = calculate_hist(energies, bin_size, density=False)
    plt.bar(center_bin, hist, width=bin_size, color='tab:blue')
    plt.xlabel("Ground state energy per atom [eV]") 
    plt.ylabel("Count") 
    if entry.pressure is None:
        plt.title(str(np.round(entry.temp)) + " K")
        plt.savefig(pathdir / f"hist_opt_T{np.round(entry.temp)}.pdf", bbox_inches='tight')
    else:
        plt.title(str(np.round(entry.temp)) + " K, " + str(np.round(entry.pressure, 1)) + " MPa")
        plt.savefig(pathdir / f"hist_opt_T{np.round(entry.temp)}_P{np.round(entry.pressure, 1)}.pdf", bbox_inches='tight')
    plt.close()


def plot_freq(entry: TaskEntry, x_np: Union[np.ndarray, list], pathdir: PATHLIKE, smearing: float = 0.2):
    (minimum, hessian) = pickle.load(open(entry.outputs[0], 'rb'))
    frequencies = _compute_frequencies(hessian, minimum) * second/ 10**12 # from ASE to THZ
    y_np = np.zeros(len(x_np))
    for i,x in enumerate(x_np):
        for freq in frequencies[3:]:
            y_np[i] += np.exp(- 0.5 * ((x-freq)/smearing)**2) / (np.sqrt(2*np.pi) * smearing)
    plt.plot(x_np , y_np, label = "smearing: " +str(smearing) + " THz")
    plt.legend()
    plt.xlabel("frequency [THz]") 
    plt.ylabel("phonon spectra") 
    plt.xlim(np.min(x_np), np.max(x_np))
    plt.ylim(0.0, np.max(y_np)*1.05)
    
    if entry.pressure is None:
        plt.title(str(np.round(entry.temp)) + " K")
        plt.savefig(pathdir / f"freq_T{np.round(entry.temp)}.pdf", bbox_inches='tight')
    else:
        plt.title(str(np.round(entry.temp)) + " K, " + str(np.round(entry.pressure, 1)) + " MPa")
        plt.savefig(pathdir / f"freq_T{np.round(entry.temp)}_P{np.round(entry.pressure, 1)}.pdf", bbox_inches='tight')
    plt.clf()

def plot_lambda_effect(states: list[ThermodynamicState], state_begin: ThermodynamicState, state_end: ThermodynamicState, pathdir: PATHLIKE):
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

    lmd = []
    e_ave = []
    e_err = []
    sorted_states = sorted(states, key=lambda state: recover_lambda(np.array(state.lambda_params), lmd_begin, lmd_end))
    for state in sorted_states:
        assert state.temperature == temperature, "temperature of all states should be the same"
        assert state.pressure == pressure, "pressure of all states should be the same"
        if pressure is None:
            assert np.allclose(cell, state.cell), "cell of all states should be the same"    
        assert state._get_hamiltonian_names() == hamilnames, "the mixed hamiltonians of all states should be the same"
        assert state.natoms == natoms, "the number of atoms of all states should be the same"
        state.calc_lambda_gradient(coefficients= coeffs)

        lmd.append(recover_lambda(np.array(state.lambda_params), lmd_begin, lmd_end))
        e_ave.append(state.gradients['lambda'] * kB * state.temperature/natoms)
        e_err.append(state.gradients_error['lambda'] * kB * state.temperature/natoms)

    plt.errorbar(lmd, e_ave, yerr=e_err)
    plt.xlabel("lambda") 
    plt.ylabel(hamilnames[1] +" - " + hamilnames[0] + " per atom [eV]") 
    plt.xlim(0, 1)
    if pressure is None:
        plt.title(str(np.round(temperature)) + " K")
        plt.savefig(pathdir / f"Delta_E_T{np.round(temperature)}.pdf", bbox_inches='tight')
    else:
        plt.title(str(np.round(temperature)) + " K, " + str(np.round(pressure,1)) + " MPa")
        plt.savefig(pathdir / f"Delta_E_T{np.round(temperature)}_P{np.round(pressure, 1)}.pdf", bbox_inches='tight')
    plt.clf()


def plot_tsweeps(states: list[ThermodynamicState], pathdir: PATHLIKE, entry_hess: TaskEntry):
    cell = states[0].cell
    pressure = states[0].pressure
    natoms = states[0].natoms
    hamiltonian = states[0].hamiltonian

    tems = []
    h_ave = []
    h_err = []
    sorted_states = sorted(states, key=lambda state: state.temperature)
    for state in sorted_states:
        assert state.pressure == pressure, "pressure of all states should be the same"
        if pressure is None:
            assert np.allclose(cell, state.cell), "cell of all states should be the same"    
        assert state.hamiltonian == hamiltonian, "the hamiltonians of all states should be the same"
        assert state.natoms == natoms, "the number of atoms of all states should be the same"
        
        (minimum, hessian) = pickle.load(open(entry_hess.outputs[0], 'rb'))
        u_opt = minimum.energy
        if pressure is not None:       # use enthalpy
            u_opt += minimum.volume * pressure * 10 * bar

        u_mean = state.properties['energy'].mean
        u_error = state.properties['energy'].error
        if pressure is not None:       # use enthalpy
            u_mean += state.properties['volume'].mean * pressure * 10 * bar
            u_error += state.properties['volume'].error * pressure * 10 * bar
        u_anharm_mean = u_mean - u_opt - (3 * natoms - 3) * kB * state.temperature / 2

        tems.append(state.temperature)
        h_ave.append(u_anharm_mean /natoms )
        h_err.append(u_error/natoms)

        #_, grad, grad_error = state.calc_temperature_gradient(dy_tem = False)
        #h_ave.append((-(grad+ (3 * natoms - 3) / state.temperature) * kB * state.temperature**2 - u_opt) /natoms )   # CHECK THIS !!!
        #h_err.append(grad_error * kB * state.temperature**2/natoms)

    plt.errorbar(tems, h_ave, yerr=h_err)
    plt.xlabel("Temperature [K]") 
    plt.ylabel("Anharmonic enthalpy per atom [eV]") 
    plt.xlim(np.min(tems), np.max(tems))
    if pressure is None:
        plt.savefig(pathdir / f"Anharn_enthalpy.pdf", bbox_inches='tight')
    else:
        plt.title(str(np.round(pressure,1)) + " MPa")
        plt.savefig(pathdir / f"Anharn_enthalpy_P{np.round(pressure, 1)}.pdf", bbox_inches='tight')
    plt.clf()


def plot_psweeps(states: list[ThermodynamicState], pathdir: PATHLIKE):
    temperature = states[0].temperature
    natoms = states[0].natoms
    hamiltonian = states[0].hamiltonian

    press = []
    v_ave = []
    v_err = []
    sorted_states = sorted(states, key=lambda state: state.pressure)
    for state in sorted_states:
        assert state.pressure is not None, "pressure should be set for all states"
        assert state.temperature == temperature, "temperature of all states should be the same"
        assert state.hamiltonian == hamiltonian, "the hamiltonians of all states should be the same"
        assert state.natoms == natoms, "the number of atoms of all states should be the same"
        #state.calc_pressure_gradient()

        press.append(state.pressure)
        v_ave.append(state.properties['volume'].mean /natoms)
        v_err.append(state.properties['volume'].error /natoms)

    plt.errorbar(press, v_ave, yerr=v_err)
    plt.xlabel("Pressure [MPa]") 
    plt.ylabel("Volume per atom [A**3]") 
    plt.xlim(np.min(press), np.max(press))
    plt.title(str(np.round(temperature)) + " K")
    plt.savefig(pathdir / f"Volume_T{np.round(temperature)}.pdf", bbox_inches='tight')
    plt.clf()



def plot_kinetic_CV(states: list[ThermodynamicState], pathdir: PATHLIKE, pressure_cell: float):
    temperature = states[0].temperature
    cell = states[0].cell
    pressure = states[0].pressure
    natoms = states[0].natoms
    hamiltonian = states[0].hamiltonian

    Ekin_clas = 3*kB*temperature/2
    nbeads_lst = sorted({state.nbeads for state in states})
    for nbeads in nbeads_lst:
        g_masses = [0]
        kincv_ave = [Ekin_clas]
        integ = []
        kincv_err = [0]
        sorted_states = sorted(states, key=lambda state: state.mass_scaling, reverse=True)
        for i, state in enumerate(sorted_states):
            if state.nbeads == nbeads:
                assert state.temperature == temperature, "temperature of all states should be the same"
                assert state.pressure == pressure, "pressure of all states should be the same"
                if pressure is None:
                    assert np.allclose(cell, state.cell), "cell of all states should be the same"    
                assert state.hamiltonian == hamiltonian, "the hamiltonians of all states should be the same"
                assert state.natoms == natoms, "the number of atoms of all states should be the same"
                #state.calc_masses_gradient()

                g_masses.append(1/np.sqrt(state.mass_scaling))
                kincv_ave.append(state.properties['kinetic_cv'].mean /natoms)
                integ.append(2*(state.properties['kinetic_cv'].mean - Ekin_clas) * np.sqrt(state.mass_scaling)/natoms)
                if len(integ) ==1:
                    integ.append(2*(state.properties['kinetic_cv'].mean - Ekin_clas) * np.sqrt(state.mass_scaling)/natoms)
                kincv_err.append(state.properties['kinetic_cv'].error /natoms)

        if len(integ) != 0:
            plt.errorbar(g_masses, kincv_ave, yerr=kincv_err, label = str(nbeads))
            plt.errorbar(g_masses, integ, yerr=kincv_err, label = str(nbeads)+"_integ")
    plt.xlabel("inverse of square root of scale factor masses") 
    plt.ylabel("centroid-virial quantum kinetic energy per atom [eV]") 
    plt.legend()
    plt.xlim(0, 1)
    if pressure is None:
        plt.title(str(np.round(temperature)) + " K")
        plt.savefig(pathdir / f"Delta_E_T{np.round(temperature)}.pdf", bbox_inches='tight')
    else:
        plt.title(str(np.round(temperature)) + " K, " + str(np.round(pressure_cell,1)) + " MPa")
        plt.savefig(pathdir / f"Delta_E_T{np.round(temperature)}_P{np.round(pressure_cell, 1)}.pdf", bbox_inches='tight')
    plt.clf()
    