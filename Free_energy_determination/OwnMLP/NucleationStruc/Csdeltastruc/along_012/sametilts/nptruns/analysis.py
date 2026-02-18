import pickle
from pathlib import Path
import numpy as np

from thermoflow.workflow import TaskDatabase, Type
from thermoflow.state import WalkerState, combine_outputs, ThermodynamicState, calc_ti_lambda, calc_ti_md, calc_ti_pimd
from execute_workflow import DIR_PREFIX, FILE_TASK_YAML, KEY_NPT, KEY_OPT, KEY_HESS, KEY_TI_LOW, KEY_INTERMEDIATE, KEY_TI_HIGH, KEY_NVT, KEY_TI_LOT, KEY_NVT_LOT, KEY_OPT_LOT, KEY_HESS_LOT
from thermoflow.plotting_tools import print_hist_npt, print_hist_opt, plot_freq, plot_lambda_effect, plot_tsweeps, plot_psweeps, plot_kinetic_CV
from psiflow.free_energy.phonons import _harmonic_free_energy, _compute_frequencies
from psiflow.hamiltonians import Harmonic
from ase.units import kB, bar
import csv
import matplotlib.pyplot as plt
from thermoflow.workflow import State

# hacky patch
import sys
from thermoflow import state
sys.modules['state'] = state

KEY_PLOTS = 'plots'
HARMONIC_HAMILTONIAN = 'Harmonic'
MACE_HAMILTONIAN = 'MACEHamiltonian'

from configuration import phases, config, TEMPERATURE_GRID, PRESSURE_GRID, HARM_T, T_MAX


def get_entries_state(tasks, **kwargs):
    all_entries = tasks.get_entries(**kwargs)
    entries = set([])
    for entry in all_entries:
        if entry.state == State.DONE:
            entries.add(entry)
        else:
            print("FAILED: ", entry)
    return entries


root = Path.cwd()
for phase in phases:
    print(f'start post-analysis of phase {phase}')
    dir_phase = root / f'{DIR_PREFIX}-{phase}'
    file_tasks = dir_phase / FILE_TASK_YAML
    tasks = TaskDatabase.from_yaml(file_tasks)
    tasks.to_yaml(file_tasks)  # To update the current state

    plots_dir = dir_phase / KEY_PLOTS
    plots_dir.mkdir(exist_ok=True)
    
    print(f'start making {KEY_NPT} plots, this might take a while')
    (workdir := plots_dir / KEY_NPT).mkdir(exist_ok=True)
    entries = get_entries_state(tasks, type=Type.MD, name='npt') #, temp=HARMONIC_TEMPERATURE_GRID[0], pressure=HARMONIC_PRESSURE_GRID[0])
    outputs = [pickle.load(open(entry.outputs[0], 'rb')) for entry in entries]
    states = combine_outputs(outputs)
    for st in states:
        print_hist_npt(st, workdir)

    print(f'start making NPT temperature sweep plots')
    (workdir := plots_dir / KEY_NPT).mkdir(exist_ok=True)
    for pressure in config[KEY_NPT]['pressures']:
        entries = get_entries_state(tasks, type=Type.MD, pressure=pressure, name=KEY_NPT)
        outputs = [pickle.load(open(entry.outputs[0], 'rb')) for entry in entries]
        states = combine_outputs(outputs)
        entry_hess = get_entries_state(tasks, type=Type.HESS, name=KEY_HESS).pop()  # TODO: With more hessians this might not be the one you want
        plot_tsweeps(states, workdir, entry_hess)
    
    print(f'start making {KEY_OPT} plots')
    (workdir := plots_dir / KEY_OPT).mkdir(exist_ok=True)
    entries = get_entries_state(tasks, type=Type.OPT, name=KEY_OPT)
    for entry in entries:
        print_hist_opt(entry, workdir)
    
    print(f'start making {KEY_HESS} plots')
    (workdir := plots_dir / KEY_HESS).mkdir(exist_ok=True)
    entries = get_entries_state(tasks, type=Type.HESS, name=KEY_HESS)
    for entry in entries:
        plot_freq(entry, np.arange(0.0, 25.0, 0.05), workdir)
    
    print(f'start making {KEY_TI_LOW} plots')
    (workdir := plots_dir / KEY_TI_LOW).mkdir(exist_ok=True)
    for pressure in config[KEY_TI_LOW]['pressures']:
        for temperature in config[KEY_TI_LOW]['temperatures']:
            try:
                entries = get_entries_state(tasks, type=Type.LAMBDA, temp=temperature, pressure=pressure, name=KEY_TI_LOW)
                outputs = [pickle.load(open(entry.outputs[0], 'rb')) for entry in entries]
                states = combine_outputs(outputs)
                entry_begin = get_entries_state(tasks, type=Type.LAMBDA, temp=temperature, pressure=pressure, lambda_param=0.0, name=KEY_TI_LOW).pop()
                state_begin = combine_outputs([pickle.load(open(entry_begin.outputs[0], 'rb'))])[0]
                entry_end = get_entries_state(tasks, type=Type.LAMBDA, temp=temperature, pressure=pressure, lambda_param=1.0, name=KEY_TI_LOW).pop()
                state_end = combine_outputs([pickle.load(open(entry_end.outputs[0], 'rb'))])[0]
                plot_lambda_effect(states, state_begin, state_end, workdir)  # TODO: You might also want to print the pressure and temperature at which the hessian is constructed
            except:
                print(f'lambda plot at {temperature} K and {pressure} MPa failed')
    
    print(f'start making NVT temperature sweep plots of the intermediate PES')
    (workdir := plots_dir / KEY_INTERMEDIATE).mkdir(exist_ok=True)
    for pressure in config[KEY_INTERMEDIATE]['pressures']:
        try:
            entries = get_entries_state(tasks, type=Type.MD, pressure=pressure, name=KEY_INTERMEDIATE)
            outputs = [pickle.load(open(entry.outputs[0], 'rb')) for entry in entries]
            states = combine_outputs(outputs)
            entry_hess = get_entries_state(tasks, type=Type.HESS, name=KEY_HESS).pop()  # TODO: With more hessians this might not be the one you want
            plot_tsweeps(states, workdir, entry_hess)
        except:
            print(f'temperature plot at {pressure} MPa failed')

    print(f'start making {KEY_TI_HIGH} plots')
    (workdir := plots_dir / KEY_TI_HIGH).mkdir(exist_ok=True)
    for pressure in config[KEY_TI_HIGH]['pressures']:
        for temperature in config[KEY_TI_HIGH]['temperatures']:
            try:
                entries = get_entries_state(tasks, type=Type.LAMBDA, temp=temperature, pressure=pressure, name=KEY_TI_HIGH)
                outputs = [pickle.load(open(entry.outputs[0], 'rb')) for entry in entries]
                states = combine_outputs(outputs)
                entry_begin = get_entries_state(tasks, type=Type.LAMBDA, temp=temperature, pressure=pressure, lambda_param=0.0, name=KEY_TI_HIGH).pop()
                state_begin = combine_outputs([pickle.load(open(entry_begin.outputs[0], 'rb'))])[0]
                entry_end = get_entries_state(tasks, type=Type.LAMBDA, temp=temperature, pressure=pressure, lambda_param=1.0, name=KEY_TI_HIGH).pop()
                state_end = combine_outputs([pickle.load(open(entry_end.outputs[0], 'rb'))])[0]
                plot_lambda_effect(states, state_begin, state_end, workdir)  # TODO: You might also want to print the pressure and temperature at which the hessian is constructed
            except:
                print(f'lambda plot at {temperature} K and {pressure} MPa failed')
    
    print(f'start making NVT temperature sweep plots')
    (workdir := plots_dir / KEY_NVT).mkdir(exist_ok=True)
    for pressure in config[KEY_NVT]['pressures']:
        try:
            entries = get_entries_state(tasks, type=Type.MD, pressure=pressure, name=KEY_NVT)
            outputs = [pickle.load(open(entry.outputs[0], 'rb')) for entry in entries]
            states = combine_outputs(outputs)
            entry_hess = get_entries_state(tasks, type=Type.HESS, name=KEY_HESS).pop()
            plot_tsweeps(states, workdir, entry_hess)
        except:
            print(f'temperature plot at {pressure} MPa failed')
    '''
    print(f'start making {KEY_OPT_LOT} plots at a different level of theory')
    (workdir := plots_dir / KEY_OPT_LOT).mkdir(exist_ok=True)
    entries = get_entries_state(tasks, type=Type.OPT, name=KEY_OPT_LOT)
    for entry in entries:
        print_hist_opt(entry, workdir)
    
    print(f'start making {KEY_HESS_LOT} plots at a different level of theory')
    (workdir := plots_dir / KEY_HESS_LOT).mkdir(exist_ok=True)
    entries = get_entries_state(tasks, type=Type.HESS, name=KEY_HESS_LOT)
    for entry in entries:
        plot_freq(entry, np.arange(0.0, 25.0, 0.05), workdir)

    print(f'start making {KEY_TI_LOT} plots')
    (workdir := plots_dir / KEY_TI_LOT).mkdir(exist_ok=True)
    for pressure in config[KEY_TI_LOT]['pressures']:
        for temperature in config[KEY_TI_LOT]['temperatures']:
            try:
                entries = get_entries_state(tasks, type=Type.LAMBDA, temp=temperature, pressure=pressure, name=KEY_TI_LOT)
                outputs = [pickle.load(open(entry.outputs[0], 'rb')) for entry in entries]
                states = combine_outputs(outputs)
                entry_begin = get_entries_state(tasks, type=Type.LAMBDA, temp=temperature, pressure=pressure, lambda_param=0.0, name=KEY_TI_LOT).pop()
                state_begin = combine_outputs([pickle.load(open(entry_begin.outputs[0], 'rb'))])[0]
                entry_end = get_entries_state(tasks, type=Type.LAMBDA, temp=temperature, pressure=pressure, lambda_param=1.0, name=KEY_TI_LOT).pop()
                state_end = combine_outputs([pickle.load(open(entry_end.outputs[0], 'rb'))])[0]
                plot_lambda_effect(states, state_begin, state_end, workdir)  # TODO: You might also want to print the pressure and temperature at which the hessian is constructed
            except:
                print(f'lambda plot at {temperature} K and {pressure} MPa failed')

    print(f'start making NVT temperature sweep plots for the other level of theory')
    (workdir := plots_dir / KEY_NVT_LOT).mkdir(exist_ok=True)
    for pressure in config[KEY_NVT_LOT]['pressures']:
        try:
            entries = get_entries_state(tasks, type=Type.MD, pressure=pressure, name=KEY_NVT_LOT)
            outputs = [pickle.load(open(entry.outputs[0], 'rb')) for entry in entries]
            states = combine_outputs(outputs)
            entry_hess = get_entries_state(tasks, type=Type.HESS, name=KEY_HESS_LOT).pop()  # TODO: With more hessians this might not be the one you want
            plot_tsweeps(states, workdir, entry_hess)
        except:
            print(f'temperature plot at {pressure} MPa failed')
    '''
    
    #For free energy
    # We assume that we only have one pressure and temperature for which we calculate the harmonic reference
    #pressure = 0.1 MPa
    try:
        temperature =  HARM_T[0]
        print(f'start making Free_energy plots')
        (workdir := plots_dir / "Free_energy").mkdir(exist_ok=True)

        free_energy = {'harm': [], 'intermediate': [], 'nvt': [], 'harm_lot': [], 'nvt_lot': []}
        free_error = {'harm': [], 'intermediate': [], 'nvt': [], 'harm_lot': [], 'nvt_lot': []}

        entries = get_entries_state(tasks, type=Type.HESS, name=KEY_HESS)
        (minimum, hessian) = pickle.load(open(list(entries)[0].outputs[0], 'rb'))
        frequencies = _compute_frequencies(hessian, minimum)
        harmonic = 1 * Harmonic(minimum, hessian)
        str_harmonic = []
        for c, h in zip(harmonic.coefficients, harmonic.hamiltonians):
            str_harmonic.append(f'{c:.4f} x {h.__class__.__name__}')
        str_harmonic = ' + '.join(str_harmonic)

        walkerstate_harm = WalkerState(
            hamiltonian = str_harmonic, 
            temperature = temperature, 
            natoms = len(minimum), 
            cell = minimum.cell
        )
        state_harm = ThermodynamicState(state=walkerstate_harm)
        state_harm.free_energy = minimum.energy/(kB*temperature) + _harmonic_free_energy(frequencies, temperature=temperature, quantum=False) 
        for tem in TEMPERATURE_GRID:
            gs_energy = (minimum.energy) / len(minimum)  # NO PV term added at the moment
            free_vib = _harmonic_free_energy(frequencies, temperature=tem, quantum=False) * kB*tem / len(minimum)
            free_energy['harm'].append(gs_energy + free_vib)
            free_error['harm'].append(0)

        entries = get_entries_state(tasks, type=Type.LAMBDA, temp=temperature, name=KEY_TI_LOW)
        outputs = [pickle.load(open(entry.outputs[0], 'rb')) for entry in entries]
        states = combine_outputs(outputs)
        entry_begin = get_entries_state(tasks, type=Type.LAMBDA, temp=temperature, lambda_param=0.0, name=KEY_TI_LOW).pop()
        state_begin = combine_outputs([pickle.load(open(entry_begin.outputs[0], 'rb'))])[0]
        state_begin.free_energy, state_begin.free_error = state_harm.free_energy, state_harm.free_error
        entry_end = get_entries_state(tasks, type=Type.LAMBDA, temp=temperature, lambda_param=1.0, name=KEY_TI_LOW).pop()
        state_end = combine_outputs([pickle.load(open(entry_end.outputs[0], 'rb'))])[0]
        state_ti_end = calc_ti_lambda(states, state_begin, state_end)

        entries = get_entries_state(tasks, type=Type.MD, name=KEY_INTERMEDIATE)
        outputs = [pickle.load(open(entry.outputs[0], 'rb')) for entry in entries]
        states = combine_outputs(outputs)
        calc_ti_md(states, state_ti_end, E_opt = minimum.energy)  # updates states by setting the free energy
        for tem in TEMPERATURE_GRID:
            free_energy['intermediate'].append(next((st.free_energy* kB*tem / st.natoms for st in states if st.temperature == tem), None))
            free_error['intermediate'].append(next((st.free_error* kB*tem / st.natoms for st in states if st.temperature == tem), None))
        for st in states:
            if st.temperature == T_MAX[0]:
                state_ti_end = st

        entries = get_entries_state(tasks, type=Type.LAMBDA, temp=T_MAX[0], name=KEY_TI_HIGH)
        outputs = [pickle.load(open(entry.outputs[0], 'rb')) for entry in entries]
        states = combine_outputs(outputs)
        entry_begin = get_entries_state(tasks, type=Type.LAMBDA, temp=T_MAX[0], lambda_param=0.0, name=KEY_TI_HIGH).pop()
        state_begin = combine_outputs([pickle.load(open(entry_begin.outputs[0], 'rb'))])[0]
        state_begin.free_energy, state_begin.free_error = state_ti_end.free_energy, state_ti_end.free_error
        entry_end = get_entries_state(tasks, type=Type.LAMBDA, temp=T_MAX[0], lambda_param=1.0, name=KEY_TI_HIGH).pop()
        state_end = combine_outputs([pickle.load(open(entry_end.outputs[0], 'rb'))])[0]
        state_ti_end = calc_ti_lambda(states, state_begin, state_end)

        entries = get_entries_state(tasks, type=Type.MD, name=KEY_NVT)
        outputs = [pickle.load(open(entry.outputs[0], 'rb')) for entry in entries]
        states = combine_outputs(outputs)
        calc_ti_md(states, state_ti_end, E_opt = minimum.energy)  # updates states by setting the free energy
        for tem in TEMPERATURE_GRID:
            free_energy['nvt'].append(next((st.free_energy* kB*tem / st.natoms for st in states if st.temperature == tem), None))
            free_error['nvt'].append(next((st.free_error* kB*tem / st.natoms for st in states if st.temperature == tem), None))
        for st in states:
            if st.temperature == T_MAX[0]:
                state_ti_end = st
        '''
        entries = get_entries_state(tasks, type=Type.LAMBDA, temp=T_MAX[0], name=KEY_TI_LOT)
        outputs = [pickle.load(open(entry.outputs[0], 'rb')) for entry in entries]
        states = combine_outputs(outputs)
        entry_begin = get_entries_state(tasks, type=Type.LAMBDA, temp=T_MAX[0], lambda_param=0.0, name=KEY_TI_LOT).pop()
        state_begin = combine_outputs([pickle.load(open(entry_begin.outputs[0], 'rb'))])[0]
        state_begin.free_energy, state_begin.free_error = state_ti_end.free_energy, state_ti_end.free_error
        entry_end = get_entries_state(tasks, type=Type.LAMBDA, temp=T_MAX[0], lambda_param=1.0, name=KEY_TI_LOT).pop()
        state_end = combine_outputs([pickle.load(open(entry_end.outputs[0], 'rb'))])[0]
        state_ti_end = calc_ti_lambda(states, state_begin, state_end)

        entries = get_entries_state(tasks, type=Type.HESS, name=KEY_HESS_LOT)
        (minimum_lot, hessian_lot) = pickle.load(open(list(entries)[0].outputs[0], 'rb'))
        frequencies_lot = _compute_frequencies(hessian_lot, minimum_lot)
        for tem in TEMPERATURE_GRID:
            gs_energy = (minimum_lot.energy) / len(minimum_lot)  # NO PV term added at the moment
            free_vib = _harmonic_free_energy(frequencies_lot, temperature=tem, quantum=False) * kB*tem / len(minimum_lot)
            free_energy['harm_lot'].append(gs_energy + free_vib)
            free_error['harm_lot'].append(0)

        entries = get_entries_state(tasks, type=Type.MD, name=KEY_NVT_LOT)
        outputs = [pickle.load(open(entry.outputs[0], 'rb')) for entry in entries]
        states = combine_outputs(outputs)
        calc_ti_md(states, state_ti_end, E_opt = minimum_lot.energy)  # updates states by setting the free energy
        for tem in TEMPERATURE_GRID:
            free_energy['nvt_lot'].append(next((st.free_energy* kB*tem / st.natoms for st in states if st.temperature == tem), None))
            free_error['nvt_lot'].append(next((st.free_error* kB*tem / st.natoms for st in states if st.temperature == tem), None))
        '''
        # No Gibbs correction

        for pes in ['harm', 'intermediate', 'nvt']:  #, 'harm_lot', 'nvt_lot']:
            temps = TEMPERATURE_GRID
            values = free_energy[pes]
            errors = free_error[pes]

            with open(workdir / f"{pes}_Tharm_{temperature:.0f}K.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Temperature_K", "FreeEnergy_eVperatom", "Error_eVperatom"])
                writer.writerows(zip(temps, values, errors))

            plt.errorbar(temps, values, yerr=errors, label=pes, capsize=3, marker='o', linestyle='-')
        plt.xlabel("Temperature [K]")
        plt.ylabel("Free Energy per atom [eV]")
        plt.title(f"Free Energy vs Temperature with reference at {temperature:.0f}K")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(workdir / f"free_energy_Tharm_{temperature:.0f}K.pdf")
        plt.clf()
    except:
        print('free energy calculation failed')
