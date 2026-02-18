from pathlib import Path
import ase.io
import numpy as np
import pickle

from parsl import python_app, File
from parsl.dataflow.futures import DataFuture
from psiflow.geometry import Geometry
from psiflow.sampling import Walker, sample
from psiflow.data.utils import _count_frames
from psiflow.sampling.walker import replica_exchange
from psiflow.hamiltonians import Harmonic, MixtureHamiltonian

from .psiflow_utils import load_hamiltonian, load_hessian, is_valid_input, save_simulation_output, fill_refs, get_hamiltonian, get_geometry
from .workflow import State, Type, TaskDatabase, TaskEntry

DEFAULT_KWARGS = {
    'multi': 1, 
    'start': 0, 
    'timestep': 0.5,
    'store_traj': False, 
    'freq_rex_temperature': None, 
    'init_struc': 'file', 
    'init_temperatures': 'all', 
    'init_pressures': 'all', 
    'init_last': False,
    'hess_name': None,
    'hess_temperatures': None,
    'hess_pressures': None,
    'other_hamiltonian': None}
REQUIRED_KWARGS = ('steps', 'step', 'temperatures', 'pressures', 'hamiltonian', 'workdir')


def check_kwargs(**kwargs) -> dict:
    EXTRA_KWARGS = []
    if 'number_of_hamiltonians' in kwargs:
        for i in range(1, kwargs['number_of_hamiltonians']):
            EXTRA_KWARGS.append(f'other_hamiltonian_{i}')
    return {k: kwargs.get(k, DEFAULT_KWARGS[k]) for k in DEFAULT_KWARGS} | {k: kwargs[k] for k in REQUIRED_KWARGS} | {k: kwargs[k] for k in EXTRA_KWARGS}


def check_task_state(tasks: TaskDatabase, config: dict, name: str) -> TaskDatabase:
    """"""
    workdir, store_traj = config['workdir'], config['store_traj']
    nframes = (config['steps'] - config['start']) // config['step']
    temps, pressures, multi = config['temperatures'], config['pressures'], config['multi']
    init_struc, init_temp, init_pressure = config['init_struc'], config['init_temperatures'], config['init_pressures']
    hess_name, hess_temp, hess_pressure = config['hess_name'], config['hess_temperatures'], config['hess_pressures']

    input_valid = True
    ref_temps = np.full(len(temps), None)
    ref_pressures = np.full(len(pressures), None)
    if hess_name is not None:
        ref_temps = fill_refs(hess_temp, len(temps))
        ref_pressures = fill_refs(hess_pressure, len(pressures))
        if init_struc != 'file':
            print('WARNING: the optimized structure of the hessians are used instead of init_struc')
    elif init_struc != 'file':
        ref_temps = fill_refs(init_temp, len(temps))
        ref_pressures = fill_refs(init_pressure, len(pressures))
    else:
        files_in = [config['init_file']]
        print(f'use initial file for {name} run')
        input_valid = is_valid_input(files_in[0])
    assert len(ref_temps) == len(temps), 'these should be the same length'
    assert len(ref_pressures) == len(pressures), 'these should be the same length'

    for (rt, t, rp, p, n) in [(rt, t, rp, p, n) for rt, t in zip(ref_temps, temps) for rp, p in zip(ref_pressures, pressures) for n in range(multi)]:

        if hess_name is not None:
            kwargs = {"type": Type.HESS, "name": hess_name}
            if rt is not None:
                kwargs["temp"] = rt
            if rp is not None:
                kwargs["pressure"] = rp
            task_hess = tasks.get_entries(**kwargs).pop()
            files_in = task_hess.outputs[0] if task_hess.state != State.MISSING_INPUT else None
            if input_valid:
                input_valid = bool(files_in)
        elif init_struc != 'file':
            # check for available input simulations
            files_in = []
            type_is_MD = True
            kwargs = {"type": Type.MD}
            if init_struc != "all":
                kwargs["name"] = init_struc
            if rt != "all":
                kwargs["temp"] = rt
            if rp != "all":
                kwargs["pressure"] = rp
            init_entries = tasks.get_entries(**kwargs)
            if init_entries == set([]):
                kwargs["type"] = Type.LAMBDA
                init_entries = tasks.get_entries(**kwargs)
                if init_entries == set([]):
                    kwargs["type"] = Type.OPT
                    init_entries = tasks.get_entries(**kwargs)
                    type_is_MD = False
            for intask in init_entries:
                if intask.state == State.MISSING_INPUT:
                    continue
                if type_is_MD:
                    if len(intask.outputs) == 2:
                        files_in.append(intask.outputs[1])
                else:
                    files_in.append(intask.outputs[0])
            # both props and trajectories are needed
            if input_valid:
                input_valid = bool(files_in)

        if p is not None:
            file_props = workdir / f'props_T{round(t)}_P{round(p, 1)}_{n}.pickle'
            file_traj = workdir / f'traj_T{round(t)}_P{round(p, 1)}_{n}.xyz'
        else:
            file_props = workdir / f'props_T{round(t)}_{n}.pickle'
            file_traj = workdir / f'traj_T{round(t)}_{n}.xyz'
        file_props_ok = file_props.is_file()        # TODO: check for length?
        file_traj_ok = (not store_traj) or (file_traj.is_file() and _count_frames([file_traj]) >= nframes)

        task = tasks.new_task(Type.MD, name=name, temp=t, pressure=p, multi=n)
        task.outputs.append(File(file_props))
        task.inputs = files_in
        if store_traj:
            task.outputs.append(File(file_traj))
        if file_props_ok and file_traj_ok:
            task.state = State.DONE
        else:
            task.state = State.WIP if input_valid else State.MISSING_INPUT
    return tasks


def main(name: str, tasks: TaskDatabase, config: dict) -> TaskDatabase:
    """"""
    tasks = check_task_state(tasks, config, name)
    if not (_ := tasks.count_states()[State.WIP]):
        tasks.reset()
        return tasks
    else:
        print(f'Submitting {_} simulations.')

    walkers = []
    
    if config['other_hamiltonian'] is not None:
        terms = config['other_hamiltonian'].split('+')
        hamiltonian_parts = []
        for term in terms:
            coeff, ham_str = term.split('x')
            hamiltonian_parts.append((float(coeff), ham_str))

    for task in tasks.get_entries(type=Type.MD, state=State.WIP, active = True):
        harmonic = None
        if  config['hess_name'] is not None:
            future = load_hessian(task.inputs)
            geom, hessian = future[0], future[1]
            harmonic = Harmonic(geom, hessian)
        else:
            if config['init_last']:
                ind = -1
            else:
                #hash_id = hashlib.md5(str(task.outputs[0]).encode('utf-8')).hexdigest()   # alternative method
                ind = int(task.hash, 16)
            geom = get_geometry(ind, *task.inputs)

        if config['other_hamiltonian'] is None:
            mixture = load_hamiltonian(Path(config['hamiltonian']))    
        else:
            (coeff, ham_str) = hamiltonian_parts[0]
            mixture = coeff * get_hamiltonian(ham_str, config, harmonic = harmonic)
            for (coeff, ham_str) in hamiltonian_parts[1:]:
                mixture += coeff * get_hamiltonian(ham_str, config, harmonic = harmonic)

        walker = Walker(geom, hamiltonian=mixture, timestep = config['timestep'], temperature=task.temp, pressure=task.pressure)
        walker.task = task                  # ensure easy coupling
        walkers.append(walker)

    if config['freq_rex_temperature'] is not None:
        group_walkers = {}
        for walker in walkers:
            if (walker.pressure, walker.task.get_properties()['multi']) not in group_walkers:
                group_walkers[(walker.pressure, walker.task.get_properties()['multi'])] = [walker]
            else:
                group_walkers[(walker.pressure, walker.task.get_properties()['multi'])].append(walker)
        for (p, n), coupled_walkers in group_walkers.items():
            replica_exchange(
                coupled_walkers,
                trial_frequency=config['freq_rex_temperature']
            )

    outputs = sample(
        walkers=walkers,
        steps=config['steps'],
        step=config['step'],
        start=config['start'],
        keep_trajectory=True,
        observables=['cell_h{angstrom}'],           # needed for cell average / histogram
        fix_com=True,
        use_unique_seeds=True,
    )

    for output, walker in zip(outputs, walkers):
        task = walker.task
        len_outputs = len(task.outputs)
        future = save_simulation_output(output, walker, output.status, outputs=task.outputs)
        task.outputs = future.outputs
        if len_outputs == 2:
            filestr = str(task.outputs[1].filepath)
            if not Path.exists(Path(filestr)):
                task.outputs[1] = output.trajectory.extxyz

    tasks.reset()
    return tasks



