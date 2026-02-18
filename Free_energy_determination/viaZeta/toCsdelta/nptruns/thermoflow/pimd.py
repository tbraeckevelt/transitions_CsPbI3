'''
This file needs an update to work in v5 branch
'''

from pathlib import Path

from parsl import python_app, File
from parsl.dataflow.futures import DataFuture
from psiflow.geometry import Geometry
from psiflow.sampling import Walker, sample
from psiflow.data.utils import _count_frames

from .psiflow_utils import load_hamiltonian, is_valid_input, PATHLIKE, make_path
from .workflow import State, Type, TaskDatabase, TaskEntry
from .sample import load_geometry, load_geometries
from .new import save_simulation_output

# TODO: can we merge this with basic sample script?

DEFAULT_KWARGS = {'multi': 1, 'start': 0, 'timestep': 0.5, 'store_traj': True, 'nbeads': 16}
REQUIRED_KWARGS = ('steps', 'step', 'temperatures', 'pressures', 'hamiltonian', 'workdir', 'mass_scale')
OBSERVABLES = ['cell_h{angstrom}', 'kinetic_cv{electronvolt}', 'kinetic_ij{electronvolt}',
               'kinetic_opsc{electronvolt}', 'kinetic_tens{electronvolt}']


def check_kwargs(**kwargs) -> dict:
    return {k: kwargs.get(k, DEFAULT_KWARGS[k]) for k in DEFAULT_KWARGS} | {k: kwargs[k] for k in REQUIRED_KWARGS}


def check_task_state(config: dict, file_in: PATHLIKE = None) -> TaskDatabase:
    """"""
    workdir, store_traj = config['workdir'], config['store_traj']
    nframes = (config['steps'] - config['start']) // config['step']
    temps, pressures, multi = config['temperatures'], config['pressures'], config['multi']
    nbeads, mass_scale = config['nbeads'], config['mass_scale']
    if not isinstance(nbeads, list):
        nbeads= [nbeads]

    tasks = TaskDatabase()
    input_valid = is_valid_input(file_in) if file_in is not None else True

    for (t, p, n, m, nb) in [(t, p, n, m, nb) for m in mass_scale for t in temps for p in pressures for n in range(multi) for nb in nbeads]:
        task = tasks.new_task(Type.PIMD, temp=t, pressure=p, nbeads=nb, mass_scale=m, multi = n)
        if p is not None:
            key = f'T{round(t)}_P{round(p, 1)}_N{nb}_M{m}_{n}'
        else:
            key = f'T{round(t)}_N{nb}_M{m}_{n}'
        file_props = workdir / f'props_{key}.pickle'
        file_traj = workdir / f'traj_{key}.xyz'
        file_props_ok = file_props.is_file()        # TODO: check for length?
        file_traj_ok = (not store_traj) or (file_traj.is_file() and _count_frames([file_traj]) >= nframes)
        task.outputs.append(File(file_props))
        if store_traj:
            task.outputs.append(File(file_traj))
        if file_props_ok and file_traj_ok:
            task.state = State.DONE
        else:
            task.state = State.WIP if input_valid else State.MISSING_INPUT
    return tasks


def main(file_in: File | DataFuture, config: dict) -> TaskDatabase:
    """"""
    tasks = check_task_state(config, file_in=file_in)
    if not (_ := tasks.count_states()[State.WIP]):
        tasks.reset()
        return tasks
    else:
        print(f'Submitting {_} PIMD simulations.')

    walkers = []
    hamiltonian = load_hamiltonian(Path(config['hamiltonian']))
    geos = load_geometries(file_in)
    #geom = load_geometry(file_in).result()
    for task in tasks.get_entries(type=Type.PIMD, state=State.WIP):
        #hash_id = hashlib.md5(str(task.outputs[0]).encode('utf-8')).hexdigest()   # alternative method
        geom = geos[int(task.hash, 16) % len(geos)]     # should be unique for each MD, so tasks should contain the multi attribute as extra property
        walker = Walker(geom, hamiltonian=hamiltonian, timestep = config['timestep'], temperature=task.temp, pressure=task.pressure,
                        nbeads=task.nbeads, masses=task.mass_scale)
        walker.task = task                  # ensure easy coupling
        walkers.append(walker)

    outputs = sample(
        walkers=walkers,
        steps=config['steps'],
        step=config['step'],
        start=config['start'],
        keep_trajectory=True,
        observables=OBSERVABLES,
        fix_com=True,
        use_unique_seeds=True,
    )

    for output, walker in zip(outputs, walkers):
        task = walker.task
        future = save_simulation_output(output, walker, output.status, outputs=task.outputs)
        task.outputs = future.outputs

    tasks.reset()
    return tasks



