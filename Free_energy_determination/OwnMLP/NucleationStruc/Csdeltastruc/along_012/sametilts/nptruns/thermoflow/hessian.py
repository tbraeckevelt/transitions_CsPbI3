import pickle
from pathlib import Path

import numpy as np
from parsl import python_app, File
from parsl.dataflow.futures import AppFuture
from psiflow.geometry import Geometry
from psiflow.data.utils import read_frames
from psiflow.free_energy.phonons import compute_harmonic

from .psiflow_utils import load_hamiltonian, get_minimum
from .workflow import State, Type, TaskDatabase, TaskEntry

# TODO: support for analytical gradient?

DEFAULT_KWARGS = {'opt_struc': 'all'}
REQUIRED_KWARGS = ('delta_shift', 'temperatures', 'pressures', 'hamiltonian', 'workdir')


def check_kwargs(**kwargs) -> dict:
    EXTRA_KWARGS = []
    if 'number_of_hamiltonians' in kwargs:
        for i in range(1, kwargs['number_of_hamiltonians']):
            EXTRA_KWARGS.append(f'other_hamiltonian_{i}')
    return {k: kwargs.get(k, DEFAULT_KWARGS[k]) for k in DEFAULT_KWARGS} | {k: kwargs[k] for k in REQUIRED_KWARGS} | {k: kwargs[k] for k in EXTRA_KWARGS}


def check_task_state(tasks: TaskDatabase, config: dict, name: str) -> TaskDatabase:
    """"""
    workdir, temps, pressures = config['workdir'], config['temperatures'], config['pressures']

    for (t, p) in [(t, p) for t in temps for p in pressures]:
        # check for available input optimisation
        files_in = []
        kwargs = {"type": Type.OPT, "temp": t, "pressure": p}
        if config['opt_struc'] != 'all':
            kwargs["name"] = config['opt_struc']
        for task in tasks.get_entries(**kwargs):
            if task.state == State.MISSING_INPUT:
                continue
            files_in += task.outputs
        input_valid = bool(files_in)
        if p is not None:
            file_hess = workdir / f'hess_T{round(t)}_P{round(p, 1)}.pickle'
        else:
            file_hess = workdir / f'hess_T{round(t)}.pickle'

        task = tasks.new_task(type=Type.HESS, name = name, temp=t, pressure=p)
        task.outputs = [File(file_hess)]
        task.inputs = files_in
        if file_hess.is_file():
            task.state = State.DONE
        else:
            task.state = State.WIP if input_valid else State.MISSING_INPUT

    return tasks


@python_app(executors=["default_threads"])
def store_hessian(geometry: Geometry, hess: np.ndarray, outputs: list[File]) -> AppFuture:
    from .psiflow_utils import _is_positive_definite
    file_out = Path(outputs[0].filepath)
    if _is_positive_definite(hess, geometry):
        pickle.dump((geometry, hess), file_out.open('wb'))
    else:
        print(f'Did not store Hessian at "{file_out.name}" because it is not positive definite..')


def main(name: str, tasks: TaskDatabase, config: dict) -> TaskDatabase:
    """"""
    tasks = check_task_state(tasks, config, name)
    if not (_ := tasks.count_states()[State.WIP]):
        tasks.reset()
        return tasks
    else:
        print(f'Submitting {_} calculations.')

    hamiltonian = load_hamiltonian(Path(config['hamiltonian']))
    for task in tasks.get_entries(type=Type.HESS, state=State.WIP, active=True):
        geometries = read_frames(inputs=task.inputs)
        minimum = get_minimum(geometries)
        hessian = compute_harmonic(minimum, hamiltonian, pos_shift=config['delta_shift'])
        future = store_hessian(minimum, hessian, outputs=task.outputs)
        task.outputs = future.outputs

    tasks.reset()
    return tasks



