from pathlib import Path

import numpy as np
from parsl import python_app, File
from psiflow.geometry import Geometry
from psiflow.data import Dataset
from psiflow.data.utils import _count_frames, _read_frames
from psiflow.sampling.optimize import optimize_dataset

from .psiflow_utils import load_hamiltonian, _read_simulation_output, cell_tril_to_cell, _scale_geometry, is_valid_input
from .workflow import State, Type, TaskDatabase, TaskEntry

# TODO: what is 1 out of multi opts fail?

KEY_PROPS = 'props'
KEY_TRAJ = 'traj'
KEY_CELL = 'cell_h{angstrom}'
DEFAULT_KWARGS = {'init_struc': 'all', 'scale_cell': True}
REQUIRED_KWARGS = ('multi', 'f_max', 'temperatures', 'pressures', 'hamiltonian', 'workdir')


def check_kwargs(**kwargs) -> dict:
    EXTRA_KWARGS = []
    if 'number_of_hamiltonians' in kwargs:
        for i in range(1, kwargs['number_of_hamiltonians']):
            EXTRA_KWARGS.append(f'other_hamiltonian_{i}')
    return {k: kwargs.get(k, DEFAULT_KWARGS[k]) for k in DEFAULT_KWARGS} | {k: kwargs[k] for k in REQUIRED_KWARGS} | {k: kwargs[k] for k in EXTRA_KWARGS}


def check_task_state(tasks: TaskDatabase, config: dict, name: str) -> TaskDatabase:
    """"""
    multi, workdir = config['multi'], config['workdir']
    temps, pressures, multi = config['temperatures'], config['pressures'], config['multi']

    if config['init_struc'] == 'file':
        print(f'use initial file for {name} run')

    for (t, p) in [(t, p) for t in temps for p in pressures]:
        if config['init_struc'] != 'file':
            # check for available input simulations
            files_props, files_traj = [], []
            kwargs = {"type": Type.MD, "temp": t, "pressure": p}
            if config['init_struc'] != 'all':
                kwargs["name"] = config['init_struc']
            for intask in tasks.get_entries(**kwargs):
                if intask.state == State.MISSING_INPUT:
                    continue
                files_props.append(intask.outputs[0])
                if len(intask.outputs) == 2:
                    files_traj.append(intask.outputs[1])
            # both props and trajectories are needed
            input_valid = bool(files_props) and bool(files_traj)
        else:
            files_traj = [config['init_file']]
            input_valid = is_valid_input(files_traj[0])
            files_props = None

        if p is not None:
            file_opt = workdir / f'opt_T{round(t)}_P{round(p, 1)}.xyz'
        else:
            file_opt = workdir / f'opt_T{round(t)}.xyz'

        task = tasks.new_task(type=Type.OPT, name = name, temp=t, pressure=p)
        task.outputs = [File(file_opt)]
        task.inputs = {KEY_PROPS: files_props, KEY_TRAJ: files_traj}
        if file_opt.is_file() and _count_frames([file_opt]) == multi:
            task.state = State.DONE
        else:
            task.state = State.WIP if input_valid else State.MISSING_INPUT

    return tasks


@python_app(executors=["default_threads"])
def sample_geometries(n: int, inputs: list[File]) -> list[Geometry]:
    """Sample n geometries from input files with equal spacing inbetween"""
    counts = np.cumsum([0] + [_count_frames([f]) for f in inputs])
    ids = np.linspace(0, counts[-1], n, endpoint=False, dtype=int)
    geometries = []
    for idx, file in enumerate(inputs):
        mask_in_bounds = (counts[idx] <= ids) * (ids < counts[idx+1])
        ids_shifted = ids[mask_in_bounds] - counts[idx]
        geometries += _read_frames(ids_shifted.tolist(), inputs=[file])
    assert len(geometries) == n
    return geometries


@python_app(executors=["default_threads"])
def get_average_cell(inputs: list[File]) -> np.ndarray:
    props = _read_simulation_output(inputs)
    cell_h_av = np.concatenate([d[KEY_CELL] for d in props]).mean(axis=0)
    return cell_tril_to_cell(cell_h_av)


@python_app(executors=["default_threads"])
def adjust_geometries_to_cell(geometries: list[Geometry], cell: np.ndarray) -> list[Geometry]:
    """"""
    return [_scale_geometry(geom, cell) for geom in geometries]


def main(name: str, tasks: TaskDatabase, config: dict) -> TaskDatabase:
    """"""
    tasks = check_task_state(tasks, config, name)
    if not (_ := tasks.count_states()[State.WIP]):
        tasks.reset()
        return tasks
    else:
        print(f'Submitting {_} optimisation batches.')

    hamiltonian = load_hamiltonian(Path(config['hamiltonian']))
    for task in tasks.get_entries(type=Type.OPT, state=State.WIP, active=True):
        geometries = sample_geometries(n=config['multi'], inputs=task.inputs[KEY_TRAJ])
        if config['scale_cell']:
            assert task.inputs[KEY_PROPS] is not None, 'the cell properties of the input trajectories should be provided'
            cell = get_average_cell(inputs=task.inputs[KEY_PROPS])
            average_geometries = adjust_geometries_to_cell(geometries, cell)
        data_opt = optimize_dataset(
            Dataset(states=average_geometries), hamiltonian, f_max=config['f_max'], mode='fix_cell'
        )
        data_opt.save(task.outputs[0].filepath)
        if not Path.exists(Path(task.outputs[0].filepath)):
            task.outputs = [data_opt.extxyz]  # suboptimal: save should return datafutures such that we could work with them instead 

    tasks.reset()
    return tasks


