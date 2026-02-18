import time
import sys
import pickle
from pathlib import Path
from typing import Any, Union, Tuple, Optional

import ase
import parsl
import typeguard
import numpy as np
from parsl import python_app, File
from parsl.dataflow.futures import DataFuture, AppFuture
from parsl.utils import Timer
from parsl.dataflow.states import FINAL_FAILURE_STATES, FINAL_STATES
from psiflow.geometry import Geometry
from psiflow.hamiltonians import MACEHamiltonian, Hamiltonian, MixtureHamiltonian
from psiflow.sampling import Walker, SimulationOutput
from psiflow.models import MACE, load_model
from psiflow.free_energy.phonons import _c, _compute_frequencies, second


from .state import ThermodynamicState, WalkerState
from .workflow import TaskDatabase


PATHLIKE = str | Path | File | DataFuture
FINAL_SUCCESS_STATES = [k for k in FINAL_STATES if k not in FINAL_FAILURE_STATES]


@typeguard.typechecked
def running_average(arr: Union[np.ndarray, list], window: int) -> np.ndarray:
    arr_avg = np.zeros(len(arr)-window+1)
    for i in range(len(arr)-window+1):
        arr_avg[i] = np.average(arr[i:(i+window)])
    return arr_avg


@typeguard.typechecked
def calculate_hist(
    arr: Union[np.ndarray, list],
    bin_size: float,
    density: bool = True
) -> Tuple[np.ndarray, np.ndarray]:

    arr_min = np.floor(np.min(arr)/bin_size)*bin_size
    arr_max = np.ceil(np.max(arr)/bin_size)*bin_size
    bin = np.arange(arr_min, arr_max+bin_size, bin_size)

    hist, _ = np.histogram(arr, bins=bin, density=density)
    center_bin = running_average(bin, 2)
    return hist, center_bin


def make_path(path: PATHLIKE) -> Path:
    """"""
    if isinstance(path, File | DataFuture):
        path = path.filepath
    return Path(path)


def _log_states(dfk, file):
    with dfk.task_state_counts_lock:
        n_tot = dfk.task_count
        n_succes = sum([dfk.task_state_counts[k] for k in FINAL_SUCCESS_STATES])
        n_fail = sum([dfk.task_state_counts[k] for k in FINAL_FAILURE_STATES])
    n_remain = n_tot - n_succes - n_fail
    msg = f'(STATUS) {time.asctime()} | total: {n_tot}, completed: {n_succes}, failed: {n_fail}, remaining: {n_remain}'
    print(msg, file=file)


def monitor_task_states(interval: float = 300, file: str | Path = sys.stdout):
    """Keep track of job progress, without crashing parsl"""
    print(f'Monitoring Parsl tasks')
    dfk = parsl.dfk()
    Timer(_log_states, dfk, file, interval=interval)


def monitor_task_db(tasks: TaskDatabase, file: str | Path, interval: float = 300):
    """Periodically update task database"""
    tasks.to_yaml(file)
    Timer(tasks.to_yaml, file, interval=interval)


def is_valid_input(file: PATHLIKE) -> bool:
    """"""
    if not isinstance(file, PATHLIKE):
        return False
    path = make_path(file)
    if not path.is_file():
        return False
    return True


def load_hamiltonian(path_model: Path) -> MACEHamiltonian:
    """"""
    assert path_model.exists()
    if (_ := Path(path_model)).is_dir():
        model: MACE = load_model(_)
        hamiltonian = model.create_hamiltonian()
    else:
        # TODO: are atomic energies important for MD?
        hamiltonian = MACEHamiltonian(File(_), {})
    return hamiltonian


@python_app(executors=["default_threads"])
def load_hessian(file: Path | File | DataFuture) -> tuple[Geometry, np.ndarray]:
    """"""
    if isinstance(file, File):
        file = file.filepath
    return pickle.load(Path(file).open('rb'))


@typeguard.typechecked
def _get_average_cell(geometries: list[Geometry]) -> np.ndarray:
    return np.mean([geo.cell for geo in geometries], axis=0)


get_average_cell = python_app(_get_average_cell, executors=["default_threads"])


@typeguard.typechecked
def _scale_geometry(geometry: Geometry, cell: np.ndarray) -> Geometry:
    atoms = ase.Atoms(positions=geometry.per_atom.positions, numbers=geometry.per_atom.numbers, cell=geometry.cell)
    atoms.set_cell(cell, scale_atoms=True)
    geometry = geometry.copy()
    geometry.per_atom.positions = atoms.positions
    geometry.cell = cell
    return geometry


scale_geometry = python_app(_scale_geometry, executors=["default_threads"])


@typeguard.typechecked
def _get_minimum(geometries: list[Geometry]) -> Geometry:
    energies = np.array([geom.energy for geom in geometries])
    return geometries[energies.argmin()]


get_minimum = python_app(_get_minimum, executors=["default_threads"])


@typeguard.typechecked
def _is_positive_definite(
        hessian: np.ndarray,
        geometry: Geometry,
        threshold: float = 1
) -> bool:
    """
    Check if the Hessian matrix is positive definite based on the
    computed frequencies.

    Returns:
        bool: True if the Hessian matrix is positive definite, False otherwise.
    """
    frequencies = _compute_frequencies(hessian, geometry)
    threshold_ase = threshold / second * (100 * _c)                     # invcm to ASE
    frequencies = frequencies[np.abs(frequencies) > threshold_ase]
    return all(frequencies > 0)


is_positive_definite = python_app(_is_positive_definite, executors=["default_threads"])

'''
@python_app(executors=["default_threads"])
def save_simulation_output(output: SimulationOutput, walker: Walker, status: int, outputs: list[File]) -> None:
    """"""
    # TODO: this function internally works with futures.. why does this work?
    from psiflow.sampling.output import potential_component_names

    if status == -1:
        # simulation encountered an unknown error
        return

    # evaluate futures
    data = {k: v.result() for k, v in output._data.items()}

    # rename hamiltonian contributions
    names_ipi = potential_component_names(len(output.hamiltonians))
    names = [h.__class__.__name__ for h in output.hamiltonians]
    for name_ipi, name in zip(names_ipi, names):
        data[name] = data.pop(name_ipi)

    # store state for easy analysis
    data['state'] = make_state_from_walker(walker).result()
    pickle.dump(data, open(outputs[0], 'wb'))

    if len(outputs) == 2:
        output.trajectory.save(outputs[1].filepath)

    return
'''

@python_app(executors=["default_threads"])
def save_simulation_output(output: SimulationOutput, walker: Walker, status: int, outputs: list[File]) -> None:
    """  """
    # TODO: nesting apps?
    @python_app(executors=["default_threads"])
    def _save_props(walkerstate, **kwargs):
        from psiflow.sampling.output import potential_component_names

        # rename hamiltonian contributions
        names_ipi = potential_component_names(len(output.hamiltonians))
        names = [h.__class__.__name__ for h in output.hamiltonians]
        for name_ipi, name in zip(names_ipi, names):
            count=0
            while name+str(count) in kwargs:
                count+=1
            kwargs[name+str(count)] = kwargs.pop(name_ipi)
        kwargs['state'] = walkerstate
        pickle.dump(kwargs, open(outputs[0], 'wb'))

    if status == -1:
        # simulation encountered an unknown error
        return
    walkerstate = make_state_from_walker(walker)
    _save_props(walkerstate, **output._data)

    if len(outputs) == 2:
        output.trajectory.save(outputs[1].filepath)

    return


def _read_simulation_output(inputs: list[PATHLIKE]) -> list[dict]:
    """"""
    return [pickle.load(open(file, 'rb')) for file in inputs]


read_simulation_output = python_app(_read_simulation_output, executors=["default_threads"])


def _state_from_args(hamiltonian: Hamiltonian,
                     geom: Geometry,
                     temperature: float,
                     pressure: Optional[float] = None) -> 'WalkerState':
    """"""
    mixture: MixtureHamiltonian = 1 * hamiltonian
    desc = []
    for c, h in zip(mixture.coefficients, mixture.hamiltonians):
        desc.append(f'{c:.4f} x {h.__class__.__name__}')
    desc = ' + '.join(desc)
    return WalkerState(
        hamiltonian=desc,
        temperature=temperature,
        pressure=pressure,
        natoms=len(geom),
        cell=geom.cell if pressure is None else None,
        )


state_from_args = python_app(_state_from_args, executors=["default_threads"])


def make_state_from_walker(walker: Walker) -> AppFuture:
    """"""
    return state_from_args(walker.hamiltonian, walker.start, temperature=walker.temperature, pressure=walker.pressure)


def cell_tril_to_cell(cell_tril: np.ndarray) -> np.ndarray:
    """"""
    cell = np.zeros(shape=(3, 3))
    cell[0, 0], cell[1, 1], cell[2, 2], cell[1, 0], cell[2, 0], cell[2, 1] = cell_tril
    return cell


def fill_refs(values, length):
    if isinstance(values, (list, np.ndarray)):
        return np.full(length, values[0]) if len(values) == 1 else np.array(values)
    return np.full(length, values)


def get_hamiltonian(ham_str: str, config: dict, harmonic=None):
    if ham_str == 'harmonic':
        if harmonic is None:
            raise ValueError("harmonic is not defined but was requested")
        return harmonic
    if ham_str not in config:
        raise KeyError(f"{ham_str} not found in config")
    return load_hamiltonian(Path(config[ham_str]))


@python_app(executors=["default_threads"])
def get_geometry(ind: int, *files: list[Geometry]) -> Geometry:
    geos = []
    for file in files:
        traj = ase.io.read(str(file), index=':')
        for atoms in traj:
            geos.append(Geometry.from_atoms(atoms))
    return geos[ind % len(geos)]


#
# @join_app
# @typeguard.typechecked
# def make_positive_definite(
#         attempt: int,
#         geometry: Geometry,
#         hessian: np.ndarray,
#         hamiltonian: Hamiltonian,
#         max_attempt: int = 10,
# ) -> AppFuture:
#     """
#     Recursively attempts to make the Hessian matrix positive definite by
#     perturbing the geometry (with increasing amplitude) and optimizing it.
#
#     Parameters:
#         attempt (int): The current attempt number.
#         geometry (Geometry): The initial geometry.
#         hessian (np.ndarray): The Hessian matrix.
#         hamiltonian (Hamiltonian): The Hamiltonian object.
#         max_attempt (int, optional): Maximum number of attempts. Default=10.
#
#     Returns:
#         AppFuture: The future object representing the result of the operation.
#     """
#     assert attempt < max_attempt, "The maximum number of attempts is reached."
#     if positive_definite(hessian, geometry):
#         return make_lstfuture(geometry, hessian)
#     else:
#         new_geometry = perturb(geometry, 0.0 1 *attempt)
#         opt_geometry = optimize(
#             new_geometry,
#             hamiltonian,
#             ftol=1e-4,
#             steps=4000,
#             mode='bfgstrm'
#         )
#         new_hessian = compute_harmonic(
#             opt_geometry,
#             hamiltonian,
#             pos_shift=0.001
#         )
#         return make_positive_definite(
#             attemp t +1,
#             opt_geometry,
#             new_hessian,
#             hamiltonian
#         )
#
#
