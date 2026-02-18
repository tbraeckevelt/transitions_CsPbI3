import enum
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from functools import cached_property
from typing import Collection, Any
from parsl import File

import yaml
import numpy as np


class State(enum.Enum):
    MISSING_INPUT = -2
    FAILED = -1
    DONE = 0
    WIP = 1
    UNKNOWN = 2


class Type(enum.Enum):
    MD = 0
    OPT = 1
    HESS = 2
    LAMBDA = 3
    PIMD = 4


@dataclass
class TomTom:
    name: str
    file: Path
    pressure_grid: np.ndarray
    temperature_grid: np.ndarray
    harmonic_pressure_grid: np.ndarray
    harmonic_temperature_grid: np.ndarray
    pimd_pressure_grid: np.ndarray
    pimd_temperature_grid: np.ndarray


# TODO: bad coupling
extra_properties = {
    Type.LAMBDA: ('lambda_param',),
    Type.PIMD: ('nbeads', 'mass_scale'),
}


class TaskEntry:
    """"""
    type: Type = None
    extra_properties: tuple[str]
    inputs: Any

    def __init__(self, type: Type, name: str, temp: float, pressure: float, **kwargs):
        self.type = type
        self.state = State.UNKNOWN
        self.name = name
        self.temp = float(temp)
        if pressure is not None:
            self.pressure = float(pressure)
        else:
            self.pressure = pressure
        self.extra_properties = tuple(sorted(kwargs))
        for key in self.extra_properties:
            setattr(self, key, kwargs[key])
        self.outputs = []
        self.id = None  # When is this used?

        assert all(key in self.extra_properties for key in extra_properties.get(self.type, []))

    def check_state(self):
        # TODO: extend this
        exists = []
        for file in self.outputs:
            try:
                exists.append(Path(file).is_file())
            except TypeError:
                exists.append(Path(file.filepath).is_file())
        if all(exists):
            self.state = State.DONE

    @cached_property
    def properties(self): return ('name', 'temp', 'pressure') + self.extra_properties

    def get(self, k: str):
        return getattr(self, k)

    def get_properties(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.properties}

    def to_dict(self) -> dict:
        return ({'type': self.type.name, 'state': self.state.name, 'outputs': [f.filepath for f in self.outputs]} |
                self.get_properties())

    @staticmethod
    def from_dict(data: dict) -> 'TaskEntry':
        type, state, outputs = Type[data.pop('type')], State[data.pop('state')], data.pop('outputs')
        task = TaskEntry(type, **data)
        task.state, task.outputs = state, [File(_) for _ in outputs]
        return task

    @cached_property
    def hash(self) -> str:
        data = self.type.name, *(getattr(self, k) for k in self.properties)
        return hashlib.md5(str(data).encode('utf-8')).hexdigest()

    @property
    def key(self) -> str:
        return f'{self.hash}_{self.id}' if self.id is not None else self.hash

    def __str__(self):
        props = ', '.join([f'{k}: {v}' for k, v in self.get_properties().items()])
        return f'{self.type.name}({props}, {self.state.name})'


class TaskDatabase:
    """"""
    def __init__(self, entries: Collection[TaskEntry] = None):
        self.entries: dict[str, set[TaskEntry]] = {}
        self.active: set[TaskEntry] = set([])
        for task in (entries or []):
            self.store(task)

    def store(self, entry: TaskEntry) -> None:
        task_hash = entry.hash
        if task_hash not in self.entries:
            self.entries[task_hash] = set([])
        entry.id = len(self.entries[entry.hash])
        self.entries[entry.hash].add(entry)
        self.active.add(entry)

    def new_task(self, type: Type, **kwargs) -> TaskEntry:
        entry = TaskEntry(type, **kwargs)
        self.store(entry)
        return entry

    def get_entries(self, type: Type, active: bool = False, **kwargs) -> set[TaskEntry]:
        selected = set([])
        for entry in (self.active if active else self.all_entries):
            if entry.type == type:
                if all(k in entry.properties for k in kwargs if k != 'state'):
                    if all(getattr(entry, k) == v for k, v in kwargs.items()):
                        selected.add(entry)
        return selected

    def reset(self) -> None:
        self.active = set([])

    def merge(self, other: 'TaskDatabase') -> None:
        assert isinstance(other, TaskDatabase)
        for entry in other.all_entries:
            self.store(entry)

    def to_yaml(self, file: str | Path) -> None:
        data = {}
        for entry in self.all_entries:
            entry.check_state()
            data[entry.key] = entry.to_dict()
        yaml.safe_dump(data, Path(file).open('w'))

    @classmethod
    def from_yaml(cls, file: str | Path):
        tasks = [TaskEntry.from_dict(data) for data in yaml.safe_load(open(file)).values()]
        return cls(tasks)

    def count_states(self, active: bool = True) -> dict[State, int]:
        """"""
        state_count = {state: 0 for state in State}
        for entry in (self.active if active else self.all_entries):
            # entry.check_state()
            state_count[entry.state] += 1

        n_tot = sum(state_count.values())
        n_done, n_todo, n_missing = state_count[State.DONE], state_count[State.WIP], state_count[State.MISSING_INPUT]
        print(f'[TODO: {n_todo}/{n_tot} | DONE: {n_done}/{n_tot} | MISSING INPUT: {n_missing}/{n_tot}]')
        return state_count

    @property
    def all_entries(self) -> set[TaskEntry]:
        return set.union(*[entries for entries in self.entries.values()])

    def __len__(self):
        return sum([len(entries) for entries in self.entries.values()])


