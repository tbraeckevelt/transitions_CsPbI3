from pathlib import Path

import ase.io
import psiflow
from parsl import File

import sys
from thermoflow.psiflow_utils import monitor_task_states, monitor_task_db
from thermoflow.workflow import TaskDatabase


# TODO: actual logging?

DIR_PREFIX = 'phase'
FILE_STRUCTURE = 'structure.xyz'
FILE_CONFIG = 'config.yaml'
FILE_TASK_YAML = 'tasks.yaml'

# these keys can be defined in the overall config to specify additional options
KEY_INITIALIZATION = 'initialization'
KEY_SAMPLING = 'sampling'
KEY_NPT = 'npt'
KEY_OPT = 'opt'
KEY_HESS = 'hess'
KEY_TI_LOW = 'ti_low'
KEY_INTERMEDIATE = 'intermediate'
KEY_TI_HIGH = 'ti_high'
KEY_NVT = 'nvt'
KEY_TI_LOT = 'ti_lot'
KEY_NVT_LOT = 'nvt_lot'
KEY_OPT_LOT = 'opt_lot'
KEY_HESS_LOT = 'hess_lot'
WORKFLOW_KEYS = (KEY_INITIALIZATION, KEY_SAMPLING, KEY_NPT, KEY_OPT, KEY_HESS, KEY_TI_LOW, KEY_INTERMEDIATE, KEY_TI_HIGH, KEY_NVT)#, KEY_TI_LOT, KEY_NVT_LOT, KEY_OPT_LOT, KEY_HESS_LOT)


def check_input():
    """"""
    assert all(key in config for key in WORKFLOW_KEYS), 'Incomplete simulation config data'


def do_MD(dir_phase, keyname, tasks, config):
    (workdir := dir_phase / keyname).mkdir(exist_ok=True)
    from thermoflow.sample import main, check_kwargs
    kwargs = check_kwargs(
        workdir=workdir,
        **(config | config[keyname])
    )
    if kwargs['init_struc'] == 'file':
        kwargs['init_file'] = File(dir_phase / FILE_STRUCTURE)
    return main(keyname, tasks, kwargs)


def do_TI(dir_phase, keyname, tasks, config):
    (workdir := dir_phase / keyname).mkdir(exist_ok=True)
    from thermoflow.ti import main, check_kwargs
    kwargs = check_kwargs(
        workdir=workdir,
        **(config | config[keyname])
    )
    if kwargs['init_struc'] == 'file':
        kwargs['init_file'] = File(dir_phase / FILE_STRUCTURE)
    return main(keyname, tasks, kwargs)


def do_OPT(dir_phase, keyname, tasks, config):
    (workdir := dir_phase / keyname).mkdir(exist_ok=True)
    from thermoflow.optimise import main, check_kwargs
    kwargs = check_kwargs(
        workdir=workdir,
        **(config | config[keyname])
    )
    if kwargs['init_struc'] == 'file':
        kwargs['init_file'] = File(dir_phase / FILE_STRUCTURE)
    return main(keyname, tasks, kwargs)


def do_HESS(dir_phase, keyname, tasks, config):
    (workdir := dir_phase / keyname).mkdir(exist_ok=True)
    from thermoflow.hessian import main, check_kwargs
    kwargs = check_kwargs(
        workdir=workdir,
        **(config | config[keyname])
    )
    return main(keyname, tasks, kwargs)


def main():
    """"""
    root = Path.cwd()
    check_input()
    for phase, file in phases.items():
        # work in a separate directory for each phase
        dir_phase = root / f'{DIR_PREFIX}-{phase}'
        try:
            dir_phase.mkdir()
            print(f'Creating working directory: {dir_phase.stem}')
        except FileExistsError:
            print(f'Current working directory: {dir_phase.stem}')
        if not (_ := dir_phase / FILE_STRUCTURE).is_file():
            ase.io.write(_, ase.io.read(file))
        (dir_phase / FILE_CONFIG).write_text(f'file: {file}\n')
        (dir_phase / FILE_CONFIG).write_text('\n'.join([f'{k}: {v}' for k, v in config.items()]))
        tasks = TaskDatabase()

        # initialization
        print('Scheduling initialization sampling step..')
        tasks = do_MD(dir_phase, KEY_INITIALIZATION, tasks, config)

        # sampling
        print('Scheduling explorative sampling step..')
        tasks = do_MD(dir_phase, KEY_SAMPLING, tasks, config)

        # npt
        print('Scheduling NPT sampling step..')
        tasks = do_MD(dir_phase, KEY_NPT, tasks, config)

        # optimizations
        print('Scheduling HO optimisation step..')
        tasks = do_OPT(dir_phase, KEY_OPT, tasks, config)

        # Hessian calculation
        print('Scheduling HO Hessian step..')
        tasks = do_HESS(dir_phase, KEY_HESS, tasks, config)

        # compute harmonic correction at low temperature
        print('Scheduling Lambda integration step at low temperature..')
        tasks = do_TI(dir_phase, KEY_TI_LOW, tasks, config)
        
        # intermediate NVT MD
        print('Scheduling intermediate NVT sampling step..')
        tasks = do_MD(dir_phase, KEY_INTERMEDIATE, tasks, config)

        # compute harmonic correction at high temperature
        print('Scheduling Lambda integration step at high temperature..')
        tasks = do_TI(dir_phase, KEY_TI_HIGH, tasks, config)

        # nvt
        print('Scheduling NVT sampling step..')
        tasks = do_MD(dir_phase, KEY_NVT, tasks, config)

        # compute LOT correction at high temperature
        #print('Scheduling Lambda integration step at high temperature between different levels of theory..')
        #tasks = do_TI(dir_phase, KEY_TI_LOT, tasks, config)

        # nvt at a different level of theory
        #print('Scheduling NVT at a different level of theory sampling step..')
        #tasks = do_MD(dir_phase, KEY_NVT_LOT, tasks, config)

        # optimizations at a different level of theory
        #print('Scheduling HO optimisation step at a different level of theory..')
        #tasks = do_OPT(dir_phase, KEY_OPT_LOT, tasks, config)

        # Hessian calculation
        #print('Scheduling HO Hessian step at a different level of theory..')
        #tasks = do_HESS(dir_phase, KEY_HESS_LOT, tasks, config)

        monitor_task_db(tasks, dir_phase / FILE_TASK_YAML, 60)
        print(f'Stored tasks: {len(tasks)}')
        print('That\'s all folks')

    return


if __name__ == '__main__':
    """"""
    from configuration import phases, config
    with psiflow.load():
        monitor_task_states(60)
        main()
    print('Done')
