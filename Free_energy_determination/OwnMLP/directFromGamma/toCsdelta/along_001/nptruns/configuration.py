import numpy as np

# NPT MD parameters
PRESSURE_GRID = np.array([0.1])
PRESSURE_NONE = np.array([None])
TEMPERATURE_GRID = np.logspace(np.log10(150), np.log10(600), 16, base=10)
HARM_T = [np.logspace(np.log10(150), np.log10(600), 16, base=10)[0]]
T_MAX =  [np.logspace(np.log10(150), np.log10(600), 16, base=10)[-1]]
STEP = 200
STEPS = 54000
FREQ_REX = 200
CALIBRATION = 4000

pathdata = '/pfs/lustrep2/scratch/project_465001125/tbraecke/CsPbI3_intermediates/OwnMLP/directFromGamma/toCsdelta/along_001/nptruns/data/'
names = ['int_1', 'int_2', 'int_1_double', 'int_2_double', 'int_3_double', 'int_4_double', 'int_5_double', 'int_6_double', 'Csdelta']
phases = {}
for name in names:
    phases[name] = pathdata + name + '.xyz'

config = dict(
    timestep = 2.0,  # For CsPbI3 a timestep of 2 fs is reasonable
    number_of_hamiltonians=1, #2,  # Specify how many hamiltonians you want to use, call the first one hamiltonian (used as default), and the other other_hamiltonian_1, other_hamiltonian_2, ...
    hamiltonian= pathdata + 'modelvasp.pth',
    #other_hamiltonian_1='/pfs/lustrep2/scratch/project_465001125/tbraecke/CsPbI3_intermediates/testmultistepprocedure/data/mace_mp_0_big.model',
    #To use the harmonic hamiltonian, the name of the hessian calculation has to be provided
    initialization=dict(
        multi=1,
        temperatures=HARM_T,
        pressures=PRESSURE_GRID,
        steps=10000,
        step=STEP,
        start=0,
        store_traj=True,
    ),
    sampling=dict(
        init_struc='initialization',
        init_last=True,
        multi=1,
        temperatures=TEMPERATURE_GRID,
        pressures=PRESSURE_GRID,
        steps=STEPS,
        step=STEP,
        start=CALIBRATION,
        store_traj=True,
    ),
    npt=dict(
        init_struc='sampling',
        multi=2,
        init_temperatures=TEMPERATURE_GRID,
        temperatures=TEMPERATURE_GRID,
        pressures=PRESSURE_GRID,
        steps=STEPS,
        step=STEP,
        start=CALIBRATION,
        store_traj=True,
        freq_rex_temperature=FREQ_REX,
    ),
    opt=dict(
        init_struc='npt',
        multi=400,
        temperatures=HARM_T,
        pressures=PRESSURE_GRID,
        f_max=1e-3,
    ),
    hess=dict(
        opt_struc='opt',
        temperatures=HARM_T,
        pressures=PRESSURE_GRID,
        delta_shift=0.001,
    ),
    ti_low=dict(
        hess_name='hess',  # if provided, the simulation always starts in its minimum
        hess_temperatures=HARM_T,
        hess_presssures=PRESSURE_GRID,
        begin_hamiltonian = '1.0xharmonic',  # this exact string format is needed: 'coefficient1xhamiltonian1+coefficient2xhamiltonian2+...'
        end_hamiltonian = '0.7xharmonic+0.3xhamiltonian',  # this exact string format is needed: 'coefficient1xhamiltonian1+coefficient2xhamiltonian2+...'
        lambda_path=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0],  # values 0.0 and 1.0 should be present in this list
        multi=4,
        temperatures=HARM_T,
        pressures=PRESSURE_NONE,
        steps=STEPS,
        step=STEP,
        start=CALIBRATION,
        store_traj=True,
    ),
    intermediate=dict(
        hess_name='hess',  # if provided, the simulation always starts in its minimum
        hess_temperatures=HARM_T,
        hess_presssures=PRESSURE_GRID,
        other_hamiltonian='0.7xharmonic+0.3xhamiltonian',  # this exact string format is needed: 'coefficient1xhamiltonian1+coefficient2xhamiltonian2+...'
        multi=4,
        temperatures=TEMPERATURE_GRID,
        pressures=PRESSURE_NONE,
        steps=STEPS,
        step=STEP,
        start=CALIBRATION,
        freq_rex_temperature=FREQ_REX,
    ),
    ti_high=dict(
        hess_name='hess',
        hess_temperatures=HARM_T,
        hess_presssures=PRESSURE_GRID,
        begin_hamiltonian='0.7xharmonic+0.3xhamiltonian',  # this exact string format is needed: 'coefficient1xhamiltonian1+coefficient2xhamiltonian2+...'
        end_hamiltonian='1.0xhamiltonian',  # this exact string format is needed: 'coefficient1xhamiltonian1+coefficient2xhamiltonian2+...'
        lambda_path=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0],  # values 0.0 and 1.0 should be present in this list
        multi=4,
        temperatures=T_MAX,
        pressures=PRESSURE_NONE,
        steps=STEPS,
        step=STEP,
        start=CALIBRATION,
        store_traj=True,
    ),
    nvt=dict(
        init_struc='opt',
        multi=4,
        temperatures=TEMPERATURE_GRID,
        pressures=PRESSURE_NONE,
        steps=STEPS,
        step=STEP,
        start=CALIBRATION,
        store_traj=True,
        freq_rex_temperature=FREQ_REX,
    ),
)
'''
    ti_lot=dict(
        init_struc='nvt',
        init_temperatures=T_MAX,
        begin_hamiltonian='1.0xhamiltonian',  # this exact string format is needed: 'coefficient1xhamiltonian1+coefficient2xhamiltonian2+...'
        end_hamiltonian='1.0xother_hamiltonian_1',  # this exact string format is needed: 'coefficient1xhamiltonian1+coefficient2xhamiltonian2+...'
        lambda_path=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],  # values 0.0 and 1.0 should be present in this list
        multi=4,
        temperatures=T_MAX,
        pressures=PRESSURE_NONE,
        steps=STEPS,
        step=STEP,
        start=CALIBRATION,
        store_traj=True,
    ),
    nvt_lot=dict(
        init_struc='nvt',
        init_temperatures=TEMPERATURE_GRID,
        other_hamiltonian='1.0xother_hamiltonian_1',
        multi=4,
        temperatures=TEMPERATURE_GRID,
        pressures=PRESSURE_NONE,
        steps=STEPS,
        step=STEP,
        start=CALIBRATION,
        store_traj=True,
        freq_rex_temperature=FREQ_REX,
    ),
    opt_lot=dict(
        init_struc='nvt_lot',
        multi=400,
        temperatures=HARM_T,
        pressures=PRESSURE_NONE,
        f_max=1e-3,
    ),
    hess_lot=dict(
        opt_struc='opt_lot',
        temperatures=HARM_T,
        pressures=PRESSURE_NONE,
        delta_shift=0.001,
    ),
    
)
'''
