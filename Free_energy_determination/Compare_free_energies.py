from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
#from execute_workflow import DIR_PREFIX
DIR_PREFIX = 'phase'
from collections import defaultdict
from ase.units import eV, kJ, mol

kJmol = kJ/mol
to_kJmol = eV/kJmol


def flatten_dct(newdict, subdict, catkey):
    for key, val in subdict.items():
        newkey = catkey+'/'+key
        if newkey[0] == '/':
            newkey = newkey[1:]
        if isinstance(val, dict):
            flatten_dct(newdict, val, newkey)
        else:
            newdict[newkey] = val

structures = {
    'viaZeta': {
        'toZeta': {
            'nvtruns': ['gamma', 'int_1', 'int_2', 'int_3'], #['gamma', 'int_1', 'int_2', 'int_3', 'zeta'],
            'nptruns': ['gamma', 'int_1', 'int_2', 'int_3'], #['gamma', 'int_1', 'int_2', 'int_3', 'zeta'],
        },
        'toCsdelta': {
            'nvtruns': ['zeta', 'zeta_shift', 'int_1', 'int_2a', 'int_2b', 'int_3', 'int_3and4', 'int_4a', 'int_4b', 'Csdelta'],
            'nptruns': ['zeta', 'zeta_shift', 'int_1', 'int_2a', 'int_2b', 'int_3', 'int_3and4', 'int_4a', 'int_4b', 'Csdelta'],
        },
        'toFAdelta': {
            'nvtruns': ['int_1', 'int_2', 'int_3', 'int_layer', 'FAdelta'], #['zeta', 'int_1', 'int_2', 'int_3', 'int_layer', 'FAdelta'],
            'nptruns': ['int_1', 'int_2', 'int_3', 'int_layer', 'FAdelta'], #['zeta', 'int_1', 'int_2', 'int_3', 'int_layer', 'FAdelta'],
        }
    },
    'directFromGamma': {
        'toFAdelta': {
            'nvtruns': ['int_1', 'int_2', 'int_3', 'FAdelta', 'interesting_struc'], #['gamma', 'int_1', 'int_2', 'int_3', 'FAdelta', 'interesting_struc'],
            'nptruns': ['int_1', 'int_2', 'int_3', 'FAdelta', 'interesting_struc'], #['gamma', 'int_1', 'int_2', 'int_3', 'FAdelta', 'interesting_struc'],
        },
        'toCsdelta': {
            'along_001': {
                'nvtruns': ['int_1', 'int_2', 'int_1_double', 'int_2_double', 'int_3_double', 'int_4_double', 'int_5_double', 'int_6_double', 'Csdelta'], #['gamma', 'int_1', 'int_2', 'int_1_double', 'int_2_double', 'int_3_double', 'int_4_double', 'int_5_double', 'int_6_double', 'Csdelta'],
                'nptruns': ['int_1', 'int_2', 'int_1_double', 'int_2_double', 'int_3_double', 'int_4_double', 'int_5_double', 'int_6_double', 'Csdelta'], #['gamma', 'int_1', 'int_2', 'int_1_double', 'int_2_double', 'int_3_double', 'int_4_double', 'int_5_double', 'int_6_double', 'Csdelta'],
            },
            #'along_012': {
            #    'sametilts': {
            #        'nvtruns': ['int_layer_1a', 'int_layer_1a_connected', 'int_layer_1b', 'int_layer_1b_connected', 'Csdelta_sametilts', 'Csdelta_sametilts_connected'], #['gamma', 'int_layer_1a', 'int_layer_1a_connected', 'int_layer_1b', 'int_layer_1b_connected', 'Csdelta_sametilts', 'Csdelta_sametilts_connected'],
            #        'nptruns': ['int_layer_1a', 'int_layer_1a_connected', 'int_layer_1b', 'int_layer_1b_connected', 'Csdelta_sametilts', 'Csdelta_sametilts_connected'], #['gamma', 'int_layer_1a', 'int_layer_1a_connected', 'int_layer_1b', 'int_layer_1b_connected', 'Csdelta_sametilts', 'Csdelta_sametilts_connected'],
            #    },
            #    'withtilts': {
            #        'nvtruns': ['int_layer_1a_connected', 'int_layer_1b_double', 'int_layer_1b_double_connected', 'int_layer_1c_double', 'int_layer_1c_double_connected', 'int_layer_2c_double', 'int_layer_2c_double_connected', 'Csdelta', 'Csdelta_connected'], #['gamma', 'int_layer_1a_connected', 'int_layer_1b_double', 'int_layer_1b_double_connected', 'int_layer_1c_double', 'int_layer_1c_double_connected', 'int_layer_2c_double', 'int_layer_2c_double_connected', 'Csdelta', 'Csdelta_connected'],
            #        'nptruns': ['int_layer_1a_connected', 'int_layer_1b_double', 'int_layer_1b_double_connected', 'int_layer_1c_double', 'int_layer_1c_double_connected', 'int_layer_2c_double', 'int_layer_2c_double_connected', 'Csdelta', 'Csdelta_connected'], #['gamma', 'int_layer_1a_connected', 'int_layer_1b_double', 'int_layer_1b_double_connected', 'int_layer_1c_double', 'int_layer_1c_double_connected', 'int_layer_2c_double', 'int_layer_2c_double_connected', 'Csdelta', 'Csdelta_connected'],
            #    }
            #}
        }
    },
    'NucleationStruc': {
        'Zetastruc': {
            '0dim': {
                'nvtruns': ['gamma', 'int_3by3_Csopt1', 'int_3by3_Csopt2', 'int_4by2_Csopt1', 'int_4by2_Csopt2', 'int_4by2_and_2by1_Csopt1', 'int_4by2_and_2by1_Csopt2', 'int_4by2_and_2by2_Csopt1', 'int_4by2_and_2by2_Csopt2', 'int_line_Csopt1', 'int_line_Csopt2'],
                'nptruns': ['gamma', 'int_3by3_Csopt1', 'int_3by3_Csopt2', 'int_4by2_Csopt1', 'int_4by2_Csopt2', 'int_4by2_and_2by1_Csopt1', 'int_4by2_and_2by1_Csopt2', 'int_4by2_and_2by2_Csopt1', 'int_4by2_and_2by2_Csopt2', 'int_line_Csopt1', 'int_line_Csopt2'],
            },
            '1dim':{
                'nvtruns': ['int_1', 'int_2a', 'int_2b', 'int_layer_1'], #['gamma', 'int_1', 'int_2a', 'int_2b', 'int_layer_1'],
                'nptruns': ['int_1', 'int_2a', 'int_2b', 'int_layer_1'], #['gamma', 'int_1', 'int_2a', 'int_2b', 'int_layer_1'],
            }
        },
        'FAdeltastruc': {
            'nvtruns': ['int_1a', 'int_1b', 'int_1c', 'int_1a_double', 'int_2a_double', 'int_3a_double', 'int_4a_double', 'int_4b_double', 'int_4c_double'], #['gamma', 'int_1a', 'int_1b', 'int_1c', 'int_1a_double', 'int_2a_double', 'int_3a_double', 'int_4a_double', 'int_4b_double', 'int_4c_double'],
            'nptruns': ['int_1a', 'int_1b', 'int_1c', 'int_1a_double', 'int_2a_double', 'int_3a_double', 'int_4a_double', 'int_4b_double', 'int_4c_double'], #['gamma', 'int_1a', 'int_1b', 'int_1c', 'int_1a_double', 'int_2a_double', 'int_3a_double', 'int_4a_double', 'int_4b_double', 'int_4c_double'],
        },
        'Csdeltastruc': {
            'along_001': {
                'nvtruns': ['gamma_double', 'int_1', 'int_1_double', 'int_2_double', 'int_3_double', 'int_4_double', 'int_5_double'],
                'nptruns': ['gamma_double', 'int_1', 'int_1_double', 'int_2_double', 'int_3_double', 'int_4_double', 'int_5_double'],
            },
            #'along_012': {
            #    'sametilts': {
            #        'nvtruns': ['int_1', 'int_1_connected', 'int_2a', 'int_2a_connected', 'int_2b', 'int_2b_connected', 'int_3a', 'int_3a_connected', 'int_3b_connected', 'int_3c', 'int_3c_connected'],
            #        'nptruns': ['int_1', 'int_1_connected', 'int_2a', 'int_2a_connected', 'int_2b', 'int_2b_connected', 'int_3a', 'int_3a_connected', 'int_3b_connected', 'int_3c', 'int_3c_connected'],
            #    },
            #    'withtilts': {
            #        'nvtruns': ['int_1_connected', 'int_2', 'int_2_connected', 'int_2b_double_connected'],
            #        'nptruns': ['int_1_connected', 'int_2', 'int_2_connected', 'int_2b_double_connected'],
            #    }
            #}
        }
    }
}

flatten_structures = {}
flatten_dct(flatten_structures, structures, '')
root =Path.cwd()
TEMPERATURE_GRID = np.logspace(np.log10(150), np.log10(600), 16, base=10)
HARM_T = [np.logspace(np.log10(150), np.log10(600), 16, base=10)[0]]

free_e_ref = {}
error_e_ref = {}
for ens in ['nvtruns', 'nptruns']:
    free_e_ref[ens] = {}
    error_e_ref[ens] = {}
    for pes in ['harm', 'intermediate', 'nvt']:  #, 'harm_lot', 'nvt_lot']:
        free_e_ref[ens][pes] = np.zeros_like(TEMPERATURE_GRID, dtype=float)
        error_e_ref[ens][pes] = np.zeros_like(TEMPERATURE_GRID, dtype=float)
        dir_phase = root / 'viaZeta' / 'toZeta' / ens / f'{DIR_PREFIX}-gamma' / 'plots' / 'Free_energy'

        with open(dir_phase / f'{pes}_Tharm_{HARM_T[0]:.0f}K.csv', "r", newline="") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for i, row in enumerate(reader):
                free_e_ref[ens][pes][i] = 5*to_kJmol*float(row[1])
                error_e_ref[ens][pes][i] = 5*to_kJmol*float(row[2])

free_e_ref_double = {}
error_e_ref_double = {}
for ens in ['nvtruns', 'nptruns']:
    free_e_ref_double[ens] = {}
    error_e_ref_double[ens] = {}
    for pes in ['harm', 'intermediate', 'nvt']:  #, 'harm_lot', 'nvt_lot']:
        free_e_ref_double[ens][pes] = np.zeros_like(TEMPERATURE_GRID, dtype=float)
        error_e_ref_double[ens][pes] = np.zeros_like(TEMPERATURE_GRID, dtype=float)
        dir_phase = root / 'NucleationStruc' / 'Csdeltastruc' / 'along_001' / ens / f'{DIR_PREFIX}-gamma_double' / 'plots' / 'Free_energy'

        with open(dir_phase / f'{pes}_Tharm_{HARM_T[0]:.0f}K.csv', "r", newline="") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for i, row in enumerate(reader):
                free_e_ref_double[ens][pes][i] = 5*to_kJmol*float(row[1])
                error_e_ref_double[ens][pes][i] = 5*to_kJmol*float(row[2])

free_e_ref_540 = {}
error_e_ref_540 = {}
for ens in ['nvtruns', 'nptruns']:
    free_e_ref_540[ens] = {}
    error_e_ref_540[ens] = {}
    for pes in ['harm', 'intermediate', 'nvt']:  #, 'harm_lot', 'nvt_lot']:
        free_e_ref_540[ens][pes] = np.zeros_like(TEMPERATURE_GRID, dtype=float)
        error_e_ref_540[ens][pes] = np.zeros_like(TEMPERATURE_GRID, dtype=float)
        dir_phase = root / 'NucleationStruc' / 'Zetastruc' / '0dim' / ens / f'{DIR_PREFIX}-gamma' / 'plots' / 'Free_energy'

        with open(dir_phase / f'{pes}_Tharm_{HARM_T[0]:.0f}K.csv', "r", newline="") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for i, row in enumerate(reader):
                free_e_ref_540[ens][pes][i] = 5*to_kJmol*float(row[1])
                error_e_ref_540[ens][pes][i] = 5*to_kJmol*float(row[2])

for pes in ['nvt']: # ['harm', 'intermediate', 'nvt']:  #, 'harm_lot', 'nvt_lot']:
    free_e = {}
    error_e = {}
    for path_traj, list_struc in flatten_structures.items():
        if 'nptruns' in path_traj:
            ens = 'nptruns'
        elif 'nvtruns' in path_traj:
            ens = 'nvtruns'
        free_e[path_traj] = {}
        error_e[path_traj] = {}
        for phase in list_struc:
            free_e[path_traj][phase] = np.zeros_like(TEMPERATURE_GRID, dtype=float)
            error_e[path_traj][phase] = np.zeros_like(TEMPERATURE_GRID, dtype=float)
            dir_phase = root / path_traj / f'{DIR_PREFIX}-{phase}' / 'plots' / 'Free_energy'

            try:
                with open(dir_phase / f'{pes}_Tharm_{HARM_T[0]:.0f}K.csv', "r", newline="") as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header
                    for i, row in enumerate(reader):
                        temperature = float(row[0])
                        assert float(row[0]) == TEMPERATURE_GRID[i]
                        if '0dim' in path_traj:
                            free_e[path_traj][phase][i] = 5*to_kJmol*float(row[1]) - free_e_ref_540[ens][pes][i]
                            error_e[path_traj][phase][i] = np.sqrt((5*to_kJmol*float(row[2]))**2 + error_e_ref_540[ens][pes][i]**2)     
                        elif 'double' in phase:
                            free_e[path_traj][phase][i] = 5*to_kJmol*float(row[1]) - free_e_ref_double[ens][pes][i]
                            error_e[path_traj][phase][i] =  np.sqrt((5*to_kJmol*float(row[2]))**2 + error_e_ref_double[ens][pes][i]**2)
                        else:
                            free_e[path_traj][phase][i] = 5*to_kJmol*float(row[1]) - free_e_ref[ens][pes][i]
                            error_e[path_traj][phase][i] = np.sqrt((5*to_kJmol*float(row[2]))**2 + error_e_ref[ens][pes][i]**2)   
            except:
                print(str(dir_phase / f"{pes}_Tharm_{HARM_T[0]:.0f}K.csv") + " not found")
                continue

        #for phase in list_struc:
        #    plt.errorbar(TEMPERATURE_GRID, free_e[phase], yerr=error_e[phase], label=phase)
        #plt.legend()
        #plt.xlabel("Temperature [K]")
        #plt.ylabel("Free Energy per atom [eV]")
        #plt.title(f"Gamma to Zeta")
        #plt.grid(True)
        #plt.savefig(root / path_traj / str(f"free_energies_diff_{pes}.pdf"))
        #plt.close()

def plot_fe_path(selection, idxs, free_e, error_e, name_plot, start_phase=None, end_phase=None, corr_surface=None):
    """
    Plots free energy along a reaction path for selected structures.

    Args:
        selection: list of tuples (path_traj, phase, fraction)
        idxs: list of indices of TEMPERATURE_GRID to plot
        free_e, error_e: dicts with free energy and error arrays
        name_plot: output filename
        start_phase, end_phase: optional indices in 'selection' for interface reference
    """

    # Plot setup
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    ls = {'nvtruns': ':', 'nptruns': '-'}
    mk = {'nvtruns': 'v', 'nptruns': 'o'}

    # Collect and plot
    for ens in ['nvtruns', 'nptruns']:
        for i, temp_idx in enumerate(idxs):
            temp_label = f"{TEMPERATURE_GRID[temp_idx]:.0f}K"

            for j, path in enumerate(selection):
                if j == 0:
                    if ens == 'nvtruns':
                        label = f"fixed {temp_label}"
                    else:
                        label = f"equilibrated {temp_label}"
                else:
                    label = None
                y_vals, y_errs, x_vals = [], [], []
                if start_phase is not None and end_phase is not None:
                    y_start = free_e[selection[0][start_phase][0]+ens][selection[0][start_phase][1]][temp_idx]
                    y_end = free_e[selection[0][end_phase][0]+ens][selection[0][end_phase][1]][temp_idx]
                    for path_traj, phase, fraction, x_val in path:
                        if phase == 'int_3' and 'toZeta' in path_traj and ens == 'nvtruns':
                            continue
                        if phase == 'int_2b' and 'Zetastruc' in path_traj:
                            y_vals.append((free_e[path_traj+ens][phase][temp_idx] - (1.0-fraction)*y_start - fraction*y_end)*corr_surface/2)
                            y_errs.append(error_e[path_traj+ens][phase][temp_idx]*corr_surface/2)
                        else:
                            y_vals.append((free_e[path_traj+ens][phase][temp_idx] - (1.0-fraction)*y_start - fraction*y_end)*corr_surface)
                            y_errs.append(error_e[path_traj+ens][phase][temp_idx]*corr_surface)
                        x_vals.append(x_val)
                else:
                    for path_traj, phase, fraction, x_val in path:
                        if phase == 'int_3' and 'toZeta' in path_traj and ens == 'nvtruns':
                            continue
                        y_vals.append(free_e[path_traj+ens][phase][temp_idx])
                        y_errs.append(error_e[path_traj+ens][phase][temp_idx])
                        x_vals.append(x_val)

                ax.errorbar(
                    x_vals,
                    y_vals,
                    yerr=y_errs,
                    marker=mk[ens],
                    label=label,
                    linestyle=ls[ens],
                    color=colors[i % len(colors)],
                    capsize=4,
                    linewidth=2,
                )

    # Add vertical grid lines and labels on top
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min
    label_height = y_max + 0.05 * y_span  # small space above top data

    for path in selection:
        for path_traj, phase, fraction, x_val in path:
            ax.axvline(x_val, color='gray', linestyle=':', alpha=0.4, linewidth=1)
            ax.text(
                x_val, label_height, phase, ha='left', va='bottom', fontsize=8
            )

    # Adjust limits to include labels
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(y_min, y_max + 0.1 * y_span)

    ax.set_xlabel("Reaction coordinate")
    if start_phase is not None and end_phase is not None:
        ax.set_ylabel("Interface free energy [kJ/(mol*A2)] ")
    else:
        ax.set_ylabel("Free energy difference with gamma [kJ/mol] pfu")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    plt.savefig(name_plot)
    plt.close()
    


idxs = [0, 8, -1]
'''
#gamma to zeta path
selection = [
    [
        ('viaZeta/toZeta/', 'gamma', 0.0, 0.0),
        ('viaZeta/toZeta/', 'int_1', 0.25, 0.25),
        ('viaZeta/toZeta/', 'int_2', 0.5, 0.5),
        ('viaZeta/toZeta/', 'int_3', 0.75, 0.75),
        ('viaZeta/toCsdelta/', 'zeta', 1.0, 1.0),
    ]
]

plot_fe_path(selection, idxs, free_e, error_e, f"free_energies_lineplot_{pes}_gammatoZeta.png")
plot_fe_path(selection, idxs, free_e, error_e, f"interface_free_energies_{pes}_gammatoZeta.png", start_phase = 0, end_phase = -1, corr_surface = 64/(25.6*25.6*np.sqrt(2)))

selection = [
    [
        ('viaZeta/toZeta/', 'gamma', 0.0, 0.0),
        ('NucleationStruc/Zetastruc/1dim/', 'int_1', 0.25, 0.25),
        ('NucleationStruc/Zetastruc/1dim/', 'int_2a', 0.5, 0.48),
        #('NucleationStruc/Zetastruc/1dim/', 'int_2b', 1.0),
        #('NucleationStruc/Zetastruc/0dim/', 'int_line_Csopt1', 1.0),
        #('NucleationStruc/Zetastruc/0dim/', 'int_line_Csopt2', 1.0),
        ('NucleationStruc/Zetastruc/1dim/', 'int_layer_1', 1.0, 1.0),
    ],
    [
        ('viaZeta/toZeta/', 'gamma', 0.0, 0.0 ),
        ('NucleationStruc/Zetastruc/1dim/', 'int_1', 0.25, 0.25),
        ('NucleationStruc/Zetastruc/1dim/', 'int_2b', 0.5, 0.52),
    ],
    #[
    #    ('NucleationStruc/Zetastruc/0dim/', 'int_line_Csopt1', 0.7),
    #],
    #[
    #    ('NucleationStruc/Zetastruc/0dim/', 'int_line_Csopt2', 0.8),
    #],
]

plot_fe_path(selection, idxs, free_e, error_e, f"free_energies_lineplot_{pes}_nucleationZeta.png")
plot_fe_path(selection, idxs, free_e, error_e, f"interface_free_energies_{pes}_nucleationZeta.png", start_phase = 0, end_phase = -1, corr_surface = 64/(2*25.6*6.4*np.sqrt(2)))
'''
#gamma to FAdelta path
selection = [
    [
        ('viaZeta/toZeta/', 'gamma', 0.0, 0.0),
        #('directFromGamma/toFAdelta/', 'int_1', 0.25, 0.23),
        #('directFromGamma/toFAdelta/', 'int_2', 0.5, 0.5),
        ('directFromGamma/toFAdelta/', 'interesting_struc', 0.25, 0.27),
        ('directFromGamma/toFAdelta/', 'int_3', 0.75, 0.75),
        ('directFromGamma/toFAdelta/', 'FAdelta', 1.0, 1.0),
    ],
    [
        ('directFromGamma/toFAdelta/', 'int_1', 0.25, 0.25),
    ],
    [
        ('directFromGamma/toFAdelta/', 'int_2', 0.5, 0.5),
    ],
]

plot_fe_path(selection, idxs, free_e, error_e, f"free_energies_lineplot_{pes}_gammatoFAdelta.pdf")
plot_fe_path(selection, idxs, free_e, error_e, f"interface_free_energies_{pes}_gammatoFAdelta.pdf", start_phase = 0, end_phase = -1, corr_surface = 64/(2*31.35347*18.10193))
'''
selection = [
    [
        ('viaZeta/toZeta/', 'gamma', 0.0, 0.0),
        ('NucleationStruc/FAdeltastruc/', 'int_1a', 0.25, 0.21),
        ('directFromGamma/toFAdelta/', 'int_3', 1.0, 1.0),
    ],
    [
        ('NucleationStruc/FAdeltastruc/', 'int_1b', 0.25, 0.25),
    ],
    [
        ('NucleationStruc/FAdeltastruc/', 'int_1c', 0.25, 0.29),
    ],
]

plot_fe_path(selection, idxs, free_e, error_e, f"free_energies_lineplot_{pes}_nucleationFAdelta.png")
plot_fe_path(selection, idxs, free_e, error_e, f"interface_free_energies_{pes}_nucleationFAdelta.png", start_phase = 0, end_phase = -1, corr_surface = 64/(2*np.sqrt(12.8**2+18*3.2**2)*18.10193))

selection = [
    [
        ('viaZeta/toZeta/', 'gamma', 0.0, 0.0),
        ('NucleationStruc/FAdeltastruc/', 'int_1a_double', 0.125, 0.125),
        ('NucleationStruc/FAdeltastruc/', 'int_2a_double', 0.25, 0.25),
        ('NucleationStruc/FAdeltastruc/', 'int_3a_double', 0.375, 0.375),
        ('NucleationStruc/FAdeltastruc/', 'int_4a_double', 0.5, 0.46),
        ('directFromGamma/toFAdelta/', 'int_3', 1.0, 1.0),
    ],
    [
        ('NucleationStruc/FAdeltastruc/', 'int_4b_double', 0.5, 0.5),
    ],
    [
        ('NucleationStruc/FAdeltastruc/', 'int_4c_double', 0.5, 0.54),
    ],
]

plot_fe_path(selection, idxs, free_e, error_e, f"free_energies_lineplot_{pes}_nucleationFAdelta_double.png")
plot_fe_path(selection, idxs, free_e, error_e, f"interface_free_energies_{pes}_nucleationFAdelta_double.png", start_phase = 0, end_phase = -1, corr_surface = 128/(2*np.sqrt(12.8**2+18*3.2**2)*18.10193))

selection = [
    [
        ('viaZeta/toZeta/', 'gamma', 0.0, 0.0),
        ('directFromGamma/toCsdelta/along_001/', 'int_1', 0.5, 0.5),
        ('directFromGamma/toCsdelta/along_001/', 'int_2', 0.75, 0.75),
        ('directFromGamma/toCsdelta/along_001/', 'Csdelta', 1.0, 1.0),
    ],
]

plot_fe_path(selection, idxs, free_e, error_e, f"free_energies_lineplot_{pes}_gammatoCsdelta.png")
plot_fe_path(selection, idxs, free_e, error_e, f"interface_free_energies_{pes}_gammatoCsdelta.png", start_phase = 0, end_phase = -1, corr_surface = 64/(25.6*25.6*np.sqrt(2)))

selection = [
    [
        ('viaZeta/toZeta/', 'gamma', 0.0, 0.0),
        ('directFromGamma/toCsdelta/along_001/', 'int_1_double', 0.25, 0.25),
        ('directFromGamma/toCsdelta/along_001/', 'int_2_double', 0.375, 0.375),
        ('directFromGamma/toCsdelta/along_001/', 'int_3_double', 0.5, 0.5),
        ('directFromGamma/toCsdelta/along_001/', 'int_4_double', 0.625, 0.625),
        ('directFromGamma/toCsdelta/along_001/', 'int_5_double', 0.75, 0.75),
        ('directFromGamma/toCsdelta/along_001/', 'int_6_double', 0.875, 0.875),
        ('directFromGamma/toCsdelta/along_001/', 'Csdelta', 1.0, 1.0),
    ],
]

plot_fe_path(selection, idxs, free_e, error_e, f"free_energies_lineplot_{pes}_gammatoCsdelta_double.png")
plot_fe_path(selection, idxs, free_e, error_e, f"interface_free_energies_{pes}_gammatoCsdelta_double.png", start_phase = 0, end_phase = -1, corr_surface = 128/(25.6*25.6*np.sqrt(2)))

selection = [
    [
        ('viaZeta/toZeta/', 'gamma', 0.0, 0.0),
        ('NucleationStruc/Csdeltastruc/along_001/', 'int_1', 0.5, 0.5),
        ('directFromGamma/toCsdelta/along_001/', 'int_1', 1.0, 1.0),
    ],
]

plot_fe_path(selection, idxs, free_e, error_e, f"free_energies_lineplot_{pes}_nucleationCsdelta.png")
plot_fe_path(selection, idxs, free_e, error_e, f"interface_free_energies_{pes}_nucleationCsdelta.png", start_phase = 0, end_phase = -1, corr_surface = 64/(25.6*25.6))

selection = [
    [
        ('viaZeta/toZeta/', 'gamma', 0.0, 0.0),
        ('NucleationStruc/Csdeltastruc/along_001/', 'int_1_double', 0.25, 0.25),
        ('NucleationStruc/Csdeltastruc/along_001/', 'int_2_double', 0.375, 0.375),
        ('NucleationStruc/Csdeltastruc/along_001/', 'int_3_double', 0.5, 0.5),
        ('NucleationStruc/Csdeltastruc/along_001/', 'int_4_double', 0.625, 0.625),
        ('NucleationStruc/Csdeltastruc/along_001/', 'int_5_double', 0.75, 0.75),
        ('directFromGamma/toCsdelta/along_001/', 'int_1', 1.0, 1.0),
    ],
]

plot_fe_path(selection, idxs, free_e, error_e, f"free_energies_lineplot_{pes}_nucleationCsdelta_double.png")
plot_fe_path(selection, idxs, free_e, error_e, f"interface_free_energies_{pes}_nucleationCsdelta_double.png", start_phase = 0, end_phase = -1, corr_surface = 128/(25.6*25.6))

#zeta to FAdelta path
selection = [
    [
        ('viaZeta/toCsdelta/', 'zeta', 0.0, 0.0),
        #('viaZeta/toFAdelta/', 'int_1', 0.25, 0.25),
        #('viaZeta/toFAdelta/', 'int_2', 0.5, 0.5),
        ('viaZeta/toFAdelta/', 'int_layer', 0.375, 0.375),
        ('viaZeta/toFAdelta/', 'int_3', 0.75, 0.75),
        ('viaZeta/toFAdelta/', 'FAdelta', 1.0, 1.0),
    ],
    [
        ('viaZeta/toFAdelta/', 'int_1', 0.25, 0.25),
    ]
    ,
    [
        ('viaZeta/toFAdelta/', 'int_2', 0.5, 0.5),
    ]
]

plot_fe_path(selection, idxs, free_e, error_e, f"free_energies_lineplot_{pes}_zetatoFAdelta.png")
#plot_fe_path(selection, idxs, free_e, error_e, f"interface_free_energies_{pes}_zetatoFAdelta.png", start_phase = 0, end_phase = -1)

#zeta to Csdelta path
selection = [
    [
        ('viaZeta/toCsdelta/', 'zeta', 0.0, 0.0),
        ('viaZeta/toCsdelta/', 'int_1', 0.125, 0.125),
        ('viaZeta/toCsdelta/', 'int_2a', 0.25, 0.23),
        #('viaZeta/toCsdelta/', 'int_2b', 0.25, 0.27),
        ('viaZeta/toCsdelta/', 'int_3', 0.375, 0.375),
        ('viaZeta/toCsdelta/', 'int_4a', 0.5, 0.46),
        #('viaZeta/toCsdelta/', 'int_4b', 0.5, 0.5),
        #('viaZeta/toCsdelta/', 'int_3and4', 0.5, 0.54),
        ('viaZeta/toCsdelta/', 'Csdelta', 1.0, 1.0),
    ],
    [
        ('viaZeta/toCsdelta/', 'zeta', 0.0, 0.0),
        ('viaZeta/toCsdelta/', 'int_1', 0.125, 0.125),
        #('viaZeta/toCsdelta/', 'int_2a', 0.25, 0.23),
        ('viaZeta/toCsdelta/', 'int_2b', 0.25, 0.27),
        ('viaZeta/toCsdelta/', 'int_3', 0.375, 0.375),
        #('viaZeta/toCsdelta/', 'int_4a', 0.5, 0.46),
        ('viaZeta/toCsdelta/', 'int_4b', 0.5, 0.5),
        #('viaZeta/toCsdelta/', 'int_3and4', 0.5, 0.54),
        ('viaZeta/toCsdelta/', 'Csdelta', 1.0, 1.0), 
    ],
    [
        ('viaZeta/toCsdelta/', 'zeta', 0.0, 0.0),
        ('viaZeta/toCsdelta/', 'int_1', 0.125, 0.125),
        ('viaZeta/toCsdelta/', 'int_2a', 0.25, 0.23),
        #('viaZeta/toCsdelta/', 'int_2b', 0.25, 0.27),
        #('viaZeta/toCsdelta/', 'int_3', 0.375, 0.375),
        #('viaZeta/toCsdelta/', 'int_4a', 0.5, 0.46),
        #('viaZeta/toCsdelta/', 'int_4b', 0.5, 0.5),
        ('viaZeta/toCsdelta/', 'int_3and4', 0.5, 0.54),
        ('viaZeta/toCsdelta/', 'Csdelta', 1.0, 1.0), 
    ],

]

plot_fe_path(selection, idxs, free_e, error_e, f"free_energies_lineplot_{pes}_zetatoCsdelta.png")
#plot_fe_path(selection, idxs, free_e, error_e, f"interface_free_energies_{pes}_zetatoCsdelta.png", start_phase = 0, end_phase = -1)
'''

