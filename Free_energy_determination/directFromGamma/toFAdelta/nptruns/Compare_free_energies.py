from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
from configuration import phases, TEMPERATURE_GRID, HARM_T
from execute_workflow import DIR_PREFIX

REF_PHASE = 'int_1'
root = Path.cwd()

for pes in ['harm', 'intermediate', 'nvt']:  #, 'harm_lot', 'nvt_lot']:
    free_e = {}
    error_e = {}
    for phase in phases:
        free_e[phase] = np.zeros_like(TEMPERATURE_GRID, dtype=float)
        error_e[phase] = np.zeros_like(TEMPERATURE_GRID, dtype=float)
        dir_phase = root / f'{DIR_PREFIX}-{phase}' / 'plots' / 'Free_energy'

        try:
            with open(dir_phase / f'{pes}_Tharm_{HARM_T[0]:.0f}K.csv', "r", newline="") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for i, row in enumerate(reader):
                    temperature = float(row[0])
                    assert float(row[0]) == TEMPERATURE_GRID[i]
                    free_e[phase][i] = float(row[1])
                    error_e[phase][i] = float(row[2])
        except:
            print(str(dir_phase / f"{pes}_Tharm_{HARM_T[0]:.0f}K.csv") + " not found")
            continue

    for phase in phases:
        plt.errorbar(TEMPERATURE_GRID, free_e[phase] - free_e[REF_PHASE], yerr=np.sqrt(error_e[phase]**2 + error_e[REF_PHASE]**2), label=phase)
    plt.legend()
    plt.xlabel("Temperature [K]")
    plt.ylabel("Free Energy per atom [eV]")
    plt.title(f"Gamma to FAdelta")
    plt.grid(True)
    plt.savefig(root/str(f"free_energies_diff_{pes}.pdf"))
    plt.close()
