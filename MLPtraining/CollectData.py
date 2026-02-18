from pathlib import Path
import numpy as np
from ase.io import read, write
import shutil
import subprocess


def flatten_dct(newdict, subdict, catkey):
    for key, val in subdict.items():
        newkey = catkey+'/'+key
        if newkey[0] == '/':
            newkey = newkey[1:]
        if isinstance(val, dict):
            flatten_dct(newdict, val, newkey)
        else:
            newdict[newkey] = val

selection0 = {}

selection1 = {
    'initialization/traj_T150_0.xyz': [-1],
    'sampling/traj_T600_0.xyz': [10, 250],
}

selection2 = {
    'initialization/traj_T150_P0.1_0.xyz': [0, -1],
    'sampling/traj_T600_P0.1_0.xyz': [10, 90, 170, 250],
}

selection3 = {
    'initialization/traj_T150_P0.1_0.xyz': [0, -1],
    'sampling/traj_T600_P0.1_0.xyz': [10, 90, 170, 250],
    'npt/traj_T600_P0.1_0.xyz': [125, 250],
    'npt/traj_T600_P0.1_1.xyz': [125, 250],
} 

selection4 = {
    'initialization/traj_T150_0.xyz': [-1],
}

selection5 = {
    'initialization/traj_T150_0.xyz': [-1],
    'sampling/traj_T600_0.xyz': [10, 90, 170, 250],
    'nvt/traj_T600_0.xyz': [125, 250],
    'nvt/traj_T600_1.xyz': [125, 250],
}

selection6 = {
    'initialization/traj_T150_P0.1_0.xyz': [0, -1],
}

selection7 = {
    'initialization/traj_T150_P0.1_0.xyz': [0, -1],
    'sampling/traj_T600_P0.1_0.xyz': [10, 250],
}

selection8 = {
    'initialization/traj_T150_0.xyz': [-1],
    'sampling/traj_T600_0.xyz': [10, 90, 170, 250],
    'nvt/traj_T600_0.xyz': [100, 150, 200, 250],
    'nvt/traj_T600_1.xyz': [100, 150, 200, 250],
}

selection9 = {
    'initialization/traj_T150_P0.1_0.xyz': [0, -1],
    'sampling/traj_T600_P0.1_0.xyz': [10, 90, 170, 250],
    'npt/traj_T600_P0.1_0.xyz': [100, 150, 200, 250],
    'npt/traj_T600_P0.1_1.xyz': [100, 150, 200, 250],
}

selection10 = {
    'initialization/traj_T150_0.xyz': [-1],
    'sampling/traj_T600_0.xyz': [10],
}

selection11 = {
    'initialization/traj_T150_P0.1_0.xyz': [0, -1],
    'sampling/traj_T600_P0.1_0.xyz': [10],
}

selection12 = {
    'initialization/traj_T150_0.xyz': [-1],
    'sampling/traj_T600_0.xyz': [250],
}

selection13 = {
    'initialization/traj_T150_P0.1_0.xyz': [0, -1],
    'sampling/traj_T600_P0.1_0.xyz': [250],
}

selection14 = {
    'initialization/traj_T150_0.xyz': [-1],
    'sampling/traj_T600_0.xyz': [10, 90, 170, 250],
}

selection15 = {
    'sampling/traj_T600_0.xyz': [250],
}

selection16 = {
    'sampling/traj_T600_P0.1_0.xyz': [250],
}

structures = {
    'viaZeta': {
        'toZeta': {
            'nvtruns': {
                'phase-gamma': selection1,
                'phase-int_1': selection1,
                'phase-int_2': selection1,
                'phase-int_3': selection1,
                'phase-zeta': selection1,
            },
            'nptruns': {
                'phase-gamma': selection2,
                'phase-int_1': selection3,
                'phase-int_2': selection3,
                'phase-int_3': selection3,
                'phase-zeta': selection2,
            }
        },
        'toCsdelta': {
            'nvtruns': {
                'phase-zeta': selection0,
                'phase-zeta_shift': selection4,
                'phase-int_1': selection1,
                'phase-int_2a': selection1,
                'phase-int_2b': selection1,
                'phase-int_3': selection1,
                'phase-int_3and4': selection1,
                'phase-int_4a': selection1,
                'phase-int_4b': selection1,
                'phase-Csdelta': selection5,
            },
            'nptruns': {
                'phase-zeta': selection0,
                'phase-zeta_shift': selection6,
                'phase-int_1': selection7,
                'phase-int_2a': selection7,
                'phase-int_2b': selection3,
                'phase-int_3': selection7,
                'phase-int_3and4': selection7,
                'phase-int_4a': selection7,
                'phase-int_4b': selection7,
                'phase-Csdelta': selection3,
            }
        },
        'toFAdelta': {
            'nvtruns': {
                'phase-zeta': selection1,
                'phase-int_1': selection1,
                'phase-int_2': selection1,
                'phase-int_3': selection1,
                'phase-int_layer': selection1,
                'phase-FAdelta': selection1,
            },
            'nptruns': {
                'phase-zeta': selection7,
                'phase-int_1': selection7,
                'phase-int_2': selection7,
                'phase-int_3': selection7,
                'phase-int_layer': selection7,
                'phase-FAdelta': selection3,
            }
        }
    },
    'directFromGamma': {
        'toFAdelta': {
            'nvtruns': {
                'phase-gamma': selection0,
                'phase-int_1': selection8,
                'phase-int_2': selection8,
                'phase-int_3': selection8,
                'phase-FAdelta': selection8,
                'phase-interesting_struc': selection8,
            },
            'nptruns': {
                'phase-gamma': selection0,
                'phase-int_1': selection9,
                'phase-int_2': selection9,
                'phase-int_3': selection9,
                'phase-FAdelta': selection9,
                'phase-interesting_struc': selection9,
            }
        },
        'toCsdelta': {
            'along_001': {
                'nvtruns': {
                    'phase-gamma': selection0,
                    'phase-int_1': selection8,
                    'phase-int_2': selection8,
                    'phase-int_1_double': selection1,
                    'phase-int_2_double': selection1,
                    'phase-int_3_double': selection1,
                    'phase-int_4_double': selection1,
                    'phase-int_5_double': selection1,
                    'phase-int_6_double': selection1,
                    'phase-Csdelta': selection8,
                },
                'nptruns': {
                    'phase-gamma': selection0,
                    'phase-int_1': selection9,
                    'phase-int_2': selection9,
                    'phase-int_1_double': selection7,
                    'phase-int_2_double': selection7,
                    'phase-int_3_double': selection7,
                    'phase-int_4_double': selection7,
                    'phase-int_5_double': selection7,
                    'phase-int_6_double': selection7,
                    'phase-Csdelta': selection9,
                }
            },
            'along_012': {
                'sametilts': {
                    'nvtruns': {
                        'phase-gamma': selection0,
                        'phase-int_layer_1a': selection10,
                        'phase-int_layer_1a_connected': selection10,
                        'phase-int_layer_1b': selection10,
                        'phase-int_layer_1b_connected': selection10,
                        'phase-Csdelta_sametilts': selection1,
                        'phase-Csdelta_sametilts_connected': selection1,
                    },
                    'nptruns': {
                        'phase-gamma': selection0,
                        'phase-int_layer_1a': selection11,
                        'phase-int_layer_1a_connected': selection11,
                        'phase-int_layer_1b': selection11,
                        'phase-int_layer_1b_connected': selection11,
                        'phase-Csdelta_sametilts': selection2,
                        'phase-Csdelta_sametilts_connected': selection7,
                    }
                },
                'withtilts': {
                    'nvtruns': {
                        'phase-gamma': selection0,
                        'phase-int_layer_1a_connected': selection10,
                        'phase-int_layer_1b_double': selection10,
                        'phase-int_layer_1b_double_connected': selection10,
                        'phase-int_layer_1c_double': selection10,
                        'phase-int_layer_1c_double_connected': selection10,
                        'phase-int_layer_2c_double': selection10,
                        'phase-int_layer_2c_double_connected': selection10,
                        'phase-Csdelta': selection1,
                        'phase-Csdelta_connected': selection1,
                    },
                    'nptruns': {
                        'phase-gamma': selection0,
                        'phase-int_layer_1a_connected': selection11,
                        'phase-int_layer_1b_double': selection11,
                        'phase-int_layer_1b_double_connected': selection11,
                        'phase-int_layer_1c_double': selection11,
                        'phase-int_layer_1c_double_connected': selection11,
                        'phase-int_layer_2c_double': selection11,
                        'phase-int_layer_2c_double_connected': selection11,
                        'phase-Csdelta': selection2,
                        'phase-Csdelta_connected': selection7,
                    }
                }
            }
        }
    },
    'NucleationStruc': {
        'Zetastruc': {
            '0dim': {
                'nvtruns': {
                    'phase-gamma': selection0,
                    'phase-int_3by3_Csopt1': selection12,
                    'phase-int_3by3_Csopt2': selection12,
                    'phase-int_4by2_Csopt1': selection12,
                    'phase-int_4by2_Csopt2': selection12,
                    'phase-int_4by2_and_2by1_Csopt1': selection12,
                    'phase-int_4by2_and_2by1_Csopt2': selection12,
                    'phase-int_4by2_and_2by2_Csopt1': selection12,
                    'phase-int_4by2_and_2by2_Csopt2': selection12,
                    'phase-int_line_Csopt1': selection12,
                    'phase-int_line_Csopt2': selection12,
                },
                'nptruns': {
                    'phase-gamma': selection0,
                    'phase-int_3by3_Csopt1': selection13,
                    'phase-int_3by3_Csopt2': selection13,
                    'phase-int_4by2_Csopt1': selection13,
                    'phase-int_4by2_Csopt2': selection13,
                    'phase-int_4by2_and_2by1_Csopt1': selection13,
                    'phase-int_4by2_and_2by1_Csopt2': selection13,
                    'phase-int_4by2_and_2by2_Csopt1': selection13,
                    'phase-int_4by2_and_2by2_Csopt2': selection13,
                    'phase-int_line_Csopt1': selection13,
                    'phase-int_line_Csopt2': selection13,
                }
            },
            '1dim':{
                'nvtruns': {
                    'phase-gamma': selection0,
                    'phase-int_1': selection5,
                    'phase-int_2a': selection5,
                    'phase-int_2b': selection1,
                    'phase-int_layer_1': selection5,
                },
                'nptruns': {
                    'phase-gamma': selection0,
                    'phase-int_1': selection3,
                    'phase-int_2a': selection3,
                    'phase-int_2b': selection7,
                    'phase-int_layer_1': selection2,
                }
            }
        },
        'FAdeltastruc': {
            'nvtruns': {
                'phase-gamma': selection0,
                'phase-int_1a': selection14,
                'phase-int_1b': selection1,
                'phase-int_1c': selection1,
                'phase-int_1a_double': selection1,
                'phase-int_2a_double': selection14,
                'phase-int_3a_double': selection1,
                'phase-int_4a_double': selection5,
                'phase-int_4b_double': selection5,
                'phase-int_4c_double': selection5,
            },
            'nptruns': {
                'phase-gamma': selection0,
                'phase-int_1a': selection7,
                'phase-int_1b': selection7,
                'phase-int_1c': selection7,
                'phase-int_1a_double': selection7,
                'phase-int_2a_double': selection7,
                'phase-int_3a_double': selection7,
                'phase-int_4a_double': selection3,
                'phase-int_4b_double': selection3,
                'phase-int_4c_double': selection3,
            }
        },
        'Csdeltastruc': {
            'along_001': {
                'nvtruns': {
                    'phase-gamma_double': selection15,
                    'phase-int_1': selection1,
                    'phase-int_1_double': selection1,
                    'phase-int_2_double': selection1,
                    'phase-int_3_double': selection5,
                    'phase-int_4_double': selection5,
                    'phase-int_5_double': selection1,
                },
                'nptruns': {
                    'phase-gamma_double': selection16,
                    'phase-int_1': selection7,
                    'phase-int_1_double': selection7,
                    'phase-int_2_double': selection7,
                    'phase-int_3_double': selection3,
                    'phase-int_4_double': selection3,
                    'phase-int_5_double': selection7,
                }
            },
            'along_012': {
                'sametilts': {
                    'nvtruns': {
                        'phase-int_1': selection10,
                        'phase-int_1_connected': selection1,
                        'phase-int_2a': selection10,
                        'phase-int_2a_connected': selection12,
                        'phase-int_2b': selection10,
                        'phase-int_2b_connected': selection1,
                        'phase-int_3a': selection10,
                        'phase-int_3a_connected': selection12,
                        'phase-int_3b_connected': selection12,
                        'phase-int_3c': selection10,
                        'phase-int_3c_connected': selection12,
                    },
                    'nptruns': {
                        'phase-int_1': selection11,
                        'phase-int_1_connected': selection13,
                        'phase-int_2a': selection11,
                        'phase-int_2a_connected': selection13,
                        'phase-int_2b': selection11,
                        'phase-int_2b_connected': selection13,
                        'phase-int_3a': selection11,
                        'phase-int_3a_connected': selection13,
                        'phase-int_3b_connected': selection13,
                        'phase-int_3c': selection11,
                        'phase-int_3c_connected': selection13,
                    }
                },
                'withtilts': {
                    'nvtruns': {
                        'phase-int_1_connected': selection12,
                        'phase-int_2': selection10,
                        'phase-int_2_connected': selection12,
                        'phase-int_2b_double_connected': selection12,
                    },
                    'nptruns': {
                        'phase-int_1_connected': selection13,
                        'phase-int_2': selection11,
                        'phase-int_2_connected': selection13,
                        'phase-int_2b_double_connected': selection13,
                    }
                }
            }
        }
    }
}


flatten_structures = {}
flatten_dct(flatten_structures, structures, '')
root =Path.cwd()
vasp_folder = root / 'DoVASPcalc'

traj_calc = []
for path_traj, indices in flatten_structures.items():
    for i, ind in enumerate(indices):

        para_shift = int(path_traj[-5])*len(indices)
        path_traj_folder = ''
        for partpath in path_traj.split('/')[:-1]:
            path_traj_folder += partpath + '/'
        path_traj_folder = path_traj_folder[:-1]
        strucfolder = vasp_folder / path_traj_folder / f'struc_{i+para_shift}'

        file_calculated = strucfolder / 'calc_struc.xyz'

        if file_calculated.exists():
            print(f'{strucfolder} collect data')
            traj_calc.append(read(file_calculated))
        else:
            print(f'{strucfolder} calculation FAILED!')

write(vasp_folder/'vaspdata.xyz', traj_calc)