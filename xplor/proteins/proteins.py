import mdtraj as md
import numpy as np
import pandas as pd
import importlib, itertools

with importlib.resources.path("xplor-functions", 'data/amino_acid_names_3_1_letter.txt') as data_path:
    amino_acids_3_letter_1 = data_path

def get_column_names_from_pdb(pdbid='1UBQ'):
    _ = md.load_pdb(f'https://files.rcsb.org/view/{pdbid}.pdb')
    fasta = [i for i in _.top.to_fasta()[0]]
    AAs = np.genfromtxt(amino_acids_3_letter_1, delimiter=' - ', dtype=str)
    AAs = pd.DataFrame(AAs, columns=['name', '3letter', '1letter'])
    AAs = AAs.set_index('1letter')
    residues = np.array([f'{n}{i + 1}' for i, n in enumerate(AAs.loc[fasta]['3letter'].str.upper())])

    column_names = [f'{i} {k} {j}' for i, j, k in itertools.product(['proximal', 'distal'], ['sPRE', '15N_relax_600', '15N_relax_800'], residues)]
    return column_names