import mdtraj as md
import numpy as np
import pandas as pd
import  itertools, os, sys


AMINO_ACIDS_3_LETTER_1 = os.path.join(os.path.dirname(sys.modules['xplor'].__file__), "data/amino_acid_names_3_1_letter.txt")


def get_column_names_from_pdb(pdbid='1UBQ', return_residues=False, return_AAs=False):
    """Get column names for sPRE/600MHz 15N and 800MHz 15N from a pdbid.

    Keyword Args:
        pdbid (str, optional): The pdb id of the protein for which the column names
            should be generated.
        return_residues (bool, optional): Whether to only return residues.
            Defaults to False.
        return_AAs (bool, optional): Whether to only return the AAs of `pdbid`.
            Defaults to False.

    Returns:
        list: A list with the column names

    """
    _ = md.load_pdb(f'https://files.rcsb.org/view/{pdbid}.pdb')
    fasta = [i for i in _.top.to_fasta()[0]]
    AAs = np.genfromtxt(AMINO_ACIDS_3_LETTER_1, delimiter=' - ', dtype=str)
    AAs = pd.DataFrame(AAs, columns=['name', '3letter', '1letter'])
    AAs = AAs.set_index('1letter')
    if return_AAs:
        return AAs

    residues = np.array([f'{n}{i + 1}' for i, n in enumerate(AAs.loc[fasta]['3letter'].str.upper())])
    if return_residues:
        return residues

    column_names = [f'{i} {k} {j}' for i, j, k in itertools.product(['proximal', 'distal'], ['sPRE', '15N_relax_600', '15N_relax_800'], residues)]
    return column_names