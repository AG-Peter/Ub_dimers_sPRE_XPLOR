################################################################################
# Imports
################################################################################


import mdtraj as md
import numpy as np
import parmed as pmd
from parmed.charmm import CharmmPsfFile
import pandas as pd
import os, sys, glob, multiprocessing, copy, ast, subprocess, yaml, shutil, scipy
from ..proteins.proteins import get_column_names_from_pdb
from io import StringIO
from joblib import Parallel, delayed
from .custom_gromacstopfile import CustomGromacsTopFile
from xplor.argparse.argparse import YAML_FILE, write_argparse_lines_from_yaml_or_dict
from ..misc import get_local_or_proj_file, get_iso_time
from .psf_parser import PSFParser
from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile
import builtins
from scipy.optimize import minimize, NonlinearConstraint, Bounds, nnls
from typing import Callable
from ..misc import delete_old_csvs


################################################################################
# Globals
################################################################################


__all__ = ['parallel_xplor', 'get_series_from_mdtraj', 'call_xplor_with_yaml',
           'normalize_sPRE', 'test_conect', 'create_psf_files', 'make_linear_combination_from_clusters']

MASSES = {'1.00800': 'H', '14.0070': 'N', '12.0110': 'C', '15.9990': 'O', '32.0600': 'S'}
# mapping is pdb: psf
H_MAPPING = {'H': 'HT1', 'H2': 'HT2', 'H3': 'HT3'}
OXT_MAPPING = {'O': 'OT1', 'OXT': 'OT2'}
# MANUAL_MAPPING = {'H': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'O': 'OT1', 'OXT': 'OT2'}
#                   'CA': 'CA', 'CB': 'CB', 'CG': 'CG', 'CD': 'CD', 'CE': 'CE',
#                   'C': 'C', 'HA': 'HA',
#                   'HB1': 'HB1', 'HB2': 'HB2', 'HB3': 'HB3',
#                   'HG1': 'HG1', 'HG2': 'HG2', 'HG3': 'HG3',
#                   'HD1': 'HD1', 'HD2': 'HD2', 'HD3': 'HD3',
#                   'HE1': 'HE1', 'HE2': 'HE2', 'HE3': 'HE3',
#                   'N': 'N', 'NE2': 'NE2',
#                   'OE1': 'OE1',}

import os
import builtins
from io import StringIO

def _redefine_os_path_exists():
    orig_func = os.path.exists
    def new_func(path):
        if path.lower() == 'tmp_stringio.pdb':
            return True
            return True
        else:
            return orig_func(path)
    os.path.exists = new_func


def _redefine_open():
    orig_func = builtins.open
    def new_func(*args, **kwargs):
        if args[0].lower() == 'tmp_stringio.pdb':
            return args[0]
        else:
            return orig_func(*args, **kwargs)
    builtins.open = new_func


class RAMFile(StringIO):
    def lower(self):
        return 'tmp_stringio.pdb'
    def close(self):
        pass


_redefine_os_path_exists()
_redefine_open()
print('Redefining os.path.isfile to work with tmp_stringio.pdb')
print('Redefining open() to work with tmp_stringio.pdb')


################################################################################
# Utils
################################################################################


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


################################################################################
# Functions
################################################################################


def create_psf_files(ubq_sites):
    """Creates psf files by ubiquitylation site and saves it to data.

    Because the psf files stay the same for a ubiquitylation site, this
    function takes the computational overload from creating new psf files for
    every frame.

    Args:
        ubq_sites (list[str]): The ubiquitylation sites to consider.

    """
    for ubq_site in ubq_sites:
        gromos_top_file = f'/home/andrejb/Research/DEVA/2017_04_27_UBQ_TOPOLOGIES/top_G54A7/diUBQ/{ubq_site}_01/system.top'
        with Capturing() as output:
            omm_top = CustomGromacsTopFile(gromos_top_file, includeDir='/home/andrejb/Software/gmx_forcefields')

        # mdtraj part
        # pdb_file = f'/home/andrejb/Research/DEVA/2017_04_27_UBQ_TOPOLOGIES/top_G54A7/diUBQ/{ubq_site}_01/{ubq_site}_01.pdb'
        pdb_file = glob.glob(f'/home/andrejb/Research/SIMS/2017_*_G_2ub_{ubq_site}_01_01/start.pdb')
        assert len(pdb_file) == 1
        pdb_file = pdb_file[0]
        traj = md.load(pdb_file)
        # print(traj.n_atoms)
        # print(md.Topology.from_openmm(omm_top.topology).n_atoms)
        traj.top = md.Topology.from_openmm(omm_top.topology)

        isopeptide_bonds = []
        for r in traj.top.residues:
            if r.name == 'GLQ':
                r.name = 'GLY'
                for a in r.atoms:
                    if a.name == 'C':
                        isopeptide_bonds.append(f"{a.residue.name} {a.residue.resSeq} {a.name}")
            if r.name == 'LYQ':
                r.name = 'LYS'
                for a in r.atoms:
                    if a.name == 'CQ': a.name = 'CE'
                    if a.name == 'NQ':
                        a.name = 'NZ'
                        isopeptide_bonds.append(f"{a.residue.name} {a.residue.resSeq} {a.name}")
                    if a.name == 'HQ': a.name = 'HZ1'

        assert os.path.isdir('xplor/data/')
        tmp_pdb = f'tmp_{ubq_site}_pdb_for_pdb2psf.pdb'
        tmp_pdb_with_dir = f'xplor/data/{tmp_pdb}'
        tmp_psf = tmp_pdb.replace('.pdb', '.psf')
        tmp_psf_with_dir = tmp_pdb_with_dir.replace('.pdb', '.psf')
        final_psf = f'xplor/data/{ubq_site.lower()}_psf_for_xplor_with_added_bond.psf'
        traj.save_pdb(tmp_pdb_with_dir)
        try:
            call_pdb2psf(tmp_pdb, 'xplor/data/', os.getcwd())
            _add_bond_to_psf(tmp_psf_with_dir, isopeptide_bonds)
            _rename_atoms_according_to_charmm(tmp_psf_with_dir, tmp_pdb_with_dir)
            # call pdb2psf again when the termini are fixed
            call_pdb2psf(tmp_pdb, 'xplor/data/', os.getcwd())
            shutil.copyfile(tmp_psf_with_dir, final_psf)
        except OSError as e:
            raise OSError(f"Probably not working due to path problems. For this"
                          f"function to work you should be in the xplor_functions"
                          f"root dir. cwd: {os.getcwd()}. Error: {e}")
        finally:
            if os.path.isfile(tmp_pdb_with_dir): os.remove(tmp_pdb_with_dir)
            if os.path.isfile(tmp_psf_with_dir): os.remove(tmp_psf_with_dir)


def check_pdb_and_psf_integrity(pdb_file, psf_file, ubq_site):
    """Checks, whether a combination of psf_file and pdb_file have the same atoms."""
    with open(psf_file, 'r') as f:
        psf_file = f.read()
    hunks = psf_file.split('\n\n')
    atom_hunk = hunks[['!NATOM' in hunk for hunk in hunks].index(True)].splitlines()[1:]

    with open(pdb_file, 'r') as f:
        lines_pdb = list(filter(lambda x: x.startswith('ATOM'), f.read().splitlines()))

    psf_atoms = {int(i[0]): i[4] for i in map(str.split, atom_hunk)}
    pdb_atoms = {int(i[1]): i[2] for i in map(str.split, lines_pdb)}

    add = 0
    ubq_site_resid = int(ubq_site.lstrip('k')) + 76

    for i in range(1, 1 + max([len(psf_atoms), len(pdb_atoms)])):
        ii = i + add
        if '68   HIS  HD1' in atom_hunk[i - 1]:
            add -= 1
            continue
        if f'{ubq_site_resid:<5}LYS  HZ3' in atom_hunk[i - 1]:
            add -= 1
            continue
        if '144  HIS  HD1' in atom_hunk[i - 1]:
            add -= 1
            continue
        if psf_atoms[i] != pdb_atoms[ii]:
            print('pdb atom:', pdb_atoms[ii], 'psf atom:', psf_atoms[i])
            print('\n')
            print(lines_pdb[ii - 1])
            print('\n')
            print(atom_hunk[i - 1])
            print('\n')
            for j in range(ii-5, ii+5):
                print(lines_pdb[j])
            print('\n')
            for j in range(i - 5, i + 5):
                print(atom_hunk[j])
            return False
    return True


def datetime_windows_and_linux_compatible():
    import datetime
    from sys import platform
    if platform == "linux" or platform == "linux2" or platform == "darwin":
        return datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()
    elif platform == "win32":
        return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def is_aa_sim(file):
    """From a traj_nojump.xtc, decides, whether sim is an aa sim."""
    if '/home/kevin/projects/molsim/diUbi_aa' in file:
        return True
    directory = '/'.join(file.split('/')[:-1])
    contents = os.listdir(directory)
    gro_file = os.path.join(directory, 'start.gro')
    with open(gro_file, 'r') as f:
        content = f.read()
    if 'MARTINI' in content:
        return False
    elif 'Protein in water' in content:
        return True
    else:
        print(content)
        raise Exception("Can not decide whether sim is AA or CG.")


def get_ubq_site(traj_file):
    """Returns ubq_site and basename for a traj_file."""
    return get_ubq_site_and_basename(traj_file)[1]


def get_ubq_site_and_basename(traj_file):
    """Returns ubq_site and basename for a traj_file."""
    basename = traj_file.split('/')[-2]
    if basename == 'data':
        basename = traj_file.split('/')[-1].split('.')[0]
    substrings = basename.split('_')
    try:
        ubq_site = substrings[[substr.lower().startswith('k') for substr in substrings].index(True)]
    except ValueError:
        if 'm1' in traj_file or 'M1' in traj_file:
            return basename, 'm1'
        else:
            print(basename)
            raise
    return basename, ubq_site


def normalize_sPRE(df_comp, df_obs, kind='var', norm_res_count=10,
                   get_factors=False, print_factors=True):
    """Normalizes a dataframe with sPRE values in it and returns a new df.

    Args:
        df_comp (pd.DataFrame): A pandas Dataframe with computed values.
            Such a dataframe can be obtained with the `parallel_xplor` function.
        df_obs (pd.DataFrame_: A dataframe with observables. Such a dataframe
            can be obtained with `xplor.functions.parse_input_files.get_observed_df`

    Keyword Args:
        kind (str, optional): Can be one of the two:
            * 'var': For choosing the residues with the smallest variance.
            * 'mean': For choosing the residues with the smallest mean.
        norm_res_count (int, optional): How many residues should be used to
            normalize the data.

    Returns:
        tupel: A tupel containing the following:
        
            pd.DataFrame: The new dataframe.
            np.ndarray: The centers of the proximal subunit.
            np.ndarray: The centers of the distal subunit.

    """
    out = []
    missing = []
    factors_out = {}

    if print_factors:
        print(f"After declaration factors is {factors_out}. "
              f"Considered ubq_sites are: {pd.value_counts(df_comp['ubq_site']).index}")

    residues = get_column_names_from_pdb(return_residues=True)

    for ubq_site in pd.value_counts(df_comp['ubq_site']).index:
        if ubq_site not in df_obs.keys():
            print(f"ubq_site {ubq_site} not in df_obs {df_obs.keys()}.")
            missing.append(ubq_site)
            continue
        print(ubq_site)
        # put the needed values into their own df
        sPRE_ind = df_comp.columns[['sPRE' in c and 'norm' not in c for c in df_comp.columns]]
        sPRE_comp = df_comp[df_comp['ubq_site'] == ubq_site][sPRE_ind]
        print('Shape of computed data:', sPRE_comp.shape)

        # get measured data
        v_obs = df_obs.loc[sPRE_ind][ubq_site].values
        print('Shape of observed data:', v_obs.shape)

        prox_columns = [i for i in sPRE_comp.columns if 'prox' in i and 'sPRE' in i and 'norm' not in i]
        dist_columns = [i for i in sPRE_comp.columns if 'dist' in i and 'sPRE' in i and 'norm' not in i]

        sPRE_calc_norm = []

        def sorting_func(index):
            new_index = index.map(lambda x: int(x.split()[1][3:]))
            return new_index

        # get the mean values along the columns
        # get the threshold of the <norm_res_count> lowest values
        if kind == 'var':
            v_calc = sPRE_comp.var(axis='rows')
            if np.any(np.isnan(v_calc)):
                v_calc[np.isnan(v_calc)] = 0
            v_calc_prox = v_calc[prox_columns].sort_index(key=sorting_func)
            v_calc_dist = v_calc[dist_columns].sort_index(key=sorting_func)
            assert v_calc_prox.shape == v_calc_dist.shape
            prox_nonzero = v_calc_prox[v_calc_prox != 0]
            dist_nonzero = v_calc_dist[v_calc_dist != 0]
            assert v_calc_prox.shape[0] > prox_nonzero.shape[0]
            assert v_calc_dist.shape[0] > dist_nonzero.shape[0]
            threshold_prox = prox_nonzero.nsmallest(norm_res_count).max()
            residues_prox = prox_nonzero.nsmallest(norm_res_count).sort_index(key=sorting_func)
            threshold_dist = dist_nonzero.nsmallest(norm_res_count).max()
            residues_dist = dist_nonzero.nsmallest(norm_res_count).sort_index(key=sorting_func)
            print(f"Proximal threshold = {threshold_prox}, Distal threshold = {threshold_dist}")
        elif kind == 'mean':
            raise Exception("Rework similar to `var` method.")
            v_calc = sPRE_comp.var(axis='rows')
            v_calc_prox = v_calc[prox_columns].values
            v_calc_dist = v_calc[dist_columns].values
            threshold_prox = np.partition(v_calc_prox[np.nonzero(v_calc_prox)], norm_res_count)[norm_res_count - 1]
            threshold_dist = np.partition(v_calc_dist[np.nonzero(v_calc_dist)], norm_res_count)[norm_res_count - 1]
            print(f"Proximal threshold = {threshold_prox}, Distal threshold = {threshold_dist}")
        else:
            raise Exception("`kind` must be either 'var' or 'mean'.")

        # get the columns that fulfill this condition
        min_values_prox = (v_calc_prox <= threshold_prox) & (v_calc_prox != 0.0)
        centers_prox = min_values_prox[np.where(min_values_prox)[0]].index
        min_values_dist = (v_calc_dist <= threshold_dist) & (v_calc_dist != 0.0)
        centers_dist = min_values_dist[np.where(min_values_dist)[0]].index

        print(f"Considered residues are Prox: {residues_prox.index.tolist()} with these variances: {np.round(residues_prox.values, 4).tolist()}"
              f"\nand Dist: {residues_dist.index.tolist()} with these variances: {np.round(residues_prox.values, 4).tolist()}")

        # test + 76
        a = v_calc_dist[min_values_dist]
        b = v_calc[centers_dist]
        if not a.equals(b):
            # print(v_calc)
            raise Exception(f"Centers do not match. Maybe prox and dist was "
                            f"mixed up? a is {a}, b is {b}, min_values_dist "
                            f"is {min_values_dist} and centers_dist is "
                            f"{centers_dist}")

        # get the factors
        v_calc = sPRE_comp.mean(axis='rows')
        factors_prox = df_obs[ubq_site][residues_prox.index].values / v_calc[residues_prox.index].values
        factors_dist = df_obs[ubq_site][residues_dist.index].values / v_calc[residues_dist.index].values
        f_prox = np.mean(factors_prox)
        f_dist = np.mean(factors_dist)
        factors_out[ubq_site] = {'proximal': f_prox, 'distal': f_dist}
        print(f"Proximal factor = {f_prox}, Distal factor = {f_dist}")

        if print_factors:
            print(f"At ubq_site: {ubq_site} factors is {factors_out}.")

        # copy the existing values and multiply
        new_values = sPRE_comp.copy()
        new_values[prox_columns] *= f_prox
        new_values[dist_columns] *= f_dist

        sPRE_norm = new_values
        rename_columns = {k: 'normalized '+k for k in sPRE_comp.columns}
        sPRE_norm = sPRE_norm.rename(columns=rename_columns)
        # sPRE_norm.to_csv(f'sPRE_{ubq_site}_normalized_via_10_min_variance.csv')

        # a check
        for check in ['proximal', 'distal']:
            norm_columns = [c for c in sPRE_norm.columns if check in c]
            assert len(norm_columns) == 76
            non_norm_columns = [c for c in sPRE_comp.columns if check in c]
            assert len(norm_columns) == len(non_norm_columns)
            a = sPRE_norm[norm_columns].values
            a = a[~np.isnan(a)].flatten()
            b = sPRE_comp[non_norm_columns].values
            b = b[~np.isnan(b)].flatten()
            test = np.unique((a / b).round(10))
            test = test[~np.isnan(test)]
            if check == 'proximal':
                print(test, f_prox)
                assert np.isclose(test, f_prox)
                assert np.isclose(test, factors_out[ubq_site][check])
            else:
                print(test, f_dist)
                assert np.isclose(test, f_dist)
                assert np.isclose(test, factors_out[ubq_site][check])

        # append to new frame
        out.append(sPRE_norm)
        print('\n')

    if get_factors:
        return factors_out

    if print_factors:
        print(f"Normally {factors_out} would have been returned.")

    # set the norm columns to 0, even if they exist
    norm_cols = []
    df_comp_w_norm = df_comp.copy()
    # for c in df_comp_w_norm.columns:
    #     if 'sPRE' in c and 'norm' not in c:
    #         new_name = f'normalized {c}'
    #         norm_cols.append(new_name)
    #         df_comp_w_norm[new_name] = 0

    # check if all is 0
    assert (df_comp_w_norm[norm_cols] == 0).all(None)

    def sort_columns(type_, normed=False):
        if not normed:
            def sort(x):
                if 'sPRE' in x and type_ in x and 'norm' not in x:
                    return True
                else:
                    return False
        else:
            def sort(x):
                if 'sPRE' in x and type_ in x and 'norm' in x:
                    return True
                else:
                    return False
        return sort

    # make the values
    for ubq_site in pd.value_counts(df_comp['ubq_site']).index:
        for type_ in ['proximal', 'distal']:
            if print_factors:
                print(f"In checking ubq_site: {ubq_site}, type_: {type_}, factors_out is {factors_out}.")
            try:
                factor = factors_out[ubq_site][type_]
            except KeyError:
                print(factors_out)
                raise
            non_norm_cols = list(filter(sort_columns(type_, normed=False), df_comp_w_norm.columns))
            assert all([type_ in _ for _ in non_norm_cols]), print(non_norm_cols)
            assert len(non_norm_cols) == 76, print(len(non_norm_cols))
            norm_cols = ['normalized ' + c for c in non_norm_cols]
            assert len(non_norm_cols) == len(norm_cols), print(norm_cols, non_norm_cols)
            old_values = df_comp_w_norm[df_comp_w_norm['ubq_site'] == ubq_site][non_norm_cols].values
            new_values = old_values * factor

            new_df = df_comp.copy()[['ubq_site', 'traj_file', 'frame']]
            new_df = new_df[new_df['ubq_site'] == ubq_site]
            assert len(new_df.columns) == 3

            # test whether numpy multiplication works
            test = np.unique(np.unique(new_values / old_values).round(10))
            mask = np.logical_and(~np.isnan(test), test != 0)
            test_ = test[mask]
            assert np.isclose(test_, factor), print(test_, factor)
            assert len(norm_cols) == new_values.T.shape[0]

            new_df[norm_cols] = new_values
            assert len(new_df.columns) == 76 + 3

            df_comp_w_norm = df_comp_w_norm.combine_first(new_df)
            values = np.unique(df_comp_w_norm[df_comp_w_norm['ubq_site'] == ubq_site][norm_cols].values)
            assert len(values) > 1

            # and check
            a = df_comp_w_norm[df_comp_w_norm['ubq_site'] == ubq_site][norm_cols]
            b = df_comp_w_norm[df_comp_w_norm['ubq_site'] == ubq_site][non_norm_cols]
            test = np.unique(np.unique(a / b).round(9))
            mask = np.logical_and(~np.isnan(test), test != 0)
            test_ = np.unique(test[mask])
            try:
                check = np.isclose(test_, factor)
            except ValueError as e:
                e2 = Exception(f"Can not compare factor {factor} to test_ {test_}. Seems like a rework is needed.")
                raise e2 from e
            if not np.all(check):
                msg = (f"I tried to check the normalization of {ubq_site} {type_}, "
                       f"by getting the normalized columns: {norm_cols} and the non-normalized "
                       f"columns: {non_norm_cols}. Dividing these values yields the value {test_}. "
                       f"However, this is not the value, that is stored in the factors_out dict: {factor}")
                raise Exception(msg)
            else:
                print(f"{ubq_site} {type_} was correctly normalized")

    if missing:
        raise Exception("Don't know, whether this still works with the new factor assignments.")
        df_comp_w_norm = df_comp_w_norm[~ df_comp_w_norm['ubq_site'].str.contains('|'.join(missing))]

    df_comp_w_norm = df_comp_w_norm.fillna(0)

    return df_comp_w_norm, centers_prox, centers_dist # - 76 changed a second time. Make this function prox dist agnostic # changed due to new nmin method from pandas


def call_xplor_with_yaml(pdb_file, psf_file=None, yaml_file='', from_tmp=False,
                         testing=False, fix_isopeptides=True, **kwargs):
    """Calls the xplor script with values from a yaml file.

    Arguments for XPLOR can be provided by either specifying a yaml file, or
    providing them as keyword_args to this function.

    Args:
        pdb_file (str): File path to a pdb file. Can be either a local file
            or a project file in xplor/data

    Keyword Args:
        yaml_file (str, optional): Path to a yaml file. If an empty string is provided
            the defaults.yml file from xplor/data is loaded. Defaults to ''.
        from_tmp (bool, optional): Changes the executable of the command to
            work with an ssh interpreter to 134.34.112.158. If set to false,
            the executable will be taken from xplor/scripts.
            Defaults to False
        testing (bool, optional): Adds the '-testing' flag to the command.
            Defaults to False.
        fix_isopeptides (bool, optional): Whether to fix isopeptide bonds using the
            technique developed in `check_conect`. Defaults to True.
        psf_file (Union[str, None], optional): Provide the path to a psf file
            when setting `fix_isopeptides` True. Defaults to None and will
            raise an Error, if `fix_isopeptides` is set to True and `psf_file` is None.
        **kwargs: Arbitrary keyword arguments. Keywords that are not flags
            of the xplor/scripts/xplor_single_struct_script.py will be discarded.

    Returns:
        str: The string which the xplor/scripts/xplor_single_struct_script.py
            prints to stdout

    """
    # load defaults
    if not yaml_file:
        yaml_file = YAML_FILE
    with open(yaml_file, 'r') as stream:
        defaults = yaml.safe_load(stream)

    pdb_file = get_local_or_proj_file(pdb_file)

    # overwrite tbl files, if present
    defaults.update((k, kwargs[k]) for k in set(kwargs).intersection(defaults))

    # get the datafiles
    for pot in ['psol', 'rrp600', 'rrp800']:
        if pot in defaults:
            filename = get_local_or_proj_file(defaults[pot]['call_parameters']['restraints']['value'])
            if not os.path.exists(filename) and not 'empty' in filename:
                raise Exception(f"The restraint file {filename} does not exist.")
            elif not os.path.exists(filename) and 'empty' in filename:
                raise NotImplementedError("Write a function to write an empty tbl file to /tmp and pass that")
            defaults[pot]['call_parameters']['restraints']['value'] = filename

    # make arguments out of them
    arguments = write_argparse_lines_from_yaml_or_dict(defaults)

    if from_tmp:
        executable = f'/home/kevin/software/xplor-nih/xplor-nih-3.2/bin/pyXplor {os.getcwd()}/xplor/scripts/xplor_single_struct.py'
    else:
        executable = '/home/kevin/software/xplor-nih/xplor-nih-3.2/bin/pyXplor ' + get_local_or_proj_file('scripts/xplor_single_struct.py')
    if fix_isopeptides:
        if psf_file is None:
            raise Exception("Please provide the path to a psf file, when running in `fix_isopeptides`-mode.")
        cmd = f'{executable} -pdb {pdb_file} -psf {psf_file} {arguments}'
        cmd += ' -struct_loading_method initStruct'
    else:
        cmd = f'{executable} -pdb {pdb_file} {arguments}'
        cmd += ' -struct_loading_method loadPDB'
    if testing:
        cmd += ' -testing'
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    return_code = process.poll()
    out = out.decode(sys.stdin.encoding)
    err = err.decode(sys.stdin.encoding)
    if return_code > 0:
        print(out)
        pdb_save = os.path.join(os.getcwd(), '_'.join(pdb_file.split('_')[:5]) + '.pdb')
        shutil.copyfile(pdb_file, pdb_save)
        psf_save = 'Protocol did not use psf file.'
        if psf_file is not None:
            psf_save = pdb_save.replace('.pdb', '.psf')
            shutil.copyfile(psf_file, psf_save)
        raise Exception(f"Call to subprocess using pdb file {pdb_file} did not succeed."
                        f"Here's the error: {err}, and the return code: {return_code}. I"
                        f"saved the files that causes this error here:"
                        f"pdb: {pdb_save},"
                        f"psf: {psf_save}")
    out = out.split('findImportantAtoms: done')[1]
    return out


def parallel_xplor(ubq_sites, simdir='/home/andrejb/Research/SIMS/2017_*', n_threads='max-2',
                   df_outdir='/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package/',
                   suffix='_df_no_conect.csv', write_csv=True, fix_isopeptides=True, specific_index=None, parallel=False,
                   subsample=5, yaml_file='', testing=False, from_tmp=False, max_len=-1, break_after=False,
                   delete_csvs=None, **kwargs):
    """Runs xplor on many simulations in parallel.

    This function is somewhat specific and there are some hardcoded directories
    in it. It uses MDTraj and OpenMM to load trajectories from Andrej's sim
    directory (/home/andrejb/Research/SIMS/). These trajectories are provided
    in a joblib Parallel/delayed construct to `get_series_from_mdtraj`, which
    results in a list of pandas Series, that are stacked to a long dataframe.

    The dataframe is periodically saved (to not loose anything). Check out the
    function `xplor.delete_old_csvs` to remove the unwanted intermediate csvs,
    produced by this function.

    Args:
        ubq_sites (list): A list of ubiquitination sites, that should be recognized.

    Keyword Args:
        simdir (str, optional): Path to the sims, that contain the ubq_site substring.
            Defaults to '/home/andrejb/Research/SIMS/2017_*'
        n_threads (Union[int, str], optional): The number of threads to run.
            Can be an int, but also 'max' or 'max-2', where 'max' will give
            make this function use the maximum number of cores. 'max-2' will use
            all but 2 cores. Defaults to 'max-2'
        df_outdir (str, optional): Where to save the csv files to. Defaults to
            '/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package/'.
        suffix (str, optional): Suffix of the csv files, used to sort different
            runs. Defaults to '_df_no_conect.csv'.
        write_csv (bool, optional): Whether to write the csv to disk. Defaults
            to True.
        subsample (int, optional): Whether to subsample trajectories. Give an
            int and only use every `subsample`-th frame. Defaults to 5.
        max_len (int, optional): Only go to that maximum length of a trajectory.
            Defaults to -1, wich will use the full length of the trajectories.
        yaml_file (str, optional): Path to a yaml file. If an empty string is provided
            the defaults.yml file from xplor/data is loaded. Defaults to ''.
        from_tmp (bool, optional): Changes the executable of the command to
            work with an ssh interpreter to 134.34.112.158. If set to false,
            the executable will be taken from xplor/scripts.
            Defaults to False
        testing (bool, optional): Adds the '-testing' flag to the command.
            Defaults to False.
        specific_index (Union[int, None], optional): If given, only that Index will
            be used in the parallel loop. For Debugging. Defaults to None.
        parallel (bool): Whether to do the calculations in parallel. Defaults to False.
        break_after (Union[bool, int], optional): Whether to break the for loop early, stopping the
            calculation after the specified number of loops. Defaults to False.
        fix_isopeptides (Union[bool, int], optional): Whether to fix isopeptide bonds using the
            technique developed in `check_conect`. If set to True all calculations use this
            techniyue. Can also take an int, that is larger than `subsample`, in that
            case, only every `fix_isopeptided` frame will use this technique, the other
            frames will use the old, faster protocol. Defaults to 25.
        delete_old_csvs (Union[int, None]): To delete old csvs and save some disk-space.
            Uses misc.delete_old_csvs with `keep=delete_old_csvs` to keep the specified
            number of dataframes.
        **kwargs: Arbitrary keyword arguments. Keywords that are not flags
            of the xplor/scripts/xplor_single_struct_script.py will be discarded.

    Returns:
        pd.Dataframe: A pandas Dataframe.

    """
    if n_threads == 'max-2':
        n_threads = multiprocessing.cpu_count() - 2
    elif n_threads == 'max':
        n_threads = multiprocessing.cpu_count()

    if np.any(fix_isopeptides):
        print("Using procedure from `check_conect` to fix isopeptide bonds.")

    # get list of already existing dfs
    files = glob.glob(f'{df_outdir}*{suffix}')
    if files and write_csv:
        highest_datetime_csv = sorted(files, key=get_iso_time)[-1]
        df = pd.read_csv(highest_datetime_csv, index_col=0)
    else:
        df = pd.DataFrame({})

    # run loop
    for i, ubq_site in enumerate(ubq_sites):
        for j, dir_ in enumerate(glob.glob(f"{simdir}{ubq_site}_*") + glob.glob(f'/home/kevin/projects/molsim/diUbi_aa/{ubq_site.upper()}_*')):
            traj_file = dir_ + '/traj_nojump.xtc'
            if 'andrejb' in traj_file:
                if not is_aa_sim(traj_file):
                    print(f"{traj_file} is not an AA sim")
                    continue
            basename = traj_file.split('/')[-2]
            if 'andrejb' in traj_file:
                top_file = dir_ + '/start.pdb'
            else:
                top_file = dir_ + '/init.gro'
            with Capturing() as output:
                top_aa = CustomGromacsTopFile(f'/home/andrejb/Software/custom_tools/topology_builder/topologies/gromos54a7-isop/diUBQ_{ubq_site.upper()}/system.top',
                                              includeDir='/home/andrejb/Software/gmx_forcefields')
            traj = md.load(traj_file, top=top_file)
            traj.top = md.Topology.from_openmm(top_aa.topology)

            # check if traj is complete
            try:
                value_counts = pd.value_counts(df['traj_file'])
                frames = value_counts[traj_file]
            except KeyError:
                frames = 0
            if frames >= traj[:max_len:subsample].n_frames:
                print(f"traj {basename} already finished")
                continue
            else:
                print(f"traj {basename} NOT FINISHED")

            isopeptide_bonds = []
            isopeptide_indices = []
            for r in traj.top.residues:
                if r.name == 'GLQ':
                    r.name = 'GLY'
                    for a in r.atoms:
                        if a.name == 'C':
                            isopeptide_indices.append(a.index + 1)
                            isopeptide_bonds.append(f"{a.residue.name} {a.residue.resSeq} {a.name}")
                if r.name == 'LYQ':
                    r.name = 'LYS'
                    for a in r.atoms:
                        if a.name == 'CQ': a.name = 'CE'
                        if a.name == 'NQ':
                            a.name = 'NZ'
                            isopeptide_indices.append(a.index + 1)
                            isopeptide_bonds.append(f"{a.residue.name} {a.residue.resSeq} {a.name}")
                        if a.name == 'HQ': a.name = 'HZ1'

            # create an array of Trues for fix_isopeptides
            if fix_isopeptides is True:
                fix_isopeptides_arr = np.full(traj.n_frames, True)[:max_len:subsample]
            elif fix_isopeptides is False:
                fix_isopeptides_arr = np.full(traj.n_frames, False)[:max_len:subsample]
            elif isinstance(fix_isopeptides, int):
                if fix_isopeptides < subsample:
                    raise Exception("`fix_isopeptides` is smaller than `subsample`. Please fix.")
                elif specific_index is not None:
                    pass
                else:
                    if fix_isopeptides % subsample != 0:
                        raise Exception("`fix_isopeptides` % `subsample` should be 0. Please change them to be divisible.")
                    _ = np.full(traj.n_frames, False)[:max_len:subsample]
                    _[::int(fix_isopeptides/subsample)] = True
                    fix_isopeptides_arr = _

            # parallel call
            if parallel:
                if specific_index is None:
                    out = Parallel(n_jobs=n_threads, prefer='threads')(delayed(get_series_from_mdtraj)(frame,
                                                                                                       traj_file,
                                                                                                       top_file,
                                                                                                       frame_no,
                                                                                                       testing=testing,
                                                                                                       from_tmp=from_tmp,
                                                                                                       yaml_file=yaml_file,
                                                                                                       fix_isopeptides=fix,
                                                                                                       isopeptide_bonds=isopeptide_bonds,
                                                                                                       **kwargs) for
                                                                       frame, frame_no, fix in zip(traj[:max_len:subsample],
                                                                                              np.arange(traj.n_frames)[:max_len:subsample],
                                                                                              fix_isopeptides_arr))
                else:
                    out = Parallel(n_jobs=n_threads, prefer='threads')(delayed(get_series_from_mdtraj)(frame,
                                                                                                       traj_file,
                                                                                                       top_file,
                                                                                                       frame_no,
                                                                                                       testing=testing,
                                                                                                       from_tmp=from_tmp,
                                                                                                       yaml_file=yaml_file,
                                                                                                       fix_isopeptides=fix_isopeptides,
                                                                                                       isopeptide_bonds=isopeptide_bonds,
                                                                                                       **kwargs) for
                                                                       frame, frame_no in zip(traj[specific_index], [specific_index]))
            else:
                if specific_index is None:
                    out = []
                    for frame, frame_no, fix in zip(traj[:max_len:subsample], np.arange(traj.n_frames)[:max_len:subsample], fix_isopeptides_arr):
                        out.append(get_series_from_mdtraj(frame, traj_file, top_file, frame_no,
                                                          testing=testing, from_tmp=from_tmp, yaml_file=yaml_file,
                                                          fix_isopeptides=fix, isopeptide_bonds=isopeptide_bonds,
                                                          **kwargs))
                else:
                    out = []
                    for frame, frame_no in zip(traj[specific_index], [specific_index]):
                        out.append(get_series_from_mdtraj(frame, traj_file, top_file, frame_no,
                                                          testing=testing, from_tmp=from_tmp, yaml_file=yaml_file,
                                                          fix_isopeptides=fix_isopeptides,
                                                          isopeptide_bonds=isopeptide_bonds,
                                                          **kwargs))

            # continue working with output
            now = datetime_windows_and_linux_compatible()
            df_name = os.path.join(df_outdir, f"{now}{suffix}")
            df = df.append(out, ignore_index=True)
            if write_csv:
                df.to_csv(df_name)
            if isinstance(delete_old_csvs, int):
                delete_old_csvs(df_outdir=df_outdir, suffix=suffix, keep=delete_csvs)
            if testing:
                break
            if break_after:
                if j >= break_after:
                    break
        if break_after:
            break
    return df


def _start_series_with_info(frame, traj_file, top_file, frame_no):
    """Pre-formats a pandas series to which the XPLOR output can be appended.

    Args:
        frame (mdtraj.Trajectory): The traj which will be saved,
        traj_file (str): The name of the parent traj (this will be an .xtc file).
        top_file (str): The topology used for this traj file.
        frame_no (int): The number of the frame.

    Returns:
        tuple: A tuple containing the following:
            pd.Series: A pre-formatted pandas series.
            str: The basename of the `traj_file`.
            str: The name of the temporary pdb file including a hash.

    """
    # define a pre-formatted pandas series
    data = {'traj_file': traj_file, 'top_file': top_file, 'frame': frame_no, 'time': frame.time[0]}
    column_names = get_column_names_from_pdb()
    columns = {column_name: np.nan for column_name in column_names}
    data.update(columns)
    series = pd.Series(data=data)

    # get data
    basename = os.path.basename(traj_file).split('.')[0]
    pdb_file = os.path.join(os.getcwd(), f'tmp_{basename}_frame_{frame_no}_hash_{abs(hash(frame))}.pdb')

    return series, basename, pdb_file


def _test_xplor_with_pdb(pdb_file):
    tbl_file = get_local_or_proj_file('data/diUbi_k6_800_mhz_relaxratiopot_in.tbl')
    executable = '/home/kevin/software/xplor-nih/executables/pyXplor -c '
    cmd = f""""import protocol
    from diffPotTools import readInRelaxData
    from relaxRatioPotTools import create_RelaxRatioPot
    protocol.loadPDB('{pdb_file}', deleteUnknownAtoms=True)
    protocol.initParams('protein')
    relax_data_in = readInRelaxData('{tbl_file}', pattern=['resid', 'R1', 'R1_err', 'R2', 'R2_err', 'NOE', 'NOE_err'])
    rrp = create_RelaxRatioPot(name='rrp600', data_in=relax_data_in, freq=600.0)
    print([[600, r.name().split()[1], r.calcd()] for r in rrp.restraints()])"
    """
    cmd = '; '.join([i.lstrip() for i in cmd.splitlines()])
    cmd = executable + cmd.rstrip('; ')
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    return_code = process.poll()
    out = out.decode(sys.stdin.encoding)
    err = err.decode(sys.stdin.encoding)
    if return_code != 0:
        print(err)
        raise Exception(f"xplor with pdb and rrp failed with exit code {return_code}")
    else:
        return out, True


def _test_xplor_with_psf(psf_file, pdb_file):
    tbl_file = get_local_or_proj_file('data/diUbi_k6_800_mhz_relaxratiopot_in.tbl')
    executable = '/home/kevin/software/xplor-nih/executables/pyXplor -c '
    cmd = f""""import protocol
        from diffPotTools import readInRelaxData
        from relaxRatioPotTools import create_RelaxRatioPot
        protocol.initStruct('{psf_file}')
        print('struct loaded')
        protocol.initCoords('{pdb_file}')
        print('pdb loaded')
        protocol.initParams('protein')
        relax_data_in = readInRelaxData('{tbl_file}', pattern=['resid', 'R1', 'R1_err', 'R2', 'R2_err', 'NOE', 'NOE_err'])
        rrp = create_RelaxRatioPot(name='rrp600', data_in=relax_data_in, freq=600.0)
        print([[600, r.name().split()[1], r.calcd()] for r in rrp.restraints()])"
        """
    cmd = '; '.join([i.lstrip() for i in cmd.splitlines()])
    cmd = executable + cmd.rstrip('; ')
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    return_code = process.poll()
    out = out.decode(sys.stdin.encoding)
    err = err.decode(sys.stdin.encoding)
    if return_code != 0:
        print(err)
        raise Exception(f"xplor with psf, pdb and rrp failed with exit code {return_code}")
    elif '%PARSER-ERR: Too many parsing errors --> termination' in out:
        print(f"xplor died with parser error.")
        return out, False
    else:
        return out, True


def call_pdb2psf(file, dir_, cwd):
    try:
        os.chdir(dir_)
        cmd = f'/home/kevin/software/xplor-nih/executables/pdb2psf {file}'
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        out, err = process.communicate()
        return_code = process.poll()
        out = out.decode(sys.stdin.encoding)
        err = err.decode(sys.stdin.encoding)
    finally:
        os.chdir(cwd)
    return out, err, return_code


def test_conect(traj_file, pdb_file, remove_after=True, frame_no=0, ast_print=0,
                top='', dir_='/home/kevin/projects/tobias_schneider/test_parmed/'):
    cwd = os.getcwd()

    # create OpenMM topology
    if not top:
        basename, ubq_site = get_ubq_site_and_basename(traj_file)
        # gromos_top_file = f'/home/andrejb/Software/custom_tools/topology_builder/topologies/gromos54a7-isop/diUBQ_{ubq_site.upper()}/system.top'
        gromos_top_file = f'/home/andrejb/Research/DEVA/2017_04_27_UBQ_TOPOLOGIES/top_G54A7/diUBQ/{ubq_site}_01/system.top'
        with Capturing() as output:
            omm_top = CustomGromacsTopFile(gromos_top_file, includeDir='/home/andrejb/Software/gmx_forcefields')
    else:
        with Capturing() as output:
            omm_top = CustomGromacsTopFile(top, includeDir='/home/soft/gromacs/gromacs-2021.1/src/share/top/')

    # mdtraj part
    traj = md.load_frame(traj_file, index=frame_no, top=pdb_file)
    print('[DEBUG]: traj.n_atoms:', traj.n_atoms, 'omm_top.n_atoms:', md.Topology.from_openmm(omm_top.topology).n_atoms)
    traj.top = md.Topology.from_openmm(omm_top.topology)

    isopeptide_bonds = []
    for r in traj.top.residues:
        if r.name == 'GLQ':
            r.name = 'GLY'
            for a in r.atoms:
                if a.name == 'C':
                    isopeptide_bonds.append(f"{a.residue.name} {a.residue.resSeq} {a.name}")
        if r.name == 'LYQ':
            r.name = 'LYS'
            for a in r.atoms:
                if a.name == 'CQ': a.name = 'CE'
                if a.name == 'NQ':
                    a.name = 'NZ'
                    isopeptide_bonds.append(f"{a.residue.name} {a.residue.resSeq} {a.name}")
                if a.name == 'HQ': a.name = 'HZ1'

    # save and call xplor
    mdtraj_pdb = os.path.join(dir_, 'test_mdtraj.pdb')
    mdtraj_psf = os.path.join(dir_, 'test_mdtraj.psf')
    traj.save_pdb(mdtraj_pdb)
    out_straight_pdb, _ = _test_xplor_with_pdb(mdtraj_pdb)
    if _:
        print('[DEBUG]: straight XPLOR from pdb succeeded.')
        print('[VALUES]: ', ast.literal_eval(out_straight_pdb.splitlines()[-1])[ast_print])
    else:
        print('[ERROR]: straight XPLOR from pdb DID NOT SUCCEED.')
        print('[ERROR]: ', out_straight_pdb)

    # call pdb2psf and try again
    call_pdb2psf('test_mdtraj.pdb', dir_, cwd)

    # call xplor with this file
    out_psf_from_pdb2psf, _ = _test_xplor_with_psf(mdtraj_psf, mdtraj_pdb)

    # check the output
    if _:
        print('[DEBUG]: XPLOR with psf (from pdb2psf) and pdb (from MDTraj) succeeded.')
        print('[VALUES]: ', ast.literal_eval(out_psf_from_pdb2psf.splitlines()[-1])[ast_print])
    else:
        print('[ERROR]: XPLOR with psf (from pdb2psf) and pdb (from MDTraj) DID NOT SUCCEED.')
        print('[ERROR]: ', out_psf_from_pdb2psf)

    if 'm1' in traj_file:
        traj_atom_slice = traj.atom_slice(traj.top.select('not residue 76 and not residue 77'))
        mdtraj_atom_slice_pdb = os.path.join(dir_, 'test_mdtraj_atom_slice.pdb')
        mdtraj_atom_slice_psf = os.path.join(dir_, 'test_mdtraj_atom_slice.psf')
        traj_atom_slice.save_pdb(mdtraj_atom_slice_pdb)
        call_pdb2psf('test_mdtraj_atom_slice.pdb', dir_, cwd)
        out_removed_linkage_from_m1, _ = _test_xplor_with_psf(mdtraj_atom_slice_psf, mdtraj_atom_slice_pdb)

        if _:
            print('[DEBUG]: XPLOR from M1-linked diUBQ (without GLY76 and MET77) with psf (from pdb2psf) and pdb (from MDTraj) succeeded.')
            value = ast.literal_eval(out_removed_linkage_from_m1.splitlines()[-1])[ast_print]
            print('[VALUES]: ', value)
        else:
            print('[ERROR]: XPLOR from M1-linked diUBQ (without GLY76 and MET77) with psf (from pdb2psf) and pdb (from MDTraj) DID NOT SUCCEED.')
            print('[ERROR]: ', out_removed_linkage_from_m1)

        test1 = ast.literal_eval(out_psf_from_pdb2psf.splitlines()[-1])[ast_print][-1]
        test2 = ast.literal_eval(out_removed_linkage_from_m1.splitlines()[-1])[ast_print][-1]

        if test1 != test2:
            segid = int(value[1])
            residue = traj_atom_slice.top.atom(traj_atom_slice.top.select(f'resSeq {segid}')[0]).residue
            print("[SUCCESS]: After removing GLY76 and MET77 from M1-linked diUBQ, the "
                  "Tensor of inertia changed and thus the R2/R2 relax ratios changed. "
                  f"From {test1} ({residue} of normal M1-diUBQ) to {test2} ({residue} of M1-diUBQ w/o GLY76 and MET77). "
                  "How can this effect be brought to K6-diUBQ?")
            return

    # add the bond to the psf from xplor
    mdtraj_copy_psf = os.path.join(dir_, 'test_mdtraj_copy.psf')
    shutil.copyfile(mdtraj_psf, mdtraj_copy_psf)

    before_atoms = _count_atoms(mdtraj_psf)
    before_bonds, before_should_be_bonds = _count_bonds(mdtraj_psf)

    _add_bond_to_psf(mdtraj_copy_psf, isopeptide_bonds)

    after_atoms = _count_atoms(mdtraj_copy_psf)
    after_bonds, after_should_be_bonds = _count_bonds(mdtraj_copy_psf)

    print(f'[DEBUG]: The number of atom-lines went from {before_atoms} to {after_atoms}')
    print(f'[DEBUG]: The number of bonds went from {before_bonds} to {after_bonds}')
    print(f'[DEBUG]: The number before the !NBOND directive went from {before_should_be_bonds} to {after_should_be_bonds}')

    # try pdbfixer and check the number of atoms
    mdtraj_copy_pdb = os.path.join(dir_, 'test_mdtraj_copy.pdb')
    shutil.copyfile(mdtraj_pdb, mdtraj_copy_pdb)
    _rename_atoms_according_to_charmm(mdtraj_copy_psf, mdtraj_copy_pdb)

    print(f"[DEBUG]: After also fixing and aligning the pdb, it now contains {md.load_pdb(mdtraj_copy_pdb).n_atoms} atoms.")

    out_manually_added_bonds, _ = _test_xplor_with_psf(mdtraj_copy_psf, mdtraj_copy_pdb)
    if _:
        if any([i[-1] != j[-1] for i, j in zip(out_psf_from_pdb2psf, out_manually_added_bonds)]):
            # from out_manually added bonds one aa is missing
            data1 = ast.literal_eval(out_psf_from_pdb2psf.splitlines()[-1])
            data2 = ast.literal_eval(out_manually_added_bonds.splitlines()[-1])

            # data2 does not contain aa resSeq 144
            values1 = np.array([i[2] for i in data1 if int(i[1]) <= 143])
            values2 = np.array([i[2] for i in data2 if int(i[1]) <= 143])
            if not any(np.equal(values1, values2)):
                mean = np.mean(values2 - values1)
                std = np.std(values2 - values1)
                print("[DEBUG]: All values between 'out_psf_from_pdb2psf' differ compared to 'out_manually_added_bonds'")
                print(f"[DEBUG]: The mean difference is {mean}, the standard deviation is {std}")
            print("[SUCCESS]: The command succeeded and the values differ at some positions")
            return
        else:
            print('[DEBUG]: XPLOR with manually added bond to psf file succeeded, but the values are identical.')
            print('[VALUES]: ', ast.literal_eval(out_manually_added_bonds.splitlines()[-1])[ast_print])
    else:
        print('[ERROR]: XPLOR with  manually added bond to psf file DID NOT SUCCEED.')
        print('[ERROR]: ', out_manually_added_bonds)

    # parmed part
    struct = pmd.openmm.load_topology(omm_top.topology)
    struct.coordinates = traj.xyz[0] * 10  # needed for nm to angstrom

    for r in struct.residues:
        if r.name == 'GLQ':
            r.name = 'GLY'
        if r.name == 'LYQ':
            r.name = 'LYS'
            for a in r.atoms:
                if a.name == 'CQ': a.name = 'CE'
                if a.name == 'NQ': a.name = 'NZ'
                if a.name == 'HQ': a.name = 'HZ1'

    # save
    pdb_file = 'test.pdb'
    psf_file = 'test.psf'
    struct.save(os.path.join(dir_, pdb_file), format='pdb', overwrite=True)
    struct.save(os.path.join(dir_, psf_file), format='psf', vmd=True, overwrite=True)

    # try this psf
    out3, _ = _test_xplor_with_psf(os.path.join(dir_, psf_file), os.path.join(dir_, pdb_file))
    if _:
        print('XPLOR with psf succeeded.')
        print(ast.literal_eval(out3.splitlines()[-1])[ast_print])
    else:
        print('XPLOR with psf did not succeed. As expected.')

    print('before')
    with open(os.path.join(dir_, psf_file), 'r') as f:
        print('\n'.join(f.read().splitlines()[:5]))

    _fix_parmed_psf(os.path.join(dir_, psf_file))

    print('after')
    with open(os.path.join(dir_, psf_file), 'r') as f:
        print('\n'.join(f.read().splitlines()[:5]))

    try:
        out5, _ = _test_xplor_with_psf(os.path.join(dir_, psf_file), os.path.join(dir_, pdb_file))
    except Exception as e:
        print("XPLOR with manually parsed psf file still fails. No idea, why... Here's the original Error:")
        print(e)
        _ = False
    if _:
        print('XPLOR with fixed psf succeeded.')
        print(ast.literal_eval(out5.splitlines()[-1])[ast_print])
    else:
        print('XPLOR with manually parsed psf did not succeed. Still don\'t know why')

    # clean up
    if remove_after:
        for file in ['test.pdb', 'test.psf', 'test_mdtraj.pdb', 'test_mdtraj.psf', 'test_mdtraj_copy.psf']:
            file = os.path.join(dir_, file)
            if not os.path.isfile(file):
                print(f"The file {file} was not present.")
            else:
                os.remove(file)


def _rename_atoms_according_to_charmm(psf_file, pdb_file, saveloc=None):
    """Takes a psf file and a pdb file. Adds Hydrogens to the pdb file with
    pdbfixer, overwrites the pdb file and renames atoms, like they are named in
    the psf file.

    Args:
        psf_file (str): The path to the psf file.
        pdb_file (str): The path to the pdb file.

    Keyword Args:
        saveloc (Union[str, None], optional): Where to save the pdb file to.
            This is useful, when `pdb_file` is an instance of `RAMFile`.
            If None is provided, `pdb_file` will be overwritten.
            Defaults to None.

    Returns:
        parmed.Structure: The ouptut structure.

    """
    fixer = PDBFixer(pdb_file)
    try:
        fixer.addMissingHydrogens(pH=13.0)
    except Exception:
        _ = md.load(pdb_file)
        for r in _.top.residues:
            print(r)
        raise

    if not isinstance(pdb_file, RAMFile):
        os.remove(pdb_file)
    else:
        pdb_file.seek(0)

    buffer = RAMFile()
    PDBFile.writeFile(fixer.topology, fixer.positions, buffer)
    buffer.seek(0)
    lines = buffer.read().splitlines()
    buffer.seek(0)
    atom_lines = [line.startswith('ATOM') for line in lines].count(True)

    # for i, (line1, line2) in enumerate(zip(pdb_file.read().splitlines(), lines)):
    #     print('line1: ', line1)
    #     print('line2: ', line2)
    #     if i == 10:
    #         break
    # print(_count_atoms(psf_file))
    # print(fixer.topology.getNumAtoms())

    with open(psf_file, 'r') as f:
        full_file = f.read()
    hunks = full_file.split('\n\n')
    atom_hunk = hunks[['!NATOM' in hunk for hunk in hunks].index(True)].splitlines()[1:]
    columns = ['Atom ID', 'Segment ID', 'Residue ID', 'Res. Name', 'Atom Name', 'Atom Type', 'Charge', 'Mass', 'Extra']
    psf_atoms = pd.DataFrame([x.split() for x in atom_hunk], columns=columns)
    psf_atoms['element'] = psf_atoms['Mass'].map(MASSES)
    assert not psf_atoms['element'].isna().any(), print(psf_atoms[psf_atoms['element'].isna()])

    new_pdb = []
    i = 0
    remove = 0
    for line in lines:
        if line.startswith('ATOM'):

            if line.split()[2] == 'H' and int(line.split()[5]) == 77:
                line = line.replace('  H   ', '  HN  ')
                lines[i] = line

            if line.split()[2] == 'H2' and int(line.split()[5]) == 77:
                remove += 1
                continue

            if line.split()[2] == 'H3' and int(line.split()[5]) == 77:
                remove += 1
                continue

            # fix the number names from pdbfixer
            if any([x.isdigit() for x in line.split()[2]]):
                atom_name = line.split()[2]
                resname = line.split()[3]
                atom = atom_name[:-1]
                number = int(atom_name[-1])
                atom_before = lines[i - 1].split()[-1]
                atom_name_before = lines[i - 1].split()[2]

                # get number before
                if any([x.isdigit() for x in lines[i - 1].split()[2]]):
                    number_before = int(lines[i - 1].split()[2][-1])
                else:
                    number_before = -1

                # decide how to change the name of the atom
                if 'N' in atom_name or 'C' in atom_name:
                    pass
                elif atom_name == 'HD2' and resname == 'PRO':
                    pass
                elif atom_name == 'HD3' and resname == 'PRO':
                    line = line.replace('HD3', 'HD1')
                    lines[i] = line
                elif number == number_before and (atom_before == 'C' or atom_before == 'N'):
                    pass
                elif number == 1 and atom_name_before != 'HZ1':
                    pass
                elif number == 1 and atom_name_before == 'HZ1':
                    # flip HZ1 and HZ2
                    line = line.replace('HZ1', 'HZ2')
                    lines[i] = line
                elif not f'{atom}{number - 1}' in lines[i-1] and not lines[i - 1].split()[2] == atom:
                    line = line.replace(atom_name, f'{atom}{number - 1}')
                    lines[i] = line

            # change the atom id for lines after the isopeptide MET77 N hydrogens have been dropped
            if remove > 0:
                atom_id = line.split()[1]
                new_atom_id = str(int(atom_id) - remove)
                line = line.replace(atom_id, new_atom_id)
                lines[i] = line

            # check where the lines are identical
            identical_elements = psf_atoms['element'] == line.split()[-1]
            identical_resid = psf_atoms['Residue ID'] == line.split()[5]
            identical_resname = psf_atoms['Res. Name'] == line.split()[3]
            identical_name = psf_atoms['Atom Name'] == line.split()[2]
            identical = identical_elements & identical_resid & identical_resname & identical_name

            # overwrite the lines
            if not identical.any():
                # MANUAL_MAPPING = {'H': 'HT1', 'H2': 'HT2', 'H3': 'HT3', 'O': 'OT1', 'OXT': 'OT2'}
                # for HT1
                if line.split()[2] in ['H', 'H2', 'H3'] and (int(line.split()[5]) == 1 or int(line.split()[5]) == 77):
                    psf_atom = H_MAPPING[line.split()[2]]
                # for isop HA1, HA2
                # elif line.split()[2] in ['HA1', 'HA2'] and int(line.split()[5]) == 77:
                #     print(line)
                #     raise NotImplementedError()
                # for OXT
                elif line.split()[2] in ['O', 'OXT'] and  int(line.split()[5]) == 152:
                    psf_atom = OXT_MAPPING[line.split()[2]]
                # for inbetween HN
                elif line.split()[2] == 'H' and int(line.split()[5]) != 1  and int(line.split()[5]) != 77:
                    psf_atom = 'HN'
                else:
                    print(line)
                    for _ in lines[i - 3:i + 10]:
                        print(_)
                    index = np.where(identical)[0]
                    for _ in range(i - 3, i + 10):
                        print(atom_hunk[_])
                    raise Exception(f"Could not find ANY matching atom for\n{line}")
            elif pd.value_counts(identical)[True] == 1:
                index = np.where(identical)[0][0]
                psf_atom = atom_hunk[index].split()[4]
            else:
                print('\n', line, '\n')
                for _ in lines[i-1:i+10]:
                    print(_)
                index = np.where(identical)[0]
                for idx in index:
                    print(atom_hunk[idx])
                raise Exception("Could not determine atom. Possible atoms are:")
            if len(psf_atom) > 4:
                raise Exception(f"This atom is too long: {psf_atom}")
            if len(psf_atom) == 4:
                line = line[:12] + f"{psf_atom:<3}" + line[16:]
            else:
                line = line[:13] + f"{psf_atom:<3}" + line[16:]
        i += 1
        new_pdb.append(line)

    # for i, (line1, line2) in enumerate(zip(lines, new_pdb)):
    #     print('line1: ', line1)
    #     print('line2: ', line2)
    #     if i == 10:
    #         break

    if saveloc is None:
        saveloc = pdb_file
    with open(saveloc, 'w') as f:
        for line in new_pdb:
            f.write(line + '\n')


def _count_atoms(psf_file):
    """Counts the number of atoms in a psf file."""
    with open(psf_file, 'r') as f:
        full_file = f.read()
    hunks = full_file.split('\n\n')
    atom_hunk = hunks[['!NATOM' in hunk for hunk in hunks].index(True)].splitlines()
    return len(atom_hunk) - 1 # subtract one for the line with !NATOM in it


def _count_bonds(psf_file):
    """Counts the number of bonds and atoms in bonds in a psf file.

    Args:
        psf_file (str): The path to the psf file.

    Returns:
        tuple: A tuple containing the following:
            int: Number of bonds.
            int: Number that occurs in XXXX !NBOND line in psf file.

    """
    with open(psf_file, 'r') as f:
        full_file = f.read()
    n_bonds = 0
    hunks = full_file.split('\n\n')
    bond_hunk = hunks[['!NBOND' in hunk for hunk in hunks].index(True)].splitlines()
    bond_lines = bond_hunk[1:]
    should_be = int(bond_hunk[0].split()[0])
    for bond_line in bond_lines:
        entries = len(bond_line.split())
        n_bonds += int(entries / 2)
    return n_bonds, should_be


def _add_bond_to_psf(psf_file, bond_atoms):
    """Adds a bond to a XPLOR (pdb2psf) psf file.

    Args:
        psf_file (str): The path to the psf file (will be overwritten).
        bond_atoms (list[str]): The atoms between which a bond will be added.

    """
    with open(psf_file, 'r') as f:
        full_file = f.read()
    hunks = full_file.split('\n\n')
    atom_hunk = hunks[['!NATOM' in hunk for hunk in hunks].index(True)].splitlines()
    lines = full_file.splitlines()
    bond_integers = _find_atoms(atom_hunk, bond_atoms)

    os.remove(psf_file)

    with open(psf_file, 'w') as f:
        for i, line in enumerate(lines):
            if i < len(lines) - 2:
                if '!NTHETA' in lines[i + 2]:
                    if len(line.split()) == 8:
                        line += '\n'
                    line += f'    {bond_integers[0]:>4}    {bond_integers[1]:>4}'
            if '!NBOND' in line:
                current_bond_count = line.split()[0]
                new_bond_count = str(int(current_bond_count) + 1)
                line = line.replace(current_bond_count, new_bond_count)
            f.write(line + '\n')


def _find_atoms(lines, bond_atoms):
    """Tries to find the atoms from bond_atoms in lines."""
    atom_integers = []
    for atom in bond_atoms:
        resname, resseq, atom_name = atom.split()
        for line in lines:
            if resname in line and resseq in line:
                if any([i == atom_name for i in line.split()]):
                    atom_integers.append(int(line.split()[0]))
    if not len(atom_integers) == 2:
        raise Exception("Could not unambigously determine CAs")
    return atom_integers


def isfloat(value):
    """Returns true if value is float"""
    try:
        float(value)
        return True
    except ValueError:
        return False


def _get_psf_atom_line(line):
    """Uses a line from a ParmEd psf file and formats it to an XPLOR psf file.

    (Can also use an XPLOR psf file line, to check whether the formatting works.)

    Args:
        line (str): The ParmEd psf file line.

    Returns:
        str: The new line.

    """
    data = {}
    datafields = ['no', 'chain', 'resseq', 'resname', 'name', 'type_', 'charge', 'mass', 'extra']
    for key, value in zip(datafields, line.split()):
        if value.isdigit():
            data[key] = int(value)
        elif isfloat(value):
            data[key] = float(value)
        else:
            data[key] = value
    new_line = f"    {data['no']:>4} {data['chain']}    {data['resseq']:<4} "
    new_line += f"{data['resname']}  {data['name']:<3}  {data['type_']:<3}   {data['charge']: .6f}"
    new_line += "       {:.7s}".format('{:0.5f}'.format(data['mass']))
    if len(data) == 9:
        new_line += f"           {data['extra']}"
    elif len(data) == 8:
        new_line += f"           0"
    else:
        raise Exception(f"Could not convert dict with {len(data)} keys to line. Here's the dict: {data}")
    return new_line


def _fix_parmed_psf(psf_file):
    """Fixes a ParmEd psf file.

    Args:
        psf_file (str): Path to the file. Will be repalced.

    """
    with open(psf_file, 'r') as f:
        lines = f.read().splitlines()

    os.remove(psf_file)

    with open(psf_file, 'w') as f:
        for line in lines:
            if 'PSF CHEQ EXT XPLOR' in line:
                line = 'PSF'
            if 'SYS' in line:
                line = line.replace('SYS', 'A')
                line = _get_psf_atom_line(line)
            if '!N' in line:
                line = line[2:]
            if '!NTITLE' in line:
                line += '\n REMARKS Created by ParmEd and manually fixed by Kevin Sawade (kevin.sawade at uni-konstanz.de)'
            if '!NTITLE' in line:
                f.write(line)
            else:
                f.write(line + '\n')


class RAMFile(StringIO):
    def lower(self):
        return 'tmp_stringio.pdb'

    def close(self):
        pass


def test_mdtraj_stringio(frame, traj_file, top_file, frame_no, testing=False,
                           from_tmp=False, yaml_file='', fix_isopeptides=True,
                           check_fix_isopeptides=False, isopeptide_bonds=None, **kwargs):
    """Function to test implementation of stringIO buffers as mdtraj save locations."""
    series, basename, pdb_file = _start_series_with_info(frame, traj_file, top_file, frame_no)

    file = RAMFile()
    frame.save_pdb(file)
    file.seek(0)
    print('\n'.join(file.read().splitlines()[:5]))



def get_series_from_mdtraj(frame, traj_file, top_file, frame_no, testing=False,
                           from_tmp=False, yaml_file='', fix_isopeptides=True,
                           check_fix_isopeptides=False, isopeptide_bonds=None,
                           print_raw_out=False, test_single_residue=False, **kwargs):
    """Saves a temporary pdb file which will then be passed to call_xplor_with_yaml

    Arguments for XPLOR can be provided by either specifying a yaml file, or
    providing them as keyword_args to this function. For example you can use
    provide `psol_call_parameters_tauc = 0.4` to set the correlation time of the
    psol potential to 0.4. The difference between `psol_call_parameters_tauc`
    and `psol_set_parameters_TauC` is, that the first is provided to psol, when
    it is instantiated (called):

    ```python
    psol = create_PSolPot(name=kwargs['psol_call_parameters_name'],
                          file=kwargs['psol_call_parameters_restraints'],
                          tauc=kwargs['psol_call_parameters_tauc'],
                          ...)
    ```

    The other is provided in a for loop with call signatures similar to:

    ```python
    prefix = 'psol_set_parameters_'
    for key, value in kwargs.items():
    if key.startswith(prefix):
        key = 'set' + key.replace(prefix, '')
        if isinstance(value, bool) and value:
            if key == 'setVerbose':
                getattr(psol, key)(True)
            # getattr(psol, key)() # currently not working
        else:
            getattr(psol, key)(value)
    ```

    In this code block the parameters that are set with code like this:

    ```python
    psol.setTauC(0.4)
    psol.setGammaI(26.752196)
    ```

    are set.

    Args:
        frame (md.Trajectory): An `mdtraj.core.Trajectory` instance with 1 frame.
        traj_file (str): The location of the original trajectory.
        top_file (str): The location of the original topology.
        frame_no (int): The frame number the trajectory originates from.

    Keyword Args:
        yaml_file (str, optional): Path to a yaml file. If an empty string is provided
            the defaults.yml file from xplor/data is loaded. Defaults to ''.
        from_tmp (bool, optional): Changes the executable of the command to
            work with an ssh interpreter to 134.34.112.158. If set to false,
            the executable will be taken from xplor/scripts.
            Defaults to False
        testing (bool, optional): Adds the '-testing' flag to the command.
            Defaults to False.
        fix_isopeptides (bool, optional): Whether to fix isopeptide bonds using the
            technique developed in `check_conect`. Defaults to True.
        check_fix_isopeptides (bool, optional): Whether to compare the results
            from a whole system (`fix_isopeptides` True) and a disjoint
            system (`fix_isopeptides` False). When this option is set to True,
            the function does not return anything, but rather prints a message,
            if the two different isopeptides give identical or different results.
            This function is meant for testing. Defaults to False.
        isopeptide_bonds (Union[list[str], None], optional): When you choose to fix
            the isopeptides (`fix_isopeptides` set to True), you
        **kwargs: Arbitrary keyword arguments. Keywords that are not flags
            of the xplor/scripts/xplor_single_struct_script.py will be discarded.

    Returns:
        pd.Series: A pandas Series instance.

    """
    # pre-formatted series
    series, basename, pdb_file = _start_series_with_info(frame, traj_file, top_file, frame_no)
    _, ubq_site = get_ubq_site_and_basename(traj_file)

    # how many residues.
    should_be_residue_number = frame.n_residues

    if fix_isopeptides:
        if isopeptide_bonds is None:
            raise Exception("Con only fix psf file when provided a list of str as argument `isopeptide_bonds`.")
        psf_file = get_local_or_proj_file(f'data/{ubq_site.lower()}_psf_for_xplor_with_added_bond.psf')
        pdb_stringio = RAMFile()
        frame.save_pdb(pdb_stringio)
        pdb_stringio.seek(0)
        _rename_atoms_according_to_charmm(psf_file, pdb_stringio, saveloc=pdb_file)
        pdb_stringio.seek(0)
        try:
            pdb_stringio.write(pdb_file)
            if check_fix_isopeptides:
                if not check_pdb_and_psf_integrity(pdb_file, psf_file, ubq_site):
                    raise Exception("psf and pdb are not integer.")
                else:
                    print('psf and pdb are integer.')
            out = call_xplor_with_yaml(pdb_file, psf_file=psf_file, from_tmp=from_tmp, testing=testing,
                                       yaml_file=yaml_file, fix_isopeptides=True, **kwargs)
            out = ast.literal_eval(out)
        finally:
            if os.path.isfile(pdb_file): os.remove(pdb_file)

        values1 = np.array([i[2] for i in out if i[0] == 'rrp600' and int(i[1]) <= 143])

    if not fix_isopeptides or check_fix_isopeptides:
        frame.save_pdb(pdb_file)

        try:
            out = call_xplor_with_yaml(pdb_file, psf_file=None, from_tmp=from_tmp, testing=testing,
                                       yaml_file=yaml_file, fix_isopeptides=False, **kwargs)
            try:
                out = ast.literal_eval(out)
            except SyntaxError:
                print(out)
                raise
        finally:
            os.remove(pdb_file)

        if series['proximal ILE3 sPRE'] == 0 and series['proximal PHE4 sPRE'] == 0:
            print(cmd)
            print(out)
            print(psol)
            raise Exception(f"This psol value should not be 0. Traj is {traj_file}, frame is {frame_no}")

    if print_raw_out:
        print(out)

    for o in out:
        # switched from <= to > because prox and dist was mixed up.
        if int(o[1]) > (should_be_residue_number / 2):
            resSeq = int(int(o[1]) - (should_be_residue_number / 2)) # changed from int(o[1]) to this to fix prox dist misunderstanding
            position = 'proximal'
        else:
            resSeq = int(o[1]) # changed from int(int(o[1]) - (should_be_residue_number / 2)) to this to fix prox dist misunderstanding
            position = 'distal'

        try:
            resname = frame.top.residue(int(o[1]) - 1).name
        except TypeError:
            print(o)
            raise

        if resSeq <= 0:
            raise Exception(f"A resSeq can not be 0, but {resSeq} was computed for {o}")

        if o[0] == 'rrp600':
            series[f'{position} {resname}{resSeq} 15N_relax_600'] = o[2]
        elif o[0] == 'rrp800':
            series[f'{position} {resname}{resSeq} 15N_relax_800'] = o[2]
        elif o[0] == 'psol':
            series[f'{position} {resname}{resSeq} sPRE'] = o[2]

        if test_single_residue:
            if f'{position} {resname}{resSeq} sPRE' == test_single_residue:
                print(o[2])
                raise Exception("STOP")

    basename, ubq_site = get_ubq_site_and_basename(traj_file)
    series['basename'] = basename
    series['ubq_site'] = ubq_site
    series['isopeptide'] = fix_isopeptides

    sPRE_ind = [c for c in series.index if 'sPRE' in c]
    rrp600_ind = [c for c in series.index if '600' in c]
    rrp800_ind = [c for c in series.index if '800' in c]

    if series[sPRE_ind + rrp600_ind + rrp800_ind].isnull().all():
        raise Exception("STOP")

    if check_fix_isopeptides:
        values2 = np.array([i[2] for i in out if i[0] == 'rrp600' and int(i[1]) <= 143])
        if not any(np.equal(values1, values2)):
            mean = np.mean(values1 - values2)
            std = np.std(values1 - values2)
            print("Values for rrp600 with isopeptide bond")
            print(values1)
            print("Values for rrp600 without isopeptide bonds")
            print(values2)
            print(f"The mean difference is {mean}, the standard deviation is {std}")
            print("The command succeeded and the values differ at some positions")
            return
        else:
            raise Exception("Values are identical for with and without isopeptide.")
            
    return series


def make_linear_combination_from_clusters(trajs, df, df_obs, fast_exchangers, ubq_site, return_means=False, cluster_nums=None,
                                          exclusions=[], new_method=True, return_non_norm_means=False, return_pandas=False,
                                          exclude_non_exp_values=False, manual_fix_columns=False):
    """Makes a linear combination from sPRE values and clustered trajs.

    Args:
        trajs (Union[encodermap.Info_all, None]): The trajs with cluster membership inside them.
            Can also be None. In which case, the df should have a column called 'cluster_membership'.
        df (pandas.DataFrame): The XPLOR data.
        df_obs (pandas.DataFrame): The observerd NMR values.
        fast_exchangers (pandas.DataFrame): Boolean dataframe with residues that exchnage rapidly.
        ubq_site (str): The ubiquitination site currenlty worked on.

    Keyword Args:
        return_means (bool, optional): Whether to return the cluster means additional to the linear solution.
        new_method (bool, optional): Whether to use the nerw method using scipy minimize.

    Returns:
        np.ndarray: The linear combination to build df_obs.

    """
    df = df.copy()
    df = df.fillna(0)
    df_obs = df_obs.copy()
    obs = df_obs[df_obs.index.str.contains('sPRE')][ubq_site].values

    df = df[df['ubq_site'] == ubq_site]
    sPRE_ind = [i for i in df.columns if 'sPRE' in i and 'norm' in i]
    non_norm_sPRE_ind =  [i for i in df.columns if 'sPRE' in i and 'norm' not in i]

    if manual_fix_columns:
        from xplor.proteins.proteins import get_column_names_from_pdb
        residues = get_column_names_from_pdb(return_residues=True)
        sPRE_ind = []
        non_norm_sPRE_ind = []
        for pos in ['proximal', 'distal']:
            for r in residues:
                sPRE_ind.append(f"normalized {pos} {r} sPRE")
                non_norm_sPRE_ind.append(f"{pos} {r} sPRE")
        obs = df_obs.loc[non_norm_sPRE_ind][ubq_site].values

    # put cluster membership into df
    if trajs is not None:
        cluster_membership_sPRE = []
        for i, traj in enumerate(trajs):
            frames = df[df['traj_file'] == traj.traj_file]['frame'].values
            cluster_membership_sPRE.append(traj.cluster_membership[frames])
        cluster_membership_sPRE = np.hstack(cluster_membership_sPRE)
        df['cluster_membership'] = cluster_membership_sPRE
    else:
        assert 'cluster_membership' in df

    # calculcate the per-cluster per-residue means
    cluster_means = []
    cluster_means_out = np.zeros((df['cluster_membership'].max() + 1, 152))
    non_norm_means_out = np.zeros((df['cluster_membership'].max() + 1, 152))
    if cluster_nums is None:
        solv = np.zeros(df['cluster_membership'].max() + 1)
        for cluster_num in np.unique((df['cluster_membership'])):
            if cluster_num == -1:
                continue
            if cluster_num in exclusions:
                continue
            mean = np.median(df[sPRE_ind][df['cluster_membership'] == cluster_num], axis=0)
            non_norm_mean = np.median(df[non_norm_sPRE_ind][df['cluster_membership'] == cluster_num], axis=0)
            cluster_means.append(mean)
            cluster_means_out[cluster_num] = mean
            non_norm_means_out[cluster_num] = non_norm_mean
    else:
        for cluster_num in cluster_nums:
            mean = np.median(df[sPRE_ind][df['cluster_membership'] == cluster_num], axis=0)
            non_norm_mean = np.median(df[non_norm_sPRE_ind][df['cluster_membership'] == cluster_num], axis=0)
            cluster_means.append(mean)
            cluster_means_out[cluster_num] = mean
            non_norm_means_out[cluster_num] = non_norm_mean
    cluster_means = np.vstack(cluster_means)
    # print(cluster_means.shape, obs.shape)
    # print(np.any(np.isnan(cluster_means)))

    # exclude fast exchangers
    fast_exchange = fast_exchangers[ubq_site].values

    if exclude_non_exp_values:
        fast_exchange = np.logical_or(fast_exchange, obs == 0)

    assert np.all(~np.isnan(cluster_means))
    assert np.all(~np.isnan(obs))

    # test numpy lsrq
    # solv = np.linalg.lstsq(cluster_means.T[~fast_exchange], obs[~fast_exchange])
    # print(solv)

    # test scipy nnls
    solv_ = scipy.optimize.nnls(cluster_means.T[~fast_exchange], obs[~fast_exchange])[0]

    # Feli approved
    if cluster_nums is None:
        i = 0
        for cluster_num in np.unique((df['cluster_membership'])):
            if cluster_num == -1:
                continue
            if cluster_num in exclusions:
                continue
            solv[cluster_num] = solv_[i]
            i += 1

        print('solv_: ', solv_.shape)
        print('clu_membership: ', np.unique(df['cluster_membership']))
        print('solv: ', solv.shape)

    if new_method:
        res = minimize(get_objective_function(cluster_means.T[~fast_exchange], obs[~fast_exchange]), solv_,
                       constraints=[constraint], bounds=bounds)
        assert np.isclose(np.sum(res.x), 1)

        if cluster_nums is None:
            i = 0
            for cluster_num in np.unique((df['cluster_membership'])):
                if cluster_num == -1:
                    continue
                if cluster_num in exclusions:
                    continue
                solv[cluster_num] = res.x[i]
                i += 1
        else:
            solv = res.x

    if return_pandas:
        test = df.groupby(['cluster_membership']).median()
        # remove cluster membership -1
        test = test.loc[test.index.tolist()[1:]]
        if len(test) < len(cluster_means_out):
            for filler_name in list(set(range(len(cluster_means_out))) - set(test.index)):
                filler = pd.Series(0, index=test.columns)
                filler.name = filler_name
                test = test.append(filler)
            test = test.sort_index()
        if not np.array_equal(test[sPRE_ind].values, cluster_means_out):
            raise Exception(f"You wanted to return a pandas Dataframe, but the values in the dataframe: "
                            f"{test[sPRE_ind].values[:5, :5]} does not match the values that would have been "
                            f"returned, if a numpy array was requested: {cluster_means_out[:5, :5]}. These two "
                            f"need to be the same to ensure reproducibility. Maybe the columns in the df are misaligned. "
                            f"Here they are: {test.columns.tolist()}")
        cluster_means_out = test[sPRE_ind]
        cluster_means_out['ubq_site'] = ubq_site

    # make linear combination
    # x = scipy.optimize.lsq_linear(cluster_means.T[~fast_exchange], obs[~fast_exchange], bounds=(0, 1))
    # argsort = np.argsort(x.x)[::-1]
    # with np.printoptions(suppress=True):
    #    print(argsort)
    #     print(x.x[argsort])

    assert np.isclose(np.sum(solv), 1)
    try:
        if return_means and not return_non_norm_means:
            return solv, cluster_means_out
        elif return_non_norm_means:
            return solv, cluster_means_out, non_norm_means_out
        return solv
    except ValueError:
        print(return_means)
        print(type(return_means))
        raise

def get_objective_function(cluster_means: np.ndarray, observed_values: np.ndarray) -> Callable:
    def obj_fun(x: np.ndarray) -> float:
        return np.linalg.norm(np.sum(x * cluster_means, 1) - observed_values)
    return obj_fun

constraint = NonlinearConstraint(np.sum, 1., 1.)
bounds = Bounds(0, 1)