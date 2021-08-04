################################################################################
# Imports
################################################################################


import mdtraj as md
import numpy as np
import pandas as pd
import os, sys, glob, multiprocessing, copy
from ..proteins.proteins import get_column_names_from_pdb
from joblib import Parallel, delayed
from .custom_gromacstopfile import CustomGromacsTopFile
from xplor.argparse.argparse import YAML_FILE, write_argparse_lines_from_yaml_or_dict
from ..misc import get_local_or_proj_file, get_iso_time

################################################################################
# Globals
################################################################################

__all__ = ['parallel_xplor', 'get_series_from_mdtraj', 'call_xplor_with_yaml',
           'normalize_sPRE']

################################################################################
# Functions
################################################################################


def datetime_windows_and_linux_compatible():
    import datetime
    from sys import platform
    if platform == "linux" or platform == "linux2" or platform == "darwin":
        return datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()
    elif platform == "win32":
        return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def is_aa_sim(file):
    """From a traj_nojump.xtc, decides, whether sim is an aa sim."""
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


def get_ubq_site_and_basename(traj_file):
    """Returns ubq_site and basename for a traj_file."""
    basename = traj_file.split('/')[-2]
    substrings = basename.split('_')
    try:
        ubq_site = substrings[[substr.startswith('k') for substr in substrings].index(True)]
    except ValueError:
        print(basename)
        raise
    return basename, ubq_site


def normalize_sPRE(df_comp, df_obs, kind='var', norm_res_count=10):
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

    residues = get_column_names_from_pdb(return_residues=True)

    for ubq_site in pd.value_counts(df_comp['ubq_site']).index:
        if ubq_site not in df_obs.keys():
            missing.append(ubq_site)
            continue
        print(ubq_site)
        # put the needed values into their own df
        sPRE_ind = df_comp.columns[['sPRE' in c for c in df_comp.columns]]
        sPRE_comp = df_comp[df_comp['ubq_site'] == ubq_site][sPRE_ind]
        print('Shape of computed data:', sPRE_comp.shape)

        # get measured data
        v_obs = df_obs.loc[sPRE_ind][ubq_site].values
        print('Shape of observed data:', v_obs.shape)

        sPRE_calc_norm = []

        # get the mean values along the columns
        # get the threshold of the <norm_res_count> lowest values
        if kind == 'var':
            v_calc = sPRE_comp.var(axis='rows').values
            v_calc_prox, v_calc_dist = np.split(v_calc, 2)
            threshold_prox = np.partition(v_calc_prox[np.nonzero(v_calc_prox)], norm_res_count)[norm_res_count - 1]
            threshold_dist = np.partition(v_calc_dist[np.nonzero(v_calc_dist)], norm_res_count)[norm_res_count - 1]
            print(f"Proximal threshold = {threshold_prox}, Distal threshold = {threshold_dist}")
        elif kind == 'mean':
            v_calc = sPRE_comp.mean(axis='rows').values
            v_calc_prox, v_calc_dist = np.split(v_calc, 2)
            threshold_prox = np.partition(v_calc_prox[np.nonzero(v_calc_prox)], norm_res_count)[norm_res_count - 1]
            threshold_dist = np.partition(v_calc_dist[np.nonzero(v_calc_dist)], norm_res_count)[norm_res_count - 1]
            print(f"Proximal threshold = {threshold_prox}, Distal threshold = {threshold_dist}")
        else:
            raise Exception("`kind` must be either 'var' or 'mean'.")

        # get the columns that fulfill this condition
        min_values = (v_calc_prox <= threshold_prox) & (v_calc_prox != 0.0)
        centers_prox = np.where(min_values)[0]
        min_values = (v_calc_dist <= threshold_dist) & (v_calc_dist != 0.0)
        centers_dist = np.where(min_values)[0] + 76

        print(f"Considered residues are Prox: {residues[centers_prox]} and Dist: {residues[centers_dist - 76]}")

        # test + 76
        a = v_calc_dist[min_values][0]
        b = v_calc[centers_dist][0]
        assert a == b

        # get the factors
        v_calc = sPRE_comp.mean(axis='rows').values
        factors_prox = v_obs[centers_prox] / v_calc[centers_prox]
        factors_dist = v_obs[centers_dist] / v_calc[centers_dist]
        f_prox = np.mean(factors_prox)
        f_dist = np.mean(factors_dist)
        print(f"Proximal factor = {f_prox}, Distal factor = {f_dist}")

        # copy the existing values and multiply
        new_values = copy.deepcopy(sPRE_comp.values)
        new_values[:, :76] *= f_prox
        new_values[:, 76:] *= f_dist

        sPRE_norm = pd.DataFrame(new_values, columns=sPRE_comp.columns)
        sPRE_norm.columns = [f'normalized {c}' for c in sPRE_norm.columns]
        # sPRE_norm.to_csv(f'sPRE_{ubq_site}_normalized_via_10_min_variance.csv')

        # append to new frame
        out.append(sPRE_norm)
        print('\n')
    out = pd.concat(out)
    df_comp_w_norm = df_comp.copy()
    df_comp_w_norm = df_comp_w_norm[~ df_comp_w_norm['ubq_site'].str.contains('|'.join(missing))]
    df_comp_w_norm[sPRE_norm.columns] = out.values

    return df_comp_w_norm, centers_prox, centers_dist


def call_xplor_with_yaml(pdb_file, yaml_file='', from_tmp=False, testing=False, **kwargs):
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
        **kwargs: Arbitrary keyword arguments. Keywords that are not flags
            of the xplor/scripts/xplor_single_struct_script.py will be discarded.

    Returns:
        str: The string which the xplor/scripts/xplor_single_struct_script.py
            prints to stdout

    """
    # load defaults
    import yaml, subprocess
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
           defaults[pot]['call_parameters']['restraints']['value'] = filename

    # make arguments out of them
    arguments = write_argparse_lines_from_yaml_or_dict(defaults)

    if from_tmp:
        executable = '/home/kevin/software/xplor-nih/xplor-nih-3.2/bin/pyXplor /tmp/pycharm_project_13/xplor/scripts/xplor_single_struct.py'
    else:
        executable = '/home/kevin/software/xplor-nih/xplor-nih-3.2/bin/pyXplor ' + get_local_or_proj_file('scripts/xplor_single_struct.py')
    cmd = f'{executable} -pdb {pdb_file} {arguments}'
    if testing:
        cmd += ' -testing'
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    return_code = process.poll()
    out = out.decode(sys.stdin.encoding)
    err = err.decode(sys.stdin.encoding)
    if return_code > 0:
        print(out)
        raise Exception(f"Call to subprocess did not succeed. Here's the error: {err}, and the return code: {return_code}")
    out = out.split('findImportantAtoms: done')[1]
    return out


def parallel_xplor(ubq_sites, simdir='/home/andrejb/Research/SIMS/2017_*', n_threads='max-2',
                   df_outdir='/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package/',
                   suffix = '_df_no_conect.csv', write_csv=True,
                   subsample=5, yaml_file='', testing=False, from_tmp=False, max_len=-1, **kwargs):
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
        **kwargs: Arbitrary keyword arguments. Keywords that are not flags
            of the xplor/scripts/xplor_single_struct_script.py will be discarded.

    Returns:
        pd.Dataframe: A pandas Dataframe.

    """
    if n_threads == 'max-2':
        n_threads = multiprocessing.cpu_count() - 2
    elif n_threads == 'max':
        n_threads = multiprocessing.cpu_count()

    # get list of already existing dfs
    files = glob.glob(f'{df_outdir}*{suffix}')
    if files and write_csv:
        highest_datetime_csv = sorted(files, key=get_iso_time)[-1]
        df = pd.read_csv(highest_datetime_csv, index_col=0)
    else:
        df = pd.DataFrame({})

    # run loop
    for i, ubq_site in enumerate(ubq_sites):
        for j, dir_ in enumerate(glob.glob(f"{simdir}{ubq_site}_*")):
            traj_file = dir_ + '/traj_nojump.xtc'
            if not is_aa_sim(traj_file):
                print(f"{traj_file} is not an AA sim")
                continue
            basename = traj_file.split('/')[-2]
            top_file = dir_ + '/start.pdb'
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

            isopeptide_indices = []
            for r in traj.top.residues:
                if r.name == 'GLQ':
                    r.name = 'GLY'
                    for a in r.atoms:
                        if a.name == 'C':
                            isopeptide_indices.append(a.index + 1)
                if r.name == 'LYQ':
                    r.name = 'LYS'
                    for a in r.atoms:
                        if a.name == 'CQ': a.name = 'CE'
                        if a.name == 'NQ':
                            a.name = 'NZ'
                            isopeptide_indices.append(a.index + 1)
                        if a.name == 'HQ': a.name = 'HZ1'

            # parallel call
            out = Parallel(n_jobs=n_threads, prefer='threads')(delayed(get_series_from_mdtraj)(frame,
                                                                                               traj_file,
                                                                                               top_file,
                                                                                               frame_no,
                                                                                               testing=testing,
                                                                                               from_tmp=from_tmp,
                                                                                               yaml_file=yaml_file,
                                                                                               **kwargs) for
                                                               frame, frame_no in zip(traj[:max_len:subsample],
                                                                                      np.arange(traj.n_frames)[
                                                                                      :max_len:subsample]))

            # continue working with output
            now = datetime_windows_and_linux_compatible()
            df_name = os.path.join(df_outdir, f"{now}{suffix}")
            df = df.append(out, ignore_index=True)
            if write_csv:
                df.to_csv(df_name)
            if testing:
                break
    return df


def get_series_mdanalysis(frame, traj_file, top_file, frame_no, testing=False, from_tmp=False, yaml_file='', **kwargs):
    """Similar to `get_series_from_mdtraj`, but using MDAnalysis and saving a psf file.
    
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
        **kwargs: Arbitrary keyword arguments. Keywords that are not flags
            of the xplor/scripts/xplor_single_struct_script.py will be discarded.
    
    Returns:
        pd.Series: A pandas series instance
    
    """
    pass


def get_series_from_mdtraj(frame, traj_file, top_file, frame_no, testing=False, from_tmp=False, yaml_file='', **kwargs):
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
        **kwargs: Arbitrary keyword arguments. Keywords that are not flags
            of the xplor/scripts/xplor_single_struct_script.py will be discarded.

    Returns:
        pd.Series: A pandas Series instance.

        """
    # function specific imports
    import ast

    # define a pre-formatted pandas series
    data = {'traj_file': traj_file, 'top_file': top_file, 'frame': frame_no, 'time': frame.time[0]}
    column_names = get_column_names_from_pdb()
    columns = {column_name: np.nan for column_name in column_names}
    data.update(columns)
    series = pd.Series(data=data)

    should_be_residue_number = frame.n_residues
    
    # get data
    basename = os.path.basename(traj_file).split('.')[0]
    pdb_file = os.path.join(os.getcwd(), f'tmp_{basename}_frame_{frame_no}_hash_{abs(hash(frame))}.pdb')
    frame.save_pdb(pdb_file)

    try:
        out = call_xplor_with_yaml(pdb_file, from_tmp=from_tmp, testing=testing, yaml_file=yaml_file, **kwargs)
        out = ast.literal_eval(out)
    finally:
        os.remove(pdb_file)

    if series['proximal ILE3 sPRE'] == 0 and series['proximal PHE4 sPRE'] == 0:
        print(cmd)
        print(out)
        print(psol)
        raise Exception(f"This psol value should not be 0. Traj is {traj_file}, frame is {frame_no}")

    for o in out:
        if int(o[1]) <= (should_be_residue_number / 2) - 1:
            resSeq = int(o[1])
            position = 'proximal'
        else:
            resSeq = int(int(o[1]) - (should_be_residue_number / 2))
            position = 'distal'
        try:
            resname = frame.top.residue(int(o[1]) - 1).name
        except TypeError:
            print(o)
            raise
        if o[0] == 'rrp600':
            series[f'{position} {resname}{resSeq} 15N_relax_600'] = o[2]
        elif o[0] == 'rrp800':
            series[f'{position} {resname}{resSeq} 15N_relax_800'] = o[2]
        elif o[0] == 'psol':
            series[f'{position} {resname}{resSeq} sPRE'] = o[2]

    basename, ubq_site = get_ubq_site_and_basename(traj_file)
    series['basename'] = basename
    series['ubq_site'] = ubq_site
            
    return series