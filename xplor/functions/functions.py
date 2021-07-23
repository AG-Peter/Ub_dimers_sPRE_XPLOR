import mdtraj as md
import numpy as np
import pandas as pd
import os, sys, errno, string, glob, itertools, multiprocessing, dateutil, datetime
from ..proteins.proteins import get_column_names_from_pdb
from joblib import Parallel, delayed

YAML_FILE = os.path.join(os.path.dirname(sys.modules['xplor'].__file__), "data/defaults.yml")

def datetime_windows_and_linux_compatible():
    import datetime
    from sys import platform
    if platform == "linux" or platform == "linux2" or platform == "darwin":
        return datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()
    elif platform == "win32":
        return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

def get_local_or_proj_file(files):
    """Gets a file either from local storage or from project files.

    If a file is a local file, this function will simply return the input.
    If a file is a project file, you can specify it with data/filename*.

    Args:
        files (str): The filename. Can contain a wildcard *.

    Returns:
        str: The full filepath

    Raises:
        FileNotFoundError: When the provided `file` is neither a local nor a project file.

    """
    glob_files = glob.glob(files)
    if not glob_files:
        e = FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), files)
        proj_files = os.path.join(os.path.dirname(sys.modules['xplor'].__file__), files)
        datafiles = glob.glob(proj_files)
        if not datafiles:
            print(proj_files)
            raise e
        else:
            glob_files = datafiles
    if len(glob_files) == 1:
        return glob_files[0]
    else:
        return glob_files


def write_argparse_lines_from_yaml_or_dict(input='', print_argparse=False):
    """Uses the values from a yaml file to either build argparse lines, or build a string
    that sets these argparse lines.

    Keyword Args:
        yaml_file (Union[str, dict], optional): The path to the yaml file to parse.
            If empty string ('') is provided, the module's default yaml at xplor/data/defaults.yml will
            be used. Can also be fed with a dict. Defaults to ''.
        print_argparse (bool, optional): Whether to print argparse lines from
            that yaml. Defaults to False.

    Returns:
        str: A string, that can be provided to the XPLOR scirpt.

    """
    import yaml, subprocess

    flags = {}

    if not input:
        input = YAML_FILE
    if isinstance(input, str):
        with open(input, 'r') as stream:
            input = yaml.safe_load(stream)
    input_dict = input

    for pot in input_dict.keys():
        for arg_type in input_dict[pot].keys():
            for param, data in input_dict[pot][arg_type].items():
                flag = f'-{pot}_{arg_type}_{param}'
                required = 'False'
                type_ = f"{data['type']}"
                default = f"{data['value']}"
                help_ = f"{data['descr']}"
                if type_ == 'file':
                    flags[flag] = default
                    line = f'parser.add_argument("{flag}", required={required}, type=str, help="""{help_}""")'
                elif type_ == 'str':
                    flags[flag] = f'"{default}"'
                    line = f'parser.add_argument("{flag}", required={required}, type={type_}, default="{default}", help="""{help_}""")'
                elif type_ == 'bool':
                    flags[flag] = default
                    line = f'parser.add_argument("{flag}", required={required}, type=str2bool, default="{default}", help="""{help_}""")'
                else:
                    flags[flag] = default
                    line = f'parser.add_argument("{flag}", required={required}, type={type_}, default={default}, help="""{help_}""")'
                if print_argparse: print(line)
    return ' '.join([f'{key} {val}' for key, val in flags.items()])


def make_15_N_table(in_file, out_file=None, return_df=True, split_prox_dist=False):
    """Creates a pandas dataframe from a 15N relax file provided by Tobias Schneider.

    The input is expected to be a .txt file akin to this layout::
        %<residue number><chain id><P atom name><Q atom name><freq of spectrometer (MHz)><R1 rate (1/s)><R1 rate error (1/s)><R2 rate (1/s)><R2 rate error (1/s)><NOE><NOE error>
        2 A N H 600.00 1.401544 0.009619 12.445858 0.285990 0.779673 0.025396
        3 A N H 600.00 1.439783 0.016894 11.625145 0.153587 0.790700 0.032749
        4 A N H 600.00 1.442767 0.014602 12.005927 0.442207 0.759039 0.033966
        5 A N H 600.00 1.345446 0.011759 11.814099 0.120512 0.742896 0.028782

    This gets parsed and put into a dataframe with the option out_file, the dataframe
    is put into a .tbl file, which can be read by XPLOR.

    Args:
        in_file (str): The input file. Can be a project data resource.

    Keyword Args:
        out_file (Union[str, None], optional): Where to put the tbl file.
            If None is provided, the .tbl file will not be written to disk.
            Defaults to None.
        return_df (bool, optional): Whether to return a pandas dataframe.
            Defaults to True.
        split_prox_dist (bool, optional): Whether to split prox and dist into their own
            .tbl files or combine them.

    Returns:
        Union[None, pd.Dataframe]: Either None or the pandas dataframe.

    """
    in_file = get_local_or_proj_file(in_file)
    with open(in_file, 'r') as f:
        lines = f.read().splitlines()
    lines = list(filter(lambda x: False if x == '' else True, lines))

    names = lines[0].lstrip('%<').rstrip('>').split('><') + ['is in secondary']
    df = {n: [] for n in names}

    for line in lines[1:]:
        for i, data in enumerate(line.split()):
            if '*' in data and i == 0:
                df['is in secondary'].append(False)
                data = data.lstrip('*')
            elif '*' not in data and i == 0:
                df['is in secondary'].append(True)
            try:
                df[names[i]].append(float(data))
            except ValueError:
                df[names[i]].append(data)
    df['position'] = list(map(lambda x: 'proximal' if x == 'A' else 'distal', df['chain id']))
    df = pd.DataFrame(df)
    
    mhzs = df['freq of spectrometer (MHz)'].unique()
    for mhz in mhzs:
        rho = []
        rho2 = []
        for i, resnum in enumerate(df['residue number']):
            if df['freq of spectrometer (MHz)'][i] != mhz:
                continue
            if not df['is in secondary'][i]:
                continue
            if df['chain id'][i] == 'B' and not split_prox_dist:
                resnum += 76
            R1 = df['R1 rate (1/s)'][i]
            R1_err = df['R1 rate error (1/s)'][i]
            R2 = df['R2 rate (1/s)'][i]
            R2_err = df['R1 rate error (1/s)'][i]
            NOE = df['NOE'][i]
            NOE_err = df['NOE error'][i]
            if not split_prox_dist:
                rho.append([int(resnum), R1, R1_err, R2, R2_err, NOE, NOE_err])
            else:
                if df['chain id'][i] == 'A':
                    rho.append([int(resnum), R1, R1_err, R2, R2_err, NOE, NOE_err])
                else:
                    rho2.append([int(resnum), R1, R1_err, R2, R2_err, NOE, NOE_err])
        if out_file is not None:
            if not '*' in out_file and split_prox_dist:
                raise Exception("Method will write two files. Please provide a filename with wildcard (*).")
            filename = out_file.replace('*', str(int(mhz)))
            if not split_prox_dist:
                with open(filename, 'w+') as f:
                    for r in rho:
                        f.write(' '.join(map(str, r)) + '\n')
                print(filename, 'written')
            else:
                fnames = [filename.replace('.tbl', '_proximal.tbl'), filename.replace('.tbl', '_distal.tbl')]
                for fname, r in zip(fnames, [rho, rho2]):
                    with open(fname, 'w+') as f:
                        for _ in r:
                            f.write(' '.join(map(str, _)) + '\n')
                print(fnames, 'written')
        else:
            out = []
            for r in rho + rho2:
                out.append(' '.join(map(str, r)) + '\n')
    if return_df:
        df = df.rename(columns={'residue number': 'resSeq'})
        df['resSeq'] = df['resSeq'].astype(int)
        max_ = df[df['chain id'] == 'A']['resSeq'].max()
        df['resSeq'] = df.apply(lambda row: row.resSeq + max_ if row['chain id'] == 'B' else row.resSeq, axis=1)
        df = df.sort_values(by=['freq of spectrometer (MHz)', 'chain id', 'resSeq'], axis='rows')
        return(df)


def label(resSeq, sPRE, err=0.01):
    """Label for sPRE.tbl files."""
    return f"assign (resid {resSeq:<2} and name HN)	{sPRE:5.3f}	{err:5.3f}"


def getResSeq(lines):
    """Returns Resseqs for lines in sPRE .txt files provided by Tobias Schneider."""
    return list(map(lambda x: int(x.split('\t')[0][3:]), filter(lambda x: False if (x == '' or 'mM' in x or 'primary' in x) else True, lines)))


def make_sPRE_table(in_files, out_file=None, return_df=True, split_prox_dist=False, omit_erros=True):
    """Creates a pandas dataframe from a sPRE results file provided by Tobias Schneider.

    The input is expected to be a .txt file akin to this layout::
        primary sequence	sPRE	err
            mM\+(-1)s\+(-1)
        Gln2	3.67659401129085	3.49776478934889
        Ile3	2.03877683593419	1.24739785462518
        Phe4	--	--
        Val5	2.00930109656374	2.94263017846034
        Lys6	2.75217278575404	1.10804033122158
        Thr7	6.0981182022504		0.785394525309821
        Leu8	2.94922624345896	1.53018589439035
        Thr9	8.38046508538605	0.855337600823175

    This gets parsed and put into a dataframe with the option out_file, the dataframe
    is put into a .tbl file, which can be read by XPLOR.

    Args:
        in_file (str): The input file. Can be a project data resource.

    Keyword Args:
        out_file (Union[str, None], optional): Where to put the tbl file.
            If None is provided, the .tbl file will not be written to disk.
            Defaults to None.
        return_df (bool, optional): Whether to return a pandas dataframe.
            Defaults to True.
        split_prox_dist (bool, optional): Whether to split prox and dist into their own
            .tbl files or combine them.

    Returns:
        Union[None, pd.Dataframe]: Either None or the pandas dataframe.

    """
    residues_with_errors = [78, 79]
    in_files = get_local_or_proj_file(in_files)
    files = sorted(in_files, key=lambda x: 1 if ('proximal' in x or 'prox' in x) else 2)
    assert len(files) == 2, print(f"I need a proximal and a distal file. I found {files}")
    proximal, distal = files
    
    file = proximal
    new_lines = []
    with open(file, 'r') as f:
        lines = f.read().splitlines()
    for i, l in enumerate(lines):
        if l == '' or i == 1:
            continue
        if i == 0:
            main_labels = l.split('\t')
            df = {k: [] for k in main_labels}
            continue
        for k, v in zip(main_labels, l.split('\t')):
            df[k].append(v)
    df['resSeq'] = getResSeq(lines)
    df['position'] = np.full(len(df['resSeq']), 'proximal').tolist()

    file = distal
    with open(file, 'r') as f:
        lines = f.read().splitlines()
    for i, l in enumerate(lines):
        if l == '' or i == 0 or i == 1:
            continue
        for k, v in zip(main_labels, l.split('\t')):
            df[k].append(v)
    max_ = max(df['resSeq'])
    resSeq = list(map(lambda x: x + max_, getResSeq(lines)))
    df['resSeq'].extend(resSeq)
    df['position'].extend(np.full(len(resSeq), 'distal').tolist())
    df = pd.DataFrame(df)
    
    df['sPRE'] = pd.to_numeric(df['sPRE'], errors='coerce')
    df['err'] = pd.to_numeric(df['err'], errors='coerce')
    
    if out_file is not None:
        if not split_prox_dist:
            with open(out_file, 'w') as f:
                for i, row in df.iterrows():
                    if any(pd.isna(row)):
                        continue
                    new_line = label(row['resSeq'], row['sPRE'], row['err'])
                    if omit_erros and row['resSeq'] in residues_with_errors:
                        continue
                    f.write(new_line + '\n')
            print(out_file, 'written')
        else:
            fnames = [out_file.replace('.tbl', '_proximal.tbl'), out_file.replace('.tbl', '_distal.tbl')]
            for i, fname in enumerate(fnames):
                if i == 0:
                    sub_df = df[df['position'] == 'proximal']
                    print(sub_df)
                else:
                    sub_df = df[df['position'] == 'distal']
                with open(fname, 'w') as f:
                    for j, row in sub_df.iterrows():
                        if any(pd.isna(row)):
                            continue
                        if i == 0:
                            new_line = label(row['resSeq'], row['sPRE'], row['err'])
                        else:
                            new_line = label(row['resSeq'] - max_, row['sPRE'], row['err'])
                        f.write(new_line + '\n')
            print(fnames, 'written')
    if return_df:
        return df


def parse_atom_line(line):
    data = {}
    data['record'], data['serial_no'], data['name'] = line[:4], int(line[6:11]), line[12:16]
    data['alternate_loc_ind'], data['residue'], data['chain'] = line[16], line[17:20], line[21]
    data['res_seq'], data['code_for_insertions'] = int(line[22:26]), line[26]
    data['x'], data['y'], data['z'] = float(line[30:38]), float(line[38:46]), float(line[46:54])
    data['occ'], data['temp'], data['elem'] = float(line[54:60]), float(line[60:66]), line[76:78]
    
    for key, value in data.items():
        if isinstance(value, str):
            data[key] = value.strip()
    return data


def split_and_order_chains(datas):
    no_chains = len(list(filter(lambda x: True if x == 'OXT' else False, (x.get('name') for x in datas))))
    no_atoms = len(list(filter(lambda x: True if x == 'ATOM' else False, (x.get('record') for x in datas))))
    i = 0
    chains = [[] for x in range(no_chains)]
    for data in datas:
        if data['record'] == 'TER':
            chains[i - 1].append(data)
            continue
        chains[i].append(data)
        if data['name'] == 'OXT':
            i += 1
    order = []
    for i, chain in enumerate(chains):
        if any([x.get('residue') == 'GLQ' for x in chain]):
            order.insert(0, i)
        else:
            order.append(i)
    chains = [chains[i] for i in order]
    for chain, new_chain in zip(chains, string.ascii_uppercase):
        for value in chain:
            value['chain'] = new_chain
    return chains


def add_ter(chains):
    for i, chain in enumerate(chains):
        for j, element in enumerate(chain):
            if element['record'] == 'ATOM':
                if element['residue'] == 'GLQ' and element['name'] == 'OXT':
                    data = {}
                    data['record'], data['serial_no'] = 'TER', element['serial_no']
                    data['residue'], data['chain'] = element['residue'], element['chain']
                    data['res_seq'], data['code_for_insertions'] = element['res_seq'], element['code_for_insertions']
                    chains[i][j] = data
                if element['residue'] == 'GLY' and j == len(chain) - 1:
                    data = {}
                    data['record'], data['serial_no'] = 'TER', 0
                    data['residue'], data['chain'] = element['residue'], element['chain']
                    data['res_seq'], data['code_for_insertions'] = element['res_seq'], element['code_for_insertions']
                    chains[i].append(data)
    return chains


def build_pdb_line(data, atom_nr, res_number):
    if data['record'] == 'ATOM':
        line = f"{data['record']}  {atom_nr:>5}  {data['name']:<4}{data['alternate_loc_ind']}{data['residue']}"
        line += f" {data['chain']}{res_number:>4}{data['code_for_insertions']}    {data['x']:8.3f}{data['y']:8.3f}{data['z']:8.3f}"
        line += f"{data['occ']:6.2f}{data['temp']:6.2f}          {data['elem']:>2}"
    else:
        line = f"{data['record']}   {atom_nr:>5}      {data['residue']}"
        line += f" {data['chain']}{res_number:>4}{data['code_for_insertions']}"
    return line


def parse_ter_line(line):
    data = {}
    data['record'], data['serial_no'] = line[:4], int(line[6:11])
    data['residue'], data['chain'] = line[17:20], line[21]
    data['res_seq'], data['code_for_insertions'] = int(line[22:26]), line[26]
    
    for key, value in data.items():
        if isinstance(value, str):
            data[key] = value.strip()

    return data


def create_connect(data):
    line = "CONECT"
    for d in data:
        line += f" {d:>4}"
    return line + '\n'


def parse_pdb_line(line):
    if 'ATOM' in line:
        return parse_atom_line(line)
    elif 'TER' in line:
        return parse_ter_line(line)
    else:
        raise Exception(f"Unknown Record Type in line {line}")


def prepare_pdb_for_gmx(file, verification=None):
    """Function from expansion elephant. Might be deprecated."""
    if verification:
        with open(verification, 'r') as f:
            verification_lines = f.readlines()
    with open(file, 'r') as f:
        lines = f.readlines()
        
    # define
    leading = []
    datas = []
    old_lines = []
    connects = []
    
    # parse
    for i, line in enumerate(lines):
        if 'REMARK' in line or 'CRYST' in line or 'MODEL' in line:
            leading.append(line)
        elif 'ATOM' in line or 'TER' in line:
            old_lines.append(line)
            data = parse_pdb_line(line)
            datas.append(data)
        elif 'ENDMDL' in line or 'END' in line:
            pass
        elif 'CONECT' in line:
            connects.append(line)
        else:
            print(i, line)
            raise Exception(f"Unkonwn Record Type in line {line}")
         
    # rearrange chains
    chains = split_and_order_chains(datas)
    chains = add_ter(chains)
    data = list(itertools.chain(*chains))
    
    # split some connects
    connects = list(map(lambda x: [int(y) for y in x.split()[1:]], connects))
    replacements = {}
    
    # build new pdb
    residue = 1
    for i, d in enumerate(data):
        # print(old_lines[i])
        if i > 0:
            if d['residue'] != data[i -1]['residue']:
                residue += 1
        i += 1
        line = build_pdb_line(d, i, residue) +'\n'
        leading.append(line)
        
        if any(d['serial_no'] in c for c in connects):
            replacements[d['serial_no']] = i
        
        if verification:
            if line[:22] != verification_lines[i + 3][:22]:
                print(line, verification_lines[i + 3])
                raise Exception('STOP')
        
    leading.append('ENDMDL\n')
    # fix connects
    new_connects = []
    for co in connects:
        new = []
        for c in co:
            new.append(replacements[c])
        new_connects.append(new)
            
    for connect in new_connects:
        leading.append(create_connect(connect))
    leading.append('END')
    
    with open(file, 'w') as f:
        for line in leading:
            f.write(line)


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


def get_iso_time(in_str):
    """Returns the datetime of a file that starts with an iso time.

    For example this one:
        '/path/to/file/2021-07-23T04:10:27+02:00_df_no_conect.csv'

    Args:
        in_str (str): The filename

    Returns:
        datetime.time: A datetime timestamp.

    """
    time = dateutil.parser.isoparse(os.path.split(in_str)[-1].split('_')[0])
    return time


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
                   suffix = '_df_no_conect.csv',
                   subsample=5, yaml_file='', testing=False, from_tmp=False, max_len=-1, **kwargs):
    """Runs xplor on many simulations in parallel.

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

    # store output here
    list_of_pandas_series = []

    # get list of already existing dfs
    files = glob.glob(f'{df_outdir}*{suffix}')
    highest_datetime_csv = sorted(files, key=get_iso_time)[-1]
    check_df = pd.read_csv(highest_datetime_csv, index_col=0)

    raise Exception("STOP")

    # run loop
    for i, ubq_site in enumerate(ubq_sites):
        for j, dir_ in enumerate(glob.glob(f"{simdir}{ubq_site}_*")):
            traj_file = dir_ + '/traj_nojump.xtc'
            basename = traj_file.split('/')[-2]
            top_file = dir_ + '/start.pdb'
            traj = md.load(traj_file, top=top_file)

            # check if traj is complete
            try:
                value_counts = pd.value_counts(check_df['traj_file'])
                frames = value_counts[traj_file]
            except KeyError:
                frames = 0
            if frames == traj[::subsample].n_frames:
                print(f"traj {basename} already finished")
                continue
            else:
                print(f"traj {basename} NOT FINISHED")

            for r in traj.top.residues:
                if r.index > 75 and r.resSeq < 76:
                    r.resSeq += 76
                if r.name == 'GLQ':
                    r.name = 'GLY'
                if r.name == 'LYQ':
                    r.name = 'LYS'
                    for a in r.atoms:
                        if a.name == 'CQ': a.name = 'CE'
                        if a.name == 'NQ': a.name = 'NZ'
                        if a.name == 'HQ': a.name = 'HZ1'



            out = Parallel(n_jobs=n_threads, prefer='threads')(delayed(get_prox_dist_from_mdtraj)(frame,
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
            list_of_pandas_series.extend(out)
            now = datetime_windows_and_linux_compatible()
            df_name = os.path.join(df_outdir, f"{now}{suffix}")
            df = pd.concat(list_of_pandas_series, axis=1).T
            df.to_csv(df_name)
    return df


def get_prox_dist_from_mdtraj(frame, traj_file, top_file, frame_no, testing=False, from_tmp=False, yaml_file='', **kwargs):
    """Saves a temporary pdb file which will then be passed to call_xplor_with_yaml

    Arguments for XPLOR can be provided by either specifying a yaml file, or
    providing them as keyword_args to this function.

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

    basename, ubq_site = get_ubq_site_and_basename(traj_file)

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

    series['basename'] = basename
    series['ubq_site'] = ubq_site
            
    return series