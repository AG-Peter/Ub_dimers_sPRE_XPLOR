import mdtraj as md
import numpy as np
import pandas as pd
import os, sys, errno, string, glob, itertools
from ..proteins.proteins import get_column_names_from_pdb

YAML_FILE = os.path.join(os.path.dirname(sys.modules['xplor'].__file__), "data/defaults.yml")


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
        datafiles = glob.glob(os.path.join(os.path.dirname(sys.modules['xplor'].__file__), files))
        if not datafiles:
            raise e
        else:
            glob_files = datafiles
    if len(glob_files) == 1:
        return glob_files[0]
    else:
        return glob_files


def write_argparse_lines_from_yaml(yaml_file=''):
    import yaml, subprocess


    if not yaml_file:
        yaml_file = YAML_FILE
    with open(yaml_file, 'r') as stream:
        input_dict = yaml.safe_load(stream)

    for pot in input_dict.keys():
        for arg_type in input_dict[pot].keys():
            for param, data in input_dict[pot][arg_type].items():
                flag = f'-{pot}_{arg_type}_{param}'
                required = 'False'
                type_ = f"{data['type']}"
                default = f"{data['value']}"
                help_ = f"{data['descr']}"
                if type_ == 'file':
                    line = f'parser.add_argument("{flag}", required={required}, type=str, help="""{help_}""")'
                elif type_ == 'str':
                    line = f'parser.add_argument("{flag}", required={required}, type={type_}, default="{default}", help="""{help_}""")'
                else:
                    line = f'parser.add_argument("{flag}", required={required}, type={type_}, default={default}, help="""{help_}""")'
                print(line)


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
    return f"assign (resid {resSeq:<2} and name HN)	{sPRE:5.3f}	{err:5.3f}"


def getResSeq(lines):
    return list(map(lambda x: int(x.split('\t')[0][3:]), filter(lambda x: False if (x == '' or 'mM' in x or 'primary' in x) else True, lines)))


def make_sPRE_table(in_files, out_file=None, return_df=True, split_prox_dist=False):
    files = glob.glob(in_files)
    files = sorted(files, key=lambda x: 1 if ('proximal' in x or 'prox' in x) else 2)
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


def call_xplor_with_yaml(pdb_file, yaml_file='', from_tmp=False, **kwargs):
    # load defaults
    import yaml, subprocess
    if not yaml_file:
        yaml_file = YAML_FILE
    with open(yaml_file, 'r') as stream:
        defaults = yaml.safe_load(stream)

    pdb_file = get_local_or_proj_file(pdb_file)

    # overwrite tbl files, if present
    defaults.update(kwargs)

    # get the datafiles
    for pot in ['psol', 'rrp600', 'rrp800']:
        if pot in defaults:
           filename = get_local_or_proj_file(defaults[pot]['call_parameters']['restraints']['value'])
           defaults[pot]['call_parameters']['restraints']['value'] = filename

    if from_tmp:
        executable = '/home/kevin/software/xplor-nih/xplor-nih-3.2/bin/pyXplor /tmp/pycharm_project_13/xplor/scripts/xplor_single_struct.py'
    else:
        executable = '.' + get_local_or_proj_file('scripts/xplor_single_struct.py')
    cmd = f'{executable} -pdb lol -testing'
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=False)
    out, err = process.communicate()
    return_code = process.poll()
    out = out.decode(sys.stdin.encoding)
    err = err.decode(sys.stdin.encoding)
    if return_code > 0:
        raise Exception(f"Call to subprocess did not succeed. Here's the error: {err}, and the return code: {return_code}")
    # out = out.split('findImportantAtoms: done')[1]
    # out = ast.literal_eval(out)

def parallel_xplor(trajs, yaml_file='', from_tmp=False, **kwargs):
    pass

    
def get_prox_dist_from_mdtraj(frame, traj_file, top_file, frame_no, testing=False,
                             pdb_file='values_from_every_frame/tmp_full_frame_lyq_and_glq_removed_fixed_algo.pdb'):
    # function specific imports
    import subprocess, sys, ast

    # define a pre-formatted pandas series
    data = {'traj_file': traj_file, 'top_file': top_file, 'frame': frame_no, 'time': frame.time[0]}
    column_names = get_column_names_from_pdb()
    columns = {column_name: np.nan for column_name in column_names}
    data.update(columns)
    series = pd.Series(data=data)
    
    # [y for x in non_flat for y in x]
    substrings = [item for part in traj_file.split('/') for item in part.split('_')]
    ubq_site = substrings[[substr.startswith('k') for substr in substrings].index(True)]
    
    should_be_residue_number = frame.n_residues
    
    # get data
    sPRE_tbl = f'values_from_every_frame/diUbi_{ubq_site}_empty_sPRE_prox_in.tbl'
    relax_600_tbl = f'values_from_every_frame/diUbi_empty_600_mhz_relaxratiopot_prox_in.tbl'
    relax_800_tbl = f'values_from_every_frame/diUbi_empty_800_mhz_relaxratiopot_prox_in.tbl'
    
    cmd = f"xplor_single_struct.py -pdb {pdb_file} -spre_tbl {sPRE_tbl} -relax_600_tbl {relax_600_tbl} -relax_800_tbl {relax_800_tbl}"
    if testing:
        cmd += ' -testing'
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    return_code = process.poll()
    out = out.decode(sys.stdin.encoding)
    err = err.decode(sys.stdin.encoding)
    out = out.split('findImportantAtoms: done')[1]
    out = ast.literal_eval(out)

    for o in out:
        resname = frame.top.residue(int(o[1]) - 1)
        if o[0] == 'rrp600':
            series[f'proximal {resname} 15N_relax_600'] = o[2]
        elif o[0] == 'rrp800':
            series[f'proximal {resname} 15N_relax_800'] = o[2]
        elif o[0] == 'psol':
            series[f'proximal {resname} sPRE'] = o[2]

    if series['proximal ILE3 sPRE'] == 0 and series['proximal PHE4 sPRE'] == 0:
        print(cmd)
        print(out)
        print(psol)
        raise Exception(f"This psol value should not be 0. Traj is {traj_file}, frame is {frame_no}")

    sPRE_tbl = sPRE_tbl.replace('prox', 'dist')
    relax_600_tbl = relax_600_tbl.replace('prox', 'dist')
    relax_800_tbl = relax_800_tbl.replace('prox', 'dist')

    cmd = f"xplor_single_struct.py -pdb {pdb_file} -spre_tbl {sPRE_tbl} -relax_600_tbl {relax_600_tbl} -relax_800_tbl {relax_800_tbl}"
    if testing:
        cmd += ' -testing'
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    return_code = process.poll()
    out = out.decode(sys.stdin.encoding)
    err = err.decode(sys.stdin.encoding)
    out = out.split('findImportantAtoms: done')[1]
    out = ast.literal_eval(out)

    for o in out:
        try:
            resname = frame.top.residue(int(o[1]) - 1 - should_be_residue_number)
        except TypeError:
            print(o)
            raise
        if o[0] == 'rrp600':
            series[f'distal {resname} 15N_relax_600'] = o[2]
        elif o[0] == 'rrp800':
            series[f'distal {resname} 15N_relax_800'] = o[2]
        elif o[0] == 'psol':
            series[f'distal {resname} sPRE'] = o[2]
            
    return series