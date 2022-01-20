################################################################################
# Imports
################################################################################


import mdtraj as md
import numpy as np
import pandas as pd
import itertools
from ...misc import get_local_or_proj_file
from ...proteins import get_column_names_from_pdb
from ..functions import RAMFile


################################################################################
# Globals
################################################################################


__all__ = ['make_15_N_table', 'make_sPRE_table', 'get_observed_df',
           'get_fast_exchangers', 'get_in_secondary']


################################################################################
# Functions
################################################################################


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
        return (df)


def label(resSeq, sPRE, err=0.01):
    """Label for sPRE.tbl files."""
    return f"assign (resid {resSeq:<2} and name HN)	{sPRE:5.3f}	{err:5.3f}"


def getResSeq(lines):
    """Returns Resseqs for lines in sPRE .txt files provided by Tobias Schneider."""
    return list(map(lambda x: int(x.split('\t')[0][3:]),
                    filter(lambda x: False if (x == '' or 'mM' in x or 'primary' in x) else True, lines)))


def make_sPRE_table(in_files, out_file=None, return_df=True, split_prox_dist=False,
                    omit_erros=True, print_instead_of_write=False):
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

    Example:
        >>> import xplor
        >>> xplor.functions.parse_input_files.make_15_N_table(f'data/spre_and_relaxation_data_k6_k29_k33/relaxation_file_ub2_{ubq_site}.txt',
        ...                                                   out_file=f'{os.getcwd()}/xplor/data/diUbi_sPRE_{ubq_site}_w_CONECT.tbl')

    Returns:
        Union[None, pd.Dataframe]: Either None or the pandas dataframe.

    """
    # raise Exception("The parser does not parse lines with <space><tab> as separators. Fix this.")
    residues_with_errors = [78, 79]
    in_files = get_local_or_proj_file(in_files)
    files = sorted(in_files, key=lambda x: 1 if ('proximal' in x or 'prox' in x) else 2)
    assert 'prox' in files[0] and 'dist' not in files[0]
    assert 'dist' in files[1] and 'prox' not in files[1]
    assert len(files) == 2, print(f"I need a proximal and a distal file. I found {files}")
    proximal, distal = files

    file = proximal
    df = pd.read_csv(file, skiprows=[0, 1], delimiter=r'\s+', na_values=['--'],
                     names=['primary sequence', 'sPRE', 'err'], header=None)
    df['resSeq'] = df['primary sequence'].apply(lambda x: int(x[3:]))
    df['position'] = 'proximal'


    file = distal
    df2 = pd.read_csv(file, skiprows=[0, 1], delimiter=r'\s+', na_values=['--'],
                     names=['primary sequence', 'sPRE', 'err'], header=None)
    df2['resSeq'] = df['primary sequence'].apply(lambda x: int(x[3:]))
    df2['position'] = 'distal'

    # print(df2)
    # raise Exception("STOP")

    df = pd.concat([df, df2])

    if print_instead_of_write:
        out_file = RAMFile()

    if out_file is not None:
        if not split_prox_dist:
            with open(out_file, 'w') as f:
                for i, row in df.iterrows():
                    if row['primary sequence'] == 'Phe4':
                        print(row)
                    if any(pd.isna(row)):
                        continue
                    new_line = label(row['resSeq'], row['sPRE'], row['err'])
                    if omit_erros and row['resSeq'] in residues_with_errors:
                        print(f"Line {label(row['resSeq'], row['sPRE'], row['err'])} was ommitted, due to chosen option "
                              f"`omit_errors`, which is {omit_erros} and excluded {row['resSeq']}, because it's listed "
                              f"in `residues_with_errors`, which is {residues_with_errors}.")
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
                            # the max() in this line stems from a way of obtaining the highest distal resSeq
                            # See this line for more info
                            # https://github.com/kevinsawade/xplor_functions/blob/e9382db966026b1ada1dff53d25374ed620e908d/xplor/functions/parse_input_files/parse_input_files.py#L207
                            new_line = label(row['resSeq'] - max(df[df['position'] == 'distal']['resSeq']), row['sPRE'], row['err'])
                        f.write(new_line + '\n')
            print(fnames, 'written')

    # if print_instead_of_write:
    #     out_file.seek(0)
    #     print(out_file.read())

    if return_df:
        return df


def get_observed_df(ubq_sites, sPRE='data/spre_and_relaxation_data_k6_k29_k33/di_ub2_ubq_site_*_sPRE.txt',
                    relax='data/spre_and_relaxation_data_k6_k29_k33/relaxation_file_ub2_ubq_site.txt'):
    """Returns a nicely formatted dataframe from the observable values in data.

    The formatting of this dataframe matches the dataframe/series returned bey the
    xplor.functions submodule.

    Args:
        ubq_sites (List[str]): A list of string which ubq sites should be regarded.
         Best thing would be to provide ['k6', 'k29'].

    Keyword Args:
        sPRE (str, optional): The formatting of the sPRE string. Defaults to
            'data/spre_and_relaxation_data_k6_k29_k33/di_ub2_ubq_site_*_sPRE.txt',
            where the substring 'ubq_site' will be replaced with the strings from
            `ubq_sites`. The wildcard (*) needs to be here, because the sPRE
            values were provied in two separate files.
        relax (str, optional): The same, as `sPRE`, but wiht 15N relax data. Defaults to
            'data/spre_and_relaxation_data_k6_k29_k33/relaxation_file_ub2__ubq_site_.txt`.

    Returns:
        pd.DataFrame: A pandas dataframe.

    """
    # get residues from PDB ID
    residues = get_column_names_from_pdb(return_residues=True)

    # define an index
    index = []
    relax_indices = []
    for type_ in ['sPRE', '15N_relax_600', '15N_relax_800']:
        for pos in ['proximal', 'distal']:
            for r in residues:
                index.append(f"{pos} {r} {type_}")
                if 'relax' in type_:
                    relax_indices.append(f"{pos} {r} {type_}")

    # sPRE
    dfs_out = []
    dfs = [make_sPRE_table(sPRE.replace('ubq_site', ubq_site)) for ubq_site in ubq_sites]
    for i, (df, ubq_site) in enumerate(zip(dfs, ubq_sites)):
        df['ubq_site'] = ubq_site
        df.index = df['position'] + ' ' + df['primary sequence'].str.upper() + ' sPRE'
        df = df.rename(columns={'sPRE': ubq_site})[ubq_site].to_frame()
        dfs_out.append(df)
    df1 = pd.concat(dfs_out, axis='columns')

    # relax ratios
    dfs_out = []
    dfs = [make_15_N_table(relax.replace('ubq_site', ubq_site)) for ubq_site in ubq_sites]
    for i, (df, ubq_site) in enumerate(zip(dfs, ubq_sites)):
        df['ubq_site'] = ubq_site
        # print(df[(df['position'] == 'proximal') & (df['freq of spectrometer (MHz)'] == 600.0)]['resSeq'].unique())
        df.at[df['position'] == 'distal', 'resSeq'] = df[df['position'] == 'distal']['resSeq'] - 76
        # print(df[(df['position'] == 'distal') & (df['freq of spectrometer (MHz)'] == 600.0)]['resSeq'].unique())
        # print(len(residues))
        # print(residues[-1])
        df.index = df['position'] + ' ' + np.array(residues)[df['resSeq'] - 1] + ' 15N_relax_' + df['freq of spectrometer (MHz)'].apply(int).apply(str)
        df[ubq_site] = df['R2 rate (1/s)'] / df['R1 rate (1/s)']
        df = df[ubq_site].to_frame()

        # find duplicates
        u, c = np.unique(df.index.values, return_counts=True)
        dup = u[c > 1]
        assert dup.size == 0

        # find duplicates?
        dfs_out.append(df)
    df2 = pd.concat(dfs_out, axis='columns')

    # concat
    df = pd.concat([df1, df2], axis='rows')
    # df = df.reindex(index, fill_value=0)
    df = df.fillna(0)
    print(df)
    raise Exception("STOP")

    for i, ubq_site in enumerate(ubq_sites):
        df = make_sPRE_table(sPRE.replace('ubq_site', ubq_site))
        for position in ['proximal', 'distal']:
            for residue in residues:
                resSeq = int(residue[3:])
                if i == 0:
                    labels.append(f"{position} {residue} sPRE")
                if position == 'distal':
                    resSeq += 76
                if resSeq not in df['resSeq'].values:
                    df_obs[i].append(0.0)
                    continue
                idx = (df['position'] == position) & (df['resSeq'] == resSeq)
                assert idx.sum() == 1
                if idx.any():
                    value = float(df[idx]['sPRE'])
                    df_obs[i].append(value)
                else:
                    print(mhz, position, residue)
                    raise Exception("STOP")

    # proximal MET1 15N_relax_600
    for i, ubq_site in enumerate(ubq_sites):
        df = make_15_N_table(relax.replace('ubq_site', ubq_site))
        for mhz, name in zip([600, 800], ['15N_relax_600', '15N_relax_800']):
            for position in ['proximal', 'distal']:
                for residue in residues:
                    resSeq = int(residue[3:])
                    if i == 0:
                        labels.append(f"{position} {residue} {name}")
                    if position == 'distal':
                        resSeq += 76
                    if resSeq not in df['resSeq'].values:
                        df_obs[i].append(0.0)
                        continue
                    idx = (df['freq of spectrometer (MHz)'] == mhz) & (df['resSeq'] == resSeq)
                    assert idx.sum() == 1
                    if idx.any():
                        value = float(df[idx]['R2 rate (1/s)'] / df[idx]['R1 rate (1/s)'])
                        df_obs[i].append(value)
                    else:
                        print(mhz, position, residue)
                        raise Exception("STOP")
    df_obs = pd.DataFrame(data=np.array(df_obs).T, index=labels, columns=ubq_sites)
    df_obs = df_obs.fillna(0)
    df_obs = df_obs.reindex(index, fill_value=0)
    return df_obs


def get_fast_exchangers(ubq_sites):
    """Returns a pandas dataframe with booleans, depending on whether an AA is
    a fast exchanger or not.

    """
    fast_exchangers_dict = dict(
    prox_k6 = 'Q2, L8, T9, G10, K11, T12, T14, V17, T22, D39, A46, G47, Q49, E51, K63, T66, R72, L73, R74, G75, G76',
    dist_k6 = 'L8, T9, G10, K11, T12, T14, E16, S20, T22, D39, A46, G47, Q49, E51, K63, T66, L73, R74',
    prox_k29 = 'L8, T9, G10, K11, T12, T14, V17, S20, T22, A46, G47, Q49, E51, D58, T66, R72, L73, R74, G75, G76',
    dist_k29 = 'L8, T9, G10, K11, T12, T14, S20, T22, R42, A46, G47, Q49, E51, T55, K63, S65, T66, L73, R74',
    prox_k33 =  'L8, T9, G10, K11, T12, V17, T22, A46, G47, Q49, E51, T66, R72, L73, R74, G75, G76',
    dist_k33 = 'L8, T9, G10, K11, T12, T14, E16, S20, T22, Q31, A46, G47, Q49, E51, D58, K63, T66, R72, L73, R74'
    )

    AAs = get_column_names_from_pdb(return_AAs=True)
    residues = get_column_names_from_pdb(return_residues=True)

    fast_exchangers = []

    for ubq_site in ubq_sites:
        for data in [fast_exchangers_dict[f'prox_{ubq_site}'], fast_exchangers_dict[f'dist_{ubq_site}']]:
            append = []
            aas = data.split(', ')
            aas = [AAs.loc[a[0]]['3letter'].upper() + a[1:] for a in aas]
            for r in residues:
                if r in aas:
                    append.append(True)
                else:
                    append.append(False)
            fast_exchangers.append(append)

    data = [fast_exchangers[0] + fast_exchangers[1], fast_exchangers[2] + fast_exchangers[3], fast_exchangers[4] + fast_exchangers[5]]
    columns = [f'{i} {j} fast_exchange' for i, j in itertools.product(['proximal', 'distal'], residues)]

    fast_exchangers = pd.DataFrame(data, columns=columns, index=ubq_sites).T
    return fast_exchangers


def get_in_secondary(ubq_sites, relax='data/spre_and_relaxation_data_k6_k29_k33/relaxation_file_ub2_ubq_site.txt'):
    """Returns a pandas dataframe depending on whether an AA is in a sec struct motif."""

    residues = get_column_names_from_pdb(return_residues=True)

    in_secondary = []

    # proximal MET1 15N_relax_600
    for i, ubq_site in enumerate(ubq_sites):
        df = make_15_N_table(relax.replace('ubq_site', ubq_site))
        _ = []
        labels = []
        mhz = 600
        for position in ['proximal', 'distal']:
            for residue in residues:
                resSeq = int(residue[3:])
                labels.append(f"{position} {residue} 15N_relax_600")
                if position == 'distal':
                    resSeq += 76
                if resSeq not in df['resSeq'].values:
                    _.append(False)
                    continue
                idx = (df['freq of spectrometer (MHz)'] == mhz) & (df['resSeq'] == resSeq)
                assert idx.sum() == 1
                if idx.any():
                    value = df[idx]['is in secondary'].any()
                    _.append(value)
                else:
                    print(mhz, position, residue)
                    raise Exception("STOP")
        in_secondary.append(_)

    columns = [f'{i} {j} in_secondary' for i, j in itertools.product(['proximal', 'distal'], residues)]

    in_secondary = pd.DataFrame(in_secondary, columns=columns, index=ubq_sites).T
    return in_secondary