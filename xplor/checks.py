import os, re, sys, glob
import numpy as np
import mdtraj as md
import pandas as pd
import tempfile
from .misc import get_iso_time
from rich import print as rprint
from .functions.parse_input_files.parse_input_files import make_sPRE_table
from .functions.custom_gromacstopfile import CustomGromacsTopFile
from .functions.functions import Capturing
from .functions.functions import RAMFile


RESIDUE_THAT_WORKS = 4


def remove_prox_dist(sequence):
    return np.array([i for i in sequence if 'prox' not in i and 'dist' not in i])


def remove_unnamed_from_df(df):
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]


def return_filtered_first_row(df):
    return df.iloc[0][remove_prox_dist(df.columns)]


def equal_both_nan_or_zero(v1, v2):
    pass


def have_columns_been_switched(df1, df2, positions=None, csv1=None, csv2=None):
    if positions is None:
        positions = ['prox', 'dist']

    # make sure they have the same keys
    assert set(df1.keys()) == set(df2.keys())

    # make sure the trajs are the same
    index1 = ~df1.columns.str.contains('prox')
    index2 = ~df1.columns.str.contains('dist')
    index = np.logical_and(index1, index2)
    print(f"Possible non-value rows are: {df1.columns[index].values}")
    df1_index = df1[['traj_file', 'frame', 'time']]
    df2_index = df2[['traj_file', 'frame', 'time']]
    if not df1_index.equals(df2_index):
        if len(df1) > len(df2):
            raise Exception(f"Normally, the older one can't be longer: old: {len(df1)}, newer: {len(df2)}")
        # df2 = pd.read_csv(csv2, skiprows=lambda x: x not in df1.index)
        df_ = df2.merge(df1)
        test = df2.iloc[np.arange(len(df_))]
        assert test.equals(df_)
        df2 = df2.merge(df1)

    cols = {p: df1.columns[df1.columns.str.contains(p)] for p in positions}

    # replace nans with zeros to ensure equality checks
    df1 = df1.fillna(0)
    df2 = df2.fillna(0)

    if np.allclose(df1[cols['prox']].values, df2[cols['prox']].values) and np.allclose(df1[cols['dist']].values, df2[cols['dist']].values):
        return False
    elif np.allclose(df1[cols['prox']].values, df2[cols['dist']].values) and np.allclose(df1[cols['dist']].values, df2[cols['prox']].values):
        return True
    else:
        for i in range(len(df1)):
            row1 = df1.iloc[i]
            row2 = df2.iloc[i]
            assert row1['basename'] == row2['basename']
            assert row1['frame'] == row2['frame']
            for col_name in cols['prox'].tolist() + cols['dist'].tolist():
                if 'prox' in col_name:
                    replace_tuple = ('proximal', 'distal')
                else:
                    replace_tuple = ('distal', 'proximal')

                if row1[col_name] == row2[col_name]:
                    pass
                elif row1[col_name] == row2[col_name.replace(*replace_tuple)]:
                    pass
                else:
                    msg = (f"The value {col_name} = {row1[col_name]} of the file "
                          f"{row1['basename']} at frame {row1['frame']} does "
                          f"neither match the value {col_name} = {row2[col_name]} "
                          f"nor the value {col_name.replace(*replace_tuple)} = "
                          f"{row2[col_name.replace(*replace_tuple)]}. This is bad.")
                    raise Exception(msg)

        raise Exception("Fin")
        where = np.where(val1_prox != val2_prox)
        for i, (w1, w2) in enumerate(zip(where[0], where[1])):
            print(val1_prox[w1, w2], val2_prox[w1, w2])
            file1, basename1, frame1 = df1.iloc[w1][['traj_file', 'basename', 'frame']]
            file2, basename2, frame2 = df2.iloc[w1][['traj_file', 'basename', 'frame']]
            print(basename1, frame1, basename2, frame2)
            # print(f"The dataframes are different at position ({w1}, {w2}) which "
            #       f"corresponds to {cols['prox'][w2]} for file {df1.iloc[w1]['basename']} "
            #       f"at frame {df1.iloc[w1]['frame']}. The value in df1 is {df1.values[w1, w2]} "
            #       f"and in df2 it is {df2.values[w1, w2]}")
            if i == 2:
                break
        raise Exception("Work this out.")


def determine_changed_dfs(skiprows=2500):
    dirs = glob.glob('/home/kevin/projects/tobias_schneider/values_from_every_frame/*/')
    for dir_ in dirs:
        if 'render' in dir_:
            continue
        print(f"Loading .csv files from directory {dir_.split('/')[-2]}.")
        files = glob.glob(os.path.join(dir_, '*.csv'))
        files = list(sorted(files, key=get_iso_time))
        for older_file, newer_file in zip(files[:-1], files[1:]):
            print(f"Comparing the older file {older_file.split('/')[-1]} with "
                  f"the newer file {newer_file.split('/')[-1]}.")

            older_df = remove_unnamed_from_df(pd.read_csv(older_file, skiprows=lambda x: not x % skiprows == 0))
            newer_df = remove_unnamed_from_df(pd.read_csv(newer_file, skiprows=lambda x: not x % skiprows == 0))

            for date, df in zip(['older', 'newer'], [older_df, newer_df]):
                # length
                print(f"Only using every {skiprows}th row, the {date} dataframe contains "
                      f"{len(df)} rows.")

                # ubq sites
                if 'ubq_site' in df:
                    print(f"The {date} dataframe contains {np.unique(df['ubq_site'].str.lower())} as ubiuitination sites.")
                else:
                    print(f"The {date} dataframe contains no info about ubiquitylation sites.")

                # first row
                print(f"The first row of data in the {date} dataframe contains\n"
                      f"{return_filtered_first_row(df)}")

                print()

            if have_columns_been_switched(older_df, newer_df, csv1=older_file, csv2=newer_file):
                rprint(f"[bold magenta]The columns between {older_file.split('/')[-1]} and "
                       f"{newer_file.split('/')[-1]} have SUCCESSFULLY BEEN SWITCHED.[/bold magenta]")
            else:
                rprint(f"[bold blue]The columns between {older_file.split('/')[-1]} and "
                       f"{newer_file.split('/')[-1]} have not been switched.[/bold blue]")
            print()


def value_and_column_match(v, col):
    if np.any(np.isnan(v)) or np.any(np.isnan(col)):
        raise Exception(f"Can only work with non-nans: Here is the value {v} and"
                        f" the column: {col}")
    return (v != 0 and np.mean(col) != 0) or (v == 0 and np.mean(col) == 0)


def exp_is_zero_and_sim_not(v, col):
    return v == 0 and np.mean(col) != 0


def make_empty_tbl(x):
    if isinstance(x, int):
        return f"assign (resid {x:<2} and name HN)   0.000   0.000"
    elif isinstance(x, list):
        assert all([isinstance(_, int) for _ in x])
        return '\n'.join([f"assign (resid {i:<2} and name HN)   0.000   0.000" for i in x])
    else:
        raise Exception("Arg `x` needs to be int or list of ints.")


def was_excluded_because_missing_in_opposite_position_experimental_data(exp, seq, pos):
    if pos == 'distal':
        opp_pos = 'proximal'
    elif pos == 'proximal':
        opp_pos = 'distal'
    else:
        raise Exception(f"Arg `pos` must be 'proximal', or 'distal', you supplied: {pos}.")
    df = exp[(exp['primary sequence'] == seq) & (exp['position'] == opp_pos)]

    if df.empty:
        return opp_pos
    else:
        return False


def mistake_in_parser(seq, pos, ubq_site):
    from .functions.parse_input_files.parse_input_files import label
    resSeq = int(seq[3:])
    if pos == 'proximal':
        opp_pos = 'distal'
        resSeq += 76
    elif pos == 'distal':
        opp_pos = 'proximal'
    else:
        raise Exception(f"Arg `pos` must be 'proximal', or 'dista;', you supplied: {pos}.")

    out_file = RAMFile()
    inp = f'/home/kevin/git/xplor_functions/xplor/data/spre_and_relaxation_data_k6_k29_k33/di_ub2_{ubq_site}_*_sPRE.txt'
    test_df = make_sPRE_table(in_files=inp, out_file=out_file)
    out_file.seek(0)
    out_file = out_file.read()

    # check whether in out file
    in_out_file = make_empty_tbl(resSeq) in out_file

    # check whether in df
    row = test_df[(test_df['primary sequence'] == seq) & (test_df['position'] == opp_pos)]
    print(row)

    # check whether the row in the source file contains space
    if opp_pos == 'proximal':
        short_opp_pos = 'prox'
    else:
        short_opp_pos = 'dist'
    filename = f'/home/kevin/git/xplor_functions/xplor/data/spre_and_relaxation_data_k6_k29_k33/di_ub2_{ubq_site}_{short_opp_pos}_sPRE.txt'
    with open(filename, 'r') as f:
        for line in f.read().splitlines():
            if seq in line:
                if len(line.split('\t')) >= 4:
                    return f"Line for {seq} in file {filename.split('/')[-1]} was not parsed due to having different than <tab> indentation between value and error."

    return False



def cant_be_called_with_xplor(seq, pos, df, ubq_site):
    # tbl with residue in it should fail
    # choose a random file to load
    random = np.random.choice(df.index)
    row = df.iloc[random]
    frame = md.load_frame(row['traj_file'], top=row['top_file'],
                          index=row['frame'])

    with Capturing() as output:
        top_aa = CustomGromacsTopFile(
            f'/home/andrejb/Software/custom_tools/topology_builder/topologies/gromos54a7-isop/diUBQ_{ubq_site.upper()}/system.top',
            includeDir='/home/andrejb/Software/gmx_forcefields')
    frame.top = md.Topology.from_openmm(top_aa.topology)

    isopeptide_indices = []
    isopeptide_bonds = []
    for r in frame.top.residues:
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

    # tbl with working residue should succeed
    return_code1, err1, out1 = call_xplor_from_frame_and_single_residue(frame, RESIDUE_THAT_WORKS)
    assert return_code1 == 0

    # tbl with the residue should crash
    seq = int(seq[3:])
    if pos == 'proximal':
        seq += 76
    elif pos == 'distal':
        pass
    else:
        raise Exception(f"Arg `pos` must be 'proximal', or 'dista;', you supplied: {pos}.")
    return_code2, err2, out2 = call_xplor_from_frame_and_single_residue(frame, seq)
    if return_code2 != 0:
        return True
    else:
        return False


def call_xplor_from_frame_and_single_residue(frame, residue,
                                             executable='/home/kevin/software/xplor-nih/executables/pyXplor '):
    import subprocess
    tmp_pdb_file = '/tmp/tmp.pdb'
    frame.save_pdb(tmp_pdb_file)
    tmp_tbl_file = '/tmp/tmp.tbl'
    tbl_file_content = make_empty_tbl(residue)
    with open(tmp_tbl_file, 'w') as f:
        f.write(tbl_file_content)
    try:
        cmd = f"""-c "import protocol
            from psolPotTools import create_PSolPot
            protocol.loadPDB('{tmp_pdb_file}', deleteUnknownAtoms=True)
            protocol.initParams('protein')
            psol = create_PSolPot(name='psol', file='{tmp_tbl_file}')
            print([['psol', r.name().split()[2], r.calcd()] for r in psol.restraints()])"
            """
        cmd = '; '.join([i.lstrip() for i in cmd.splitlines()])
        cmd = executable + cmd.rstrip('; ')
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        out, err = process.communicate()
        return_code = process.poll()
        out = out.decode(sys.stdin.encoding)
        err = err.decode(sys.stdin.encoding)
    finally:
        if os.path.exists(tmp_pdb_file):
            os.remove(tmp_pdb_file)
        if os.path.exists(tmp_tbl_file):
            os.remove(tmp_tbl_file)

    return return_code, err, out


def what_went_wrong(ubq_site='k6', skiprows=2500, manual_exclusions=None):
    """Function should analyze some discrepancies between simulated and experimental values.

    General course of action:
        * Identify all locations, that exhibit discrepancies by loading some
            lines of the aa_df file and the experimental values.
        * With wrong order:
            * Redo the basic XPLOR calls with some temporary pdb files.
            * Redo the parsing of the experimental files
            * Some more advanced calls
            * Try empty and full pdb files.
        * Change the order:
            * Retry the

    """
    # load an old and a new df file and make sure the sPRE values are switched up
    search_dir = 'values_from_every_frame/from_package_all'
    df_files = glob.glob(f'/home/kevin/projects/tobias_schneider/'
                         f'{search_dir}/*csv')
    df_files = list(sorted(df_files, key=get_iso_time))
    older_file = df_files[-2]
    newest_file = df_files[-1]
    print(f"The directory {search_dir} contains {older_file.split('/')[-1]} as "
          f"the 2nd newest and {newest_file.split('/')[-1]} as the newest file.")

    # with open(oldest_file) as f:
    #     num_old_lines = sum(1 for line in f)
    # with open(newest_file) as f:
    #     num_new_lines = sum(1 for line in f)
    # print(f"They contain {num_old_lines} and {num_new_lines}, respectively.")

    old_df = remove_unnamed_from_df(pd.read_csv(older_file, skiprows=lambda x: not x % skiprows == 0))
    new_df = remove_unnamed_from_df(pd.read_csv(newest_file, skiprows=lambda x: not x % skiprows == 0))
    print(f"After only using every {skiprows}th row, the dataframes have the"
          f"shapes {old_df.shape}, {new_df.shape}, respectively.")

    # check difference in columns
    print(f"The old contains these columns, that the new doesn't have: "
          f"{set(new_df.keys()) - set(old_df.keys())}")
    print(f"The new contains these columns, that the old doesn't have: "
          f"{set(old_df.keys()) - set(new_df.keys())}")

    # check whether they have been switched
    assert have_columns_been_switched(old_df, new_df)

    # what ubs_sites are in there
    print(f"The newest df contains these ubq sites:\n"
          f"{pd.value_counts(new_df['ubq_site'].str.lower())}.\nFor now, I will "
          f"focus on {ubq_site}")
    df = new_df[new_df['ubq_site'].str.lower() == ubq_site].fillna(0)

    # set up manual exclusions for data that is obviously missing from the experiments
    if manual_exclusions is None:
        manual_exclusions = [('Met1', 'distal'), ('Gly75', 'distal'),
                             ('Gly76', 'distal'), ('Phe4', 'distal'),
                             ('Phe4', 'proximal')]

    # If lines have been switched get the experimental values
    inp = f'/home/kevin/git/xplor_functions/xplor/data/spre_and_relaxation_data_k6_k29_k33/di_ub2_{ubq_site}_*_sPRE.txt'
    exp = make_sPRE_table(inp).fillna(0)
    missing_values = {}
    columns = [c for c in df.columns if 'sPRE' in c]
    for i, c in enumerate(columns):
        missing_values[c] = ''
        pos, seq, _ = c.split()
        seq = seq.capitalize()
        if (seq, pos) in manual_exclusions:
            continue
        ind = (exp['primary sequence'] == seq) & (exp['position'] == pos)
        if not ind.any(None):
            msg = (f"Found no location for {c} in exp. Here are the {pos} valuyes:\n"
                   f"{exp[exp['position'] == pos]}")
            print(msg)
            continue
        assert pd.value_counts(ind)[True] == 1
        row = exp[ind]

        if not value_and_column_match(row['sPRE'].values, df[c].values):
            if exp_is_zero_and_sim_not(row['sPRE'].values, df[c].values):
                continue
            elif opp := was_excluded_because_missing_in_opposite_position_experimental_data(exp, seq, pos):
                missing_values[c] = f'Not in experimental data for opposite ({opp}) side.'
                continue
            elif cant_be_called_with_xplor(seq, pos, df, ubq_site):
                missing_values[c] = f'Residue needed to be excluded, because XPLOR crashes.'
                continue
            elif mistake := mistake_in_parser(seq, pos, ubq_site):
                missing_values[c] = mistake
                continue
            else:
                msg = (f"Current missing values dict is {missing_values}.\n"
                       f"The column {c} seems to yield mismatching values. The experimental "
                       f"data has a value of {row['sPRE'].values}, the column has a value of:\n"
                       f"{df[['basename', 'frame', c]]}\nwhich does not match. I already checked"
                       f" whether this was due to the initial proximal/distal mixup, but to no avail.")
                raise Exception(msg)


    # aa_df_correct_order = p