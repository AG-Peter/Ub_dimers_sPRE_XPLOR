# %% Imports
import xplor
import mdtraj as md
import multiprocessing
import pandas as pd
print(multiprocessing.cpu_count())

# %% make the tbl files
# xplor.functions.make_15_N_table('data/spre_and_relaxation_data_k6_k29/relaxation_file_ub2_k6.txt', out_file='/home/kevin/git/xplor_functions/xplor/data/diUbi_sPRE_k6_w_CONECT.tbl')
# xplor.functions.make_sPRE_table('data/spre_and_relaxation_data_k6_k29/di_ub2_k6_*_sPRE.txt', out_file='/tmp/pycharm_project_13/xplor/data/diUbi_sPRE_k6_w_CONECT.tbl')

# %% Write arparse lines
xplor.functions.write_argparse_lines_from_yaml_or_dict(print_argparse=True)

# %% run on all files
# xplor.functions.call_xplor_with_yaml('data/2017_06_28_GfM_SMmin_rnd_k6_0_start.pdb', from_tmp=True)
# traj_file = xplor.functions.get_local_or_proj_file('data/2017_06_28_GfM_SMmin_rnd_k6_0_start.pdb')
# traj = md.load(traj_file)
# series = xplor.functions.get_prox_dist_from_mdtraj(traj, traj_file, traj_file, 0, from_tmp=True)
out2 = xplor.functions.parallel_xplor(['k6', 'k29', 'k33'], from_tmp=True, max_len=-1)

# %% Test the series function
series = xplor.functions.get_prox_dist_from_mdtraj(traj, traj_file, traj_file, 0, from_tmp=True)
columns = set(xplor.proteins.get_column_names_from_pdb() + ['traj_file', 'top_file', 'frame', 'time'])
assert columns == set(series.keys()), print(set(series.keys()).difference(columns))

# %%
test = np.arange(200)[::50]
print(test)

# %% Test pandas concatenation
df = pd.concat(out2, axis=1).T
columns = set(xplor.proteins.get_column_names_from_pdb() + ['traj_file', 'top_file', 'frame', 'time'])
assert columns == set(df.keys()), print(set(df.keys()).difference(columns))
df

