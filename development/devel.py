# %% Imports
import xplor
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
import mdtraj as md
import glob, os
import parmed as pmd

# %% List dirs
# glob.glob('/home/andrejb/Research/SIMS/2017_04_27_G_2ub_*/')
os.listdir('/home/andrejb/Research/SIMS/2017_04_27_G_2ub_m1_01_01/')

# %% Test psf with tetrapeptides
pdb = '/home/kevin/projects/tetrapeptides_in_meoh_h2o/tetrapeptides_single/PFFP/PFFP_new.pdb'
top = '/home/kevin/projects/tetrapeptides_in_meoh_h2o/tetrapeptides_single/PFFP/PFFP_vac.top'
xtc = '/home/kevin/projects/tetrapeptides_in_meoh_h2o/tetrapeptides_single/PFFP/PFFP_MD_20ns_center_protonly.xtc'

xplor.functions.test_conect(xtc, pdb, remove_after=False, top=top)

# %% Test psf with M1 linked diUBQ
pdb = '/home/andrejb/Research/SIMS/2017_04_27_G_2ub_m1_01_01/start.pdb'
top = '/home/andrejb/Research/SIMS/2017_04_27_G_2ub_m1_01_01/traj.top'
xtc = '/home/andrejb/Research/SIMS/2017_04_27_G_2ub_m1_01_01/traj_nojump.xtc'

xplor.functions.test_conect(xtc, pdb, remove_after=False)

# %% try some more stuff anew
from xplor.functions.functions import _get_psf_atom_line
pdb = '/home/andrejb/Research/SIMS/2017_04_27_G_2ub_k6_01_01/start.pdb'
top = '/home/andrejb/Research/SIMS/2017_04_27_G_2ub_k6_01_01/traj.top'
xtc = '/home/andrejb/Research/SIMS/2017_04_27_G_2ub_k6_01_01/traj_nojump.xtc'

xplor.functions.test_conect(xtc, pdb, remove_after=True, ast_print=0)


# %% make the tbl files
# xplor.functions.parse_input_files.make_15_N_table('data/spre_and_relaxation_data_k6_k29/relaxation_file_ub2_k6.txt',
#                                 out_file='/home/kevin/git/xplor_functions/xplor/data/diUbi_sPRE_k6_w_CONECT.tbl')
# xplor.functions.parse_input_files.make_sPRE_table('data/spre_and_relaxation_data_k6_k29/di_ub2_k6_*_sPRE.txt',
#                                 out_file='/tmp/pycharm_project_13/xplor/data/diUbi_sPRE_k6_w_CONECT.tbl')

# %% Write arparse lines
# xplor.argparse.write_argparse_lines_from_yaml_or_dict(print_argparse=True)

# %% run on all files
# xplor.functions.call_xplor_with_yaml('data/2017_06_28_GfM_SMmin_rnd_k6_0_start.pdb', from_tmp=True)
# traj_file = xplor.functions.get_local_or_proj_file('data/2017_06_28_GfM_SMmin_rnd_k6_0_start.pdb')
# traj = md.load(traj_file)
# series = xplor.functions.get_series_from_mdtraj(traj, traj_file, traj_file, 0, from_tmp=True)
out2 = xplor.functions.parallel_xplor(['k6'], from_tmp=True, max_len=20, write_csv=False, testing=True)

# %% Test the series function
series = xplor.functions.get_series_from_mdtraj(traj, traj_file, traj_file, 0, from_tmp=True)
columns = set(xplor.proteins.get_column_names_from_pdb() + ['traj_file', 'top_file', 'frame', 'time'])
assert columns == set(series.keys()), print(set(series.keys()).difference(columns))

# %% Delete old dataframes
xplor.delete_old_csvs()

# %% Streamline observables and computation

df_comp = pd.read_csv('/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package/2021-07-23T16:49:44+02:00_df_no_conect.csv',
                      index_col=0)
df_obs = xplor.functions.parse_input_files.get_observed_df(['k6', 'k29'])
fast_exchangers = xplor.functions.parse_input_files.get_fast_exchangers(['k6', 'k29'])
in_secondary = xplor.functions.parse_input_files.get_in_secondary(['k6', 'k29'])

# %% Normalize
df_comp_norm, centers_prox, centers_dist = xplor.functions.normalize_sPRE(df_comp, df_obs)

# %% Plot
ubq_site = 'k6'
plt.close('all')
fig, (ax1, ax2,) = plt.subplots(nrows=2, figsize=(15, 10))

fake_legend_dict1 = {'line': [{'label': 'NMR experiment', 'color': 'C1'}],
                    'hatchbar': [{'label': 'Fast exchanging residues', 'color': 'lightgrey', 'alpha': 0.3, 'hatch': '//'}],
                    'text': [{'label': 'Residue used for normalization', 'color': 'red', 'text': 'Ile3'}],
                    'envelope': [{'label': 'XPLOR calculation', 'color': 'lightgrey'}]}

for ax, centers in zip([ax1, ax2], [centers_prox, centers_dist - 76]):
    ax = xplor.nmr_plot.add_sequence_to_xaxis(ax)
    ax = xplor.nmr_plot.color_labels(ax, positions=centers)
    ax.set_ylabel(r'sPRE in $\mathrm{mM^{-1}ms^{-1}}$')

(ax1, ax2) = xplor.nmr_plot.plot_line_data((ax1, ax2), df_obs, {'rows': 'sPRE', 'cols': ubq_site})
(ax1, ax2) = xplor.nmr_plot.plot_hatched_bars((ax1, ax2), fast_exchangers, {'cols': 'k6'}, color='k')

plt.tight_layout()
plt.show()
