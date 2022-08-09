# %% Imports
import copy
import pyemma
import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import mdtraj as md
import encodermap as em
import scipy.ndimage
import seaborn as sns
import scipy
import xplor
import cartopy.crs as ccrs
from pathlib import import Path
import sys, re, glob, os, itertools, pyemma, json, hdbscan, copy

# %% delete old csvs
xplor.misc.delete_old_csvs(f"{Path(xplor.__file__).parent.parent}/data/values_from_every_frame/from_package/",
                            suffix='_df_no_conect.csv', )

# %% Get data from last working encodermap
for root, dirs, files in os.walk("{Path(xplor.__file__).parent.parent}/xplor_analysis_files/runs/{ubq_site}/"):
    for file in files:
        if file.endswith(".json"):
            print(os.path.join(root, file))
            with open(os.path.join(root, file)) as f:
                _ = json.load(f)
            if 'dist_sig_parameters' in _:
                print(_['dist_sig_parameters'])
                print(_['learning_rate'])
                print(_['n_neurons'])

# %% load and plot fitness assessment extra
import json, os
import matplotlib.pyplot as plt
import numpy as np

ubq_sites = ['k6', 'k29', 'k33']
overwrite = True
quality_factor_means = {ubq_site: [] for ubq_site in ubq_sites}

json_savefile = f"{Path(xplor.__file__).parent.parent}/xplor_analysis_files/quality_factors.json"
with open(json_savefile, 'r') as f:
    all_quality_factors = json.load(f)

for ubq_site in ubq_sites:
    image_file = f"{Path(xplor.__file__).parent.parent}/xplor_analysis_files/quality_factors_{ubq_site}.png"
    if not os.path.isfile(image_file) or overwrite or overwrite_image:
        plt.close('all')
        # data = np.array(self.quality_factor_means[ubq_site])
        fig, ax = plt.subplots()
        for key, value in all_quality_factors[ubq_site].items():
            quality_factor_means[ubq_site].append(np.mean(list(value.values())))
        ax.boxplot([[v for v in value.values()] for value in all_quality_factors[ubq_site].values()],
                   positions=list(map(int, all_quality_factors[ubq_site].keys())))
        # ax.plot(np.arange(len(data)), data)

        ax.set_title(f"Quality factor for n clusters for {ubq_site}")
        ax.set_xlabel("n")
        ax.set_ylabel("Mean abs difference between exp and sim")
        plt.savefig(image_file)

# %% fitness assessment
import xplor
if not 'analysis' in globals():
    analysis = xplor.functions.EncodermapSPREAnalysis(['k6', 'k29', 'k33'])
    analysis.df_comp = 'new_psol'
    analysis.ubq_sites = ['k6', 'k29', 'k33']
    analysis.make_large_df(False)
    analysis.add_count_ids(False)
    # analysis.fitness_assessment(soft_overwrite=True)
    # analysis.plot_fitness_assessment()
# analysis.find_lowest_diffs_in_all_quality_factors(overwrite=True, which_clusters=None)
# analysis.get_values_of_combinations()

# %% what columns does aa_df have:
cols = analysis.aa_df.columns[~ (analysis.aa_df.columns.str.contains('prox|dist'))]
# print(cols)
traj_files = analysis.aa_df[analysis.aa_df['ubq_site'].str.lower().str.contains('k33')]['traj_file'].unique().tolist()
# print(traj_files)
print(analysis.aa_df['distal THR9 sPRE'])

# %% manual save
analysis.aa_df.to_csv(f"{Path(xplor.__file__).parent.parent}/data/aa_df_sPRE_manual_with_new_psol.csv")
analysis.aa_df.to_csv(analysis.large_df_file)

 # %% develop an analysis function
import xplor
if not 'analysis' in globals():
    analysis = xplor.functions.EncodermapSPREAnalysis(['k6', 'k29', 'k33'])
    analysis.df_comp = 'new_psol'
    analysis.ubq_sites = ['k6', 'k29', 'k33']
    analysis.make_large_df(False)
    analysis.add_count_ids(False)
    analysis.load_and_overwrite_sub_dfs_for_saving()
    analysis.add_centroids_to_df(overwrite=False, overwrite_sub_dfs=False, testing=False)
analysis.run_per_cluster_analysis(overwrite=True,
                                  overwrite_final_correlation=True,
                                  overwrite_final_combination=True)
raise Exception("STOP")
analysis.get_mixed_correlation_plots(overwrite=True, exclude_0_and_nan=True)
analysis.ubq_sites = ['k6', 'k29', 'k33']
print("The only_needed_counts redo counts uses the same clusters.")
raise Exception("Check whether THR9 is there")
analysis.cluster_analysis(overwrite=True,
                          save_pdb_only_needed_count_ids={'k6': [0, 1, 2, 11],
                                                          'k29': [0, 11, 12],
                                                          'k33': [0, 1, 9, 19]})
analysis.stack_all_clusters()
analysis.prepare_csv_files(check_empty_and_zero_columns=False)
analysis.fix_broken_pdbs()
# analysis.train_encodermap_sPRE(True)
# analysis.analyze_mean_abs_diff_all()

# %% Check whether series_from_mdtraj contains distal THR9
import ast
from xplor.functions.functions import get_series_from_mdtraj, Capturing
from xplor.functions.custom_gromacstopfile import CustomGromacsTopFile
from xplor.checks import call_xplor_from_frame_and_single_residue

# read rge empty tbl
ints_in_empty_tbl = []
with open(f"{Path(xplor.__file__).parent.parent}/xplor/data/diUbi_empty.tbl") as f:
    for l in f.read().splitlines():
        i = int(l.split()[2])
        ints_in_empty_tbl.append(i)

xtc_file, pdb_file = analysis.aa_df.iloc[0][['traj_file', 'top_file']]
frame = md.load_frame(xtc_file, 0, top=pdb_file)

with Capturing() as output:
    top_aa = CustomGromacsTopFile(
        f"forcefields will be made available upon request/custom_tools/topology_builder/topologies/gromos54a7-isop/diUBQ_K6/system.top",
        includeDir=f"forcefields will be made available upon request/gmx_forcefields")
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

# first use the function from checks.py
_, __, out = call_xplor_from_frame_and_single_residue(frame, ints_in_empty_tbl)
out = out.replace('findImportantAtoms: done\n', '')
out = ast.literal_eval(out)
index = [i[1] == str(9) for i in out].index(True)
print(out[index])

# then use the get_series_from_mdtraj_function
series = get_series_from_mdtraj(frame, xtc_file, pdb_file, 0, fix_isopeptides=False)
for col in series.index:
    if 'sPRE' in col:
        print(col, series[col])

# %% Why does the new_psol data have bad coefficients?
ubq_site = 'k6'
import pandas as pd
import numpy as np
from xplor.functions.functions import make_linear_combination_from_clusters
from xplor.functions.analysis import h5load
from xplor.proteins.proteins import get_column_names_from_pdb
from xplor.functions.parse_input_files.parse_input_files import get_fast_exchangers
fast_exchangers = get_fast_exchangers(['k6', 'k29', 'k33'])

# check how the all_frames dataframe can be transformed into the for_saving dataframes
# find the data that produces
# if 'find_this' not in globals():
#     find_this = pd.read_csv(f"{Path(xplor.__file__).parent.parent}/xplor_analysis_files/sub_df_for_saving_{ubq_site}.csv.back_before_new_psol")
# print(find_this.columns.tolist())
#
# raise Exception("STOP")

# first try the old data
from xplor.functions.parse_input_files.parse_input_files_legacy import get_observed_df as get_observed_df_legacy
df_obs_old = get_observed_df_legacy(['k6', 'k29', 'k33'])
if 'aa_df_old' not in globals():
    sub_dfs = [pd.read_csv(f"{Path(xplor.__file__).parent.parent}/xplor_analysis_files/sub_df_for_saving_{i}.csv.back_before_new_psol") for i in ['k6', 'k29', 'k33']]
    aa_df_old = pd.concat(sub_dfs)
    # aa_df_old = pd.read_csv(f"{Path(xplor.__file__).parent.parent}/xplor_analysis_files/lowd_and_xplor_df.csv.back")
df_sim_old = aa_df_old[aa_df_old['ubq_site'] == ubq_site]
assert df_sim_old['distal THR9 sPRE'].mean() == 0, print(df_sim_old['distal THR9 sPRE'].mean())
coefficients = make_linear_combination_from_clusters(None, df_sim_old, df_obs_old,
                                                     fast_exchangers, ubq_site=ubq_site)
with pd.HDFStore(f"{Path(xplor.__file__).parent.parent}/data/new_images/clusters.h5") as store:
    clu_df, _ = h5load(store)
clu_df = clu_df[clu_df['ubq site'] == ubq_site][['cluster id', 'N frames']]
clu_df = clu_df.sort_values('N frames', ascending=False)
clu_df['count id'] = range(len(clu_df))
print('old data and old obs', np.round(coefficients, 2)[clu_df['cluster id']])

# try the old data with manual fixing columns
coefficients = make_linear_combination_from_clusters(None, df_sim_old, df_obs_old,
                                                     fast_exchangers, ubq_site=ubq_site,
                                                     manual_fix_columns=True)
print('old data and old obs with manual fixed columns', np.round(coefficients, 2)[clu_df['cluster id']])

# try the new data with the old df_obs
if not 'aa_df_new' in globals():
    aa_df_new = pd.read_csv(f"{Path(xplor.__file__).parent.parent}/data/aa_df_sPRE_manual_with_new_psol.csv")
df_sim_new = aa_df_new[aa_df_new['ubq_site'] == ubq_site]
coefficients = make_linear_combination_from_clusters(None, df_sim_new, df_obs_old,
                                                     fast_exchangers, ubq_site=ubq_site)
print('new data old obs', np.round(coefficients, 2)[clu_df['cluster id']])

# try the new data with the new df_obs
from xplor.functions.parse_input_files.parse_input_files import get_observed_df
df_obs_new = get_observed_df(['k6', 'k29', 'k33'])
coefficients = make_linear_combination_from_clusters(None, df_sim_new, df_obs_new,
                                                     fast_exchangers, ubq_site=ubq_site)
print('new data new obs', np.round(coefficients, 2)[clu_df['cluster id']])

# exclude fast exchangers and 0/nan values
coefficients = make_linear_combination_from_clusters(None, df_sim_new, df_obs_new,
                                                     fast_exchangers, ubq_site=ubq_site,
                                                     exclude_non_exp_values=True)
print('new data new obs exclude zeros', np.round(coefficients, 2)[clu_df['cluster id']])

# compare the old and new values. How much do they differ on average
residues = get_column_names_from_pdb(return_residues=True)
index = []
for pos in ['proximal', 'distal']:
    for r in residues:
        index.append(f"{pos} {r} sPRE")
old = df_sim_old[index]
cols = old.columns[old.mean('rows') != 0]
diff = (old[cols] - df_sim_new[cols]).mean(None).mean(None)
print(f"Average diff between old and new is {diff}")

# %% Compare exp and sim
analysis.compare_exp_sim('k6', 'proximal PHE4 sPRE')

# %% count the sims in k6 sount_id == 0
analysis.aa_df[(analysis.aa_df['ubq_site'] == 'k6') & (analysis.aa_df['count id'] == 0)]

# %%
analysis.write_clusters(f"{Path(xplor.__file__).parent.parent}/data/cluster_analysis_with_fixed_normalization",
                        {'k6': [0, 1, 2, 11], 'k29': [0, 11, 12], 'k33': [19]}, which='count_id',
                        pdb=None, max_frames=200)




# %% new per cluster analysis
# analysis.write_clusters(directory=f"{Path(xplor.__file__).parent.parent}/data/new_cluster_analysis",
#                         clusters={'k6': [5, 9], 'k29': [0, 12], 'k33': [20]}, which='count_id')
analysis.run_per_cluster_analysis(overwrite_final_combination=True)

# %% Single line calls
analysis.plot_cluster_rmsds()

# %% Save cluster_means
from xplor.functions.analysis import h5load, h5store, make_linear_combination_from_clusters
import pandas as pd
import numpy as np

old_df_file = f"{Path(xplor.__file__).parent.parent}/data/new_images/clusters.h5"
new_df_file = f"{Path(xplor.__file__).parent.parent}/data/new_images/clusters_with_and_without_coeff.h5"

with pd.HDFStore(old_df_file) as store:
    df, metadata = h5load(store)
    if isinstance(df.loc[0, 'internal RMSD'], np.ndarray):
        df['internal RMSD'] = df['internal RMSD'].apply(np.var)

sub_dfs = []
for i, ubq_site in enumerate(analysis.ubq_sites):
    sub_df = df[df['ubq site'] == ubq_site]
    aa_cluster_nums = np.unique(analysis.aa_trajs[ubq_site].cluster_membership)
    linear_combination, cluster_means = make_linear_combination_from_clusters(analysis.trajs[ubq_site],
                                                                              analysis.df_comp_norm,
                                                                              analysis.df_obs,
                                                                              analysis.fast_exchangers,
                                                                              ubq_site=ubq_site,
                                                                              return_means=True)
    exp_values = analysis.df_obs[ubq_site][analysis.df_obs[ubq_site].index.str.contains('sPRE')].values
    mean_abs_diff_no_coeff = []
    for cluster_num, cluster_mean in enumerate(cluster_means):
        if cluster_num not in aa_cluster_nums:
            print(f"Cluster {cluster_num} not in aa trajs")
            continue
        mean_abs_diff_no_coeff.append(np.mean(np.abs(cluster_mean - exp_values)))
    print(len(cluster_means), len(mean_abs_diff_no_coeff))

    sub_df = sub_df.rename(columns={'mean abs diff to exp': 'mean abs diff to exp w/ coef'})
    sub_df['mean abs diff to exp w/o coef'] = mean_abs_diff_no_coeff
    sub_dfs.append(sub_df)

df = pd.concat(sub_dfs, ignore_index=True)
h5store(new_df_file, df, **metadata)


# %% Plot a single cluster
analysis.plot_cluster(2, 'k6', overwrite=True, out_file=f"{Path(xplor.__file__).parent.parent}/data/new_images/summary_single_cluster.png")

# %%

# for ubq_site in analysis.ubq_sites:
#     np.save(f"{Path(xplor.__file__).parent.parent}/xplor_analysis_files/lowd_{ubq_site}_aa.npy", analysis.aa_trajs[ubq_site].lowd)
#     np.save(f"{Path(xplor.__file__).parent.parent}/xplor_analysis_files/lowd_{ubq_site}_cg.npy", analysis.cg_trajs[ubq_site].lowd)

for ubq_site in analysis.ubq_sites:
    name_arr = []
    for traj in analysis.aa_trajs[ubq_site]:
        for frame in traj:
            name_arr.append(frame.traj_file)
    name_arr = np.array(name_arr)
    np.save(f"{Path(xplor.__file__).parent.parent}/xplor_analysis_files/lowd_{ubq_site}_aa_rotamers.npy", analysis.aa_trajs[ubq_site].lowd[index])
    index = np.array(['GfM_SMmin' in i and 'rnd' not in i for i in name_arr])
    np.save(f"{Path(xplor.__file__).parent.parent}/xplor_analysis_files/lowd_{ubq_site}_SMin.npy", analysis.aa_trajs[ubq_site].lowd[index])
    index = np.array(['GfM_SMmin_rnd' in i for i in name_arr])
    np.save(f"{Path(xplor.__file__).parent.parent}/xplor_analysis_files/lowd_{ubq_site}_aa_SMin_rnd.npy", analysis.aa_trajs[ubq_site].lowd[index])
    index = np.array(['G_2ub' in i for i in name_arr])
    np.save(f"{Path(xplor.__file__).parent.parent}/xplor_analysis_files/lowd_{ubq_site}_aa_extended.npy", analysis.aa_trajs[ubq_site].lowd[index])

# %%

from mpl_toolkits.axes_grid1 import make_axes_locatable

nbins = 100
H, (xedges, yedges, zedges) = np.histogramdd(analysis.inertia_tensors, bins=nbins)

xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)

X, Y = np.meshgrid(xcenters, ycenters)

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax1 = fig.add_subplot(222)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224)
# fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

# ax.scatter(*analysis.inertia_tensors[::100].T, c='k', s=1, alpha=0.01)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

for ct in np.linspace(0, nbins, 20, dtype=int, endpoint=False):
    cmap = plt.cm.get_cmap('turbo').copy()
    cmap.set_bad('w', alpha=0.0)
    hist = np.ma.masked_where(H[:, :, ct] == 0, H[:, :, ct])
    cs = ax.contourf(X, Y, hist.T, zdir='z',
                    offset=zedges[ct],
                    levels=20,
                    cmap='turbo', alpha=0.5)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='2%', pad=0.05)
# plt.colorbar(cs, cax=cax)

ax.set_xlim(xedges[[0, -1]])
ax.set_ylim(yedges[[0, -1]])
ax.set_zlim(zedges[[0, -1]])

# plt.savefig(f"{Path(xplor.__file__).parent.parent}/xplor_analysis_files/inertia_distribution.png")
ax1.hist(analysis.inertia_tensors[:, 0], bins=xedges)
ax2.hist(analysis.inertia_tensors[:, 1], bins=yedges)
ax3.hist(analysis.inertia_tensors[:, 2], bins=zedges)

for pos, ax in zip(['x', 'y', 'z'], [ax1, ax2, ax3]):
    ax.set_xlabel(r"L in $amu \cdot nm^{2}$")
    ax.set_ylabel("Count")
    ax.set_title(f"Moment of inertia along {pos} axis")

plt.tight_layout()
plt.savefig()

print(H.shape)

# %% Test plot
#
# from xplor.functions.analysis import add_reference_to_map
#
# plt.close('all')
# # fig, axes = plt.subplots(ncols=2)
# # xplor.nmr_plot.try_to_plot_15N(axes, 'k6', 800, trajs=analysis.aa_trajs['k6'], cluster_num=0)
#
# fig, ax = plt.subplots(subplot_kw={'projection': ccrs.EckertIV()})
# add_reference_to_map(ax)
#
# plt.savefig(f"{Path(xplor.__file__).parent.parent}/xplor_analysis_files/surface_coverage_k6.png")
