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

import sys, re, glob, os, itertools, pyemma, json, hdbscan, copy

# %% delete old csvs
xplor.misc.delete_old_csvs('/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package/',
                            suffix='_df_no_conect.csv', )

# %% Get data from last working encodermap
for root, dirs, files in os.walk("/home/kevin/projects/tobias_schneider/"):
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

json_savefile = '/mnt/data/kevin/xplor_analysis_files/quality_factors.json'
with open(json_savefile, 'r') as f:
    all_quality_factors = json.load(f)

for ubq_site in ubq_sites:
    image_file = f'/mnt/data/kevin/xplor_analysis_files/quality_factors_{ubq_site}.png'
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


# %% develop an analysis function
import xplor
if not 'analysis' in globals():
    analysis = xplor.functions.EncodermapSPREAnalysis(['k6', 'k29', 'k33'])
    analysis.df_comp = 'all_frames'
    analysis.make_large_df(False)
    analysis.add_count_ids(False)
# analysis.fix_broken_pdbs()
# analysis.check_normalization()
analysis.add_centroids_to_df()
# analysis.analyze_mean_abs_diff_all(True)
# analysis.run_per_cluster_analysis()
# analysis.
# analysis.run_per_cluster_analysis(overwrite_final_combination=True)
# analysis.get_mixed_correlation_plots(overwrite=True)
# analysis.prepare_csv_files(overwrite=True)


# %%
# analysis.aa_df.to_csv(analysis.large_df_file)

# %%
print([k for k in analysis.aa_df.keys() if 'prox' not in k and 'dist' not in k])

# %% old cluster analysis
analysis.make_large_df(overwrite=True)


# %% new per cluster analysis
# analysis.write_clusters(directory='/home/kevin/projects/tobias_schneider/new_cluster_analysis',
#                         clusters={'k6': [5, 9], 'k29': [0, 12], 'k33': [20]}, which='count_id')
analysis.run_per_cluster_analysis(overwrite_final_combination=True)

# %% Single line calls
analysis.plot_cluster_rmsds()

# %% Save cluster_means
from xplor.functions.analysis import h5load, h5store, make_linear_combination_from_clusters
import pandas as pd
import numpy as np

old_df_file = '/home/kevin/projects/tobias_schneider/new_images/clusters.h5'
new_df_file = '/home/kevin/projects/tobias_schneider/new_images/clusters_with_and_without_coeff.h5'

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

    sub_df = sub_df.rename(columns={'mean abs diff to exp': 'mean abs diff to exp w/ coeff'})
    sub_df['mean abs diff to exp w/o coeff'] = mean_abs_diff_no_coeff
    sub_dfs.append(sub_df)

df = pd.concat(sub_dfs, ignore_index=True)
h5store(new_df_file, df, **metadata)


# %% Plot a single cluster
analysis.plot_cluster(2, 'k6', overwrite=True, out_file='/home/kevin/projects/tobias_schneider/new_images/summary_single_cluster.png')

# %%

# for ubq_site in analysis.ubq_sites:
#     np.save(f'/mnt/data/kevin/xplor_analysis_files/lowd_{ubq_site}_aa.npy', analysis.aa_trajs[ubq_site].lowd)
#     np.save(f'/mnt/data/kevin/xplor_analysis_files/lowd_{ubq_site}_cg.npy', analysis.cg_trajs[ubq_site].lowd)

for ubq_site in analysis.ubq_sites:
    name_arr = []
    for traj in analysis.aa_trajs[ubq_site]:
        for frame in traj:
            name_arr.append(frame.traj_file)
    name_arr = np.array(name_arr)
    index = np.array(['kevin' in i for i in name_arr])
    np.save(f'/mnt/data/kevin/xplor_analysis_files/lowd_{ubq_site}_aa_rotamers.npy', analysis.aa_trajs[ubq_site].lowd[index])
    index = np.array(['GfM_SMmin' in i and 'rnd' not in i for i in name_arr])
    np.save(f'/mnt/data/kevin/xplor_analysis_files/lowd_{ubq_site}_SMin.npy', analysis.aa_trajs[ubq_site].lowd[index])
    index = np.array(['GfM_SMmin_rnd' in i for i in name_arr])
    np.save(f'/mnt/data/kevin/xplor_analysis_files/lowd_{ubq_site}_aa_SMin_rnd.npy', analysis.aa_trajs[ubq_site].lowd[index])
    index = np.array(['G_2ub' in i for i in name_arr])
    np.save(f'/mnt/data/kevin/xplor_analysis_files/lowd_{ubq_site}_aa_extended.npy', analysis.aa_trajs[ubq_site].lowd[index])

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

# plt.savefig('/mnt/data/kevin/xplor_analysis_files/inertia_distribution.png')
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
# plt.savefig('/mnt/data/kevin/xplor_analysis_files/surface_coverage_k6.png')
