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

import sys, re, glob, os, itertools, pyemma, json, hdbscan, copy

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

# %% develop an analysis function
if not 'analysis' in globals():
    analysis = xplor.functions.EncodermapSPREAnalysis(['k6'])
analysis.analyze(plot_lowd=True)

# %% Test plot
# analysis.load_specific_checkpoint()
# analysis.plot_lowd('k6', overwrite=True, outfile='/home/kevin/tmp.png')

# %% Test layout

if not 'plt' in globals():
    import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.close('all)')

cluster_num = 0
ubq_site = 'k6'
nbins = 100

fig = plt.figure(constrained_layout=True, figsize=(10, 50))
spec = fig.add_gridspec(ncols=2, nrows=10)

text_ax = fig.add_subplot(spec[0, :2])
scatter_all_ax = fig.add_subplot(spec[1, 0])
dense_all_ax = fig.add_subplot(spec[1, 1])
cg_vs_aa_contour_ax = fig.add_subplot(spec[2, 0])
clustered_ax = fig.add_subplot(spec[2, 1])
scatter_cluster_ax = fig.add_subplot(spec[3, 0])
render_ax = fig.add_subplot(spec[3, 1])
sPRE_prox_ax = fig.add_subplot(spec[4, :2])
sPRE_dist_ax = fig.add_subplot(spec[5, :2])
N15_prox_ax = fig.add_subplot(spec[6, :2])
N15_dist_ax = fig.add_subplot(spec[7, :2])
sPRE_vs_prox_ax = fig.add_subplot(spec[8, :2])
sPRE_vs_dist_ax = fig.add_subplot(spec[9, :2])

titles = [f'General Info of cluster {cluster_num}', 'Lowd scatter after Encodermap AA and CG',
          'Density plot AA and CG', 'Contours of AA and CG simulations',
          'Cluster assignment with HDBSCAN', f'Scatter plot of cluster {cluster_num}',
          f'Render of cluster {cluster_num}', 'sPRE proximal', 'sPRE distal', '15N proximal',
          '15N distal', 'sPRE proximal avg. vs geom/rmsd centroid', 'sPRE distal avg vs geom/rmsd centroid']

for ax, title in zip(fig.axes, titles):
    ax.set_title(title)
    if 'render' in title.lower() or 'general' in title.lower():
        ax.axis('off')

# scatter
# scatter_all_ax.scatter(*analysis.aa_trajs[ubq_site].lowd[::10].T, s=1, c='C0', label='aa')
# scatter_all_ax.scatter(*analysis.cg_trajs[ubq_site].lowd[::10].T, s=1, c='C1', label='cg')
# scatter_all_ax.legend()

# pyemma.plots.plot_free_energy(*analysis.trajs[ubq_site].lowd.T, cmap='turbo', cbar=False, ax=dense_all_ax, nbins=nbins)

# aa_H, xedges, yedges = np.histogram2d(*analysis.aa_trajs[ubq_site].lowd[:, :2].T, bins=nbins)
# xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
# ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)
# X, Y = np.meshgrid(xcenters, ycenters)
# aa_H[aa_H > 0] = 1
# out = copy.deepcopy((aa_H))
# for i in range(nbins - 1):
#     for j in range(nbins - 1):
#         if any([aa_H[i - 1, j], aa_H[i + 1, j], aa_H[i, j - 1], aa_H[i, j + 1]]):
#             out[i, j] = 1
# cg_vs_aa_contour_ax.contour(X, Y, out, levels=2, cmap='Blues')
# cg_H, xedges, yedges = np.histogram2d(*analysis.cg_trajs[ubq_site].lowd[:, :2].T, bins=nbins)
# xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
# ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)
# X, Y = np.meshgrid(xcenters, ycenters)
# cg_H[cg_H > 0] = 1
# out = copy.deepcopy((cg_H))
# for i in range(nbins - 1):
#     for j in range(nbins - 1):
#         if any([aa_H[i - 1, j], aa_H[i + 1, j], aa_H[i, j - 1], aa_H[i, j + 1]]):
#             out[i, j] = 1
# cg_vs_aa_contour_ax.contour(X, Y, out, levels=2, cmap='Greys')

# color_palette = sns.color_palette('deep', max(analysis.trajs[ubq_site].cluster_membership) + 1)
# cluster_colors = [(*color_palette[x], 1) if x >= 0
#                   else (0.5, 0.5, 0.5, 0.1)
#                   for x in analysis.trajs[ubq_site].cluster_membership]
# clustered_ax.scatter(*analysis.trajs[ubq_site].lowd[:, :2].T, s=1, c=cluster_colors)

# where = np.where(analysis.trajs[ubq_site].cluster_membership == cluster_num)[0]
# data = analysis.trajs[ubq_site].lowd
# ax4 = scatter_cluster_ax
# divider = make_axes_locatable(ax4)
# axHistx = divider.append_axes("top", size=1.2, pad=0.1)#, sharex=ax1)
# axHisty = divider.append_axes("right", size=1.2, pad=0.1)#, sharey=ax1)
# spines = [k for k in axHistx.spines.values()]
# spines[1].set_linewidth(0)
# spines[3].set_linewidth(0)
# axHistx.set_xticks([])
# H, edges, patches = axHistx.hist(data[:, 0][where], bins=50)
# axHistx.set_ylabel('count')
# axHistx.set_title('Scatter of Cluster')
# # y hist
# spines = [k for k in axHisty.spines.values()]
# spines[1].set_linewidth(0)
# spines[3].set_linewidth(0)
# axHisty.set_yticks([])
# H, edges, patches = axHisty.hist(data[:, 1][where], bins=50, orientation='horizontal')
# axHisty.set_xlabel('count')
# # scatter data
# ax4.scatter(x=data[where,0], y=data[where,1], s=1)
# spines = [k for k in ax4.spines.values()]
# spines[3].set_linewidth(0)
# spines[1].set_linewidth(0)
# ax4.set_xlabel('x in a.u.')
# ax4.set_ylabel('y in a.u.')

# try:
#     view, dummy_traj = em.misc.clustering.gen_dummy_traj(analysis.aa_trajs[ubq_site], 0, align_string='name CA and resid > 76', superpose=analysis.aa_trajs[ubq_site][0][0].traj, shorten=True,)
#     dummy_traj.save_pdb('/tmp/tmp.pdb')
#     image = em.plot.plotting.render_vmd('/tmp/tmp.pdb', drawframes=True, scale=1.5)
# finally:
#     os.remove('/tmp/tmp.pdb')
# render_ax.imshow(image)

ax_pairs = [[sPRE_prox_ax, sPRE_dist_ax], [N15_prox_ax, N15_dist_ax], [sPRE_vs_prox_ax, sPRE_vs_dist_ax]]

for i, ax_pair in enumerate(ax_pairs):
    ax1, ax2 = ax_pair
    for ax, centers in zip([ax1, ax2], [analysis.centers_prox, analysis.centers_dist - 76]):
        ax = xplor.nmr_plot.add_sequence_to_xaxis(ax)
        if i == 0:
            ax = xplor.nmr_plot.color_labels(ax, positions=centers)
        if i == 0 or i == 2:
            ax.set_ylabel(r'sPRE in $\mathrm{mM^{-1}ms^{-1}}$')
    if i == 0:
        (ax1, ax2) = xplor.nmr_plot.plot_line_data((ax1, ax2), analysis.df_obs, {'rows': 'sPRE', 'cols': ubq_site})
        (ax1, ax2) = xplor.nmr_plot.plot_hatched_bars((ax1, ax2), analysis.fast_exchangers, {'cols': 'k6'}, color='k')

plt.savefig('/home/kevin/tmp.png')

# %% Cluster and save

from encodermap.misc.clustering import gen_dummy_traj, rmsd_centroid_of_cluster
from encodermap.plot.plotting import render_vmd
from xplor.functions.functions import make_linear_combination_from_clusters

raise Exception("There is new better code in one of the notebooks")

def ckpt_step(file):
    file = os.path.basename(file).split('.')[0].split('_')[-1]
    return int(file)

overwrite = False

for i, ubq_site in enumerate(['k6', 'k29', 'k33']):
    if not ubq_site in trajs:
        _ = em.Info_all(trajs=traj_files[ubq_site], tops=refs[ubq_site],
                        common_str=[ubq_site], basename_fn=lambda x: x.split('/')[-2])
        trajs[ubq_site] = _
    else:
        print("Trajs already loaded")

    parameters_file = f'runs/{ubq_site}/production_run_tf2/parameters.json'
    if not os.path.isfile(parameters_file):
        print("Not yet finished")
        continue

    if not hasattr(trajs[ubq_site], 'lowd'):
        checkpoints = glob.glob(f'/tmp/pycharm_project_462/runs/{ubq_site}/production_run_tf2/*encoder')
        checkpoint = list(sorted(checkpoints, key=ckpt_step))[-1].replace('model_encoder', '*')

        loaded_e_map = em.EncoderMap.from_checkpoint(checkpoint)

        lowd = loaded_e_map.encode(arrays[ubq_site])
        trajs[ubq_site].load_CVs(lowd, attr_name='lowd')

    else:
        print("Lowd already in trajs")

    if not hasattr(trajs[ubq_site], 'cluster_membership'):
        clusterer = hdbscan.HDBSCAN(min_cluster_size=250, cluster_selection_method='leaf').fit(trajs[ubq_site].lowd)
        trajs[ubq_site].load_CVs(clusterer.labels_, 'cluster_membership')
    else:
        print('Cluster_membership already in trajs')

    if not os.path.isfile(f'tmp{i}.png') or overwrite:
        plt.close('all')

        # fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5), sharex=True, sharey=True)
        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223, projection='3d')
        ax4 = fig.add_subplot(224)

        ax1.scatter(*trajs[ubq_site].lowd[:,:2].T)
        pyemma.plots.plot_free_energy(*trajs[ubq_site].lowd[:,:2].T, ax=ax2, cbar=False, cmap='turbo')

        color_palette = sns.color_palette('deep', max(trajs[ubq_site].cluster_membership) + 1)
        cluster_colors = [color_palette[x] if x >= 0
                          else (0.5, 0.5, 0.5)
                          for x in trajs[ubq_site].cluster_membership]
        ax3.scatter(*trajs[ubq_site].lowd[:, :3].T, c=cluster_colors, alpha=0.2)

        try:
            view, dummy_traj = gen_dummy_traj(trajs[ubq_site], 0, superpose=True, shorten=True)
            dummy_traj.save_pdb('tmp.pdb')
            image = render_vmd('tmp.pdb', drawframes=True, renderer='tachyon', scale=2)
            ax4.imshow(image)
            ax4.axis('off')
        except AssertionError:
            pass
        finally:
            os.remove('tmp.pdb')

        plt.savefig(f'tmp{i}.png')
    else:
        print(f"Not overwriting tmp{i}.png")

