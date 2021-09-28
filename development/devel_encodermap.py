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

# %%
xplor.misc.delete_old_csvs('/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package_all/',
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

# %% develop an analysis function
if not 'analysis' in globals():
    analysis = xplor.functions.EncodermapSPREAnalysis(['k6', 'k29', 'k33'])
    # analysis = xplor.functions.EncodermapSPREAnalysis(['k6'])
    # analysis = xplor.functions.EncodermapSPREAnalysis(['k29'])
    # analysis = xplor.functions.EncodermapSPREAnalysis(['k33'])
# analysis.analyze()
# analysis.load_xplor_data(overwrite=True)
# analysis.cluster_analysis()
# analysis.plot_cluster(0, 'k6', overwrite=True)
# analysis.fitness_assessment(overwrite=True)
# analysis.get_surface_coverage(overwrite_image=True)
# analysis.get_mean_tensor_of_inertia(overwrite=True)
# analysis.distance_vs_pseudo_torsion(overwrite_image=True)
# analysis.cluster_pseudo_torsion(overwrite_struct_files=True)

# %% Develop inertia tensors
mean = np.mean(analysis.inertia_tensors['k6'], axis=0)
std = np.std(analysis.inertia_tensors['k6'], axis=0)
lower = mean - 0.75 * std
upper = mean + 0.75 * std

test = np.logical_and(analysis.inertia_tensors['k6'] >= lower, analysis.inertia_tensors['k6'] <= upper)
test = np.all(test, axis=(1, 2))
np.unique(test, return_counts=True)

# %% Test volumetric plot
%matplotlib qt5

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

# %%

for atom in analysis.aa_trajs['k6'][0].top.atoms:
    print(atom)
    print(atom.element.mass)
    break

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
