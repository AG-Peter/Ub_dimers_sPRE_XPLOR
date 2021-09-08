# %% Imports
import pyemma
import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import mdtraj as md
import encodermap as em
import seaborn as sns
import xplor

import sys, re, glob, os, itertools, pyemma, json, hdbscan

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
if not 'xplor' in globals():
    import xplor
if not 'analysis' in globals():
    analysis = xplor.functions.EncodermapSPREAnalysis(['k6'])
analysis.analyze(cluster=True)

# %% Plot
# analysis.load_specific_checkpoint('10000')
analysis.plot_lowd('k6')


# %%

trajs = {}

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

    os.makedirs(f'clustering_{ubq_site}/', exist_ok=True)

    print(
        f"Ubq site {ubq_site} has {max(trajs[ubq_site].cluster_membership)} clusters. I'm gonna save the RMSD centroids of them now.")
    linear_combination = make_linear_combination_from_clusters(trajs[ubq_site], df_comp_norm, df_obs, fast_exchangers, ubq_site=ubq_site)

    for prob, cluster_no in zip(linear_combination, np.unique(trajs[ubq_site].cluster_membership)):
        if cluster_no == -1:
            continue
        if not os.path.isfile(f'clustering_{ubq_site}/{ubq_site}_cluster_{cluster_no}_linear_combination_{prob}_percent.pdb'):
            view, dummy_traj = gen_dummy_traj(trajs[ubq_site], cluster_no, max_frames=500, superpose=True)
            index, mat, centroid = rmsd_centroid_of_cluster(dummy_traj)
            centroid.save_pdb(f'clustering_{ubq_site}/{ubq_site}_cluster_{cluster_no}_linear_combination_{prob}_percent.pdb')
            print(f"Cluster {cluster_no} contributes {prob}% to the sPRE NMR ensemble.")

# %%

print(trajs['k6'].n_trajs)
