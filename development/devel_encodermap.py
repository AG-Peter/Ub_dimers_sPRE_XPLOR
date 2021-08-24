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

# %% Create data
import xplor.functions
from xplor.functions.functions import is_aa_sim
from xplor.functions.custom_gromacstopfile import CustomGromacsTopFile

traj_files = {'k6': [], 'k29': [], 'k33': []}
refs = {}
arrays = {}
overwrite = False
simdir = '/home/andrejb/Research/SIMS/2017_*'

for i, ubq_site in enumerate(['k6', 'k29', 'k33']):
    for j, dir_ in enumerate(glob.glob(f"{simdir}{ubq_site}_*")):
        traj_file = dir_ + '/traj_nojump.xtc'
        if not is_aa_sim(traj_file):
            continue
        traj_files[ubq_site].append(dir_ + '/traj_nojump.xtc')
        if len(refs) <= i:
            refs[ubq_site] = dir_ + '/start.pdb'

    highd_file = f'xplor/data/highd_ca_pairwise_exclude_2_neighbors_input_{ubq_site}.npy'
    if not os.path.isfile(highd_file) or overwrite:
        feat = pyemma.coordinates.featurizer(refs[ubq_site])
        feat.add_distances_ca(periodic=False)
        out = pyemma.coordinates.load(traj_files[ubq_site], feat)
        out = np.vstack(out)
        np.save(highd_file, out)
    else:
        out = np.load(highd_file)
    arrays[ubq_site] = out

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


# %%

print(len(trajs))
print(len(trajs[0]))
print(refs)


# %% Plot encodermap
em1.plot.distance_histogram(arrays['k6'][::1000], float('inf'), [40, 12, 10, 1, 2, 5])
plt.show()


# %% Train Encodermap
for i, ubq_site in enumerate(['k6', 'k29', 'k33']):
    path = f'runs/{ubq_site}/production_run_tf2/'
    os.makedirs(path, exist_ok=True)
    parameters = em.Parameters()
    parameters.main_path = path
    parameters.n_neurons = [500, 500, 500, 500, 5]
    parameters.activation_functions = ['', 'tanh', 'tanh', 'tanh', 'tanh',  '']
    parameters.periodicity = float('inf')
    parameters.dist_sig_parameters = [40, 12, 10, 1, 2, 5]
    parameters.batch_size = 64
    parameters.learning_rate = 0.00001

    e_map = em.EncoderMap(parameters, arrays[ubq_site])

    e_map.train()

# %%
parameters = em1.Parameters.load('runs/k6/production_run/parameters.json')
print(parameters.n_neurons)
e_map = em1.EncoderMap(parameters, arrays[ubq_site], checkpoint_path='/tmp/pycharm_project_462/runs/k6/production_run/checkpoints/step10000.ckpt', read_only=True)
lowd = e_map.encode(arrays[ubq_site])
print(lowd.shape)

# %% Load info for linear combination

from xplor.functions.functions import get_ubq_site
df_comp = pd.read_csv('/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package/2021-07-23T16:49:44+02:00_df_no_conect.csv', index_col=0)
if not 'ubq_site' in df_comp.keys():
    df_comp['ubq_site'] = df_comp['traj_file'].map(get_ubq_site)
df_obs = xplor.functions.parse_input_files.get_observed_df(['k6', 'k29'])
fast_exchangers = xplor.functions.parse_input_files.get_fast_exchangers(['k6', 'k29'])
in_secondary = xplor.functions.parse_input_files.get_in_secondary(['k6', 'k29'])
df_comp_norm, centers_prox, centers_dist = xplor.functions.normalize_sPRE(df_comp, df_obs)

# %%

trajs = {}

# %% Cluster and save

from encodermap.misc.clustering import gen_dummy_traj, rmsd_centroid_of_cluster
from encodermap.plot.plotting import render_vmd

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

    parameters_file = f'runs/{ubq_site}/production_run/parameters.json'
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
        view, dummy_traj = gen_dummy_traj(trajs[ubq_site], cluster_no, max_frames=500, superpose=True)
        index, mat, centroid = rmsd_centroid_of_cluster(dummy_traj)
        centroid.save_pdb(f'{ubq_site}_cluster_{cluster_no}_linear_combination_{prob}_percent.pdb')
        print(f"Cluster {cluster_no} contributes {prob}% to the sPRE NMR ensemble.")
