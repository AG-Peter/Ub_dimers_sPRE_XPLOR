################################################################################
# Imports
################################################################################
import functools
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import mdtraj as md
import numpy as np
import pandas as pd
import os, sys, glob, multiprocessing, copy, ast, subprocess, yaml, shutil, scipy, json
from .custom_gromacstopfile import CustomGromacsTopFile
import warnings
from scipy.spatial.distance import cdist
from .functions import is_aa_sim, Capturing, normalize_sPRE, make_linear_combination_from_clusters
import encodermap as em
from encodermap.misc.clustering import gen_dummy_traj, rmsd_centroid_of_cluster
from ..nmr_plot.nmr_plot import *
import dateutil
import pyemma
import hdbscan
from .parse_input_files.parse_input_files import get_observed_df, get_fast_exchangers, get_in_secondary
from string import ascii_uppercase
from ..misc import get_iso_time

################################################################################
# Globals
################################################################################


__all__ = ['EncodermapSPREAnalysis']


################################################################################
# Functions
################################################################################

def ckpt_step(file):
    file = os.path.basename(file).split('.')[0].split('_')[-1]
    try:
        return int(file)
    except ValueError as e:
        try:
            time = dateutil.parser.isoparse(file)
        except Exception as e2:
            raise e2 from e1
    return 0


################################################################################
# Classes
################################################################################


class EncodermapSPREAnalysis:
    """ A python singleton """

    def __init__(self, ubq_sites,
                 sim_dirs=['/home/andrejb/Research/SIMS/2017_*',
                          '/home/kevin/projects/molsim/diUbi_aa/'],
                 analysis_dir='/mnt/data/kevin/xplor_analysis_files/'):
        """Sets the ubq_sites of the singleton.

        Args:
            ubq_sites (list[str]): The ubq_sites to be looked at during this run.

        Keyword Args:
            sim_dirs (list[str], optional): Where to search for sims.
            analysis_dir (str, optional): Where to save intermediate analysis steps.

        """
        self.ubq_sites = ubq_sites
        self.sim_dirs = sim_dirs
        self.analysis_dir = analysis_dir
        if not os.path.isdir(self.analysis_dir):
            os.makedirs(self.analysis_dir)

    def analyze(self, **overwrite_dict):
        """Start the analysis

        Stages:
            * load_trajs: Find the files and create encodermap.Info_all from them.
            * load_highd: Use pyEMMA to load the RWMD.

        Keyword Args:
            overwrite_dict (Dict[str, bool], optional): A dictionary of booleans
                telling the analysis methods, whether to redo already carried out
                analysis. Defaults to a dict full of str: False value pairs. The
                keys are named according to the stages.

        """
        ow_dict = {'load_trajs': False, 'load_highd': False, 'train_encodermap': False,
                   'load_xplor_data': False, 'cluster': False, 'plot_lowd': False,
                   'cluster_analysis': False, 'fitness_assessment': False}
        ow_dict.update(overwrite_dict)
        self.load_trajs(overwrite=ow_dict['load_trajs'])
        self.load_highd(overwrite=ow_dict['load_highd'])
        self.train_encodermap(overwrite=ow_dict['train_encodermap'])
        self.load_xplor_data(overwrite=ow_dict['load_xplor_data'])
        self.cluster(overwrite=ow_dict['cluster'])
        self.cluster_analysis(overwrite=ow_dict['cluster_analysis'])
        self.fitness_assessment(overwrite=ow_dict['fitness_assessment'])

    def load_trajs(self, overwrite=False, with_cg=True):
        if hasattr(self, 'trajs'):
            if not overwrite and bool(self.trajs):
                print("Trajs already loaded")
                return

        # setting some schtuff
        self.trajs = {}
        self.aa_trajs = {}
        self.cg_trajs = {}
        self.aa_traj_files = {ubq_site: [] for ubq_site in self.ubq_sites}
        self.aa_references = {ubq_site: [] for ubq_site in self.ubq_sites}
        self.cg_traj_files = {ubq_site: [] for ubq_site in self.ubq_sites}
        self.cg_references = {ubq_site: [] for ubq_site in self.ubq_sites}

        print("Loading trajs")
        for i, ubq_site in enumerate(self.ubq_sites):
            for j, dir_ in enumerate(glob.glob(f"{self.sim_dirs[0]}{ubq_site}_*") + glob.glob(f'{self.sim_dirs[1]}{ubq_site.upper()}_*')):
                # define names
                traj_file = dir_ + '/traj_nojump.xtc'
                basename = traj_file.split('/')[-2]
                if 'andrejb' in traj_file:
                    top_file = dir_ + '/start.pdb'
                else:
                    top_file = dir_ + '/init.gro'

                is_aa = True
                if 'andrejb' in traj_file:
                    if not is_aa_sim(traj_file): is_aa = False

                if is_aa:
                    self.aa_traj_files[ubq_site].append(traj_file)
                    self.aa_references[ubq_site].append(top_file)
                else:
                    self.cg_traj_files[ubq_site].append(traj_file)
                    self.cg_references[ubq_site].append(top_file)

            self.aa_trajs[ubq_site] = em.Info_all(trajs = self.aa_traj_files[ubq_site],
                                                  tops = self.aa_references[ubq_site],
                                                  basename_fn=lambda x: x.split('/')[-2],
                                                  common_str=[ubq_site, ubq_site.upper()])
            self.cg_trajs[ubq_site] = em.Info_all(trajs = self.cg_traj_files[ubq_site],
                                                  tops = self.cg_references[ubq_site],
                                                  basename_fn=lambda x: x.split('/')[-2],
                                                  common_str=[ubq_site])
            self.trajs[ubq_site] = (self.aa_trajs[ubq_site] + self.cg_trajs[ubq_site])


    def load_highd(self, overwrite=False):
        if not hasattr(self, 'aa_dists') or overwrite:
            self.aa_dists = {}
        if not hasattr(self, 'cg_dists') or overwrite:
            self.cg_dists = {}

        for i, ubq_site in enumerate(self.ubq_sites):
            aa_highd_file = os.path.join(self.analysis_dir, f'highd_rwmd_aa_{ubq_site}.npy')
            cg_highd_file = os.path.join(self.analysis_dir, f'highd_rwmd_cg_{ubq_site}.npy')
            highd_file = os.path.join(self.analysis_dir, f'highd_rwmd{ubq_site}.npy')

            if not os.path.isfile(aa_highd_file) or overwrite:
                with Capturing() as output:
                    top_aa = CustomGromacsTopFile(f'/home/andrejb/Software/custom_tools/topology_builder/topologies/gromos54a7-isop/diUBQ_{ubq_site.upper()}/system.top',
                                                  includeDir='/home/andrejb/Software/gmx_forcefields')
                top = md.Topology.from_openmm(top_aa.topology)

                CAs_prox = top.select('name CA and resid >= 76')
                CAs_dist = top.select('name CA and resid < 76')
                assert CAs_prox.shape == CAs_dist.shape

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    _ = self.aa_trajs[ubq_site].xyz
                    dists_0_1 = np.vstack([np.min(cdist(_[i, CAs_prox], _[i, CAs_dist]), axis=1) for i in range(len(_))])
                    dists_1_0 = np.vstack([np.min(cdist(_[i, CAs_dist], _[i, CAs_prox]), axis=1) for i in range(len(_))])
                self.aa_dists[ubq_site] = np.hstack([dists_0_1, dists_1_0])
                np.save(aa_highd_file, self.aa_dists[ubq_site])
            elif ubq_site not in self.aa_dists:
                self.aa_dists[ubq_site] = np.load(aa_highd_file)
            else:
                print("AA highd already loaded")

            if not os.path.isfile(cg_highd_file) or overwrite:
                BBs_prox = self.cg_trajs[ubq_site][0].top.select('name BB and resid >= 76')
                BBs_dist = self.cg_trajs[ubq_site][0].top.select('name BB and resid < 76')
                assert BBs_prox.shape == BBs_dist.shape

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    _ = self.cg_trajs[ubq_site].xyz
                    dists_0_1 = np.vstack(
                        [np.min(cdist(_[i, BBs_prox], _[i, BBs_dist]), axis=1) for i in range(len(_))])
                    dists_1_0 = np.vstack(
                        [np.min(cdist(_[i, BBs_dist], _[i, BBs_prox]), axis=1) for i in range(len(_))])
                self.cg_dists[ubq_site] = np.hstack([dists_0_1, dists_1_0])
                np.save(cg_highd_file, self.cg_dists[ubq_site])
            elif ubq_site not in self.cg_dists:
                self.cg_dists[ubq_site] = np.load(cg_highd_file)
            else:
                print("CG highd already loaded")

            if not hasattr(self.trajs[ubq_site], 'highd') or overwrite:
                self.aa_trajs[ubq_site].load_CVs(self.aa_dists[ubq_site], 'highd')
                self.cg_trajs[ubq_site].load_CVs(self.cg_dists[ubq_site], 'highd')
            else:
                print("Highd already loaded")

    def train_encodermap(self, overwrite=False):
        for i, ubq_site in enumerate(self.ubq_sites):
            train_path = os.path.join(self.analysis_dir, f'runs/{ubq_site}/production_run_tf2/')
            checkpoints = glob.glob(f'{train_path}*encoder')
            if checkpoints:
                print("There are checkpoints present")
            if overwrite:
                if hasattr(self.trajs[ubq_site], 'lowd'):
                    print('Deleting directory to prepare new run.')
                    del self.trajs[ubq_site]._CVs['lowd']
                if os.path.isdir(train_path):
                    shutil.rmtree(train_path)
            if hasattr(self.trajs[ubq_site], 'lowd') and not overwrite:
                print("Lowd already loaded")
                return
            elif checkpoints and not overwrite:
                checkpoint = list(sorted(checkpoints, key=ckpt_step))[-1].replace('model_encoder', '*')
                e_map = em.EncoderMap.from_checkpoint(checkpoint)
            else:
                os.makedirs(train_path, exist_ok=True)
                parameters = em.Parameters()
                parameters.main_path = train_path
                parameters.n_neurons = [250, 250, 125, 2]
                parameters.activation_functions = ['', 'tanh', 'tanh', 'tanh', '']
                parameters.periodicity = float('inf')
                parameters.dist_sig_parameters = [5.9, 12, 4, 5.9, 2, 4]
                parameters.learning_rate = 0.00001
                print("training")
                array = np.vstack([self.aa_dists[ubq_site], self.cg_dists[ubq_site]])
                e_map = em.EncoderMap(parameters, array)
                e_map.add_images_to_tensorboard(array[::1000])
                e_map.train()
                e_map.save()

            aa_lowd = e_map.encode(self.aa_trajs[ubq_site].highd)
            cg_lowd = e_map.encode(self.cg_trajs[ubq_site].highd)
            self.aa_trajs[ubq_site].load_CVs(aa_lowd, attr_name='lowd')
            self.cg_trajs[ubq_site].load_CVs(cg_lowd, attr_name='lowd')

    def load_xplor_data(self, overwrite=False,
                        csv='conect'):
        if csv == 'conect':
            files = glob.glob(
                '/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package_with_conect/*.csv')
            sorted_files = sorted(files, key=get_iso_time)
            csv = sorted_files[-1]
        elif csv == 'no_conect':
            files = glob.glob(
                '/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package/*.csv')
            sorted_files = sorted(files, key=get_iso_time)
            csv = sorted_files[-1]
        df_comp = pd.read_csv(csv, index_col=0)
        if not 'ubq_site' in df_comp.keys():
            df_comp['ubq_site'] = df_comp['traj_file'].map(get_ubq_site)
        df_comp['ubq_site'] = df_comp['ubq_site'].str.lower()
        if hasattr(self, 'df_comp_norm') and not overwrite:
            print("XPLOR data already loaded")
            return
        self.df_obs = get_observed_df(['k6', 'k29', 'k33'])
        self.fast_exchangers = get_fast_exchangers(['k6', 'k29', 'k33'])
        self.in_secondary = get_in_secondary(['k6', 'k29', 'k33'])
        self.df_comp_norm, self.centers_prox, self.centers_dist = normalize_sPRE(df_comp, self.df_obs)
        self.norm_factors = normalize_sPRE(df_comp, self.df_obs, get_factors=True)

    def cluster(self, overwrite=False):
        for i, ubq_site in enumerate(self.ubq_sites):
            if not hasattr(self.trajs[ubq_site], 'cluster_membership') or overwrite:
                aa_cluster_file = os.path.join(self.analysis_dir, f'cluster_membership_aa_{ubq_site}.npy')
                cg_cluster_file = os.path.join(self.analysis_dir, f'cluster_membership_cg_{ubq_site}.npy')
                if not all([os.path.isfile(file) for file in [aa_cluster_file, cg_cluster_file]]) or overwrite:
                    raise Exception("Run HDBSCAN on 4GPU")
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=2500, cluster_selection_method='leaf').fit(
                        self.trajs[ubq_site].lowd)
                    if max(clusterer.labels_) > 20:
                        raise Exception(f"Too many clusters {np.unique(clusterer.labels_)}")
                    else:
                        print(f"Found {np.unique(clusterer.labels_)} clusters.")
                    aa_index = np.arange(self.aa_trajs[ubq_site].n_frames)
                    cg_index = np.arange(self.aa_trajs[ubq_site].n_frames,
                                         self.cg_trajs[ubq_site].n_frames + self.aa_trajs[ubq_site].n_frames)
                    self.aa_trajs[ubq_site].load_CVs(clusterer.labels_[aa_index], 'cluster_membership')
                    self.cg_trajs[ubq_site].load_CVs(clusterer.labels_[cg_index], 'cluster_membership')
                    np.save(aa_cluster_file, clusterer.labels_[aa_index])
                    np.save(cg_cluster_file, clusterer.labels_[cg_index])
                else:
                    self.aa_trajs[ubq_site].load_CVs(np.load(aa_cluster_file), 'cluster_membership')
                    self.cg_trajs[ubq_site].load_CVs(np.load(cg_cluster_file), 'cluster_membership')
            else:
                print('Cluster_membership already in trajs')

    def cluster_analysis(self, overwrite=False, overwrite_image=False):
        """Analyzes the clusters.

        Steps:
            * Check how much CG and AA is distrubuted over the clusters.
            * Find the geometric center and the closest AA structure
            * Calculate the mean sPRE in that cluster
            * compare mean sPRE and sPRE of geometric center

        """
        for i, ubq_site in enumerate(self.ubq_sites):
            linear_combination_file = os.path.join(self.analysis_dir, f'clusters_{ubq_site}_sPRE_linear_combi.npy')
            means_file = os.path.join(self.analysis_dir, f'clusters_{ubq_site}_means.npy')
            if not os.path.isfile(linear_combination_file) or overwrite:
                linear_combination, means = make_linear_combination_from_clusters(self.trajs[ubq_site], self.df_comp_norm, self.df_obs,
                                                                           self.fast_exchangers, ubq_site=ubq_site, return_means=True)
                np.save(linear_combination_file, linear_combination)
                np.save(means_file, means)
            else:
                print("Loading linear combinations")
                linear_combination = np.load(linear_combination_file)
                means = np.load(means_file)

            for cluster_num, linear, mean in zip(np.unique(self.trajs[ubq_site].cluster_membership), linear_combination, means):
                if cluster_num == -1:
                    continue
                cluster_analysis_outdir = os.path.join(self.analysis_dir,
                                                       f'cluster_analysis/{ubq_site}/cluster_{cluster_num}')
                os.makedirs(cluster_analysis_outdir, exist_ok=True)
                image_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_summary.png')

                # Check how much AA and CG is distributed in this cluster
                aa_cluster_points_ind = np.where(self.aa_trajs[ubq_site].cluster_membership == cluster_num)[0]
                cg_cluster_points_ind = np.where(self.cg_trajs[ubq_site].cluster_membership == cluster_num)[0]
                sum_cluster = len(cg_cluster_points_ind) + len(aa_cluster_points_ind)
                aa_percent = len(aa_cluster_points_ind) / sum_cluster * 100
                cg_percent = len(cg_cluster_points_ind) / sum_cluster * 100
                coeff = str(np.format_float_scientific(linear, 2))
                percentage = sum_cluster / self.trajs[ubq_site].n_frames * 100
                print(f"Cluster {cluster_num} consists of {aa_percent:.2f}% aa structures and {cg_percent:.2f}% of cg structures.")

                # rmsd centroid
                cluster_pdb_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_linear_coefficient_{coeff}_ensemble_contribution_{percentage:.1f}_percent.pdb')
                rmsd_centroid_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_rmsd_centroid.pdb')
                rmsd_centroid_index_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_rmsd_centroid.npy')
                if not os.path.isfile(rmsd_centroid_file) or overwrite:
                    max_frames = 500
                    view, dummy_traj = gen_dummy_traj(self.aa_trajs[ubq_site], cluster_num, max_frames=max_frames, superpose=True)
                    where = np.where(self.aa_trajs[ubq_site].cluster_membership == cluster_num)[0]
                    idx = np.round(np.linspace(0, len(where) - 1, max_frames)).astype(int)
                    where = where[idx]
                    index, mat, rmsd_centroid = rmsd_centroid_of_cluster(dummy_traj)
                    rmsd_centroid_index = where[index]
                    rmsd_centroid.save_pdb(rmsd_centroid_file)
                    rmsd_centroid.save_pdb(cluster_pdb_file)
                    np.save(rmsd_centroid_index_file, rmsd_centroid_index)

                # geometric centroid
                geom_centroid_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_geom_centroid.pdb')
                geom_centroid_index_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_geom_centroid.npy')
                if not os.path.isfile(geom_centroid_file) or overwrite:
                    geom_center = np.mean(self.aa_trajs[ubq_site].lowd[aa_cluster_points_ind], axis=0)
                    sums = np.sum((self.aa_trajs[ubq_site].lowd[aa_cluster_points_ind] - geom_center) ** 2, axis=1)
                    geom_centroid_index = aa_cluster_points_ind[np.argmin(sums)]
                    geom_centroid = self.aa_trajs[ubq_site].get_single_frame(geom_centroid_index).traj
                    geom_centroid.save_pdb(geom_centroid_file)
                    np.save(geom_centroid_index_file, geom_centroid_index)

                if not os.path.isfile(image_file) or overwrite or overwrite_image:
                    self.plot_cluster(cluster_num, ubq_site, out_file=image_file, overwrite=True)

    def plot_cluster(self, cluster_num, ubq_site, out_file=None, overwrite=False, nbins=100, mhz=800):
        """
        Layout:
            +-------------------------------------------------------+
            |sPRE linear combination factor for this cluster is     |
            |                                                       |
            |cluster occupies X% of ubq_site ensemble               |
            |                                                       |
            +-------------------------------------------------------+
            |                                                       |
            |                       proximal                        |
            |    per-residue difference between mean sPRE and       |
            |    sPRE of geometric center                           |
            |                                                       |
            +-------------------------------------------------------+
            |                                                       |
            |                       distal                          |
            |    per-residue difference between mean sPRE and       |
            |    sPRE of geometric center                           |
            |                                                       |
            +--------------------------+-+--------------------------+
            |                          | |                          |
            |   scatter cg and aa      | |   density with pyemma    |
            |   with different colors  | |   of all lowd            |
            |                          | |                          |
            +--------------------------+ +--------------------------+
            |                          | |                          |
            |    contour of cg and     | |   using seabiornighted   |
            |    contour of aa         | |                          |
            |                          | |                          |
            +--------------------------+ +--------------------------+
            |    histogram             | |                          |
            |                          | |                          |
            +---------------------+----+ |                          |
            |                     | h  | |                          |
            |     cg and aa scatte| i  | |                          |
            |     this cluster    | s  | |  vmd render              |
            |     see encodermap  | t  | |                          |
            |                     | o  | |                          |
            +---------------------+----+-+--------------------------+
            |                                                       |
            |       sPRE proximal                                   |
            |       everything and this cluster                     |
            |                                                       |
            +-------------------------------------------------------+
            |                                                       |
            |        sPRE distal                                    |
            |        everything and this cluster                    |
            |                                                       |
            +-------------------------------------------------------+
            |                                                       |
            |            available 15N data proximal                |
            |            everything and this cluster                |
            |                                                       |
            +-------------------------------------------------------+
            |                                                       |
            |           available 15N data distal                   |
            |           everything and this cluster                 |
            |                                                       |
            +-------------------------------------------------------+

        Args:
            cluster_num (int): Number of the cluster to be plotted.
            ubq_site (str): The ubiquitination site to be considered.

        Keyword Args:
            out_file (Union[str, None], optional): Where to save the file to.
                String should end in '.png'. A .pdf will be also be saved.
                If None is provided will be saved at
                self.analysis_dir/cluster_analysis/{ubq_site}/cluster_{cluster_num}
                Defaults to None
            overwrite (bool, optional): Whether to overwrite if file is already present.
                Defaults to False
            nbins (int, optional): The number of bins to be used in 2D binning.
                Defaults to 100.
            mhz (Union[int, float], optional): The MHz for the 15N relax data to be
                considered.

        """
        cluster_analysis_outdir = os.path.join(self.analysis_dir,
                                               f'cluster_analysis/{ubq_site}/cluster_{cluster_num}')
        # set file
        if out_file is None:
            out_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_summary.png')
        if os.path.isfile(out_file) and not overwrite:
            print(f"Image for cluster {cluster_num} already exists.")
            return

        # get some data
        aa_cluster_points_ind = np.where(self.aa_trajs[ubq_site].cluster_membership == cluster_num)[0]
        cg_cluster_points_ind = np.where(self.cg_trajs[ubq_site].cluster_membership == cluster_num)[0]
        cluster_analysis_outdir = os.path.join(self.analysis_dir,
                                               f'cluster_analysis/{ubq_site}/cluster_{cluster_num}')
        rmsd_centroid_index_file = os.path.join(cluster_analysis_outdir,
                                                f'{ubq_site}_cluster_{cluster_num}_rmsd_centroid.npy')
        geom_centroid_index_file = os.path.join(cluster_analysis_outdir,
                                                f'{ubq_site}_cluster_{cluster_num}_geom_centroid.npy')
        rmsd_centroid_index = np.load(rmsd_centroid_index_file)
        geom_centroid_index = np.load(geom_centroid_index_file)

        plt.close('all)')

        fig = plt.figure(constrained_layout=False, figsize=(20, 50))
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
                  '15N distal', 'sPRE proximal median vs geom/rmsd centroid', 'sPRE distal median vs geom/rmsd centroid']

        for i, (ax, title) in enumerate(zip(fig.axes, titles)):
            ax.set_title(title)
            if i > 0:
                txt = ax.text(0.02, 0.85, ascii_uppercase[i - 1], transform=ax.transAxes, fontsize=26, va='center', ha='center', color='k')
            if 'render' in title.lower() or 'general' in title.lower():
                ax.axis('off')

        # text
        sum_cluster = len(cg_cluster_points_ind) + len(aa_cluster_points_ind)
        aa_percent = len(aa_cluster_points_ind) / sum_cluster * 100
        cg_percent = len(cg_cluster_points_ind) / sum_cluster * 100
        percentage = sum_cluster / self.trajs[ubq_site].n_frames * 100
        total_time = sum([traj.time[-1] for traj in self.trajs[ubq_site]]) / 1000 / 1000
        text = f"""\
        This is the summary for cluster {cluster_num} of ubiquitination site {ubq_site}
        For these images {self.cg_trajs[ubq_site].n_trajs} coarse-grained and {self.aa_trajs[ubq_site].n_trajs} all-atom trajectories were analyzed.
        These trajs cover a total of {total_time:.2f} micro seconds of simulated time. Using encodermap every frame of these simulations was projected into a 2D
        landscape (seen in A, B, C) representing the conformational ensemble. These points were clustered in 2D using the HDBSCAN clustering algorithm (D),
        resulting in {int(np.max(self.trajs[ubq_site].cluster_membership) + 1)} clusters. This cluster (id {cluster_num}) contains {percentage:.2f}% of all
        sampled points, of which {aa_percent:.2f}% (total: {len(aa_cluster_points_ind)}) are AA structures and {cg_percent:.2f} %(total: {len(cg_cluster_points_ind)}) are CG structures.
        The clustering in the structure space yields a reasonable tight structure ensemble. In (F) 10 evenly spaced all-atom structures are rendered.
        Only the AA structures could be used for sPRE calculation, because back-mapping from CG to AA requires a minimization step which takes 10 minutes per structure.
        The results are in figs G-L show measured sPRE values vs all (AA) calculated sPRE values (G and H), measured 15N relaxation times at {mhz} MHz and the
        calculated 15 N times (I and J) and the difference between median sPRE values of clustered points vs the sPRE values of the RMSD centroid of
        this cluster and the structure closest to the geom. centroid of the cluster (K and L). Considered trajectories in this cluster are:
        {np.unique(self.trajs[ubq_site].name_arr[self.trajs[ubq_site].cluster_membership == cluster_num]).tolist()[:3]} and more.
        Trajectories with `G_2ub_{ubq_site}` are atomistic sims started from extended structures.
        `GfM` are atomistic, started from back-mapped CG trajs.
        `GfM` and `SMin` are extracted from Sketchmap minima.
        `GfM` and `SMin_rnd` are extracted from random positions in Sketchmap minima.
        Trajectories with `2ub` without `G` are CG trajectories.
        Trajectories without date-code are also all-atom from extended. These simulations were started to increase the number of all-atom structures.
        """
        text_ax.text(0.05, 0.95, text, transform=text_ax.transAxes, fontsize=14,
                verticalalignment='top')


        # scatter
        scatter_all_ax.scatter(*self.aa_trajs[ubq_site].lowd[::10].T, s=1, c='C0', label='aa')
        scatter_all_ax.scatter(*self.cg_trajs[ubq_site].lowd[::10].T, s=1, c='C1', label='cg')
        scatter_all_ax.legend(loc=2)

        pyemma.plots.plot_free_energy(*self.trajs[ubq_site].lowd.T, cmap='turbo', cbar=False, ax=dense_all_ax, nbins=nbins)

        aa_H, xedges, yedges = np.histogram2d(*self.aa_trajs[ubq_site].lowd[:, :2].T, bins=nbins)
        xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
        ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)
        X, Y = np.meshgrid(xcenters, ycenters)
        aa_H[aa_H > 0] = 1
        out = copy.deepcopy((aa_H))
        for i in range(nbins - 1):
            for j in range(nbins - 1):
                if any([aa_H[i - 1, j], aa_H[i + 1, j], aa_H[i, j - 1], aa_H[i, j + 1]]):
                    out[i, j] = 1
        cg_vs_aa_contour_ax.contour(X, Y, out, levels=2, cmap='Blues')
        cg_H, xedges, yedges = np.histogram2d(*self.cg_trajs[ubq_site].lowd[:, :2].T, bins=nbins)
        xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
        ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)
        X, Y = np.meshgrid(xcenters, ycenters)
        cg_H[cg_H > 0] = 1
        out = copy.deepcopy((cg_H))
        for i in range(nbins - 1):
            for j in range(nbins - 1):
                if any([aa_H[i - 1, j], aa_H[i + 1, j], aa_H[i, j - 1], aa_H[i, j + 1]]):
                    out[i, j] = 1
        cg_vs_aa_contour_ax.contour(X, Y, out, levels=2, cmap='Greys')

        color_palette = sns.color_palette('deep', max(self.trajs[ubq_site].cluster_membership) + 1)
        cluster_colors = [(*color_palette[x], 1) if x >= 0
                          else (0.5, 0.5, 0.5, 0.1)
                          for x in self.trajs[ubq_site].cluster_membership]
        clustered_ax.scatter(*self.trajs[ubq_site].lowd[:, :2].T, s=1, c=cluster_colors)
        ind_ = np.where(self.trajs[ubq_site].cluster_membership == cluster_num)[0]
        lower_left = np.min(self.trajs[ubq_site].lowd[ind_], axis=0)
        upper_right = np.max(self.trajs[ubq_site].lowd[ind_], axis=0)
        width = upper_right[0] - lower_left[0]
        height = upper_right[1] - lower_left[1]
        rect = mpl.patches.Rectangle(lower_left, width, height, fill=False, ec='red', ls='--')
        clustered_ax.add_artist(rect)

        where = np.where(self.trajs[ubq_site].cluster_membership == cluster_num)[0]
        data = self.trajs[ubq_site].lowd
        ax4 = scatter_cluster_ax
        divider = make_axes_locatable(ax4)
        axHistx = divider.append_axes("top", size=1.2, pad=0.1)#, sharex=ax1)
        axHisty = divider.append_axes("right", size=1.2, pad=0.1)#, sharey=ax1)
        spines = [k for k in axHistx.spines.values()]
        spines[1].set_linewidth(0)
        spines[3].set_linewidth(0)
        axHistx.set_xticks([])
        H, edges, patches = axHistx.hist(data[:, 0][where], bins=50)
        axHistx.set_ylabel('count')
        axHistx.set_title('Scatter of Cluster')
        # y hist
        spines = [k for k in axHisty.spines.values()]
        spines[1].set_linewidth(0)
        spines[3].set_linewidth(0)
        axHisty.set_yticks([])
        H, edges, patches = axHisty.hist(data[:, 1][where], bins=50, orientation='horizontal')
        axHisty.set_xlabel('count')
        # scatter data
        ax4.scatter(x=data[where,0], y=data[where,1], s=1)
        spines = [k for k in ax4.spines.values()]
        spines[3].set_linewidth(0)
        spines[1].set_linewidth(0)
        ax4.set_xlabel('x in a.u.')
        ax4.set_ylabel('y in a.u.')

        # add geom and rmsd centroid
        scatter_cluster_ax.scatter(*self.trajs[ubq_site].lowd[rmsd_centroid_index], marker='H', c='red', label='rmsd centroid')
        scatter_cluster_ax.scatter(*self.trajs[ubq_site].lowd[geom_centroid_index], marker='D', c='k', label='geom centroid')
        scatter_cluster_ax.legend()

        try:
            view, dummy_traj = em.misc.clustering.gen_dummy_traj(self.aa_trajs[ubq_site], 0, align_string='name CA and resid > 76', superpose=self.aa_trajs[ubq_site][0][0].traj, shorten=True,)
            dummy_traj.save_pdb('/tmp/tmp.pdb')
            image = em.plot.plotting.render_vmd('/tmp/tmp.pdb', drawframes=True, scale=1.5)
        finally:
            os.remove('/tmp/tmp.pdb')
        render_ax.imshow(image)

        ax_pairs = [[sPRE_prox_ax, sPRE_dist_ax], [N15_prox_ax, N15_dist_ax], [sPRE_vs_prox_ax, sPRE_vs_dist_ax]]

        for i, ax_pair in enumerate(ax_pairs):
            ax1, ax2 = ax_pair
            if i == 0: # should be 0
                fake_legend_dict1 = {'line': [{'label': 'sPRE NMR experiment', 'color': 'C1'}],
                                     'hatchbar': [
                                         {'label': 'Fast exchanging residues', 'color': 'lightgrey', 'alpha': 0.3,
                                          'hatch': '//'}],
                                     'text': [
                                         {'label': 'Residue used for normalization', 'color': 'red', 'text': 'Ile3'}]}

                (ax1, ax2) = plot_line_data((ax1, ax2), self.df_obs, {'rows': 'sPRE', 'cols': ubq_site})
                (ax1, ax2) = plot_hatched_bars((ax1, ax2), self.fast_exchangers, {'cols': ubq_site}, color='k')
                (ax1, ax2), color_1 = plot_confidence_intervals((ax1, ax2), self.df_comp_norm, {'rows': 'sPRE', 'cols': ubq_site}, cbar_axis=0)
                (ax1, ax2), color_2 = plot_confidence_intervals((ax1, ax2), self.df_comp_norm, {'rows': 'sPRE', 'cols': ubq_site}, cmap='Oranges', trajs=self.aa_trajs[ubq_site], cluster_num=0, cbar_axis=1, outliers_offset=0.25)

                fake_legend_dict2 = {'envelope': [{'label': 'sPRE calculation with quantiles', 'color': 'lightblue'},
                                                  {'label': f'sPRE for cluster {cluster_num} with quantile', 'color': 'bisque'}],
                                     'line': [{'label': 'median sPRE calculation', 'color': color_1},
                                              {'label': f'median sPRE for cluster {cluster_num}', 'color': color_2}],
                                     'scatter': [{'label': 'outliers sPRE calculation', 'color': color_1, 'marker': 'o'},
                                                 {'label': f'outliers sPRE for cluster {cluster_num}', 'color': color_2, 'marker': 'o'}]}

                ax1 = fake_legend(ax1, fake_legend_dict1)
                ax2 = fake_legend(ax2, fake_legend_dict2)


            if i == 1: # should be 1
                (ax1, ax2) = plot_line_data((ax1, ax2), self.df_obs, {'rows': f'15N_relax_{mhz}', 'cols': ubq_site})
                (ax1, ax2), color_1 = try_to_plot_15N((ax1, ax2), ubq_site, mhz, cbar_axis=0)
                (ax1, ax2), color_2 = try_to_plot_15N((ax1, ax2), ubq_site, mhz, cmap='Oranges',
                                                      trajs=self.aa_trajs[ubq_site], cluster_num=0,
                                                      cbar_axis=1, outliers_offset=0.25)

                fake_legend_dict = {'envelope': [{'label': f'15N {mhz} calculation with quantiles', 'color': 'lightblue'}],
                                     'line': [{'label': f'15N {mhz} NMR experiment', 'color': color_1},
                                              {'label': f'median 15N {mhz} calculation', 'color': color_2}],
                                    'scatter': [{'label': f'outliers 15N {mhz} calculation', 'color': color_1, 'marker': 'o'},
                                                {'label': f'outliers 15N {mhz} for cluster {cluster_num}', 'color': color_2,
                                                 'marker': 'o'}]}

                ax1 = fake_legend(ax1, fake_legend_dict)


            if i == 2: # should be 2
                (ax1, ax2), color = plot_confidence_intervals((ax1, ax2), self.df_comp_norm,
                                                                {'rows': 'sPRE', 'cols': ubq_site}, cmap='Oranges',
                                                                trajs=self.aa_trajs[ubq_site], cluster_num=0,)

                fake_legend_dict = {'envelope': [{'label': f'sPRE for cluster {cluster_num} with quantile',
                                                   'color': 'bisque'}],
                                     'line': [{'label': f'median sPRE for cluster {cluster_num}', 'color': color},
                                              {'label': f'sPRE of rmsd center of cluster {cluster_num}', 'color': 'red'},
                                              {'label': f'sPRE of geom. center of cluster {cluster_num}', 'color': 'black'}]}

                fake_legend_dict2 = {'scatter': [{'label': f'outliers sPRE for cluster {cluster_num}', 'color': color, 'marker': 'o'}]}

                ax1 = fake_legend(ax1, fake_legend_dict)
                ax2 = fake_legend(ax2, fake_legend_dict2)

                (ax1, ax2) = plot_single_struct_sPRE((ax1, ax2), self.trajs[ubq_site].get_single_frame(int(rmsd_centroid_index)), self.norm_factors[ubq_site], ubq_site=ubq_site, color='red')
                (ax1, ax2) = plot_single_struct_sPRE((ax1, ax2), self.trajs[ubq_site].get_single_frame(int(geom_centroid_index)), self.norm_factors[ubq_site], ubq_site=ubq_site, color='k')


        # add more fancy stuff to axes
        for i, ax_pair in enumerate(ax_pairs):
            ax1, ax2 = ax_pair
            for ax, centers in zip([ax1, ax2], [self.centers_prox, self.centers_dist - 76]):
                ax = add_sequence_to_xaxis(ax)
                ax = color_labels(ax, positions=centers)
                if i == 0 or i == 2:
                    ax.set_ylabel(r'sPRE in $\mathrm{mM^{-1}ms^{-1}}$')

        print(f"Saving image at {out_file}")
        plt.tight_layout()
        plt.savefig(out_file)
        plt.savefig(out_file.replace('.png', '.pdf'))

    def fitness_assessment(self, overwrite=False):
        json_savefile = os.path.join(self.analysis_dir, f'quality_factors.json')
        self.quality_factor_means = {ubq_site: [] for ubq_site in self.ubq_sites}
        if not os.path.isfile(json_savefile):
            self.all_quality_factors = {ubq_site: {} for ubq_site in self.ubq_sites}
        else:
            with open(json_savefile, 'r') as f:
                self.all_quality_factors = json.load(f)

        for ubq_site in self.ubq_sites:
            image_file = os.path.join(self.analysis_dir, f'quality_factors_{ubq_site}.png')
            # get cluster means
            df = self.df_comp_norm.copy()
            df = df.fillna(0)
            df_obs = self.df_obs.copy()
            obs = df_obs[df_obs.index.str.contains('sPRE')][ubq_site].values

            df = df[df['ubq_site'] == ubq_site]
            sPRE_ind = [i for i in df.columns if 'sPRE' in i and 'norm' in i]

            # put cluster membership into df
            cluster_membership_sPRE = []
            for i, traj in enumerate(self.trajs[ubq_site]):
                frames = df[df['traj_file'] == traj.traj_file]['frame'].values
                cluster_membership_sPRE.append(traj.cluster_membership[frames])
            cluster_membership_sPRE = np.hstack(cluster_membership_sPRE)
            df['cluster_membership'] = cluster_membership_sPRE

            # calculcate the per-cluster per-residue median
            cluster_means = []
            for cluster_num in np.unique((df['cluster_membership'])):
                if cluster_num == -1:
                    continue
                mean = np.median(df[sPRE_ind][df['cluster_membership'] == cluster_num], axis=0)
                cluster_means.append(mean)
            cluster_means = np.vstack(cluster_means)

            fast_exchange = self.fast_exchangers[ubq_site].values

            n_clust = self.trajs[ubq_site].cluster_membership.max() + 1
            for no_of_considered_clusters in range(2, n_clust + 1):
                combinations = itertools.combinations(range(n_clust), no_of_considered_clusters)
                if str(no_of_considered_clusters) in self.all_quality_factors[ubq_site] and not overwrite:
                    print(f"{no_of_considered_clusters} already in json")
                    continue
                else:
                    self.all_quality_factors[ubq_site][str(no_of_considered_clusters)] = []
                for combination in combinations:
                    combination = np.asarray(combination)
                    solv = scipy.optimize.nnls(cluster_means[combination].T[~fast_exchange], obs[~fast_exchange])[0]
                    result = np.sum(solv * cluster_means[combination].T, 1)
                    diff = float(np.mean(np.abs(result[~fast_exchange] - obs[~fast_exchange])))
                    self.all_quality_factors[ubq_site][str(no_of_considered_clusters)].append(diff)
                print("Saving json")
                with open(json_savefile, 'w') as f:
                    json.dump(self.all_quality_factors, f)

            if not os.path.isfile(image_file) or overwrite:
                plt.close('all')
                for key, value in self.all_quality_factors[ubq_site].items():
                    self.quality_factor_means[ubq_site].append(np.mean(value))
                data = np.array(self.quality_factor_means[ubq_site])
                fig, ax = plt.subplots()
                ax.plot(np.arange(len(data)), data)
                ax.set_title(f"Quality factor for n clusters for {ubq_site}")
                ax.set_xlabel("Number of considered clusters")
                ax.set_ylabel("Mean abs difference between observed and caluclated")
                plt.savefig(image_file)


    def load_specific_checkpoint(self, step=None):
        for i, ubq_site in enumerate(self.ubq_sites):
            if step is not None:
                checkpoint = os.path.join(self.analysis_dir, f'runs/{ubq_site}/production_run_tf2/saved_model_{step}.model_*')
            else:
                train_path = os.path.join(self.analysis_dir, f'runs/{ubq_site}/production_run_tf2/')
                checkpoints = glob.glob(f'{train_path}*encoder')
                checkpoint = list(sorted(checkpoints, key=ckpt_step))[-1].replace('model_encoder', '*')
            e_map = em.EncoderMap.from_checkpoint(checkpoint)
            aa_lowd = e_map.encode(self.aa_trajs[ubq_site].highd)
            cg_lowd = e_map.encode(self.cg_trajs[ubq_site].highd)
            self.aa_trajs[ubq_site].load_CVs(aa_lowd, attr_name='lowd')
            self.cg_trajs[ubq_site].load_CVs(cg_lowd, attr_name='lowd')