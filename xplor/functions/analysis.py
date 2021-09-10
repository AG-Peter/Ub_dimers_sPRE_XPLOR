################################################################################
# Imports
################################################################################
import matplotlib.pyplot as plt
import seaborn as sns
import mdtraj as md
import numpy as np
import pandas as pd
import os, sys, glob, multiprocessing, copy, ast, subprocess, yaml, shutil, scipy
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
                   'cluster_analysis': False}
        ow_dict.update(overwrite_dict)
        self.load_trajs(overwrite=ow_dict['load_trajs'])
        self.load_highd(overwrite=ow_dict['load_highd'])
        self.train_encodermap(overwrite=ow_dict['train_encodermap'])
        self.load_xplor_data(overwrite=ow_dict['load_xplor_data'])
        self.cluster(overwrite=ow_dict['cluster'])
        self.cluster_analysis(overwrite=ow_dict['cluster_analysis'])
        print("Finish plotting")
        # for ubq_site in self.ubq_sites:
        #     self.plot_lowd(ubq_site, overwrite=ow_dict['plot_lowd'])

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
                        csv='/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package/2021-07-23T16:49:44+02:00_df_no_conect.csv'):
        df_comp = pd.read_csv(csv, index_col=0)
        if not 'ubq_site' in df_comp.keys():
            df_comp['ubq_site'] = df_comp['traj_file'].map(get_ubq_site)
        if hasattr(self, 'df_comp_norm') and not overwrite:
            print("XPLOR data already loaded")
            return
        self.df_obs = get_observed_df(['k6', 'k29', 'k33'])
        self.fast_exchangers = get_fast_exchangers(['k6', 'k29', 'k33'])
        self.in_secondary = get_in_secondary(['k6', 'k29', 'k33'])
        self.df_comp_norm, self.centers_prox, self.centers_dist = normalize_sPRE(df_comp, self.df_obs)

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

    def cluster_analysis(self, overwrite=False):
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
                rmsd_centroid_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_linear_coefficient_{coeff}_ensemble_contribution_{percentage:.1f}_percent.pdb')
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
                    np.save(rmsd_centroid_index_file, rmsd_centroid_index)
                else:
                    rmsd_centroid = md.load(rmsd_centroid_file)
                    rmsd_centroid_index = np.load(rmsd_centroid_index_file)

                # geometric centroid
                geom_centroid_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_geom_centroid.pdb')
                geom_centroid_index_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_geom_centroid.npy')
                if not os.path.isfile(geom_centroid_file) or overwrite:
                    geom_center = np.mean(self.aa_trajs[ubq_site].lowd[aa_cluster_points_ind], axis=0)
                    sums = np.sum((self.aa_trajs[ubq_site].lowd[aa_cluster_points_ind] - geom_center) ** 2, axis=1)
                    geom_center_index = aa_cluster_points_ind[np.argmin(sums)]
                    geom_centroid = self.aa_trajs[ubq_site].get_single_frame(geom_center_index).traj
                    geom_centroid.save_pdb(geom_centroid_file)
                    np.save(geom_centroid_index_file, geom_center_index)
                else:
                    geom_centroid = md.load(geom_centroid_file)
                    geom_centroid_index = np.load(geom_centroid_index_file)

                # self.plot_cluster(ubq_site, cluster_num, rmsd_centroid_index, geom_centroid_index, overwrite=overwrite)
                break

    def plot_cluster(self, ubq_site, cluster_num, rmsd_centroid_index, geom_centroid_index, overwrite=False):
        plt.close('all)')

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

        axes = None

        plt.show()


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

    def plot_lowd(self, ubq_site, overwrite=False, nbins=100, outfile=None):
        """Plots all the accumulated data:

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

        """
        if outfile is None:
            outfile = os.path.join(self.analysis_dir, f'lowd_plots_{ubq_site}.png')
        if not os.path.isfile(outfile) or overwrite:
            plt.close('all')
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5), sharex=True, sharey=True)

            color_palette = sns.color_palette('deep', max(self.trajs[ubq_site].cluster_membership) + 1)
            cluster_colors = [color_palette[x] if x >= 0
                              else (0.5, 0.5, 0.5)
                              for x in self.trajs[ubq_site].cluster_membership]

            ax1.scatter(*self.trajs[ubq_site].lowd[:, :2].T, s=1, c=cluster_colors)
            pyemma.plots.plot_free_energy(*self.trajs[ubq_site].lowd[:, :2].T, cmap='turbo', ax=ax2, nbins=nbins, levels=50, cbar=False)

            aa_H, xedges, yedges = np.histogram2d(*self.aa_trajs[ubq_site].lowd[:, :2].T, bins=nbins)
            xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
            ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)
            X, Y = np.meshgrid(xcenters, ycenters)

            aa_H[aa_H > 0] = 1
            ax3.contour(X, Y, aa_H, levels=2, cmap='Blues')

            cg_H, xedges, yedges = np.histogram2d(*self.cg_trajs[ubq_site].lowd[:, :2].T, bins=nbins)
            xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
            ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)
            X, Y = np.meshgrid(xcenters, ycenters)

            cg_H[cg_H > 0] = 1
            ax3.contour(X, Y, cg_H, levels=2, cmap='plasma')


            plt.savefig(outfile)
        else:
            print("File already present")