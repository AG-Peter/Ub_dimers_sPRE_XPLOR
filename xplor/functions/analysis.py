################################################################################
# Imports
################################################################################
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
import os, sys, glob, multiprocessing, copy, ast, subprocess, yaml, shutil, scipy
from .custom_gromacstopfile import CustomGromacsTopFile
import warnings
from scipy.spatial.distance import cdist
from .functions import is_aa_sim, Capturing, normalize_sPRE
import encodermap as em
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
                   'load_xplor_data': False, 'cluster': False}
        ow_dict.update(overwrite_dict)
        self.load_trajs(overwrite=ow_dict['load_trajs'])
        self.load_highd(overwrite=ow_dict['load_highd'])
        self.train_encodermap(overwrite=ow_dict['train_encodermap'])
        self.load_xplor_data(overwrite=ow_dict['load_xplor_data'])
        self.cluster(overwrite=ow_dict['cluster'])

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

        warnings.warn("Currently testing")
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
                parameters.dist_sig_parameters = [6, 12, 10, 1, 2, 10]
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
            if not hasattr(self.trajs[ubq_site], 'cluster_membership'):
                aa_cluster_file = os.path.join(self.analysis_dir, f'cluster_membership_aa_{ubq_site}.npy')
                cg_cluster_file = os.path.join(self.analysis_dir, f'cluster_membership_cg_{ubq_site}.npy')
                if not all([os.path.isfile(file) for file in [aa_cluster_file, cg_cluster_file]]) or overwrite:
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=750, cluster_selection_method='leaf').fit(
                        self.trajs[ubq_site].lowd)
                    if max(clusterer.labels_) > 10:
                        raise Exception(f"Too many clusters {np.unique(clusterer.labels_)}")
                    else:
                        print(f"Found {np.unique(clusterer.labels_)} clusters.")
                    aa_index = np.arange(self.aa_trajs[ubq_site].n_frames)
                    cg_index = np.arange(self.aa_trajs[ubq_site].n_frames, np.arange(self.cg_trajs[ubq_site].n_frames))
                    print(aa_index.shape, self.aa_trajs[ubq_site].n_frames, cg_index.shape, self.cg_trajs[ubq_site].n_frames)
                    self.trajs[ubq_site].load_CVs(clusterer.labels_, 'cluster_membership')
                else:
                    self.aa_trajs[ubq_site].load_CVs(np.load(aa_cluster_file), 'cluster_membership')
                    self.cg_trajs[ubq_site].load_CVs(np.load(cg_cluster_file), 'cluster_membership')
            else:
                print('Cluster_membership already in trajs')

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

    def plot_lowd(self, ubq_site, nbins=100):
        plt.close('all')
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5), sharex=True, sharey=True)


        ax1.scatter(*self.aa_trajs[ubq_site].lowd[::10, :2].T, s=1, label='aa')
        ax1.scatter(*self.cg_trajs[ubq_site].lowd[::10, :2].T, s=1, label='cg')
        ax1.legend()
        pyemma.plots.plot_free_energy(*self.trajs[ubq_site].lowd[:, :2].T, cmap='turbo', ax=ax2, nbins=nbins, levels=50, cbar=False)

        aa_H, xedges, yedges = np.histogram2d(*self.aa_trajs[ubq_site].lowd[:, :2].T, bins=nbins)
        xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
        ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)
        X, Y = np.meshgrid(xcenters, ycenters)

        aa_H[aa_H > 0] = 1
        ax3.contour(X, Y, aa_H, levels=2, cmap='Blues', label='aa')

        cg_H, xedges, yedges = np.histogram2d(*self.cg_trajs[ubq_site].lowd[:, :2].T, bins=nbins)
        xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
        ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)
        X, Y = np.meshgrid(xcenters, ycenters)

        cg_H[cg_H > 0] = 1
        ax3.contour(X, Y, cg_H, levels=2, cmap='plasma', label='aa')


        plt.savefig('/home/kevin/tmp.png')