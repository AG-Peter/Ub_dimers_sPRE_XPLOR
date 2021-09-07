################################################################################
# Imports
################################################################################


import mdtraj as md
import numpy as np
import parmed as pmd
from parmed.charmm import CharmmPsfFile
import pandas as pd
import os, sys, glob, multiprocessing, copy, ast, subprocess, yaml, shutil, scipy
from ..proteins.proteins import get_column_names_from_pdb
from io import StringIO
from joblib import Parallel, delayed
from .custom_gromacstopfile import CustomGromacsTopFile
from xplor.argparse.argparse import YAML_FILE, write_argparse_lines_from_yaml_or_dict
from ..misc import get_local_or_proj_file, get_iso_time
from .psf_parser import PSFParser
from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile
import builtins
from .functions import is_aa_sim
import encodermap as em

################################################################################
# Globals
################################################################################


__all__ = ['EncodermapSPREAnalysis']


################################################################################
# Functions
################################################################################

class EncodermapSPREAnalysis:
    """ A python singleton """

    class __impl:
        """ Implementation of the singleton interface """

        def get_id(self):
            return id(self)

        def set_settings(self, ubq_sites, simdirs=['/home/andrejb/Research/SIMS/2017_*',
                                                   '/home/kevin/projects/molsim/diUbi_aa/'],
                         analysis_files='/mnt/data/kevin/xplor_analysis_files'):
            """Sets the ubq_sites of the singleton.

            Args:
                ubq_sites (list[str]): The ubq_sites to be looked at during this run.

            """
            self.ubq_sites = ubq_sites
            self.simdirs = simdirs
            self.analysis_files = analysis_files
            if not os.path.isdir(self.analysis_files):
                os.makedirs(self.analysis_files)

        def load_trajs(self, overwrite=False, with_cg=True):
            if hasattr(self, 'trajs'):
                if not overwrite and bool(self.trajs):
                    print("trajs already loaded")
                    return

            # setting some schtuff
            self.trajs = {}
            self.aa_traj_files = {ubq_site: [] for ubq_site in self.ubq_sites}
            self.aa_references = {ubq_site: [] for ubq_site in self.ubq_sites}
            self.cg_traj_files = {ubq_site: [] for ubq_site in self.ubq_sites}
            self.cg_references = {ubq_site: [] for ubq_site in self.ubq_sites}

            print("Loading trajs")
            for i, ubq_site in enumerate(self.ubq_sites):
                for j, dir_ in enumerate(glob.glob(f"{self.simdirs[0]}{ubq_site}_*") + glob.glob(f'{self.simdirs[1]}{ubq_site.upper()}_*')):
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

                self.trajs[ubq_site] = (em.Info_all(trajs = self.aa_traj_files[ubq_site],
                                                   tops = self.aa_references[ubq_site],
                                                   basename_fn=lambda x: x.split('/')[-2],
                                                   common_str=[ubq_site, ubq_site.upper()]) +
                                       em.Info_all(trajs = self.cg_traj_files[ubq_site],
                                                   tops = self.cg_references[ubq_site],
                                                   basename_fn=lambda x: x.split('/')[-2],
                                                   common_str=[ubq_site]))


        def load_highd(self, overwrite=False):
            if all([hasattr(self.trajs[ubq_site], 'highd') for ubq_site in self.ubq_sites]) and not overwrite:
                print("highd already loaded")
                return
            for i, ubq_site in enumerate(self.ubq_sites):
                aa_highd_file = os.path.join(self.analysis_files, f'highd_rwmd_aa_{ubq_site}.npy')
                print(aa_highd_file)




        def spam(self):
            """ Test method, return singleton id """
            return id(self)

    # storage for the instance reference
    __instance = None

    def __init__(self):
        """ Create singleton instance """
        # Check whether we already have an instance
        if EncodermapSPREAnalysis.__instance is None:
            # Create and remember instance
            EncodermapSPREAnalysis.__instance = EncodermapSPREAnalysis.__impl()

        # Store instance reference as the only member in the handle
        self.__dict__['_EncodermapSPREAnalysis__instance'] = EncodermapSPREAnalysis.__instance

    def __getattr__(self, attr):
        """ Delegate access to implementation """
        return getattr(self.__instance, attr)

    def __setattr__(self, attr, value):
        """ Delegate access to implementation """
        return setattr(self.__instance, attr, value)
