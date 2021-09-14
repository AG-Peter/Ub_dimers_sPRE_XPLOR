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
# analysis.analyze()
# analysis.load_xplor_data(overwrite=True)
analysis.cluster_analysis(overwrite_image=True)
# analysis.plot_cluster(0, 'k6', overwrite=True)
# analysis.fitness_assessment(overwrite=False)

# %% Test plot
# plt.close('all')
# fig, axes = plt.subplots(ncols=2)
# xplor.nmr_plot.try_to_plot_15N(axes, 'k6', 800, trajs=analysis.aa_trajs['k6'], cluster_num=0)
