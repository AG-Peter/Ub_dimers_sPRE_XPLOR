# %% Imports
import pyemma
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtraj as md
import encodermap.encodermap_tf1 as em1

import sys, re, glob, os, itertools

# %% Create data
from xplor.functions.functions import is_aa_sim
from xplor.functions.custom_gromacstopfile import CustomGromacsTopFile

trajs = [[], [], []]
refs = []
simdir = '/home/andrejb/Research/SIMS/2017_*'

for i, ubq_site in enumerate(['k6', 'k29', 'k33']):
    for j, dir_ in enumerate(glob.glob(f"{simdir}{ubq_site}_*")):
        traj_file = dir_ + '/traj_nojump.xtc'
        if not is_aa_sim(traj_file):
            print(f"{traj_file} is not an AA sim")
            continue
        trajs[i].append(dir_ + '/traj_nojump.xtc')
        if len(refs) <= i:
            refs.append(dir_ + '/start.pdb')

# %%

print(len(trajs))
print(refs)

# %% Data generation with pyemma
out = []
for i, ubq_site in enumerate(['k6', 'k29', 'k33']):
    feat = pyemma.coordinates.featurizer(refs[i])
    feat.add_distances_ca(periodic=False)
    out.append(pyemma.coordinates.load(trajs[i], feat))

# %%

for o in out:
    arr = np.vstack(o)
    print(arr.shape)

