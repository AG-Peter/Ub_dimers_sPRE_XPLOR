# %% Imports
import pyemma
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtraj as md
import encodermap.encodermap_tf1 as em1

import sys, re, glob, os, itertools, pyemma, json

%matplotlib tk

# %% Create data
from xplor.functions.functions import is_aa_sim
from xplor.functions.custom_gromacstopfile import CustomGromacsTopFile

trajs = [[], [], []]
refs = []
out = []
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

    if not os.path.isfile(f'xplor/data/highd_{ubq_site}.npy'):
        feat = pyemma.coordinates.featurizer(refs[i])
        feat.add_distances_ca(periodic=False)
        out.append(np.vstack(pyemma.coordinates.load(trajs[i], feat)))

        np.save(f'xplor/data/highd_{ubq_site}.npy', out[i])
    else:
        out.append(np.load(f'xplor/data/highd_{ubq_site}.npy'))

# %% Get data from last working encodermap
for root, dirs, files in os.walk("/home/kevin/projects/tobias_schneider/"):
    for file in files:
        if file.endswith(".json"):
            print(os.path.join(root, file))
            with open(os.path.join(root, file)) as f:
                _ = json.load(f)
            if 'dist_sig_parameters' in _:
                print(_['dist_sig_parameters'])



# %% Plot encodermap
em1.plot.distance_histogram(out[0][::1000], float('inf'), [40, 12, 10, 1, 2, 5])
plt.show()


# %% Train Encodermap
ubq_site = 'k6'

parameters = em1.Parameters()
parameters.main_path = em1.misc.run_path(f'runs/{ubq_site}')
parameters.n_neurons = [500, 500, 500, 2]
parameters.activation_functions = ['', 'tanh', 'tanh', 'tanh', '']
parameters.periodicity = float('inf')
parameters.dist_sig_parameters =
parameters.__dict__

