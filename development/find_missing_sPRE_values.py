# %% Imports

import os, re, sys, glob
import xplor
import numpy as np
import mdtraj as md
import pandas as pd
import tempfile

# %% A function to compare all dfs from all directories
from xplor.checks import determine_changed_dfs
determine_changed_dfs()

# %% A function to analyse what went wrong
from xplor.checks import what_went_wrong
what_went_wrong()

