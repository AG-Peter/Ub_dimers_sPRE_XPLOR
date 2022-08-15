# Ub_dimers_sPRE_XPLOR

Repository containing the code for an upcoming publication.

## System requirements

OS: Linux

Software dependencies can be found in `env.yml`. Rebuild the environment with `conda env create -f environment.yml`.

Also, encodermap is needed. Install with `pip install encodermap`.

XPLOR version 3.3

## Installation

The paths in the repository have been adjusted to use the repository as a base. To run the code yourself, you need to first clone the repository:

```bash
git clone git@github.com:AG-Peter/Ub_dimers_sPRE_XPLOR.git && cd Ub_dimers_sPRE_XPLOR
```

Next, you need to install the python package containing the helper functions:

```bash
pip install -e .
```

This will add a python-package called `xplor` to your importable python packages. The install-time of this is less than a minute.

### Sample data

A small sample dataset of MD simulations is available under https://sawade.io/Ub_dimers_sPRE_XPLOR/ Please refer to the publication for simulation details and software versions. To get the sample data to test the code yourself, run these commands inside the `Ub_dimers_sPRE_XPLOR` repository (the download can take a while):

```bash
mkdir molsim && cd molsim
wget --recursive --no-parent https://sawade.io/Ub_dimers_sPRE_XPLOR/
mv sawade.io/Ub_dimers_sPRE_XPLOR/K* .
rm -rf sawade.io/
```

## Installation of XPLOR

As this python package heavily relies on the software-package XPLOR, you need to install XPLOR, by following the installation instructions on the XPLOR homepage. The xplor installation comes with a pyXplor executable, that will be used as a secondary python kernel. To make your xplor installation work with the helper functions set the environment variable `PYXPLOR_EXECUTABLE` before calling any other code (i.e. during your imports):

```python
import numpy as np
import os
os.environ['PYXPLOR_EXECUTABLE'] = '/path/to/pyxplor/executable'
import xplor
xplor.functions.parallel_xplor(['k6'])
```

The same applies to XPLOR's `pdb2psf` program. Set it with:
```python
import os
os.environ['PDB2PSF_EXECUTABLE'] = '/path/to/pdb2psf/executable'
```

# Run this Repo

## Analysis pipeline

The Analysis pipeline starts with MD simulations. A small subset is available and the full set can be made available upon request. A simulation is defined by a `.pdb` (topology) and `.xtc` (trajectory) file. Both of these files should reside in their own directory. The monolithic function `xplor.functions.parallel_xplor()` takes the argument `simdir` which should point to the directory containing all the other simulation directories (the determination of this directory is somewhat smart and uses the location from which xplor was installed. eg. /home/user/git/xplor/molsim or /home/user/python_packages/xplor/molsim). It is also recommended to choose a very large subsample (10 000 or even more, otherwise your PC will be busy for quite some time).

```python
df = xplor.functions.parallel_xplor(['k6', 'k29', 'k33'], subsample=10000, parallel=True)
```

On a machine with a 4-core CPU this command took 5 minutes, 30 seconds to complete. The resulting dataframe can be used to work through the analysis and image creation in the jupyter notebook `development/analysis.ipynb`.

Starting from MD simulations of the three ubiquitylation sites (`K6`, `K29`, and `K33`).

The analysis of the simulation data was done with the 

## HDBSCAN

Using HDBSCAN on the full dataset might use a substantial amount of system memory.
