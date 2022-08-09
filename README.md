# xplor_functions

For functions (and classes) that might be useful when calling XPLOR's python interpreter from within normal python.

## Installation

To install the python helper functions call:

```bash
pip install -e .
```

This will add a python-package called `xplor` to your importable python packages.

## Installation of XPLOR

As this python package heavily relies on the software-package XPLOR, you need to install XPLOR, by following the installation instructions on the XPLOR homepage. The xplor installation comes with a pyXplor executable, that will be used as a secondary python kernel.

## Analysis pipeline

The Analysis pipeline starts with MD simulations, which will be made available upon request. A simulation is defined by a `.pdb` (topology) and `.xtc` (trajectory) file. Both of these files should reside in their own directory. The monolithic function `xplor.functions.parallel_xplor()` takes the argument `simdir` which should point to the directory containing all the other simulation directories.

```python
xplor.functions.parallel_xplor(['k6', 'k29', 'k33'], df_outdir='.', suffix='_df.csv', parallel=True)
```
Starting from MD simulations of the three ubiquitylation sites (`K6`, `K29`, and `K33`).

The analysis of the simulation data was done with the 
