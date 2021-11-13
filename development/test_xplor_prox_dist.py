# %% Imports
import xplor
import os
import mdtraj as md
import glob
import pandas as pd

# %% load a diUbi aa traj and check, where there are GLQ and LYQ
file = glob.glob('/home/andrejb/Research/SIMS/2017_04_27_G_2ub_k6_01_01/*.gro')[0]
andrej_traj = md.load(file)
kevin_traj = md.load('/home/kevin/projects/molsim/diUbi_aa/K6_0/init.gro')
print(andrej_traj, kevin_traj)

# %% iterate over resiudes
for r in andrej_traj.top.residues:
    if 'Q' in r.name:
        print(r.index, r)

for r in kevin_traj.top.residues:
    if 'Q' in r.name:
        print(r.index, r)

# %% get the series
traj = andrej_traj
from xplor.functions.functions import Capturing
from xplor.functions.custom_gromacstopfile import CustomGromacsTopFile

with Capturing() as output:
    top_aa = CustomGromacsTopFile(
        f'/home/andrejb/Software/custom_tools/topology_builder/topologies/gromos54a7-isop/diUBQ_K6/system.top',
        includeDir='/home/andrejb/Software/gmx_forcefields')
traj.top = md.Topology.from_openmm(top_aa.topology)

isopeptide_indices = []
isopeptide_bonds = []
for r in traj.top.residues:
    if r.name == 'GLQ':
        r.name = 'GLY'
        for a in r.atoms:
            if a.name == 'C':
                isopeptide_indices.append(a.index + 1)
                isopeptide_bonds.append(f"{a.residue.name} {a.residue.resSeq} {a.name}")
    if r.name == 'LYQ':
        r.name = 'LYS'
        for a in r.atoms:
            if a.name == 'CQ': a.name = 'CE'
            if a.name == 'NQ':
                a.name = 'NZ'
                isopeptide_indices.append(a.index + 1)
                isopeptide_bonds.append(f"{a.residue.name} {a.residue.resSeq} {a.name}")
            if a.name == 'HQ': a.name = 'HZ1'

series = xplor.functions.get_series_from_mdtraj(traj, file, file, 0, fix_isopeptides=False)

# %% analyse the series
for index, value in series.iteritems():
    if ('prox' in index or 'dist' in index) and 'sPRE' in index:
        print(index, value)

# %% flip prox and dist
from xplor.functions.functions import datetime_windows_and_linux_compatible
import pandas as pd
now = datetime_windows_and_linux_compatible()
new_filename = f'/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package_all/{now}_df_no_conect.csv'
df = pd.read_csv('/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package_all/2021-10-13T16:19:28+02:00_df_no_conect.csv')
df_copy = df.copy()

prox_columns = [i for i in df.columns if 'prox' in i]
dist_columns = [i for i in df.columns if 'dist' in i]

for prox, dist in zip(prox_columns, dist_columns):
    prox_suffix = ' '.join(prox.split()[1:])
    dist_suffix = ' '.join(dist.split()[1:])
    assert prox_suffix == dist_suffix

reindex_dict = {i: j for i, j in zip(prox_columns, dist_columns)}
reindex_dict.update({j: i for i, j in zip(prox_columns, dist_columns)})

df = df.rename(columns=reindex_dict)

print(df.loc[0, 'proximal LEU8 sPRE'])
print(df_copy.loc[0, 'distal LEU8 sPRE'])

df.to_csv(new_filename)
