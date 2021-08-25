# %% Imports
import xplor
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
import mdtraj as md
import glob, os
import parmed as pmd

# %%
test = 'tmp_traj_nojump_frame_95_hash_481079684092461366.psf'
print(test.split('_')[:5])


# %% List dirs
# glob.glob('/home/andrejb/Research/SIMS/2017_04_27_G_2ub_*/')
os.listdir('/home/andrejb/Research/SIMS/2017_04_27_G_2ub_m1_01_01/')


# %% Test psf with M1 linked diUBQ
pdb = '/home/andrejb/Research/SIMS/2017_04_27_G_2ub_m1_01_01/start.pdb'
top = '/home/andrejb/Research/SIMS/2017_04_27_G_2ub_m1_01_01/traj.top'
xtc = '/home/andrejb/Research/SIMS/2017_04_27_G_2ub_m1_01_01/traj_nojump.xtc'

xplor.functions.test_conect(xtc, pdb, remove_after=False)

# %% try some more stuff anew
pdb = '/home/andrejb/Research/SIMS/2017_04_27_G_2ub_k6_01_01/start.pdb'
top = '/home/andrejb/Research/SIMS/2017_04_27_G_2ub_k6_01_01/traj.top'
xtc = '/home/andrejb/Research/SIMS/2017_04_27_G_2ub_k6_01_01/traj_nojump.xtc'

xplor.functions.test_conect(xtc, pdb, remove_after=True, ast_print=0)

# %% This works without problems. Something else must cause the error:
# check the files
from xplor.functions.functions import check_pdb_and_psf_integrity
pdb_file = '/home/kevin/git/xplor_functions/xplor/scripts/2021-08-14_crash_tmp_traj_nojump_frame.pdb'
psf_file = '/home/kevin/git/xplor_functions/xplor/scripts/2021-08-14_crash_tmp_traj_nojump_frame.psf'
check_pdb_and_psf_integrity(pdb_file, psf_file)

# %% The atom HB1 in LEU8 is not named correclty
# In the psf: HB1
# In the pdb: HHB1
# Why is that?
# Origin: 2017_04_27_G_2ub_k6_01_02, frame 475
# Fixed with new _rename_atoms_according_to_charmm function
from xplor.functions.functions import Capturing
from xplor.functions.custom_gromacstopfile import CustomGromacsTopFile

pdb = '/home/andrejb/Research/SIMS/2017_04_27_G_2ub_k6_01_02/start.pdb'
xtc = '/home/andrejb/Research/SIMS/2017_04_27_G_2ub_k6_01_02/traj_nojump.xtc'

frame = md.load_frame(xtc, 475, top=pdb)

gromos_top_file = f'/home/andrejb/Research/DEVA/2017_04_27_UBQ_TOPOLOGIES/top_G54A7/diUBQ/k6_01/system.top'
with Capturing() as output:
    omm_top = CustomGromacsTopFile(gromos_top_file, includeDir='/home/andrejb/Software/gmx_forcefields')

frame.top = md.Topology.from_openmm(omm_top.topology)

isopeptide_bonds = []
isopeptide_indices = []
for r in frame.top.residues:
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

series = xplor.functions.get_series_from_mdtraj(frame, xtc, pdb, 0, from_tmp=True,
                                                check_fix_isopeptides=True, isopeptide_bonds=isopeptide_bonds)

# %%
pd.options.display.min_rows = 40

# %% Why are the series with isopeptides empty?
_ = xplor.functions.parallel_xplor(['k6', 'k29', 'k33'], from_tmp=True, max_len=20, write_csv=False,
                                  df_outdir='/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package_with_conect/',
                                  suffix='_df.csv', parallel=False, break_after=3)

# %% Prepare the psf files
# xplor.functions.create_psf_files(['k6', 'k29', 'k33'])

# %% make the tbl files
# xplor.functions.parse_input_files.make_15_N_table('data/spre_and_relaxation_data_k6_k29_k33/relaxation_file_ub2_k6.txt',
#                                 out_file='/home/kevin/git/xplor_functions/xplor/data/diUbi_sPRE_k6_w_CONECT.tbl')
# xplor.functions.parse_input_files.make_sPRE_table('data/spre_and_relaxation_data_k6_k29_k33/di_ub2_k6_*_sPRE.txt',
#                                 out_file='/tmp/pycharm_project_13/xplor/data/diUbi_sPRE_k6_w_CONECT.tbl')

# %% Write arparse lines
# xplor.argparse.write_argparse_lines_from_yaml_or_dict(print_argparse=True)

# %% Develop a stringIO compatibility with mdtraj
from xplor.functions.functions import test_mdtraj_stringio
traj_file = xplor.misc.get_local_or_proj_file('data/2017_06_28_GfM_SMmin_rnd_k6_0_start.pdb')
traj = md.load(traj_file)
isopeptide_bonds_for_k6 = ['GLY 76 C', 'LYS 82 NZ']
series = test_mdtraj_stringio(traj, traj_file, traj_file, 0, isopeptide_bonds=isopeptide_bonds_for_k6)

# %% develop a get_series function
# xplor.functions.call_xplor_with_yaml('data/2017_06_28_GfM_SMmin_rnd_k6_0_start.pdb', from_tmp=True)
traj_file = xplor.misc.get_local_or_proj_file('data/2017_06_28_GfM_SMmin_rnd_k6_0_start.pdb')
traj = md.load(traj_file)
isopeptide_bonds_for_k6 = ['GLY 76 C', 'LYS 82 NZ']
series = xplor.functions.get_series_from_mdtraj(traj, traj_file, traj_file, 0, from_tmp=True,
                                                check_fix_isopeptides=True, isopeptide_bonds=isopeptide_bonds_for_k6)

# %% Run on everything
_ = xplor.functions.parallel_xplor(['k6', 'k29', 'k33'], from_tmp=True, max_len=20, write_csv=False,
                                  df_outdir='/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package_with_conect/',
                                  suffix='_df.csv', parallel=False, break_after=3)

# %%
df = _.copy()
cols = [c for c in df.columns if 'distal' in c or 'proximal' in c]
df[cols]

# %%
print(os.listdir())

# %%
series.values

# %% Test the series function
series = xplor.functions.get_series_from_mdtraj(traj, traj_file, traj_file, 0, from_tmp=True)
columns = set(xplor.proteins.get_column_names_from_pdb() + ['traj_file', 'top_file', 'frame', 'time'])
assert columns == set(series.keys()), print(set(series.keys()).difference(columns))

# %% Delete old dataframes
xplor.delete_old_csvs()
