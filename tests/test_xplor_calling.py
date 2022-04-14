import unittest
import mdtraj as md
import numpy as np
from xplor.functions import get_series_from_mdtraj
from xplor.proteins import get_column_names_from_pdb

class TestXPLORCorrectColumns(unittest.TestCase):

    def test_call_xplor(self):
        import subprocess, sys
        cmd = f"xplor_single_struct --help"
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        out, err = process.communicate()
        return_code = process.poll()
        out = out.decode(sys.stdin.encoding)
        err = err.decode(sys.stdin.encoding)
        self.assertTrue('usage: xplor_single_struct' in out, msg="xplor_single_struct script not configured correctly.")
    
    def test_columns(self):
        traj_file = '/home/andrejb/Research/SIMS/2017_06_28_GfM_SMmin_rnd_k6_0/traj_nojump.xtc'
        top_file = '/home/andrejb/Research/SIMS/2017_06_28_GfM_SMmin_rnd_k6_0/start.pdb'
        frame_no = 0
        frame = md.load_frame(traj_file, frame_no, top=top_file)

        series = get_series_from_mdtraj(frame, traj_file, top_file, frame_no, testing=True)
        sPRE_cols = [c for c in series.index if 'sPRE' in c]
        N15_600 = [c for c in series.index if '600' in c]
        N15_800 = [c for c in series.index if '800' in c]
        
        # asserts
        # [19, 37, 38]
        self.assertTrue(np.isnan(series['proximal PRO19 sPRE']))
        self.assertTrue(np.isnan(series['proximal PRO37 sPRE']))
        self.assertTrue(np.isnan(series['proximal PRO38 sPRE']))
        self.assertTrue(np.isnan(series['proximal PRO19 15N_relax_600']))
        self.assertTrue(np.isnan(series['proximal PRO37 15N_relax_600']))
        self.assertTrue(np.isnan(series['proximal PRO38 15N_relax_600']))
        self.assertTrue(np.isnan(series['proximal PRO19 15N_relax_800']))
        self.assertTrue(np.isnan(series['proximal PRO37 15N_relax_800']))
        self.assertTrue(np.isnan(series['proximal PRO38 15N_relax_800']))
        
        # replace nan
        N15_800 = (~series.isna()) & (series.index.str.contains('800'))
        series = series.fillna(0)
        self.assertTrue(all(series[sPRE_cols] < 1))
        self.assertTrue(all(series[N15_600] < 10))
        self.assertTrue(all(series[N15_800] < 20) and np.all(series[N15_800] >= 10))