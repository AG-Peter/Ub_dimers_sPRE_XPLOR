import unittest

class TestXPLORCorrectColumns(unittest.TestCase):
    
    def test_columns(self):
        traj_file = '/home/andrejb/Research/SIMS/2017_06_28_GfM_SMmin_rnd_k6_0/traj_nojump.xtc'
        top_file = '/home/andrejb/Research/SIMS/2017_06_28_GfM_SMmin_rnd_k6_0/start.pdb'
        frame_no = 0
        frame = md.load_frame(traj_file, frame_no, top=top_file)

        series = get_prox_dist_from_mdtraj(frame, traj_file, top_file, frame_no)
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
        self.assertTrue(np.all(series[sPRE_cols] < 1))
        self.assertTrue(np.all(series[N15_600] < 10))
        self.assertTrue(np.all(series[N15_800] < 20) and np.all(series[N15_800] >= 10))