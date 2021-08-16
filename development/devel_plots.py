# %% Imports
import xplor
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
import mdtraj as md
import glob, os
import parmed as pmd

# %% Delete old dataframes
# xplor.delete_old_csvs()

# %% Streamline observables and computation
from xplor.functions.functions import get_ubq_site
df_comp = pd.read_csv('/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package_with_conect/2021-08-16T12:45:19+02:00_df.csv',
                      index_col=0)
if not 'ubq_site' in df_comp.keys():
    df_comp['ubq_site'] = df_comp['traj_file'].map(get_ubq_site)
# df_comp = df_comp.fillna(0)
df_obs = xplor.functions.parse_input_files.get_observed_df(['k6', 'k29'])
fast_exchangers = xplor.functions.parse_input_files.get_fast_exchangers(['k6', 'k29'])
in_secondary = xplor.functions.parse_input_files.get_in_secondary(['k6', 'k29'])

# %% View dataframe
sPRE_ind = df_comp.columns[['sPRE' in c for c in df_comp.columns]]
df_comp[sPRE_ind].values

# %% Normalize
df_comp_norm, centers_prox, centers_dist = xplor.functions.normalize_sPRE(df_comp, df_obs)

# %% Plot
ubq_site = 'k6'
plt.close('all')
fig, (ax1, ax2,) = plt.subplots(nrows=2, figsize=(15, 10))

fake_legend_dict1 = {'line': [{'label': 'NMR experiment', 'color': 'C1'}],
                    'hatchbar': [{'label': 'Fast exchanging residues', 'color': 'lightgrey', 'alpha': 0.3, 'hatch': '//'}],
                    'text': [{'label': 'Residue used for normalization', 'color': 'red', 'text': 'Ile3'}],
                    'envelope': [{'label': 'XPLOR calculation', 'color': 'lightgrey'}]}

for ax, centers in zip([ax1, ax2], [centers_prox, centers_dist - 76]):
    ax = xplor.nmr_plot.add_sequence_to_xaxis(ax)
    ax = xplor.nmr_plot.color_labels(ax, positions=centers)
    ax.set_ylabel(r'sPRE in $\mathrm{mM^{-1}ms^{-1}}$')

(ax1, ax2) = xplor.nmr_plot.plot_line_data((ax1, ax2), df_obs, {'rows': 'sPRE', 'cols': ubq_site})
(ax1, ax2) = xplor.nmr_plot.plot_hatched_bars((ax1, ax2), fast_exchangers, {'cols': 'k6'}, color='k')

plt.tight_layout()
plt.show()