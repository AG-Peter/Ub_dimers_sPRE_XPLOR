import copy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import mdtraj as md
import pyemma.plots
import seaborn as sns
import functools
import scipy
import glob
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import statsmodels.stats.api as sms
from ..functions.functions import get_series_from_mdtraj, Capturing
from ..functions.custom_gromacstopfile import CustomGromacsTopFile
from ..misc import get_iso_time

import matplotlib.text as mpl_text

class AnyObject:
    def __init__(self, text, color, label):
        self.my_text = text
        self.my_color = color
        self.label = label
        
    def get_label(self):
        return self.label

class AnyObjectHandler:
    def __init__(self, bold=False, italic=False):
        self.fontweight = 'bold' if bold else 'normal'
        self.fontstyle = 'italic' if italic else 'normal'

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpl_text.Text(x=0, y=0, text=orig_handle.my_text, color=orig_handle.my_color, verticalalignment=u'baseline', 
                                horizontalalignment=u'left', multialignment=None, fontweight=self.fontweight,
                                fontproperties=None, rotation=0, linespacing=None, fontstyle=self.fontstyle,  
                                rotation_mode=None)
        handlebox.add_artist(patch)
        return patch


try:
    from tempfile import gettempdir
    import biotite
    import biotite.structure as struc
    import biotite.structure.io.mmtf as mmtf
    import biotite.sequence as seq
    import biotite.sequence.graphics as graphics
    import biotite.sequence.io.genbank as gb
    import biotite.database.rcsb as rcsb
    import biotite.database.entrez as entrez
    import biotite.application.dssp as dssp

    class HelixPlotter(graphics.FeaturePlotter):

        def __init__(self):
            pass

        # Check whether this class is applicable for drawing a feature
        def matches(self, feature):
            if feature.key == "SecStr":
                if "sec_str_type" in feature.qual:
                    if feature.qual["sec_str_type"] == "helix":
                        return True
            return False

        # The drawing function itself
        def draw(self, axes, feature, bbox, loc, style_param):
            # Approx. 1 turn per 3.6 residues to resemble natural helix
            n_turns = np.ceil((loc.last - loc.first + 1) / 3.6)
            x_val = np.linspace(0, n_turns * 2*np.pi, 100)
            # Curve ranges from 0.3 to 0.7
            y_val = (-0.4*np.sin(x_val) + 1) / 2

            # Transform values for correct location in feature map
            x_val *= bbox.width / (n_turns * 2*np.pi)
            x_val += bbox.x0
            y_val *= bbox.height
            y_val += bbox.y0

            # Draw white background to overlay the guiding line
            background = Rectangle(
                bbox.p0, bbox.width, bbox.height, color="white", linewidth=0
            )
            axes.add_patch(background)
            axes.plot(
                x_val, y_val, linewidth=2, color=biotite.colors["dimgreen"]
            )


    class SheetPlotter(graphics.FeaturePlotter):

        def __init__(self, head_width=0.8, tail_width=0.5):
            self._head_width = head_width
            self._tail_width = tail_width


        def matches(self, feature):
            if feature.key == "SecStr":
                if "sec_str_type" in feature.qual:
                    if feature.qual["sec_str_type"] == "sheet":
                        return True
            return False

        def draw(self, axes, feature, bbox, loc, style_param):
            x = bbox.x0
            y = bbox.y0 + bbox.height/2
            dx = bbox.width
            dy = 0

            if  loc.defect & seq.Location.Defect.MISS_RIGHT:
                # If the feature extends into the prevoius or next line
                # do not draw an arrow head
                draw_head = False
            else:
                draw_head = True

            axes.add_patch(biotite.AdaptiveFancyArrow(
                x, y, dx, dy,
                self._tail_width*bbox.height, self._head_width*bbox.height,
                # Create head with 90 degrees tip
                # -> head width/length ratio = 1/2
                head_ratio=0.5, draw_head=draw_head,
                color=biotite.colors["orange"], linewidth=0
            ))
except (ImportError):
    print("Can not use Biotite functions unless you `pip install` it.")

def _zero_runs(a, int_to_find=1):  # from link
    """Thanks to https://stackoverflow.com/questions/44790869/find-indexes-of-repeated-elements-in-an-array-python-numpy"""
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def _make_seq_annotation(traj, sequence_annotation_frame=0):

    dssp = md.compute_dssp(traj)[sequence_annotation_frame]
    extended_indices = _zero_runs(np.vectorize({'C': 1, 'E': 0, 'H':1}.get)(dssp))
    helix_indices = _zero_runs(np.vectorize({'C': 1, 'E': 1, 'H':0}.get)(dssp))

    extendes_seqs = [seq.Feature("SecStr", [seq.Location(index[0], index[1] - 1)], {"sec_str_type" : "sheet"}) for index in extended_indices]
    helix_seqs = [seq.Feature("SecStr", [seq.Location(index[0], index[1] - 1)], {"sec_str_type" : "helix"}) for index in helix_indices]

    seq_annotation = seq.Annotation(extendes_seqs + helix_seqs)
    return seq_annotation


def add_sequence_to_xaxis(ax, pdb_id='1UBQ', remove_solvent=True, sequence_annotation_frame=0,
                          sequence_subsample=1, bottom_ax_size='7%', bottom_ax_pad='20%', xlim=None):
    import biotite.sequence.graphics as graphics
    from Bio.SeqUtils import seq3

    # make labels
    traj = md.load_pdb(f"https://files.rcsb.org/view/{pdb_id}.pdb")
    if remove_solvent:
        traj = traj.atom_slice(traj.top.select('not water'))
    fasta = traj.top.to_fasta()[0]
    xlabels = [f"{seq3(f)}{i + 1}" for i, f in enumerate(fasta)][::sequence_subsample]
    xticks = np.arange(len(fasta))[::sequence_subsample]

    # set ticks and sequence elements
    ax_divider = make_axes_locatable(ax)
    seq_ax = ax_divider.append_axes('bottom', size=bottom_ax_size, pad=bottom_ax_pad)

    # get sequence
    seq_annotation = _make_seq_annotation(traj, sequence_annotation_frame)

    # plot feature
    graphics.plot_feature_map(
        seq_ax, seq_annotation, multi_line=False, loc_range=(1, traj.n_residues),
        feature_plotters=[HelixPlotter(), SheetPlotter()]
    )

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=90)
    ax.set_xlim([0, len(fasta) - 1])


    if xlim is not None:
        seq_ax.set_xlim(xlim)

    return ax


def plot_line_data(axes, df, df_index, color='C1', positions=None, mask_15N=True):
    if positions is None:
        positions = ['proximal', 'distal']
    df = df[df_index['cols']]
    out = []
    for ax, position in zip(axes, positions):
        index = [df_index['rows'] in ind and position in ind for ind in df.index]
        data = df[index].values
        if '15N' in df_index['rows'] and mask_15N:
            data = np.ma.masked_where(data == 0, data, copy=True)
        ax.plot(np.arange(len(data)), data, color=color)
        out.append(ax)
    return out


def plot_boxplots(axes, df, df_index, positions=None):
    if positions is None:
        positions = ['proximal', 'distal']
    df = df[df['ubq_site'] == df_index['cols']]
    out = []
    for ax, position in zip(axes, positions):
        index = [df_index['rows'] in col and position in col for col in df.columns]
        index = df.columns[index]
        if index.str.contains('normalized').any():
            index = index[index.str.contains('normalized')]
        data = df[index].values
        ax.boxplot(data)
        out.append(ax)
    return out


def plot_many_lines_w_alpha():
    pass


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def get_color(colorRGBA1, colorRGBA2):
    alpha = 255 - ((255 - colorRGBA1[3]) * (255 - colorRGBA2[3]) / 255)
    red   = (colorRGBA1[0] * (255 - colorRGBA2[3]) + colorRGBA2[0] * colorRGBA2[3]) / 255
    green = (colorRGBA1[1] * (255 - colorRGBA2[3]) + colorRGBA2[1] * colorRGBA2[3]) / 255
    blue  = (colorRGBA1[2] * (255 - colorRGBA2[3]) + colorRGBA2[2] * colorRGBA2[3]) / 255
    return (int(red), int(green), int(blue), int(alpha))

def plot_confidence_intervals(axes, df, df_index, cmap='Blues', cbar=True,
                              alpha=0.2, positions=None, omit_nan=False,
                              trajs=None, cluster_num=None, cbar_axis=None,
                              with_outliers=True, outliers_offset=0.0):
    """Plots an envelope around min and max values.

    Args:
        axes (tuple[mpl.axes]): Two instances of the axes to plot to.
        df (pd.DataFrame): Pandas dataframe with the data.
        df_index (Dict[str, str]): A dictionary of {'rows': 'sPRE', 'cols': f'{ubq_site}'}.
            that defines what data needs to be extracted from df.

    """
    if positions is None:
        positions = ['proximal', 'distal']

    if trajs is not None:
        test = len(np.where(trajs.cluster_membership == cluster_num)[0])
        names = np.unique(trajs.name_arr[trajs.cluster_membership == cluster_num])
        indices = trajs.index_arr[trajs.cluster_membership == cluster_num]
        traj_nums = np.unique(indices[:,0])
        if not len(names) == len(traj_nums):
            print(names)
            print(traj_nums)
            raise Exception("Could not identify traj names and indices")
        df_indices = []
        for traj_num, name in zip(traj_nums, names):
            frames = indices[indices[:, 0] == traj_num][:, 1]
            fitting_name = df['traj_file'].str.contains(name)
            fitting_frames = df['frame'].isin(frames)
            extend = np.where(fitting_name & fitting_frames)[0].tolist()
            if cluster_num is not None:
                if extend == []:
                    print(f"No points in cluster for {name}")
                else:
                    print(f"{len(extend)} points in cluster for {name}")
            df_indices.extend(extend)
        if df_indices == []:
            print(f"No cluster points for cluster {cluster_num} in {names}")
            return
        df_indices = np.array(df_indices)
    if cluster_num is not None and trajs is None:
        df = df[df['cluster_membership'] == cluster_num]
    if cluster_num is not None and trajs is not None:
        if df_index is not None:
            df = df.iloc[df_indices]
            assert len(df) == len(df_indices)
    else:
        if df_index is not None:
            df = df[df['ubq_site'] == df_index['cols']]
    if df.isna().any(None):
        df = df.fillna(0)
    out = []

    # prepare cmap
    cmap = plt.get_cmap(cmap, 5).copy()
    plot_color = cmap(4)
    cmap = cmap(np.arange(5))
    cmap[:, -1] = alpha
    cmap = mpl.colors.ListedColormap(cmap[2:4])

    for iter_, (ax, position) in enumerate(zip(axes, positions)):
        index = [df_index['rows'] in col and position in col for col in df.columns]
        index = df.columns[index]
        if index.str.contains('normalized').any():
            index = index[index.str.contains('normalized')]
        data = df[index].values
        q1, median, q3 = df[index].quantile([0.25, 0.5, 0.75], axis='rows').values

        if not len(q1) == 76:
            print(index)
            print(data.shape)
            raise Exception("No data")

        iqr = q3 - q1
        min_ = q1 - 1.5 * iqr
        max_ = q3 + 1.5 * iqr
        outliers = np.ma.masked_where((data >= min_) & (data <= max_), data)

        min_[min_ < 0] = 0
        max_[max_ < 0] = 0
        q1[q1 < 0] = 0
        q3[q3 < 0] = 0

        if not omit_nan:
            ax.fill_between(range(len(median)), min_, max_, color=cmap(0))
            ax.fill_between(range(len(median)), q1, q3, color=cmap(1))
        else:
            indices = np.arange(len(median))
            chunks = np.split(indices, np.where(median == 0)[0][1:])
            for index in chunks:
                if len(index) == 1:
                    continue
                index = index[1:]
                ax.fill_between(index, min_[index], max_[index], color=cmap(0))
                ax.fill_between(index, q1[index], q3[index], color=cmap(1))

        if with_outliers:
            XX, YY = np.meshgrid(np.arange(outliers.shape[1]), np.arange(outliers.shape[0]))
            y = outliers.ravel()
            x = XX.ravel() + outliers_offset
            ax.scatter(x, y, s=1, color=plot_color)

        # blend the colors
        # for i, color in enumerate(colors):
        #     new_color = functools.reduce(get_color, colors[i:])

        if cbar:
            if cbar_axis is None or cbar_axis == iter_:
                # cmap = mpl.colors.ListedColormap(blended_colors)
                sm = mpl.cm.ScalarMappable(cmap=cmap)
                sm.set_array([])
                divider = make_axes_locatable(ax)
                cmap_ticks = np.linspace(0, 1, 2, endpoint=False)
                cmap_ticks = cmap_ticks + (cmap_ticks[1] - cmap_ticks[0]) / 2
                cax = divider.append_axes('right', size='2%', pad=0.15)
                cbar = plt.gcf().colorbar(sm, ticks=cmap_ticks, orientation='vertical', cax=cax, label="Quartile Range")
                cax.set_yticklabels([r'$\mathrm{Q_{1/3} \mp IQR}$', 'IQR'])
        if omit_nan:
            median[median == 0] = np.nan
        ax.plot(median, color=plot_color)
        out.append(ax)
    if cbar:
        return out, plot_color
    else:
        return out, plot_color, (cmap(1), cmap(0))

def plot_single_struct_sPRE(axes, traj, factors, ubq_site, color, positions=None):
    if positions is None:
        positions = ['proximal', 'distal']

    traj_file = traj.traj_file
    top_file = traj.top_file
    traj = traj.traj
    with Capturing() as output:
        top_aa = CustomGromacsTopFile(
            f'/home/andrejb/Software/custom_tools/topology_builder/topologies/gromos54a7-isop/diUBQ_{ubq_site.upper()}/system.top',
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

    series = get_series_from_mdtraj(traj, traj_file, top_file, 0, fix_isopeptides=False)
    series = series.fillna(0)

    out = []
    for iter_, (ax, position, factor) in enumerate(zip(axes, positions, factors.values())):
        index = ['sPRE' in col and position in col for col in series.keys()]
        data = series[index].values
        try:
            data *= factor
        except TypeError:
            print(data)
            print(index)
            print(factor)
            raise
        ax.plot(np.arange(len(data)), data, color=color)
        out.append(ax)
    return out



def try_to_plot_15N(axes, ubq_site, df=None, mhz=600, cmap='Blues', cbar=True, alpha=0.2,
                    positions=None, trajs=None, cluster_num=None,
                    cbar_axis=None, with_outliers=True, outliers_offset=0.0):
    if positions is None:
        positions = ['proximal', 'distal']
    if df is None:
        files = glob.glob('/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package_with_conect/*.csv')
        sorted_files = sorted(files, key=get_iso_time)
        df = pd.read_csv(sorted_files[-1], index_col=0)
        df = df.fillna(0)
    out = plot_confidence_intervals(axes, df, df_index={'rows': f'15N_relax_{mhz}', 'cols': ubq_site}, cmap=cmap, omit_nan=True,
                                    cbar=cbar, alpha=alpha, positions=positions, trajs=trajs, cluster_num=cluster_num,
                                    cbar_axis=cbar_axis, with_outliers=with_outliers, outliers_offset=outliers_offset)
    return out


def plot_minmax_envelope(axes, df, df_index, color='lightgrey', alpha=0.8, positions=None):
    """Plots an envelope around min and max values.

    Args:
        axes (mpl.axes): Instance of the axes to plot to.
        df (pd.DataFrame): Pandas dataframe with the data.
        df_index (Dict[str, str]): A dictionary of {'rows': 'sPRE', 'cols': {ubq_site}}.
            that defines what data needs to be extracted from df.

    """
    if positions is None:
        positions = ['proximal', 'distal']

    df = df[df['ubq_site'] == df_index['cols']]
    out = []
    for ax, position in zip(axes, positions):
        index = [df_index['rows'] in col and position in col for col in df.columns]
        index = df.columns[index]
        if index.str.contains('normalized').any():
            index = index[index.str.contains('normalized')]
        data = df[index].values
        min_, max_ = np.min(data, axis=0), np.max(data, axis=0)
        _ = ax.fill_between(np.arange(len(min_)), min_, max_, color=color, alpha=alpha)
        out.append(ax)
    return out


def plot_hatched_bars(axes, df, df_index, color='lightgrey', alpha=0.3, positions=None):
    if positions is None:
        positions = ['proximal', 'distal']

    df = df[df_index['cols']]
    out = []
    for ax, position in zip(axes, positions):
        index = [position in ind for ind in df.index]
        data = df[index].values
        for i, is_fast in enumerate(data):
            if is_fast:
                _ = ax.axvspan(i - 0.5, i + 0.5, alpha=alpha, fc='none', ec='lightgrey', hatch='//', zorder=-5)
        out.append(ax)
    return out


def color_labels(ax, positions, color='red'):
    for p in positions:
        try:
            ax.get_xticklabels()[p].set_color(color)
        except IndexError:
            print(positions)
            raise
    return ax


def plot_correlation_plot(axes, df_obs, df_comp_norm, df_index, correlations,
                     colors=None, positions=None, percent=False):
    if positions is None:
        positions = ['proximal', 'distal']

    if colors is None:
        colors = ['C0', 'c', 'grey', 'tab:olive']


    df_obs = df_obs[df_index['cols']]
    df_obs_data = []
    for i, position in enumerate(positions):
        index = [df_index['rows'] in ind and position in ind for ind in df_obs.index]
        df_obs_data.append(df_obs[index].values)

    df_comp_norm = df_comp_norm[df_comp_norm['ubq_site'] == df_index['cols']]
    if df_comp_norm.isna().any(None):
        df_comp_norm = df_comp_norm.fillna(0)

    correl_data = [[], []]
    for i, corr_type in enumerate(correlations):
        if isinstance(corr_type, str):
            for j, position in enumerate(positions):
                index = [df_index['rows'] in col and position in col for col in df_comp_norm.columns]
                index = df_comp_norm.columns[index]
                if index.str.contains('normalized').any():
                    index = index[index.str.contains('normalized')]
                data = df_comp_norm[index].values
                correl_data[j].append(data)
                if corr_type == 'mean':
                    correl_data[j][i] = np.mean(correl_data[j][i], 0)
                else:
                    raise Exception("Corr currently only supports 'mean'")
        elif isinstance(corr_type, dict):
            key = list(corr_type.keys())[0]
            print(key)
            if isinstance(corr_type[key], np.ndarray):
                prox, dist = np.split(corr_type[key], 2)
                correl_data[0].append(prox)
                correl_data[1].append(dist)
            else:
                raise Exception("Corr currently only supports dict of np array.")
        else:
            raise Exception("Corr currenlty only supports str.")

    out = []
    assert len(correl_data[0]) == len(correlations), print([a.shape for a in correl_data[0]])
    if len(correlations) == 2:
        offsets = [-0.2, 0.2]
        width = 0.4
    elif len(correlations) == 3:
        offsets = [-0.2, 0, 0.2]
        width = 0.2
    elif len(correlations) == 4:
        offsets = [-0.4, -0.2, 0.0, 0.2]
        width = 0.2

    for i, dataset in enumerate(correlations):
        sim_splitted = (correl_data[0][i], correl_data[1][i])
        for j, ax in enumerate(axes):
            sim = np.asarray(sim_splitted[j])
            exp = np.asarray(df_obs_data[j])
            assert len(sim) == len(exp), print(sim.shape, exp.shape)

            if not percent:
                diff = (sim - exp).flatten()
            else:
                diff = sim / exp * 100

            if len(correlations) == 1:
                ax.bar(np.arange(len(diff)), diff, color=[colors[i] for _ in range(len(diff))])
            else:
                x = np.arange(len(diff)) + offsets[i]
                ax.bar(x, diff, align='center', width=width, color=[colors[i] for _ in range(len(diff))])

    return axes


def fake_legend(ax, dict_of_fake_labels, ncols=None):
    legend_elements = []

    func_dict = {'line': Line2D, 'envelope': Patch, 'hatchbar': Patch, 'text': AnyObject, 'scatter': Line2D}

    for type_, elements in dict_of_fake_labels.items():
        if type_ == 'line':
            for element in elements:
                legend_element = func_dict[type_]([0], [0], color=element['color'], label=element['label'])
                legend_elements.append(legend_element)
        elif type_ == 'envelope':
            for element in elements:
                legend_element = func_dict[type_](color=element['color'], label=element['label'])
                legend_elements.append(legend_element)
        elif type_ == 'hatchbar':
            for element in elements:
                legend_element = func_dict[type_](fc='none', ec=element['color'], label=element['label'], hatch=element['hatch'], alpha=element['alpha'])
                legend_elements.append(legend_element)
        elif type_ == 'text':
            for element in elements:
                legend_element_text = func_dict[type_](text=element['text'], color=element['color'], label=element['label'])
                legend_elements.append(legend_element_text)
        elif type_ == 'scatter':
            for element in elements:
                legend_element = func_dict[type_]([0], [0], color='w', marker=element['marker'], markerfacecolor=element['color'], label=element['label'])
                legend_elements.append(legend_element)
        else:
            print(f"Unknown label type {type_}")

    if 'text' in dict_of_fake_labels:
        if ncols is None:
            ax.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), handler_map={legend_element_text: AnyObjectHandler()})
        else:
            ax.legend(handles=legend_elements, ncol=ncols, handler_map={legend_element_text: AnyObjectHandler()})
    else:
        if ncols is None:
            ax.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements))
        else:
            ax.legend(handles=legend_elements, loc='upper center', ncol=int(len(legend_elements) / ncols))

    return ax
