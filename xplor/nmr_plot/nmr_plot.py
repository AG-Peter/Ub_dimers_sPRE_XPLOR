import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import mdtraj as md
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle

import matplotlib.text as mpl_text

class AnyObject:
    def __init__(self, text, color, label):
        self.my_text = text
        self.my_color = color
        self.label = label
        
    def get_label(self):
        return self.label

class AnyObjectHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        print(orig_handle)
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpl_text.Text(x=0, y=0, text=orig_handle.my_text, color=orig_handle.my_color, verticalalignment=u'baseline', 
                                horizontalalignment=u'left', multialignment=None, 
                                fontproperties=None, rotation=0, linespacing=None, 
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
                          sequence_subsample=1, bottom_ax_size='7%', bottom_ax_pad='15%'):
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

    return ax


def plot_line_data(axes, df, df_index, color='C1', positions=['proximal', 'distal']):
    df = df[df_index['cols']]
    out = []
    for ax, position in zip(axes, positions):
        index = [df_index['rows'] in ind and position in ind for ind in df.index]
        data = df[index].values
        assert len(data) == len(ax.get_xticks())
        ax.plot(np.arange(len(data)), data, color=color)
        out.append(ax)
    return out


def plot_minmax_envelope(axes, df, df_index, color='lightgrey', alpha=0.8, positions=['proximal', 'distal']):
    df = df[df['ubq_site'] == df_index['cols']]
    out = []
    for ax, position in zip(axes, positions):
        index = [df_index['rows'] in col and position in col for col in df.columns]
        index = df.columns[index]
        if index.str.contains('normalized').any():
            index = index[index.str.contains('normalized')]
        data = df[index].values
        min_, max_ = np.min(data, axis=0), np.max(data, axis=0)
        assert len(min_) == len(ax.get_xticks()), print(len(min_), int(ax.get_xlim()[1] + 1), ax.get_xticks()[-1] + 1)
        assert len(max_) == len(ax.get_xticks())
        _ = ax.fill_between(np.arange(len(min_)), min_, max_, color=color, alpha=alpha)
        out.append(ax)
    return out


def plot_hatched_bars(axes, df, df_index, color='lightgrey', alpha=0.3, positions=['proximal', 'distal']):
    df = df[df_index['cols']]
    out = []
    for ax, position in zip(axes, positions):
        index = [position in ind for ind in df.index]
        data = df[index].values
        assert len(data) == len(ax.get_xticks())
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


def fake_legend(ax, dict_of_fake_labels):
    legend_elements = []

    func_dict = {'line': Line2D, 'envelope': Patch, 'hatchbar': Patch, 'text': AnyObject}

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
                legend_element_text = func_dict[type_](element['text'], element['color'], label=element['label'])
                legend_elements.append(legend_element_text)
        else:
            print(f"Unknown label type {type_}")

    if 'text' in dict_of_fake_labels:
        ax.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), handler_map={legend_element_text: AnyObjectHandler()})
    else:
        ax.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements))

    return ax

