import mdtraj as md
import numpy as np
import pandas as pd
import encodermap as em
import loading_lizard as ll
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import nglview as ngl
import xarray as xr
import expansion_elephant as ep
import tensorflow as tf
import running_rabbit as rr
import sys
sys.path.insert(0, '/home/kevin/git/Backward/')
import backward
rr.update_gmx_environ('2021.1')

import glob, os, re, copy, pickle, hdbscan, subprocess, itertools, pathlib, pyemma, shutil, time, ast

def make_15_N_table(in_file, out_file=None, return_df=True, split_prox_dist=False):
    with open(in_file, 'r') as f:
        lines = f.read().splitlines()
    lines = list(filter(lambda x: False if x == '' else True, lines))

    names = lines[0].lstrip('%<').rstrip('>').split('><') + ['is in secondary']
    df = {n: [] for n in names}

    for line in lines[1:]:
        for i, data in enumerate(line.split()):
            if '*' in data and i == 0:
                df['is in secondary'].append(False)
                data = data.lstrip('*')
            elif '*' not in data and i == 0:
                df['is in secondary'].append(True)
            try:
                df[names[i]].append(float(data))
            except ValueError:
                df[names[i]].append(data)
    df['position'] = list(map(lambda x: 'proximal' if x == 'A' else 'distal', df['chain id']))
    df = pd.DataFrame(df)
    
    mhzs = df['freq of spectrometer (MHz)'].unique()
    for mhz in mhzs:
        rho = []
        rho2 = []
        for i, resnum in enumerate(df['residue number']):
            if df['freq of spectrometer (MHz)'][i] != mhz:
                continue
            if not df['is in secondary'][i]:
                continue
            if df['chain id'][i] == 'B' and not split_prox_dist:
                resnum += 76
            R1 = df['R1 rate (1/s)'][i]
            R1_err = df['R1 rate error (1/s)'][i]
            R2 = df['R2 rate (1/s)'][i]
            R2_err = df['R1 rate error (1/s)'][i]
            NOE = df['NOE'][i]
            NOE_err = df['NOE error'][i]
            if not split_prox_dist:
                rho.append([int(resnum), R1, R1_err, R2, R2_err, NOE, NOE_err])
            else:
                if df['chain id'][i] == 'A':
                    rho.append([int(resnum), R1, R1_err, R2, R2_err, NOE, NOE_err])
                else:
                    rho2.append([int(resnum), R1, R1_err, R2, R2_err, NOE, NOE_err])
        if out_file is not None:
            if not '*' in out_file:
                raise Exception("Method will write two files. Please provide a filename with wildcard (*).")
            filename = out_file.replace('*', str(int(mhz)))
            if not split_prox_dist:
                with open(filename, 'w') as f:
                    for r in rho:
                        f.write(' '.join(map(str, r)) + '\n')
                print(filename, 'written')
            else:
                fnames = [filename.replace('.tbl', '_proximal.tbl'), filename.replace('.tbl', '_distal.tbl')]
                for fname, r in zip(fnames, [rho, rho2]):
                    with open(fname, 'w') as f:
                        for _ in r:
                            f.write(' '.join(map(str, _)) + '\n')
                print(fnames, 'written')
        else:
            out = []
            for r in rho + rho2:
                out.append(' '.join(map(str, r)) + '\n')
    if return_df:
        df = df.rename(columns={'residue number': 'resSeq'})
        df['resSeq'] = df['resSeq'].astype(int)
        max_ = df[df['chain id'] == 'A']['resSeq'].max()
        df['resSeq'] = df.apply(lambda row: row.resSeq + max_ if row['chain id'] == 'B' else row.resSeq, axis=1)
        df = df.sort_values(by=['freq of spectrometer (MHz)', 'chain id', 'resSeq'], axis='rows')
        return(df)
            
def label(resSeq, sPRE, err=0.01):
    return f"assign (resid {resSeq:<2} and name HN)	{sPRE:5.3f}	{err:5.3f}"

def getResSeq(lines):
    return list(map(lambda x: int(x.split('\t')[0][3:]), filter(lambda x: False if (x == '' or 'mM' in x or 'primary' in x) else True, lines)))
    
def make_sPRE_table(in_files, out_file=None, return_df=True, split_prox_dist=False):
    files = glob.glob(in_files)
    files = sorted(files, key=lambda x: 1 if ('proximal' in x or 'prox' in x) else 2)
    assert len(files) == 2, print(f"I need a proximal and a distal file. I found {files}")
    proximal, distal = files
    
    file = proximal
    new_lines = []
    with open(file, 'r') as f:
        lines = f.read().splitlines()
    for i, l in enumerate(lines):
        if l == '' or i == 1:
            continue
        if i == 0:
            main_labels = l.split('\t')
            df = {k: [] for k in main_labels}
            continue
        for k, v in zip(main_labels, l.split('\t')):
            df[k].append(v)
    df['resSeq'] = getResSeq(lines)
    df['position'] = np.full(len(df['resSeq']), 'proximal').tolist()

    file = distal
    with open(file, 'r') as f:
        lines = f.read().splitlines()
    for i, l in enumerate(lines):
        if l == '' or i == 0 or i == 1:
            continue
        for k, v in zip(main_labels, l.split('\t')):
            df[k].append(v)
    max_ = max(df['resSeq'])
    resSeq = list(map(lambda x: x + max_, getResSeq(lines)))
    df['resSeq'].extend(resSeq)
    df['position'].extend(np.full(len(resSeq), 'distal').tolist())
    df = pd.DataFrame(df)
    
    df['sPRE'] = pd.to_numeric(df['sPRE'], errors='coerce')
    df['err'] = pd.to_numeric(df['err'], errors='coerce')
    
    if out_file is not None:
        if not split_prox_dist:
            with open(out_file, 'w') as f:
                for i, row in df.iterrows():
                    if any(pd.isna(row)):
                        continue
                    new_line = label(row['resSeq'], row['sPRE'], row['err'])
                    f.write(new_line + '\n')
            print(out_file, 'written')
        else:
            fnames = [out_file.replace('.tbl', '_proximal.tbl'), out_file.replace('.tbl', '_distal.tbl')]
            for i, fname in enumerate(fnames):
                if i == 0:
                    sub_df = df[df['position'] == 'proximal']
                    print(sub_df)
                else:
                    sub_df = df[df['position'] == 'distal']
                with open(fname, 'w') as f:
                    for j, row in sub_df.iterrows():
                        if any(pd.isna(row)):
                            continue
                        if i == 0:
                            new_line = label(row['resSeq'], row['sPRE'], row['err'])
                        else:
                            new_line = label(row['resSeq'] - max_, row['sPRE'], row['err'])
                        f.write(new_line + '\n')
            print(fnames, 'written')
    if return_df:
        return df

import string
import itertools

def parse_atom_line(line):
    data = {}
    data['record'], data['serial_no'], data['name'] = line[:4], int(line[6:11]), line[12:16]
    data['alternate_loc_ind'], data['residue'], data['chain'] = line[16], line[17:20], line[21]
    data['res_seq'], data['code_for_insertions'] = int(line[22:26]), line[26]
    data['x'], data['y'], data['z'] = float(line[30:38]), float(line[38:46]), float(line[46:54])
    data['occ'], data['temp'], data['elem'] = float(line[54:60]), float(line[60:66]), line[76:78]
    
    for key, value in data.items():
        if isinstance(value, str):
            data[key] = value.strip()
    return data

def split_and_order_chains(datas):
    no_chains = len(list(filter(lambda x: True if x == 'OXT' else False, (x.get('name') for x in datas))))
    no_atoms = len(list(filter(lambda x: True if x == 'ATOM' else False, (x.get('record') for x in datas))))
    i = 0
    chains = [[] for x in range(no_chains)]
    for data in datas:
        if data['record'] == 'TER':
            chains[i - 1].append(data)
            continue
        chains[i].append(data)
        if data['name'] == 'OXT':
            i += 1
    order = []
    for i, chain in enumerate(chains):
        if any([x.get('residue') == 'GLQ' for x in chain]):
            order.insert(0, i)
        else:
            order.append(i)
    chains = [chains[i] for i in order]
    for chain, new_chain in zip(chains, string.ascii_uppercase):
        for value in chain:
            value['chain'] = new_chain
    return chains

def add_ter(chains):
    for i, chain in enumerate(chains):
        for j, element in enumerate(chain):
            if element['record'] == 'ATOM':
                if element['residue'] == 'GLQ' and element['name'] == 'OXT':
                    data = {}
                    data['record'], data['serial_no'] = 'TER', element['serial_no']
                    data['residue'], data['chain'] = element['residue'], element['chain']
                    data['res_seq'], data['code_for_insertions'] = element['res_seq'], element['code_for_insertions']
                    chains[i][j] = data
                if element['residue'] == 'GLY' and j == len(chain) - 1:
                    data = {}
                    data['record'], data['serial_no'] = 'TER', 0
                    data['residue'], data['chain'] = element['residue'], element['chain']
                    data['res_seq'], data['code_for_insertions'] = element['res_seq'], element['code_for_insertions']
                    chains[i].append(data)
    return chains
    
def build_pdb_line(data, atom_nr, res_number):
    if data['record'] == 'ATOM':
        line = f"{data['record']}  {atom_nr:>5}  {data['name']:<4}{data['alternate_loc_ind']}{data['residue']}"
        line += f" {data['chain']}{res_number:>4}{data['code_for_insertions']}    {data['x']:8.3f}{data['y']:8.3f}{data['z']:8.3f}"
        line += f"{data['occ']:6.2f}{data['temp']:6.2f}          {data['elem']:>2}"
    else:
        line = f"{data['record']}   {atom_nr:>5}      {data['residue']}"
        line += f" {data['chain']}{res_number:>4}{data['code_for_insertions']}"
    return line
    
def parse_ter_line(line):
    data = {}
    data['record'], data['serial_no'] = line[:4], int(line[6:11])
    data['residue'], data['chain'] = line[17:20], line[21]
    data['res_seq'], data['code_for_insertions'] = int(line[22:26]), line[26]
    
    for key, value in data.items():
        if isinstance(value, str):
            data[key] = value.strip()

    return data

def create_connect(data):
    line = "CONECT"
    for d in data:
        line += f" {d:>4}"
    return line + '\n'

def parse_pdb_line(line):
    if 'ATOM' in line:
        return parse_atom_line(line)
    elif 'TER' in line:
        return parse_ter_line(line)
    else:
        raise Exception(f"Unknown Record Type in line {line}")

def prepare_pdb_for_gmx(file, verification=None):
    if verification:
        with open(verification, 'r') as f:
            verification_lines = f.readlines()
    with open(file, 'r') as f:
        lines = f.readlines()
        
    # define
    leading = []
    datas = []
    old_lines = []
    connects = []
    
    # parse
    for i, line in enumerate(lines):
        if 'REMARK' in line or 'CRYST' in line or 'MODEL' in line:
            leading.append(line)
        elif 'ATOM' in line or 'TER' in line:
            old_lines.append(line)
            data = parse_pdb_line(line)
            datas.append(data)
        elif 'ENDMDL' in line or 'END' in line:
            pass
        elif 'CONECT' in line:
            connects.append(line)
        else:
            print(i, line)
            raise Exception(f"Unkonwn Record Type in line {line}")
         
    # rearrange chains
    chains = split_and_order_chains(datas)
    chains = add_ter(chains)
    data = list(itertools.chain(*chains))
    
    # split some connects
    connects = list(map(lambda x: [int(y) for y in x.split()[1:]], connects))
    replacements = {}
    
    # build new pdb
    residue = 1
    for i, d in enumerate(data):
        # print(old_lines[i])
        if i > 0:
            if d['residue'] != data[i -1]['residue']:
                residue += 1
        i += 1
        line = build_pdb_line(d, i, residue) +'\n'
        leading.append(line)
        
        if any(d['serial_no'] in c for c in connects):
            replacements[d['serial_no']] = i
        
        if verification:
            if line[:22] != verification_lines[i + 3][:22]:
                print(line, verification_lines[i + 3])
                raise Exception('STOP')
        
    leading.append('ENDMDL\n')
    # fix connects
    new_connects = []
    for co in connects:
        new = []
        for c in co:
            new.append(replacements[c])
        new_connects.append(new)
            
    for connect in new_connects:
        leading.append(create_connect(connect))
    leading.append('END')
    
    with open(file, 'w') as f:
        for line in leading:
            f.write(line)
    
from tempfile import gettempdir
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
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

# Test our drawing functions with example annotation
ubq_sec_annotation = seq.Annotation([
    seq.Feature("SecStr", [seq.Location(1, 7)], {"sec_str_type" : "sheet"}),
    seq.Feature("SecStr", [seq.Location(10, 17)], {"sec_str_type" : "sheet"}),
    seq.Feature("SecStr", [seq.Location(23, 34)], {"sec_str_type" : "helix"}),
    seq.Feature("SecStr", [seq.Location(40, 45)], {"sec_str_type" : "sheet"}),
    seq.Feature("SecStr", [seq.Location(48, 50)], {"sec_str_type" : "sheet"}),
    seq.Feature("SecStr", [seq.Location(56, 59)], {"sec_str_type" : "helix"}),
    seq.Feature("SecStr", [seq.Location(64, 72)], {"sec_str_type" : "sheet"}),
])