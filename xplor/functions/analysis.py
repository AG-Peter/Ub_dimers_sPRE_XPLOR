################################################################################
# Imports
################################################################################
import datetime
import functools
import itertools
import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import mdtraj
import seaborn as sns
import mdtraj as md
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm as scipy_norm
from sklearn.mixture import GaussianMixture as GMM
import os, sys, glob, multiprocessing, copy, ast, subprocess, yaml, shutil, scipy, json
import warnings
from scipy.spatial.distance import cdist
from .functions import is_aa_sim, normalize_sPRE, make_linear_combination_from_clusters, Capturing
import encodermap as em
from encodermap.misc.clustering import gen_dummy_traj, rmsd_centroid_of_cluster, _gen_dummy_traj_single
from ..nmr_plot.nmr_plot import *
import dateutil
import pyemma
import hdbscan
from .parse_input_files.parse_input_files import get_observed_df, get_fast_exchangers, get_in_secondary
from string import ascii_uppercase
import cartopy.crs as ccrs
from ..misc import get_iso_time
from pprint import pprint
from collections import OrderedDict


################################################################################
# Imports
################################################################################


from typing import List, Union, Tuple, Optional


################################################################################
# Globals
################################################################################


__all__ = ['EncodermapSPREAnalysis', 'h5load']


################################################################################
# Functions
################################################################################


def index_of_closest_point(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def check_sPRE_normalization(df):
    for check in ['proximal', 'distal']:
        norm_columns = [c for c in df.columns if check in c and 'norm' in c and 'sPRE' in c]
        non_norm_columns = [c for c in df.columns if check in c and 'norm' not in c and 'sPRE' in c]
        print(norm_columns)
        print(non_norm_columns)
        a = df[norm_columns].values
        a = a[~np.isnan(a)].flatten()
        b = df[non_norm_columns].values
        b = b[~np.isnan(b)].flatten()
        test = np.unique((a / b).round(10))
        test = test[~np.isnan(test)]
        if check == 'proximal':
            print(len(test))
            assert np.isclose(test, f_prox)
        else:
            print(test)
            assert np.isclose(test, f_dist)


def ckpt_step(file):
    file = os.path.basename(file).split('.')[0].split('_')[-1]
    try:
        return int(file)
    except ValueError as e:
        try:
            time = dateutil.parser.isoparse(file)
        except Exception as e2:
            raise e2 from e1
    return 0


def align_principal_axes_mdtraj(traj):
    """Takes an mdtraj trajectory as input and aligns it with its principal axes.

    Transformation is done in-place. So kiss your old coordinates goodbye.

    Args:
        traj: mdtraj.Trajectory with 1 frame.

    Returns:
        None

    Raises:
        NotImplementedError: If the traj has more than 1 frame.

    Examples:
        >>> traj = md.load_pdb('https://files.rcsb.org/view/5NL0.pdb')
        >>> traj = traj.atom_slice(traj.top.select('chainid 0 to 9 or chainid 16'))
        >>> traj.xyz[0][0]
        [ 3.3407  5.9445 -3.326 ]
        >>> align_principal_axes_mdtraj(traj)
        >>> traj.xyz[0][0]
        [ 0.73253864 -7.5465813  -0.2685707 ]

    """
    from .transformations import superimposition_matrix
    if traj.n_frames != 1:
        raise NotImplementedError("Currently only works with trajs with 1 frame.")
    # get the moment of inertia tensor
    inertia_tensor = md.compute_inertia_tensor(traj)
    # the eigenvectors of thsi tensor are the principal axes
    eigvals, eigvs = np.linalg.eig(inertia_tensor)
    eigvs = eigvs[0]
    # define the normals
    normals = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    # get a homogeneous transformation from these three points
    trans_mat = superimposition_matrix(eigvs, normals)
    # make our 3D points a list of quaternions
    padded = np.pad(traj.xyz[0], ((0, 0), (0, 1)), mode='constant', constant_values=1)
    # dot_multiplication of the transformation_matrix with the quaternions gives the new points
    new_points = trans_mat.dot(padded.T).T[:,:3]
    traj.xyz[0] = new_points


def appendSphericalFrame(xyz):
    """Appends the spherical coordiantes to an array of xyz.

    Args:
        xyz (np.ndarray): An array of shape (n, 3), where n is the number of
            points.

    r e [0, inf)
    theta e [0, pi]
    phi e [-pi, pi]

    Returns:
        np.ndarray: An array of shape (n, 6), where a[:, 3] is the radius as defined
            by r = sqrt( x ** 2 + y **2 + z ** 2), a[:, 4] is the inclination as
            defined by theta = arctan( sqrt( x ** 2 + y ** 2) / z )) and a[:, 5] is
            the azimuth as defined by phi = arctan ( y / x)

    """
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    ptsnew[:, 3] = np.sqrt(xy + xyz[:, 2] ** 2)
    ptsnew[:, 4] = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:, 5] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return ptsnew


def appendSphericalTraj(pts):
    """Appends the spherical coordiantes to an array of xyz.

    Args:
        pts (np.ndarray): An array of shape (n_frames, n_atoms, 3), where n is the number of
            points.

    r e [0, inf)
    theta e [0, pi]
    phi e [-pi, pi]

    Returns:
        np.ndarray: An array of shape (n_frames, n_atoms, 6), where a[:, :, 3] is the radius as defined
            by r = sqrt( x ** 2 + y **2 + z ** 2), a[:, :, 4] is the inclination as
            defined by theta = arctan( sqrt( x ** 2 + y ** 2) / z )) and a[:, :, 5] is
            the azimuth as defined by phi = arctan ( y / x)

    """
    ptsnew = np.dstack((pts, np.zeros(pts.shape)))
    xy = pts[:, :, 0] ** 2 + pts[:, :, 1] ** 2
    ptsnew[:, :, 3] = np.sqrt(xy + pts[:, :, 2] ** 2)
    ptsnew[:, :, 4] = np.arctan2(np.sqrt(xy), pts[:, :, 2]) # for elevation angle defined from Z-axis down
    # ptsnew[:, :, 4] = np.arctan2(xyz[:, :, 2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:, :, 5] = np.arctan2(pts[:, :, 1], pts[:, :, 0])
    return ptsnew


def appendLonLatFrame(xyz):
    """Appends longitute and latitude to an array of xyz coordinates.

    Args:
        xyz (np.ndarray): An array of shape (n, 3), where n is the number of points.

    r e [0, inf]
    lons e [-180, 180]
    lats e [-90, 90]

    The transformations form `appendSpherical_np` are taken, columns 4 and 5 are swapped.
    and wrapped to the correct space.

    Returns:
        np.ndarray: An array of shape (n, 6), where a[:, 3] is the radius, a[:, 4] is the
        longitude ranging from -90 to 90 deg with 0 deg being the euqator, a[:, 5] is
        the latitudes ranging from -180 to 180 deg with 0 deg being the zero meridian.

    """
    new = appendSphericalFrame(xyz)
    new[:, 4:] = np.rad2deg(new[:, 4:])
    new[:, [4, 5]] = new[:, [5, 4]]
    new[:, 5] -= 90.0
    return new


def appendLonLatTraj(xyz):
    """Appends longitute and latitude to an array of xyz coordinates.

    Args:
        xyz (np.ndarray): An array of shape (n, 3), where n is the number of points.

    r e [0, inf]
    lons e [-180, 180]
    lats e [-90, 90]

    The transformations form `appendSpherical_np` are taken, columns 4 and 5 are swapped.
    and wrapped to the correct space.

    Returns:
        np.ndarray: An array of shape (n, 6), where a[:, 3] is the radius, a[:, 4] is the
        longitude ranging from -90 to 90 deg with 0 deg being the euqator, a[:, 5] is
        the latitudes ranging from -180 to 180 deg with 0 deg being the zero meridian.

    """
    new = appendSphericalTraj(xyz)
    new[:, :, 4:] = np.rad2deg(new[:, :, 4:])
    new[:, :, [4, 5]] = new[:, :, [5, 4]]
    new[:, :, 5] -= 90.0
    new[:, :, 5] *= -1
    return new


def center_ref_and_load(overwrite: bool = False) -> Tuple[mdtraj.Trajectory, np.ndarray, np.ndarray]:
    """Returns same basic information about 1UBQ.

    Args:
        overwrite (bool, optional): Whether to overwrite the locally stored file
            at /home/kevin/1UBQ.pdb.

    Returns:
        tupel: A tupel containing the following:
            md.Trajectory: The mdtraj.Trajectory of 1UBQ
            np.ndarray: The indices of the C-alpha atoms.
            np.ndarray: The names of the residues.

    """
    centered_1UBQ = '/mnt/data/kevin/xplor_analysis_files/centered1UBQ.pdb'
    if not os.path.isfile(centered_1UBQ) or overwrite:
        traj = md.load_pdb('/home/kevin/1UBQ.pdb')
        traj = traj.atom_slice(traj.top.select('protein'))
        align_principal_axes_mdtraj(traj)
        traj.center_coordinates()
        traj.save_pdb(centered_1UBQ)
    else:
        traj = md.load(centered_1UBQ)
    CA_indices = traj.top.select('name CA')
    resnames = np.array([str(r) for r in traj.top.residues])
    return traj, CA_indices, resnames


def add_reference_to_map(ax, step: int = 5, rotate: bool = False,
                         pole_longitude: float = 180.0,
                         pole_latitude: float = 90.0) -> matplotlib.axes.Axes:
    """"""
    traj, idx, labels = center_ref_and_load()
    scatter = traj.xyz[0, idx]
    scatter = appendLonLatFrame(scatter)
    lons, lats = scatter[:, 4:].T
    ax.gridlines()
    if rotate:
        transform = ccrs.RotatedGeodetic(pole_longitude=pole_longitude, pole_latitude=pole_latitude)
    else:
        transform = ccrs.Geodetic()
    ax.plot(lons, lats, marker='o', transform=transform)
    ax.set_global()
    for i, label in enumerate(labels):
        if (i + 1) % step == 0 or i == 0:
            ax.annotate(label, (lons[i], lats[i]), xycoords=transform._as_mpl_transform(ax))
    return ax


def new_dihedral(p):
    """Praxeolitic formula
    1 sqrt, 1 cross product.

    From: https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    """
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def _compute_n_dihedrals(pos, out=None):
    p0 = pos[:, 0]
    p1 = pos[:, 1]
    p2 = pos[:, 2]
    p3 = pos[:, 3]

    b1 = -1.0 * (p1 - p0)
    b2 = p2 - p1
    b3 = p3 - p2

    c1 = np.cross(b2, b3)
    c2 = np.cross(b1, b2)

    p1 = (b1 * c1).sum(-1)
    p1 *= (b2 * b2).sum(-1) ** 0.5
    p2 = (c1 * c2).sum(-1)

    return np.arctan2(p1, p2, out)


def h5store(filename, df, **kwargs):
    """https://stackoverflow.com/questions/29129095/save-additional-attributes-in-pandas-dataframe/29130146#29130146"""
    store = pd.HDFStore(filename)
    store.put('mydata', df)
    store.get_storer('mydata').attrs.metadata = kwargs
    store.close()

def h5load(store):
    """https://stackoverflow.com/questions/29129095/save-additional-attributes-in-pandas-dataframe/29130146#29130146"""
    data = store['mydata']
    metadata = store.get_storer('mydata').attrs.metadata
    return data, metadata

def prox_is_first(traj):
    residue = traj.top.atom(traj.top.select('resname LYQ')[0]).residue
    if residue.index < 75:
        return True
    else:
        return False

def get_prox_indices(traj, traf_file):
    CA = []
    if prox_is_first(traj):
        raise Exception(f"For the traj {traf_file}, the proximal unit is first in the chain.")
    else:
        raise Exception(f"For the traj {traf_file}, the proximal unit is later in the chain.")

def get_dist_indices(traj, traj_file):
    CA = []
    if prox_is_first(traj):
        raise Exception(f"For the traj {traf_file}, the proximal unit is first in the chain.")
    else:
        raise Exception(f"For the traj {traf_file}, the proximal unit is later in the chain.")


def replace_top_with_gromos(traj, ubq_site):
    with Capturing() as output:
        top_aa = CustomGromacsTopFile(
            f'/home/andrejb/Software/custom_tools/topology_builder/topologies/gromos54a7-isop/diUBQ_{ubq_site.upper()}/system.top',
            includeDir='/home/andrejb/Software/gmx_forcefields')
    top = md.Topology.from_openmm(top_aa.topology)
    traj.top = top


def h_bond_too_long(traj, bond):
    if any([a.element.symbol == 'H' for a in bond]):
        if get_bond_length(traj, bond) > 0.12:
            return True
    return False


def fix_pdb(traj, file, ubq_site, threshold=3, overwrite=False):
    print(file)
    sys.path.insert(0, f'/home/kevin/projects/anastasia')
    bond_lengths = {'C-C': 0.153, 'C-O': 0.123, 'O-C': 0.123, 'C-N': 0.134,
                    'N-C': 0.134, 'N-H': 0.1, 'H-N': 0.1, 'C-H': 0.109}
    from rotate import _get_far_and_near_networkx
    from transformations import translation_matrix
    replace_top_with_gromos(traj, ubq_site)
    for i, frame in enumerate(traj):
        for bond in frame.top.bonds:

            bond_length = get_bond_length(frame, bond)

            if bond_length > threshold or h_bond_too_long(frame, bond):

                # get the should be bond length
                bond_text = f"{bond[0].element.symbol}-{bond[1].element.symbol}"
                if not bond_text in bond_lengths:
                    raise Exception(f"{bond_text} not in bond_lengths")
                should_be_bond_length = bond_lengths[bond_text]

                # get near and far and swich if necessary
                near, far = _get_far_and_near_networkx(frame.top.to_bondgraph(), np.array([[bond[0].index, bond[1].index]]))
                near = near[0]
                far = far[0]
                print(bond, bond_length, len(near), len(far))
                assert bond[1].index in far
                if len(near) < len(far):
                    near, far = far, near
                # get the vector
                vector = frame.xyz[0, bond[1].index] - frame.xyz[0, bond[0].index]
                vector_length = np.linalg.norm(vector)
                new_vector = (vector / vector_length) * should_be_bond_length
                new_point = frame.xyz[0, bond[0].index] + new_vector
                translation_vector = new_point - frame.xyz[0, bond[1].index]

                # get the matrix
                trans_mat = translation_matrix(translation_vector)
                padded = np.pad(frame.xyz[0, far], ((0, 0), (0, 1)), mode='constant', constant_values=1)
                new_points = trans_mat.dot(padded.T).T[:, :3]
                traj.xyz[i, far] = new_points

                # check
                if not np.isclose(np.linalg.norm(traj.xyz[i, bond[0].index] - traj.xyz[i, bond[1].index]), should_be_bond_length, atol=0.1):
                    if len(far) == 1:
                        if bond[1].element.symbol == H:
                            atoms = np.setdiff1d(np.arange(traj.n_atoms), bond[1].index)
                            assert len(atoms) == traj.n_atoms - 1
                            traj = traj.atom_slice(atoms)
                            continue
                    raise Exception(f"Could not fix bond length for bond {bond}. It should be {should_be_bond_length}, but it is {np.linalg.norm(traj.xyz[i, bond[0].index] - traj.xyz[i, bond[1].index])}")
    if file is not None:
        traj.save_pdb(file)
    return traj


def map_nested_dicts(ob, func):
    import collections
    # https://stackoverflow.com/questions/32935232/python-apply-function-to-values-in-nested-dictionary
    if isinstance(ob, collections.Mapping):
        return {k: map_nested_dicts(v, func) for k, v in ob.items()}
    else:
        return func(ob)


################################################################################
# Classes
################################################################################


class EncodermapSPREAnalysis:
    """ A python singleton """

    def __init__(self, ubq_sites,
                 sim_dirs=['/home/andrejb/Research/SIMS/2017_*',
                          '/home/kevin/projects/molsim/diUbi_aa/'],
                 analysis_dir='/mnt/data/kevin/xplor_analysis_files/'):
        """Sets the ubq_sites of the singleton.

        Args:
            ubq_sites (list[str]): The ubq_sites to be looked at during this run.

        Keyword Args:
            sim_dirs (list[str], optional): Where to search for sims.
            analysis_dir (str, optional): Where to save intermediate analysis steps.

        """
        self.ubq_sites = ubq_sites
        self.sim_dirs = sim_dirs
        self.analysis_dir = analysis_dir
        if not os.path.isdir(self.analysis_dir):
            os.makedirs(self.analysis_dir)
        # self.cluster_exclusions = {'k6': [3], 'k29': [7], 'k33': [6]}

    @property
    def base_traj_k6(self):
        return md.load('/home/andrejb/Software/custom_tools/topology_builder/topologies/gromos54a7-isop/diUBQ_K6/0.pdb')

    @property
    def base_traj_k29(self):
        return md.load('/home/andrejb/Software/custom_tools/topology_builder/topologies/gromos54a7-isop/diUBQ_K29/0.pdb')

    @property
    def base_traj_k33(self):
        return md.load('/home/andrejb/Software/custom_tools/topology_builder/topologies/gromos54a7-isop/diUBQ_K33/0.pdb')

    @property
    def df_obs(self):
        return get_observed_df(self.ubq_sites)

    @property
    def fast_exchangers(self):
        return get_fast_exchangers(self.ubq_sites)

    @property
    def in_secondary(self):
        return get_in_secondary(self.ubq_sites)

    @property
    def df_comp(self):
        return self._df_comp

    @property
    def large_df_file(self):
        return '/mnt/data/kevin/xplor_analysis_files/lowd_and_xplor_df.csv'

    @property
    def centers_prox(self):
        return self._centers_prox

    @property
    def centers_dist(self):
        return self._centers_dist

    @df_comp.setter
    def df_comp(self, csv):
        if csv == 'conect':
            raise Exception("Need to reorder prox and dist")
            files = glob.glob(
                '/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package_with_conect/*.csv')
            sorted_files = sorted(files, key=get_iso_time)
            csv = sorted_files[-1]
            time = get_iso_time(csv)
            assert time > datetime.datetime.strptime("2021-03-11T00:00:00+0100", "%Y-%m-%dT%H:%M:%S%z")
        elif csv == 'no_conect':
            files = glob.glob(
                '/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package/*.csv')
            sorted_files = sorted(files, key=get_iso_time)
            csv = sorted_files[-1]
            time = get_iso_time(csv)
            assert time > datetime.datetime.strptime("2021-03-11T00:00:00+0100", "%Y-%m-%dT%H:%M:%S%z")
        elif csv == 'all_frames':
            files = glob.glob(
                '/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package_all/*.csv')
            sorted_files = sorted(files, key=get_iso_time)
            csv = sorted_files[-1]
            time = get_iso_time(csv)
            assert time > datetime.datetime.strptime("2021-03-11T00:00:00+0100", "%Y-%m-%dT%H:%M:%S%z")
        elif csv == 'legacy':
            raise Exception("Need to reorder prox and dist")
            csv = '/home/kevin/projects/tobias_schneider/values_from_every_frame/from_package/2021-07-23T16:49:44+02:00_df_no_conect.csv'
            time = get_iso_time(csv)
            assert time > datetime.datetime.strptime("2021-03-11T00:00:00+0100", "%Y-%m-%dT%H:%M:%S%z")
        df_comp = pd.read_csv(csv, index_col=0)
        if not 'ubq_site' in df_comp.keys():
            df_comp['ubq_site'] = df_comp['traj_file'].map(get_ubq_site)
        df_comp['ubq_site'] = df_comp['ubq_site'].str.lower()
        self._df_comp = df_comp
        _, self._centers_prox, self._centers_dist = normalize_sPRE(df_comp, self.df_obs)

    @property
    def norm_factors(self):
        file = 'norm_factors.yaml'
        if not os.path.isfile(file):
            data = normalize_sPRE(self.df_comp, self.df_obs, get_factors=True)
            data_w_strings = map_nested_dicts(data, str)
            with open(file, 'w') as f:
                yaml.safe_dump(data_w_strings, f)
        else:
            with open(file) as f:
                data_w_strings = yaml.safe_load(f)
            data = map_nested_dicts(data_w_strings, float)
        return data

    def set_cluster_exclusions(self, new_exclusions={'k6': [], 'k29': [], 'k33': []}):
        print(f"Setting {new_exclusions}")
        self.cluster_exclusions = new_exclusions
        
    def get_mixed_correlation_plots(self, type_of_corr=None, overwrite=False,
                                    exclude_0_and_nan=True):
        from scipy.stats import pearsonr
        if type_of_corr is None:
            type_of_corr = ['median_all', 'cluster_mean', 'cluster_mean', 'cluster_mean', 'cluster_combination', 'all_combination']
        for ubq_site_counting, ubq_site in enumerate(self.ubq_sites):
            image_name = f"/home/kevin/projects/tobias_schneider/cluster_analysis_with_fixed_normalization/{ubq_site}_correlation_regression.png"
            if os.path.isfile and not overwrite:
                print(f"File at {image_name} already exists.")
                continue
            print(ubq_site_counting, ubq_site)

            exp_value = self.df_obs[ubq_site][self.df_obs[ubq_site].index.str.contains('sPRE')].values

            sub_df = self.aa_df[self.aa_df['ubq_site'] == ubq_site]
            sPRE_ind = [i for i in sub_df.columns if 'sPRE' in i and 'norm' in i]

            # create plot
            plt.close('all')
            if len(type_of_corr) == 1:
                ncols = 1
            elif len(type_of_corr) <= 6:
                ncols = 2
            else:
                ncols = 3
            nrows = len(type_of_corr) // ncols
            nrows += len(type_of_corr) % ncols
            nrows += 1
            figsize = (5 * ncols, 5 * nrows)
            fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

            if len(type_of_corr) == 1:
                axes = np.array([[axes]])
            else:
                list_of_axes = axes.flatten()

            print('ncols: ', ncols, 'nrows: ', nrows)

            with pd.HDFStore(
                    '/home/kevin/projects/tobias_schneider/new_images/clusters.h5') as store:
                df, metadata = h5load(store)

            cluster_df = df[df['ubq site'] == ubq_site].copy()
            cluster_df = cluster_df.sort_values('N frames', ascending=False)
            cluster_df['count id'] = np.arange(len(cluster_df))
            cluster_df['color'] = [f'C{_}' for _ in range(1, len(cluster_df) + 1)]
            cluster_df = cluster_df.sort_values('coefficient', ascending=False)
            cluster_df = cluster_df.reset_index()

            # re-check normalization
            self.check_normalization()

            # define some sorting functions
            def sort_columns(type_, normed=False):
                if not normed:
                    def sort(x):
                        if 'sPRE' in x and type_ in x and 'norm' not in x:
                            return True
                        else:
                            return False
                else:
                    def sort(x):
                        if 'sPRE' in x and type_ in x and 'norm' in x:
                            return True
                        else:
                            return False
                return sort

            # plot prox, vs dist
            for ax, location in zip([axes[0, 0], axes[0, 1]], ['prox', 'dist']):
                non_norm_cols = list(filter(sort_columns(location, normed=False), sub_df.columns))
                non_norm_cols = list(sorted(non_norm_cols, key=lambda x: int(x.split()[-2][3:])))
                norm_cols = list(filter(sort_columns(location, normed=True), sub_df.columns))
                norm_cols = list(sorted(norm_cols, key=lambda x: int(x.split()[-2][3:])))
                x = sub_df[non_norm_cols].values.flatten()
                y = sub_df[norm_cols].values.flatten()
                mask = np.logical_and((~np.isnan(y)) & (~np.isnan(x)), (y != 0) & (x != 0))
                x = x[mask]
                y = y[mask]
                ax.scatter(x, y)
                ax.set_xlabel(r'simulated not normalized sPRE in $\mathrm{mM^{-1}ms^{-1}}$')
                ax.set_ylabel(r'simulated sPRE in $\mathrm{mM^{-1}ms^{-1}}$')
                ax.set_title(f"{ubq_site} {location} normalization.")
                ax.text(0.8, 0.95, s=r'$f_{norm, ' + ubq_site + ', ' + location + r'} = ' + f'{self.norm_factors[ubq_site][location]:.2f}$', va='center', ha='center', transform=ax.transAxes)

            linear_combination, cluster_means = make_linear_combination_from_clusters(None,
                                                                                      self.aa_df,
                                                                                      self.df_obs,
                                                                                      self.fast_exchangers,
                                                                                      ubq_site=ubq_site,
                                                                                      return_means=True)
            cluster_mean_counter = 0
            cluster_combination_ids = []
            cluster_combination_coeffs = []
            cluster_combination_count_ids = []
            for type_, ax in zip(type_of_corr, axes.flatten()[2:]):
                ax.set_xlabel(r'experimental sPRE in $\mathrm{mM^{-1}ms^{-1}}$')
                ax.set_ylabel(r'simulated sPRE in $\mathrm{mM^{-1}ms^{-1}}$')

                if type_ == 'median_all':
                    q1, median, q2 = sub_df[sPRE_ind].quantile([0.25, 0.5, 0.75], axis='rows').values
                    x = exp_value
                    y = median
                    if exclude_0_and_nan:
                        mask = np.logical_and((~np.isnan(y)) & (~np.isnan(x)), (y != 0) & (x != 0))
                        x = x[mask]
                        y = y[mask]
                        q1 = q1[mask]
                        q2 = q2[mask]
                    coef = np.polyfit(x, y, 1)
                    poly1d_fn = np.poly1d(coef)
                    ax.plot(x, y, 'o', x, poly1d_fn(x), '--k')
                    lower = q1 - y
                    upper = y - q2
                    ax.errorbar(x, y, yerr=(lower, upper), ls='none', c='C0')
                    ax.set_title("All sim values (median with IQR)")
                    corr, _ = pearsonr(x, y)
                    ax.text(0.8, 0.95, s=f'R = {corr:.2f}', va='center', ha='center', transform=ax.transAxes)

                if type_ == 'cluster_mean':
                    row = cluster_df.iloc[cluster_mean_counter]
                    cluster_mean_counter += 1
                    cluster_combination_ids.append(row['cluster id'])
                    cluster_combination_coeffs.append(row['coefficient'])
                    cluster_combination_count_ids.append(row['count id'])
                    coeff = row['coefficient']
                    color = row['color']
                    cluster_id = row['cluster id']
                    count_id = row['count id']
                    x = exp_value
                    y = cluster_means[cluster_id]
                    if exclude_0_and_nan:
                        mask = np.logical_and((~np.isnan(y)) & (~np.isnan(x)), (y != 0) & (x != 0))
                        x = x[mask]
                        y = y[mask]
                    coef = np.polyfit(x, y, 1)
                    poly1d_fn = np.poly1d(coef)
                    ax.plot(x, y, 'o', x, poly1d_fn(x), '--k')
                    ax.set_title(f"Median of cluster {count_id} (no coefficient)")
                    corr, _ = pearsonr(x, y)
                    ax.text(0.8, 0.95, s=f'R = {corr:.2f}', va='center', ha='center', transform=ax.transAxes)

                if type_ == 'all_combination':
                    sim_value = np.sum(linear_combination * cluster_means.T, 1)
                    x = exp_value
                    y = sim_value
                    if exclude_0_and_nan:
                        mask = np.logical_and((~np.isnan(y)) & (~np.isnan(x)), (y != 0) & (x != 0))
                        x = x[mask]
                        y = y[mask]
                    coef = np.polyfit(x, y, 1)
                    poly1d_fn = np.poly1d(coef)
                    ax.plot(x, y, 'o', x, poly1d_fn(x), '--k')
                    ax.set_title(f"Combination of all {len(cluster_df)} clusters (with coefficients).")
                    corr, _ = pearsonr(x, y)
                    ax.text(0.8, 0.95, s=f'R = {corr:.2f}', va='center', ha='center', transform=ax.transAxes)

                if type_ == 'cluster_combination':
                    where = np.array(cluster_combination_ids)
                    cluster_combination_count_ids = np.array(cluster_combination_count_ids).astype(str)
                    sim_value = np.sum(cluster_combination_coeffs * cluster_means[where].T, 1)
                    x = exp_value
                    y = sim_value
                    if exclude_0_and_nan:
                        mask = np.logical_and((~np.isnan(y)) & (~np.isnan(x)), (y != 0) & (x != 0))
                        x = x[mask]
                        y = y[mask]
                    coef = np.polyfit(x, y, 1)
                    poly1d_fn = np.poly1d(coef)
                    ax.plot(x, y, 'o', x, poly1d_fn(x), '--k')
                    ax.set_title(f"Combination of clusters {', '.join(cluster_combination_count_ids)} (with coefficients)")
                    corr, _ = pearsonr(x, y)
                    ax.text(0.8, 0.95, s=f'R = {corr:.2f}', va='center', ha='center', transform=ax.transAxes)

            plt.savefig(image_name)

    def prepare_csv_files(self, overwrite=False):
        with pd.HDFStore(
                '/home/kevin/projects/tobias_schneider/new_images/clusters_with_and_without_coeff.h5') as store:
            df, metadata = h5load(store)

        for ubq_count, ubq_site in enumerate(self.ubq_sites):
            csv_file = f'/home/kevin/projects/tobias_schneider/cluster_analysis_with_fixed_normalization/per_residue_values_and_combinations_{ubq_site}.csv'
            linear_combination, cluster_means_norm, cluster_mean_not_norm = make_linear_combination_from_clusters(None,
                                                                                      self.aa_df,
                                                                                      self.df_obs,
                                                                                      self.fast_exchangers,
                                                                                      ubq_site=ubq_site,
                                                                                      return_means=True,
                                                                                      return_non_norm_means=True)

            cluster_df = df[df['ubq site'] == ubq_site]
            cluster_df = cluster_df.sort_values('N frames', ascending=False)
            cluster_df['count id'] = np.arange(len(cluster_df))
            cluster_df['color'] = [f'C{_}' for _ in range(1, len(cluster_df) + 1)]
            # for i, row in cluster_df.iterrows():
            #     print(row['cluster id'], linear_combination[row['cluster id']], row['coefficient'])

            out_df = {'cluster id (count)': [], 'coefficient': []}

            columns_norm = [i for i in self.df_comp_norm.columns if 'sPRE' in i and 'norm' in i]
            test_index = columns_norm[0]
            # columns_norm = [f'median (without coeff) {i}' for i in columns_norm_]
            columns_non_norm = [i for i in self.df_comp_norm.columns if 'sPRE' in i and 'norm' not in i]
            # columns_non_norm = [f'median (without coeff) {i}' for i in columns_non_norm_]
            out_df.update({i: [] for i in columns_norm})
            out_df.update({i: [] for i in columns_non_norm})

            # iterate over clusters
            for i, row in cluster_df.iterrows():
                out_df['cluster id (count)'].append(row['count id'])
                out_df['coefficient'].append(row['coefficient'])
                for resid, (col_norm, col_non_norm, val_norm, val_non_norm) in enumerate(zip(columns_norm, columns_non_norm, cluster_means_norm[row['cluster id']], cluster_mean_not_norm[row['cluster id']])):
                    out_df[col_norm].append(val_norm)
                    out_df[col_non_norm].append(val_non_norm)
                    if resid < 76:
                        factor = self.norm_factors[ubq_site][0]
                    else:
                        factor = self.norm_factors[ubq_site][1]
                    print(val_norm / val_non_norm, factor)

            # get full combination
            final_combination = np.sum(linear_combination * cluster_means_norm.T, 1)
            out_df['cluster id (count)'].append('all combination')
            out_df['coefficient'].append(np.nan)
            for col_norm, val_combi in zip(columns_norm, final_combination):
                out_df[col_norm].append(val_combi)
            for col_non_norm in columns_non_norm:
                out_df[col_non_norm].append(np.nan)
            # print(len(out_df[test_index]))

            # get median of all
            _ = self.df_comp_norm[self.df_comp_norm['ubq_site'] == ubq_site]
            median_all_norm = np.median(_[columns_norm], 0)
            median_all_non_norm = np.median(_[columns_non_norm], 0)
            out_df['cluster id (count)'].append('all median')
            out_df['coefficient'].append(np.nan)
            for iter_, (col_norm, val_norm) in enumerate(zip(columns_norm, median_all_norm)):
                # if iter_ in [0, 1, 2, 3]:
                #     print(iter_, col_norm, val_norm)
                out_df[col_norm].append(val_norm)
            for iter_, (col_non_norm, val_non_norm) in enumerate(zip(columns_non_norm, median_all_non_norm)):
                # if iter_ in [0, 1, 2, 3]:
                #     print(iter_, col_non_norm, val_non_norm)
                out_df[col_non_norm].append(val_non_norm)
            # print(len(out_df[test_index]))

            # get limited combination
            if ubq_site == 'k6':
                clusters = [5, 9]
                clusters_str = 'combination of clusters 5 and 9'
            if ubq_site == 'k29':
                clusters = [0, 12]
                clusters_str = 'combination of clusters 0 and 12'
            if ubq_site == 'k33':
                clusters = [20]
                clusters_str = 'cluster 20 with coefficient'

            # get the cluster ids
            cluster_ids = []
            coefficients = []
            for i, row in cluster_df.iterrows():
                if row['count id'] in clusters:
                    cluster_ids.append(row['cluster id'])
                    coefficients.append(row['coefficient'])
            cluster_ids = np.array(cluster_ids)
            coefficients = np.array(coefficients)
            cluster_combination = np.sum(coefficients * cluster_means_norm[cluster_ids].T, 1)
            out_df['cluster id (count)'].append(clusters_str)
            out_df['coefficient'].append(np.nan)
            for col_norm, val_clu_combi in zip(columns_norm, cluster_combination):
                out_df[col_norm].append(val_clu_combi)
            for col_non_norm in columns_non_norm:
                out_df[col_non_norm].append(np.nan)
            # print(len(out_df[test_index]))

            out_df = pd.DataFrame(out_df)
            out_df[columns_norm + columns_non_norm] = out_df[columns_norm + columns_non_norm].replace(0, np.nan)
            out_df.to_csv(csv_file)
            out_df.to_excel(csv_file.replace('.csv', '.xlsx'))

    def add_centroids_to_df(self, overwrite=False, overwrite_rows=False, testing=False):
        if 'rmsd_centroid' in self.aa_df and 'geom_centroid' in self.aa_df and not overwrite:
            print(pd.value_counts(self.aa_df['geom_centroid']))
            print("Centroids already there")
        copied_ubq_sites = copy.deepcopy(self.ubq_sites)
        try:
            sub_dfs = []
            for ubq_num, ubq_site in enumerate(self.ubq_sites):
                self.ubq_sites = [ubq_site]
                if not os.path.isfile(f'/mnt/data/kevin/xplor_analysis_files/sub_df_for_saving_{ubq_site}.csv') or overwrite:
                    sub_df = self.aa_df[self.aa_df['ubq_site'] == ubq_site]
                    sub_df['geom_centroid'] = -1
                    sub_df['20_selected_for_rendering'] = -1
                    sub_df['rmsd_centroid'] = -1
                    sub_df['aa_percent'] = 0
                    sub_df['ensemble_percent'] = 0
                    sub_df['internal_rmsd'] = 0.0
                else:
                    sub_df = pd.read_csv(f'/mnt/data/kevin/xplor_analysis_files/sub_df_for_saving_{ubq_site}.csv', index_col=0)
                if testing:
                    if ubq_site != 'k6':
                        break
                for cluster_iter, (cluster_num, count) in enumerate(zip(*np.unique(sub_df['cluster_membership'], return_counts=True))):
                    if cluster_num == -1:
                        continue

                    if testing:
                        if cluster_num >= 3:
                            break

                    print(f"At cluster num {cluster_num} for ubq_site {ubq_site}.")
                    if 'geom_centroid' in sub_df:
                        if cluster_num in sub_df['geom_centroid'].unique() and not overwrite_rows and not overwrite:
                            print(f"Cluster {cluster_num} for ubq_site {ubq_site} already in self.aa_df")
                            continue

                    # geom_centroid
                    xy = sub_df[sub_df['cluster_membership'] == cluster_num][['x', 'y']].values
                    assert len(xy) == count
                    centroid = np.mean(xy, axis=0)
                    index_closest = index_of_closest_point(centroid, xy)
                    traj_file, frame, x, y = sub_df[sub_df['cluster_membership'] == cluster_num].iloc[index_closest][['traj_file', 'frame', 'x', 'y']]
                    test = xy[index_closest]
                    assert x == test[0] and y == test[1]
                    sub_df.at[(sub_df['traj_file'] == traj_file) & (sub_df['frame'] == frame), 'geom_centroid'] = cluster_num
                    counts = pd.value_counts(sub_df['geom_centroid'])
                    assert counts[cluster_num] == 1

                    # also add some 20 more structures with this cluster to a selected_structs_column
                    where = np.where(sub_df['cluster_membership'] == cluster_num)[0]
                    idx = np.round(np.linspace(0, len(where) - 1, 20)).astype(int)
                    where = where[idx]
                    for w in where:
                        sub_df.iat[w, sub_df.columns.get_loc('20_selected_for_rendering')] = cluster_num
                    counts = pd.value_counts(sub_df['20_selected_for_rendering'])
                    assert 19 <= counts[cluster_num] <= 21, print(ubq_site, cluster_num, counts)

                    # also add internal RMSD, when we are already here.
                    self.load_trajs()
                    self.cluster()

                    # get the RMSD centroidf for internal RMSD calculations
                    max_frames = 500
                    ref, idx, labels = center_ref_and_load()
                    warnings.warn("The creation of the dummy_traj does not use a ref and thus should not be saved. "
                                  "Don't use this code for saving pdbs. Also add another superpose to the crystal structure.")
                    view, dummy_traj = _gen_dummy_traj_single(self.aa_trajs[ubq_site], cluster_num,
                                                              max_frames=max_frames,
                                                              superpose=False, stack_atoms=False,
                                                              align_string='name CA and resid >= 76',
                                                              ref_align_string='name CA',
                                                              base_traj=getattr(self, f'base_traj_{ubq_site}'))
                    # align to cryst again
                    # only needed for actual saving
                    # assert not prox_is_first(dummy_traj)
                    # cryst = md.load('/home/kevin/1UBQ.pdb')
                    # dummy_traj.superpose(reference=cryst, atom_indices=dummy_traj.top.select('name CA and resid >= 76'),
                    #                      ref_atom_indices=cryst.top.select('name CA'))

                    # rejig where to fit onto all trajs
                    where = np.where(self.aa_trajs[ubq_site].cluster_membership == cluster_num)[0]
                    idx = np.round(np.linspace(0, len(where) - 1, max_frames)).astype(int)
                    where = where[idx]
                    index, mat, centroid = rmsd_centroid_of_cluster(dummy_traj)
                    where = where[index]

                    # this test succeeded without the supoerposing stuff.
                    test = self.aa_trajs[ubq_site].get_single_frame(where)
                    assert np.array_equal(test.xyz, centroid.xyz)

                    # get the file and frame of test
                    print(test.traj_file, test.index, type(test.index))

                    sub_df.at[(sub_df['traj_file'] == test.traj_file) & (sub_df['frame'] == test.index), 'rmsd_centroid'] = cluster_num
                    counts = pd.value_counts(sub_df['rmsd_centroid'])
                    assert counts[cluster_num] == 1

                    # get aa percent
                    aa = np.load(os.path.join(self.analysis_dir, f'cluster_membership_aa_{ubq_site}.npy'))
                    cg = np.load(os.path.join(self.analysis_dir, f'cluster_membership_cg_{ubq_site}.npy'))
                    n_aa_in_cluster = len(np.where(aa == cluster_num)[0])
                    n_cg_in_cluster = len(np.where(cg == cluster_num)[0]) # was missing the [0]
                    percent = n_aa_in_cluster / n_cg_in_cluster * 100
                    ensemble_percent = (n_aa_in_cluster + n_cg_in_cluster) / (len(aa) + len(cg)) * 100
                    del aa
                    del cg
                    sub_df.at[sub_df['rmsd_centroid'] == cluster_num, 'aa_percent'] = percent
                    sub_df.at[sub_df['rmsd_centroid'] == cluster_num, 'ensemble_percent'] = ensemble_percent

                    # internal rmsd
                    aa_where = np.where(self.aa_trajs[ubq_site].cluster_membership == cluster_num)[0]
                    indices = self.aa_trajs[ubq_site].index_arr[aa_where]
                    for j, (traj_num, frame_num) in enumerate(indices):
                        if j % 250 == 0:
                            print(f"Building cluster for ubq_site {ubq_site} traj for cluster {cluster_num}. At step {j}")
                        if j == 0:
                            traj = self.aa_trajs[ubq_site][traj_num][frame_num].traj
                        else:
                            traj = traj.join(self.aa_trajs[ubq_site][traj_num][frame_num].traj)
                    internal_rmsd = md.rmsd(traj, centroid)
                    sub_df.at[sub_df['rmsd_centroid'] == cluster_num, 'internal_rmsd'] = np.var(internal_rmsd)
                else:
                    if hasattr(self, 'trajs'): del self.trajs
                    if hasattr(self, 'aa_trajs'): del self.aa_trajs
                    if hasattr(self, 'cg_trajs'): del self.cg_trajs
                    if hasattr(self, 'aa_traj_files'): del self.aa_traj_files
                    if hasattr(self, 'cg_traj_files'): del self.cg_traj_files
                    if hasattr(self, 'aa_references'): del self.aa_references
                    if hasattr(self, 'cg_references'): del self.cg_references
                sub_dfs.append(sub_df)
                # sub_df.to_csv(f'/mnt/data/kevin/xplor_analysis_files/sub_df_for_saving_{ubq_site}.csv')
            else:
                assert len(sub_dfs) == len(copied_ubq_sites)
                new_df = pd.concat(sub_dfs)
                lengths = sum(len(sub_df) for sub_df in sub_dfs)
                assert len(new_df) == lengths
                new_df['geom_centroid'] = new_df['geom_centroid'].astype(int)
                new_df['rmsd_centroid'] = new_df['rmsd_centroid'].astype(int)
                assert 1 in new_df[new_df['ubq_site'] == ubq_site]['geom_centroid'].unique()
                assert 1 in new_df[new_df['ubq_site'] == ubq_site]['rmsd_centroid'].unique()
                assert pd.value_counts(new_df['geom_centroid'])[2] == 3
                assert pd.value_counts(new_df['rmsd_centroid'])[2] == 3
                assert 'internal_rmsd' in new_df
                if not testing:
                    self.aa_df = new_df
        finally:
            self.ubq_sites = copied_ubq_sites
            if hasattr(self, 'trajs'): del self.trajs
            if hasattr(self, 'aa_trajs'): del self.aa_trajs
            if hasattr(self, 'cg_trajs'): del self.cg_trajs
            if hasattr(self, 'aa_traj_files'): del self.aa_traj_files
            if hasattr(self, 'cg_traj_files'): del self.cg_traj_files
            if hasattr(self, 'aa_references'): del self.aa_references
            if hasattr(self, 'cg_references'): del self.cg_references

        self.aa_df.to_csv(self.large_df_file)
        self.check_normalization()

    def fix_broken_pdbs(self):
        for ubq_num, ubq_site in enumerate(self.ubq_sites):
            sub_df = self.aa_df[self.aa_df['ubq_site'] == ubq_site]
            for count_id in pd.unique(sub_df['count_id']):
                if count_id == -1:
                    continue
                # df = sub_df[sub_df['count_id'] == count_id]
                pdb_file = f'/home/kevin/projects/tobias_schneider/cluster_analysis_with_fixed_normalization/{ubq_site}/cluster_{count_id}/cluster.pdb'
                traj = md.load(pdb_file)
                if ubq_site == 'k33':
                    if count_id in [12, 16, 18, 22]:
                        fix_pdb(traj, pdb_file, ubq_site)
                if one_bond_too_long(traj, 3):
                    fix_pdb(traj, pdb_file, ubq_site)

    def analyze_mean_abs_diff_all(self, overwrite=False):
        for i, ubq_site in enumerate(self.ubq_sites):
            image_name = f"/home/kevin/projects/tobias_schneider/test_sPRE_values/{ubq_site}_scatter.png"
            os.makedirs(os.path.split(image_name)[0], exist_ok=True)
            sub_df = self.aa_df[self.aa_df['ubq_site'] == ubq_site]
            print(len(sub_df))
            # self.check_normalization()
            if 'mean_abs_diff' in sub_df:
                redo = (sub_df['mean_abs_diff'] == 0.0).all(None)
            else:
                redo = True
            if redo or overwrite:
                sim_data = sub_df[[c for c in sub_df.columns if 'sPRE' in c and 'normalized' in c]].copy()
                sim_data = sim_data.rename(columns={k: k.replace('normalized ', '') for k in sim_data.columns})
                exp_data = self.df_obs[ubq_site][[r for r in self.df_obs.index if 'sPRE' in r]].to_frame().T
                diff = set(sim_data.columns).difference(set(exp_data.columns))
                assert not diff
                sim_data = sim_data[exp_data.columns]
                diff = np.subtract(sim_data, exp_data)
                abs_ = abs(diff)
                mean = np.mean(abs_, axis=1)
                assert len(mean) == len(sub_df)
                sub_df['mean_abs_diff'] = mean
                self.aa_df = self.aa_df.combine_first(sub_df)

            plt.close('all')
            plt.scatter(sub_df['x'], sub_df['y'], c=sub_df['mean_abs_diff'], cmap='viridis', s=5)
            plt.savefig(image_name)

    def make_large_df(self, overwrite=False):
        """Loads trajs, their lowd, their cluster membership and puts it into a large df.

        To prevent mislabelling of dimensions trajs highd lowd and everything else will be
        deleted afterwards. This new df is the only access to the class from now on.

        """
        if hasattr(self, 'aa_df') and not overwrite:
            print("everything already loaded.")
            return

        if os.path.isfile(self.large_df_file) and not overwrite:
            self.aa_df = pd.read_csv(self.large_df_file, index_col=0)
        else:
            copied_ubq_sites = copy.deepcopy(self.ubq_sites)
            try:
                sub_dfs = []
                for ubq_num, ubq_site in enumerate(copied_ubq_sites):
                    sub_df = self.df_comp[self.df_comp['ubq_site'] == ubq_site]
                    print(f"Making large_df for {ubq_site}")

                    self.ubq_sites = [ubq_site]

                    self.load_trajs()
                    self.load_highd()
                    self.train_encodermap()
                    self.cluster()

                    # prepare data to be stroed in df
                    files_arr = []
                    frames_arr = []
                    index_arr = self.aa_trajs[ubq_site].index_arr
                    for traj in self.aa_trajs[ubq_site]:
                        files_arr.extend([traj.traj_file for i in range(traj.n_frames)])
                        frames_arr.extend([i for i in range(traj.n_frames)])
                    files_arr = np.array(files_arr)
                    frames_arr = np.array(frames_arr)
                    x = self.aa_trajs[ubq_site].lowd[:, 0]
                    y = self.aa_trajs[ubq_site].lowd[:, 1]
                    _, _, resnames = center_ref_and_load()
                    RWMD_columns = [f'RWMD prox {_}' for _ in resnames] + [f'RWMD dist {_}' for _ in resnames]
                    RWMD = self.aa_trajs[ubq_site].highd
                    cluster_membership = self.aa_trajs[ubq_site].cluster_membership

                    # prepare an dataframe with x, y and RWMD
                    data = {'traj_file': files_arr, 'frame': frames_arr,
                           'x': x, 'y': y, 'cluster_membership': cluster_membership,
                            **{k: v for k, v in zip(RWMD_columns, RWMD.T)}}
                    df = pd.DataFrame(data)

                    print(f"For ubq_site {ubq_site} df from all atoms is {df.shape[0] - sub_df.shape[0]} longer than df with sPRE_values")
                    sub_df = pd.merge(sub_df, df, how='inner')
                    sub_dfs.append(sub_df)

                    # delete all unwanted attributes
                    del self.trajs[ubq_site]
                    del self.aa_trajs[ubq_site]
                    del self.cg_trajs[ubq_site]
                    del self.aa_traj_files[ubq_site]
                    del self.cg_traj_files[ubq_site]
                    del self.aa_references[ubq_site]
                    del self.cg_references[ubq_site]
                    del self.aa_dists[ubq_site]
                    del self.cg_dists[ubq_site]

                self.aa_df = pd.concat(sub_dfs, ignore_index=True)
                self.aa_df = normalize_sPRE(self.aa_df, self.df_obs)
                self.aa_df.to_csv(self.large_df_file)

                del self.trajs
                del self.aa_trajs
                del self.cg_trajs
                del self.aa_traj_files
                del self.cg_traj_files
                del self.aa_references
                del self.cg_references
                del self.aa_dists
                del self.cg_dists

            finally:
                self.ubq_sites = copied_ubq_sites

    def add_count_ids(self, overwrite=False):
        if not 'count_id' in self.aa_df or overwrite:
            add_dict = {}
            for ubq_num, ubq_site in enumerate(self.ubq_sites):
                add_dict[ubq_site] = {-1: -1}
                aa = np.load(os.path.join(self.analysis_dir, f'cluster_membership_aa_{ubq_site}.npy'))
                cg = np.load(os.path.join(self.analysis_dir, f'cluster_membership_cg_{ubq_site}.npy'))
                all_clu_memberships = np.hstack([aa, cg])
                uniques, counts = np.unique(all_clu_memberships, return_counts=True)
                missing = np.in1d(uniques, np.unique(aa))
                uniques = uniques[missing]
                counts = counts[missing]
                indices = np.argsort(np.argsort(counts[1:])[::-1])
                if ubq_site == 'k29':
                    assert 18 not in indices
                for u, c, i in zip(uniques[1:], counts[1:], indices):
                    add_dict[ubq_site][u] = i
                del all_clu_memberships

            def add_count_ids(row):
                return add_dict[row['ubq_site']][int(row['cluster_membership'])]

            self.aa_df['count_id'] = self.aa_df.apply(add_count_ids, axis=1)
            df_file = '/mnt/data/kevin/xplor_analysis_files/lowd_and_xplor_df.csv'
            self.aa_df.to_csv(df_file)

    def analyze(self, **overwrite_dict):
        """Start the analysis

        Stages:
            * load_trajs: Find the files and create encodermap.Info_all from them.
            * load_highd: Use pyEMMA to load the RWMD.

        Keyword Args:
            overwrite_dict (Dict[str, bool], optional): A dictionary of booleans
                telling the analysis methods, whether to redo already carried out
                analysis. Defaults to a dict full of str: False value pairs. The
                keys are named according to the stages.

        """
        self.set_cluster_exclusions()
        ow_dict = {'load_trajs': False, 'load_highd': False, 'train_encodermap': False,
                   'load_xplor_data': False, 'cluster': False, 'plot_lowd': False,
                   'cluster_analysis': False, 'fitness_assessment': False,
                   'get_surface_coverage': False, 'distance_vs_pseudo_torsion': False,
                   'cluster_pseudo_torsion': False, 'where_is_best_fitting_point': False,
                   'find_lowest_diffs_in_all_quality_factors': False, 'plot_cluster_rmsds': False}
        ow_dict.update(overwrite_dict)
        self.load_trajs(overwrite=ow_dict['load_trajs'])
        self.load_highd(overwrite=ow_dict['load_highd'])
        self.train_encodermap(overwrite=ow_dict['train_encodermap'])
        self.load_xplor_data(overwrite=ow_dict['load_xplor_data'])
        self.cluster(overwrite=ow_dict['cluster'])
        self.cluster_analysis(overwrite=ow_dict['cluster_analysis'])
        self.where_is_best_fitting_point(overwrite=ow_dict['where_is_best_fitting_point'])
        self.find_lowest_diffs_in_all_quality_factors(overwrite=ow_dict['find_lowest_diffs_in_all_quality_factors'])
        self.plot_cluster_rmsds(overwrite=ow_dict['plot_cluster_rmsds'])
        self.fitness_assessment(overwrite=ow_dict['fitness_assessment'])
        self.get_surface_coverage(overwrite=ow_dict['get_surface_coverage'])
        self.distance_vs_pseudo_torsion(overwrite=ow_dict['distance_vs_pseudo_torsion'])
        self.cluster_pseudo_torsion(overwrite=ow_dict['cluster_pseudo_torsion'])

    def check_attr_all(self, attr_name):
        for ubq_site in self.ubq_sites:
            for _ in [self.trajs, self.cg_trajs, self.aa_trajs]:
                if not hasattr(_[ubq_site], attr_name):
                    return False
        return True

    def write_clusters(self, directory, clusters, which, max_frames=500, pdb='rmsd_centroid', align_to_cryst=True):
        df_file = '/home/kevin/projects/tobias_schneider/new_images/clusters.h5'
        with pd.HDFStore(df_file) as store:
            df, metadata = h5load(store)

        for ubq_site, cluster_ids in clusters.items():
            sub_df = df[df['ubq site'] == ubq_site]
            sub_df = sub_df.reset_index()
            if which == 'cluster_id':
                cluster_nums = cluster_ids
            elif which == 'count_id':
                sub_df['count id'] = np.argsort(np.argsort(- sub_df['N frames']))
                cluster_nums = []
                for cluster_id in cluster_ids:
                    cluster_nums.append(sub_df['cluster id'][sub_df['count id'] == cluster_id].values[0])
            else:
                raise Exception(f"Whcih needs to be either 'cluster_ids' or 'count_ids'. You supplied {which}.")

            if align_to_cryst:
                ref, idx, labels = center_ref_and_load()
            else:
                reference = True

            for cluster_num, cluster_id in zip(cluster_nums, cluster_ids):
                print(cluster_num, cluster_nums)
                cluster_dir = os.path.join(directory, f'{ubq_site}/cluster_{cluster_id}')
                os.makedirs(cluster_dir, exist_ok=True)
                pdb_file = os.path.join(cluster_dir, 'cluster.pdb')
                xtc_file = os.path.join(cluster_dir, 'cluster.xtc')

                view, dummy_traj = _gen_dummy_traj_single(self.aa_trajs[ubq_site], cluster_num, max_frames=max_frames,
                                                  superpose=ref, stack_atoms=False,
                                                  align_string='name CA and resid >= 76', ref_align_string='name CA',
                                                  base_traj=getattr(self, f'base_traj_{ubq_site}'))

                # align to cryst again
                assert not prox_is_first(dummy_traj)
                cryst = md.load('/home/kevin/1UBQ.pdb')
                dummy_traj.superpose(reference=cryst, atom_indices=dummy_traj.top.select('name CA and resid >= 76'),
                                     ref_atom_indices=cryst.top.select('name CA'))
                index = []
                for frame_no, frame in enumerate(dummy_traj):
                    if not one_bond_too_long(frame, 3):
                        index.append(frame_no)
                if len(index) > 0:
                    raise Exception("Fix the bonds")
                    print(f"Found {dummy_traj.n_frames - len(index)} bonds with too long bond lengths for {ubq_site}, {cluster_num}.")
                    index = np.array(index)
                    dummy_traj = dummy_traj[index]
                    assert dummy_traj.n_frames > 1
                else:
                    print("No bonds too long")

                if isinstance(pdb, str):
                    if pdb == 'rmsd_centroid':
                        where = np.where(self.aa_trajs[ubq_site].cluster_membership == cluster_num)[0]
                        idx = np.round(np.linspace(0, len(where) - 1, max_frames)).astype(int)
                        where = where[idx]
                        index, mat, centroid = rmsd_centroid_of_cluster(dummy_traj)
                        save_pdb = centroid
                        rmsd_centroid_index_file = os.path.join(cluster_dir, 'cluster_rmsd_centroid_index.npy')
                        rmsd_centroid_index = where[index]
                        np.save(rmsd_centroid_index_file, rmsd_centroid_index)
                    elif pdb == 'geom_centroid':
                        raise NotImplementedError()
                    else:
                        raise Exception(f"Argument `pdb` can be `int` or 'rmsd_centroid', or 'geom_centroid'. You supplied {pdb}")
                elif isinstance(pdb, int):
                    save_pdb = dummy_traj[pdb]
                else:
                    raise Exception(f"Argument `pdb` must be int, or str, you supplied {type(pdb)}..")

                save_pdb.save_pdb(pdb_file)
                print(f"Saving as {pdb_file}")
                dummy_traj.save_xtc(xtc_file)

    def reassign_df_comp(self):
        print(self.df_comp.shape)
        print(self.aa_df.shape)

        # make sure the traj_columns are identical
        aa_df = self.aa_df.sort_values(['ubq_site', 'traj_file', 'frame'])
        aa_df = aa_df.reset_index()
        df_comp = self.df_comp.sort_values(['ubq_site', 'traj_file', 'frame'])
        df_comp = df_comp.reset_index()
        assert np.all((df_comp[['traj_file', 'frame']] == aa_df[['traj_file', 'frame']]).values)

        aa_df = self.aa_df.copy()
        df_comp = self.df_comp.copy()[['ubq_site', 'traj_file', 'frame', *[c for c in df_comp if 'sPRE' in c]]]
        aa_df.combine_first(df_comp)

        aa_df = aa_df.loc[:, ~aa_df.columns.str.contains('^Unnamed')]

        self.aa_df = aa_df
        self.aa_df, self.centers_prox, self.centers_dist = normalize_sPRE(aa_df, self.df_obs)
        # self.aa_df.to_csv(self.large_df_file)

    def check_normalization(self, already_checked_approving_reassurance=False):
        for ubq_num, ubq_site in enumerate(self.ubq_sites):
            sub_df = self.aa_df[self.aa_df['ubq_site'] == ubq_site]
            for check in ['proximal', 'distal']:
                try:
                    factor_should_be = self.norm_factors[ubq_site][check]
                except KeyError:
                    if check == 'proximal':
                        c = 'prox'
                    if check == 'distal':
                        c = 'dist'
                    factor_should_be = self.norm_factors[ubq_site][c]
                non_norm_cols = [c for c in sub_df.columns if 'sPRE' in c and check in c and 'norm' not in c]
                norm_cols = [c for c in sub_df.columns if 'sPRE' in c and check in c and 'norm' in c]

                non_norm_values = sub_df[non_norm_cols].values
                norm_values = sub_df[norm_cols].values

                if non_norm_values.shape != norm_values.shape:
                    print(f"The shapes of norm {norm_values.shape} does not match the shape of non_norm {non_norm_values.shape}. I will attempt to redo the normalization.")
                    if already_checked_approving_reassurance:
                        raise Exception("After two calls to reassign df_comp I could not fix the shape.")
                    else:
                        self.reassign_df_comp()
                        self.check_normalization(True)

                factor_is = np.unique((norm_values / non_norm_values).round(10))
                mask = np.logical_and(~np.isnan(factor_is), factor_is != 0)
                factor_is = factor_is[mask]

                if np.allclose(factor_is, factor_should_be):
                    print(f'{ubq_site} at {check} is close enough.')
                    continue

                if len(factor_is) > 1:
                    if already_checked_approving_reassurance:
                        print(factor_is)
                        raise Exception("After two calls to reassign df_comp I could not fix the factors.")
                    else:
                        self.reassign_df_comp()
                        self.check_normalization(True)

                if already_checked_approving_reassurance:
                    if np.isclose(factor_is, factor_should_be):
                        print(f"{ubq_site} {check} was correctly normalized.")
                    else:
                        print(factor_is, factor_should_be)
                        raise Exception("After two calls to reassign df_comp I could not fix the factors.")

    def run_per_cluster_analysis(self, overwrite=False, overwrite_df_creation=False,
                                 overwrite_polar_plots=False, overwrite_pdb_files=False,
                                 overwrite_final_combination=False, overwrite_surface_coverage=False,
                                 overwrite_final_correlation=False,
                                 coeff_threshold=0.1):
        """Do the following steps for the clusters:

        * Render wireframe

        """
        df_file = '/home/kevin/projects/tobias_schneider/new_images/clusters.h5'

        if not hasattr(self, 'aa'):
            self.aa = {ubq_site: np.load(os.path.join(self.analysis_dir, f'cluster_membership_aa_{ubq_site}.npy')) for ubq_site in self.ubq_sites}
        if not hasattr(self, 'cg'):
            self.cg = {ubq_site: np.load(os.path.join(self.analysis_dir, f'cluster_membership_cg_{ubq_site}.npy')) for ubq_site in self.ubq_sites}

        if os.path.isfile(df_file) and not overwrite and not overwrite_df_creation:
            print("df file exists. Not overwriting")
            with pd.HDFStore(df_file) as store:
                df, metadata = h5load(store)
                if isinstance(df.loc[0, 'internal RMSD'], np.ndarray):
                    df['internal RMSD'] = df['internal RMSD'].apply(np.var)
        else:
            metadata = {}

            for ubq_num, ubq_site in enumerate(pd.unique(self.aa_df['ubq_site'])):
                sub_df = self.aa_df[self.aa_df['ubq_site'] == ubq_site]

                linear_combination, cluster_means = make_linear_combination_from_clusters(None,
                                                                                          sub_df,
                                                                                          self.df_obs,
                                                                                          self.fast_exchangers,
                                                                                          ubq_site=ubq_site,
                                                                                          return_means=True)

                # check whether normalizations still hold true
                self.check_normalization()

                assert np.isclose(np.sum(linear_combination), 1)

                # get some values true for all clusters and start the dict with data
                aa_cluster_nums = np.unique(sub_df['cluster_membership'])
                exp_values = self.df_obs[ubq_site][self.df_obs[ubq_site].index.str.contains('sPRE')].values
                mean_abs_diff_full_combination = np.mean(np.abs(np.sum(linear_combination * cluster_means.T, 1) - exp_values))

                df_data = {'cluster id': [], 'N frames': [], 'ensemble %': [],
                           'ubq site': [], 'aa %': [], 'internal RMSD': [], 'coefficient': [],
                           'mean abs diff to exp w/ coeff': [], 'mean abs diff to exp w/o coeff': []}

                for cluster_num, (coefficient, cluster_mean) in enumerate(zip(linear_combination, cluster_means)):
                    if cluster_num == -1:
                        continue
                    if cluster_num not in aa_cluster_nums:
                        print(f"Cluster {cluster_num} not in aa trajs")
                        continue

                    print(f"At cluster num {cluster_num}")
                    points_in_cluster = len(np.where(self.aa[ubq_site] == cluster_num)[0]) + len(np.where(self.cg[ubq_site] == cluster_num)[0])
                    ensemble_percent = points_in_cluster / (len(self.aa[ubq_site]) + len(self.cg[ubq_site])) * 100
                    aa_percent = len(np.where(self.aa[ubq_site] == cluster_num)[0]) / points_in_cluster * 100
                    self.aa_df.at[(self.aa_df['ubq_site'] == ubq_site) & (self.aa_df['rmsd_centroid'] == cluster_num), 'ensemble_percent'] = ensemble_percent
                    self.aa_df.at[(self.aa_df['ubq_site'] == ubq_site) & (self.aa_df['rmsd_centroid'] == cluster_num), 'aa_percent'] = aa_percent

                    # some assert
                    # assert np.isclose(aa_percent, sub_df[sub_df['rmsd_centroid'] == cluster_num]['aa_percent']).all()
                    assert np.isclose(aa_percent, self.aa_df.loc[(self.aa_df['ubq_site'] == ubq_site) & (self.aa_df['rmsd_centroid'] == cluster_num), 'aa_percent']).all()
                    assert np.isclose(ensemble_percent, self.aa_df.loc[(self.aa_df['ubq_site'] == ubq_site) & (self.aa_df['rmsd_centroid'] == cluster_num), 'ensemble_percent']).all()

                    # get the rmsd centroid
                    internal_rmsd = sub_df[sub_df['rmsd_centroid'] == cluster_num]['internal_rmsd']

                    # get the mean abs diff to the linear combination
                    mean_abs_diff = np.mean(np.abs(coefficient * cluster_mean - exp_values))
                    mean_abs_diff_no_coeff = np.mean(np.abs(cluster_mean - exp_values))

                    df_data['cluster id'].append(cluster_num)
                    df_data['N frames'].append(points_in_cluster)
                    df_data['ensemble %'].append(ensemble_percent)
                    df_data['ubq site'].append(ubq_site)
                    df_data['aa %'].append(aa_percent)
                    df_data['internal RMSD'].append(internal_rmsd)
                    df_data['coefficient'].append(coefficient)
                    df_data['mean abs diff to exp w/ coeff'].append(mean_abs_diff)
                    df_data['mean abs diff to exp w/o coeff'].append(mean_abs_diff_no_coeff)
                if ubq_num == 0:
                    df = pd.DataFrame(df_data)
                else:
                    df = pd.concat([df, pd.DataFrame(df_data)], ignore_index=True)
                metadata[f'mean_abs_diff_full_combination_{ubq_site}'] = mean_abs_diff_full_combination
            else:
                h5store(df_file, df, **metadata)

        if not hasattr(self, 'polar_coordinates_aa'):
            self.polar_coordinates_aa = {ubq_site: np.load(os.path.join(self.analysis_dir, f'polar_coordinates_aa_{ubq_site}.npy')) for ubq_site in self.ubq_sites}

        if not hasattr(self, 'polar_coordinates_cg'):
            self.polar_coordinates_cg = {ubq_site: np.load(os.path.join(self.analysis_dir, f'polar_coordinates_cg_{ubq_site}.npy')) for ubq_site in self.ubq_sites}

        for i, ubq_site in enumerate(self.ubq_sites):
            sub_df = df[df['ubq site'] == ubq_site]
            sub_df = sub_df.reset_index(drop=True)
            for j, row in sub_df.iterrows():
                cluster_num = row['cluster id']
                count_id = np.where(np.sort(sub_df['N frames'])[::-1] == row['N frames'])[0][0]
                cluster_dir = f'/home/kevin/projects/tobias_schneider/cluster_analysis_with_fixed_normalization/{ubq_site}/cluster_{count_id}/'
                os.makedirs(cluster_dir, exist_ok=True)
                image_file = os.path.join(cluster_dir, 'polar_plot.png')
                pdb_file_out = os.path.join(cluster_dir, 'cluster.pdb')
                xtc_file_out = os.path.join(cluster_dir, 'cluster.xtc')
                surface_coverage_cluster_file = os.path.join(cluster_dir, 'surface_coverage.pdb')
                rmsd_centroid_index_file = os.path.join(cluster_dir, 'cluster_rmsd_centroid_index.npy')

                if not os.path.isfile(image_file) or overwrite or overwrite_polar_plots:
                    where_aa = np.where(self.aa[ubq_site] == row['cluster id'])[0]
                    where_cg = np.where(self.cg[ubq_site] == row['cluster id'])[0]
                    plt.close('all')
                    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.EckertIV()})
                    ax = add_reference_to_map(ax)

                    nbins = 200

                    x = np.concatenate([self.polar_coordinates_aa[ubq_site][where_aa, :, 0].flatten(),
                                        self.polar_coordinates_cg[ubq_site][where_cg, :, 0].flatten()])
                    y = np.concatenate([self.polar_coordinates_aa[ubq_site][where_aa, :, 1].flatten(),
                                        self.polar_coordinates_cg[ubq_site][where_cg, :, 1].flatten()])

                    H, xedges, yedges = np.histogram2d(x=x, y=y, bins=(nbins, int(nbins / 2)), range=[[-180, 180], [-90, 90]])
                    xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
                    ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)
                    X, Y = np.meshgrid(xcenters, ycenters)

                    lon = np.linspace(-180, 180, nbins)
                    lat = np.linspace(-90, 90, int(nbins / 2))
                    lon2d, lat2d = np.meshgrid(lon, lat)

                    cmap = plt.cm.get_cmap('viridis').copy()
                    cmap.set_bad('w', 0.0)
                    H = np.ma.masked_where(H < 0.1, H)

                    ax.contourf(X, Y, H.T, transform=ccrs.PlateCarree())

                    plt.savefig(image_file)

                if not os.path.isfile(pdb_file_out) or overwrite_pdb_files:
                    # save and load the rmsd centroid
                    print(list(filter(lambda x: False if any([i in x for i in ['sPRE', '15N', 'RWMD']]) else True, self.aa_df.columns)))
                    aa_df_row = self.aa_df[(self.aa_df['rmsd_centroid'] == cluster_num) & (self.aa_df['ubq_site'] == ubq_site)]
                    xtc_file, top_file, frame = aa_df_row[['traj_file', 'top_file', 'frame']].values[0]
                    centroid = md.load_frame(xtc_file, frame, top=top_file)
                    cryst = md.load('/home/kevin/1UBQ.pdb')
                    centroid.superpose(reference=cryst, atom_indices=centroid.top.select('name CA and resid >= 76'),
                                       ref_atom_indices=cryst.top.select('name CA'))
                    if one_bond_too_long(centroid, 3):
                        print(f"Too long bonds for cluster {cluster_num} at ubq_site {ubq_site}.")
                        fix_pdb(traj, pdb_file_out, ubq_site)

                    # save and load the 20 representative structures
                    aa_df_rows = self.aa_df[(self.aa_df['20_selected_for_rendering'] == cluster_num) & (self.aa_df['ubq_site'] == ubq_site)]
                    xtc_files, top_files, frames = np.vstack(aa_df_rows[['traj_file', 'top_file', 'frame']].values).T
                    for frame_count, (xtc_file, top_file, frame) in enumerate(zip(xtc_files, top_files, frames)):
                        if frame_count == 0:
                            traj = md.load_frame(xtc_file, frame, top=top_file)
                        else:
                            traj = traj.join(md.load_frame(xtc_file, frame, top=top_file))
                    traj.superpose(reference=cryst, atom_indices=centroid.top.select('name CA and resid >= 76'),
                                   ref_atom_indices=cryst.top.select('name CA'))
                    if one_bond_too_long(centroid, 3):
                        print(f"Too long bonds for cluster {cluster_num} at ubq_site {ubq_site}.")
                        traj = fix_pdb(traj, None, ubq_site)
                    traj.save_xtc(xtc_file_out)

                if not os.path.isfile(surface_coverage_cluster_file) or overwrite_surface_coverage:
                    pass

            linear_combination, cluster_means = make_linear_combination_from_clusters(None,
                                                                                      self.aa_df,
                                                                                      self.df_obs,
                                                                                      self.fast_exchangers,
                                                                                      ubq_site=ubq_site,
                                                                                      return_means=True)

            # make linear combinations of everything and only the clusters
            where = np.where(sub_df['coefficient'] > coeff_threshold)[0]
            cluster_ids = sub_df['cluster id'].values
            count_ids = []

            for cluster_id in cluster_ids:
                count_id = np.where(np.sort(sub_df['N frames'])[::-1] == sub_df[sub_df['cluster id'] == cluster_id]['N frames'].values)[0][0]
                count_ids.append(count_id)
            count_ids = np.array(count_ids)

            coefficients_best_clusters = sub_df['coefficient'][where].values
            all_coefficients = sub_df['coefficient'].values

            cluster_combination_str = '_and_'.join(np.sort(count_ids[where]).astype(str))
            cluster_combination_str_no_underscore = ' and '.join(np.sort(count_ids[where]).astype(str))

            final_combination_image = f'/home/kevin/projects/tobias_schneider/cluster_analysis_with_fixed_normalization/{ubq_site}/combination_of_clusters_{cluster_combination_str}.png'
            final_combination_image_all = f'/home/kevin/projects/tobias_schneider/cluster_analysis_with_fixed_normalization/{ubq_site}/all_sims_confidence.png'
            final_correlation_image_all = f'/home/kevin/projects/tobias_schneider/cluster_analysis_with_fixed_normalization/{ubq_site}/correlation_plot.png'
            exp_value = self.df_obs[ubq_site][self.df_obs[ubq_site].index.str.contains('sPRE')].values
            if not os.path.isfile(final_combination_image) or overwrite_final_combination:
                plt.close('all')
                sim_value = np.sum(coefficients_best_clusters * cluster_means[where].T, 1)
                fig, (ax1, ax2,) = plt.subplots(nrows=2, figsize=(20, 10))

                from xplor.nmr_plot import plot_line_data, plot_hatched_bars

                (ax1, ax2) = plot_line_data((ax1, ax2), self.df_obs, {'rows': 'sPRE', 'cols': ubq_site})
                (ax1, ax2) = plot_hatched_bars((ax1, ax2), self.fast_exchangers, {'cols': 'k6'}, color='k')
                sim_prox, sim_dist = np.split(sim_value, 2)

                ax1.plot(sim_prox, c='C0')
                ax2.plot(sim_dist, c='C0')

                for coeff, cluster_id, count_id, color, hpos in zip(coefficients_best_clusters, cluster_ids[where], count_ids[where], ['C2', 'C3'], [0.70, 0.60]):
                    prox, dist = np.split(cluster_means[cluster_id], 2)
                    ax1.plot(prox, c=color)
                    ax2.plot(dist, c=color)
                    # old mean abs diff
                    # mean_abs_diff = sub_df['mean abs diff to exp'][cluster_id]
                    mean_abs_diff = np.mean(np.abs(sim_value - cluster_means[cluster_id])[~ self.fast_exchangers[ubq_site]])
                    print(count_id, mean_abs_diff)
                    # ax1.text(0.95, hpos, f"mean abs diff exp/sim cluster {count_id} = {mean_abs_diff:.1f}",
                    #      transform=ax1.transAxes, ha='right', va='center')

                ax1.set_title(f'{ubq_site.upper()}-diUbi proximal')
                ax2.set_title(f'{ubq_site.upper()}-diUbi distal')

                for ax, centers in zip([ax1, ax2], [self.centers_prox, self.centers_dist - 76]):
                    ax = add_sequence_to_xaxis(ax)
                    ax = color_labels(ax, positions=centers)
                    ax.set_ylabel(r'sPRE in $\mathrm{mM^{-1}ms^{-1}}$')

                mean_abs_diff = np.mean(np.abs(sim_value - exp_value)[~ self.fast_exchangers[ubq_site]])
                # ax1.text(0.95, 0.80, f"mean abs diff exp/sim {cluster_combination_str_no_underscore} = {mean_abs_diff:.1f}",
                #          transform=ax1.transAxes, ha='right', va='center')
                
                fake_legend_dict1 = {'line': [{'label': 'NMR experiment', 'color': 'C1'},
                                              {'label': f'combination of cluster {cluster_combination_str_no_underscore}', 'color': 'C0'},
                                              {'label': f'mean of cluster {count_ids[where][0]} (without coefficient)', 'color': 'C2'}],
                                    'hatchbar': [{'label': 'Fast exchanging residues', 'color': 'lightgrey', 'alpha': 0.3, 'hatch': '//'}],
                                    'text': [{'label': 'Residue used for normalization', 'color': 'red', 'text': 'Ile3'}]}

                if ubq_site != 'k33':
                    fake_legend_dict1['line'].append({'label': f'mean of cluster {count_ids[where][1]} (without coefficient)', 'color': 'C3'})
                fake_legend(ax1, fake_legend_dict1)
                plt.tight_layout()
                plt.savefig(final_combination_image)

            if not os.path.isfile(final_combination_image_all) or overwrite_final_combination:
                plt.close('all')
                fig, (ax1, ax2,) = plt.subplots(nrows=2, figsize=(20, 10))

                from xplor.nmr_plot import plot_line_data, plot_hatched_bars

                (ax1, ax2) = plot_line_data((ax1, ax2), self.df_obs, {'rows': 'sPRE', 'cols': ubq_site})
                (ax1, ax2) = plot_hatched_bars((ax1, ax2), self.fast_exchangers, {'cols': 'k6'}, color='k')
                (ax1, ax2), color, (iqr_color, iqr_plus_q_color) = plot_confidence_intervals((ax1, ax2), self.aa_df, {'rows': 'sPRE', 'cols': ubq_site}, cbar=False, with_outliers=False)

                ax1.set_title(f'{ubq_site.upper()}-diUbi proximal')
                ax2.set_title(f'{ubq_site.upper()}-diUbi distal')

                for ax, centers in zip([ax1, ax2], [self.centers_prox, self.centers_dist - 76]):
                    ax = add_sequence_to_xaxis(ax)
                    ax = color_labels(ax, positions=centers)
                    ax.set_ylabel(r'sPRE in $\mathrm{mM^{-1}ms^{-1}}$')

                index = ['sPRE' in col for col in self.aa_df.columns]
                index = self.aa_df.columns[index]
                if index.str.contains('normalized').any():
                    index = index[index.str.contains('normalized')]
                q1, median, q3 = self.aa_df[index].quantile([0.25, 0.5, 0.75], axis='rows').values

                mean_abs_diff = np.mean(np.abs(median - exp_value))
                ax1.text(0.95, 0.80, f"mean abs diff exp/(median all sims) = {mean_abs_diff:.1f}",
                         transform=ax1.transAxes, ha='right', va='center')

                fake_legend_dict1 = {'line': [{'label': 'NMR experiment', 'color': 'C1'},
                                              {'label': f'median all sims', 'color': 'C0'}],
                                     'hatchbar': [{'label': 'Fast exchanging residues', 'color': 'lightgrey', 'alpha': 0.3, 'hatch': '//'}],
                                     'text': [{'label': 'Residue used for normalization', 'color': 'red', 'text': 'Ile3'}],
                                     'envelope': [{'label': 'IQR', 'color': iqr_color},
                                                  {'label': r'$\mathrm{Q_{1/3} \mp IQR}$', 'color': iqr_plus_q_color}],
                }
                fake_legend(ax1, fake_legend_dict1)
                plt.tight_layout()
                plt.savefig(final_combination_image_all)

            if not os.path.isfile(final_correlation_image_all) or overwrite_final_correlation:
                plt.close('all')
                fig, (ax1, ax2,) = plt.subplots(nrows=2, figsize=(20, 10))

                from xplor.nmr_plot import plot_correlation_plot, plot_hatched_bars

                where = np.where(sub_df['coefficient'] > coeff_threshold)[0]
                cluster_ids = sub_df['cluster id'].values
                count_ids = []

                for cluster_id in cluster_ids:
                    count_id = np.where(np.sort(sub_df['N frames'])[::-1] == sub_df[sub_df['cluster id'] == cluster_id][
                        'N frames'].values)[0][0]
                    count_ids.append(count_id)
                count_ids = np.array(count_ids)

                fake_legend_dict1 = {'envelope': [{'label': 'median all', 'color': 'C0'},
                                                  {'label': 'cluster combination', 'color': 'c'}],
                                     'hatchbar': [
                                         {'label': 'Fast exchanging residues', 'color': 'lightgrey', 'alpha': 0.3,
                                          'hatch': '//'}]
                                     }

                clusters_to_plot = []
                for coeff, cluster_id, count_id, color in zip(coefficients_best_clusters, cluster_ids[where], count_ids[where], ['grey', 'tab:olive']):
                    print(cluster_id, count_id)
                    clusters_to_plot.append({f'cluster {count_id}': cluster_means[cluster_id]})
                    fake_legend_dict1['envelope'].append({'label': f'cluster {count_id}', 'color': color})

                print('cluster_ids: ', where, 'count_ids: ', count_ids)
                sim_value = np.sum(coefficients_best_clusters * cluster_means[where].T, 1)
                to_plot = ['mean', {'final_combination': sim_value}] + clusters_to_plot

                (ax1, ax2) = plot_hatched_bars((ax1, ax2), self.fast_exchangers, {'cols': 'k6'}, color='k')
                (ax1, ax2) = plot_correlation_plot((ax1, ax2), self.df_obs, self.aa_df, {'rows': 'sPRE', 'cols': ubq_site},
                                                   correlations=to_plot)

                for ax, centers in zip([ax1, ax2], [self.centers_prox, self.centers_dist - 76]):
                    ax = add_sequence_to_xaxis(ax)
                    ax = color_labels(ax, positions=centers)
                    ax.set_ylabel(r'\Delta sPRE in $\mathrm{mM^{-1}ms^{-1}}$')
                fake_legend(ax1, fake_legend_dict1)

                plt.tight_layout()
                plt.savefig(final_correlation_image_all)

        if hasattr(self, 'aa'): del self.aa
        if hasattr(self, 'cg'): del self.cg

    def load_trajs(self, overwrite=False):
        """Loads the trajs from disk and adds the following attributes:

        Attrs:
            trajs (Dict[str, em.Info_all]): Dict of Info_all objects containing all CG and AA trajs.
                Mapped to their respective ubiquitination sites.
            aa_trajs (Dict[str, em.Info_all]): Dict of Info_all objects containing only AA trajs.
                Mapped to their respective ubiquitination sites.
            cg_trajs (Dict[str, em.Info_all]): Dict of Info_all objects containing only CG trajs.
                Mapped to their respective ubiquitination sites.
            aa_traj_files (Dict[str, List[str]]): A dict mapping the ubiquitination sites to a list
                of strings pointing to all AA files.
            cg_traj_files (Dict[str, List[str]]): A dict mapping the ubiquitination sites to a list
                of strings pointing to all CG files.
            aa_references (Dict[str, List[str]]): A dict mapping the ubiquitination sites to a list
                of strings pointing to the reference files for the AA simulations. These
                files can either be .gro or .pdb files.
            cg_references (Dict[str, List[str]]): A dict mapping the ubiquitination sites to a list
                of strings pointing to the reference files for the CG simulations. These
                files can either be .gro or .pdb files.

        """
        if hasattr(self, 'trajs'):
            if not overwrite and bool(self.trajs):
                print("Trajs already loaded")
                return

        # setting some schtuff
        self.trajs = {}
        self.aa_trajs = {}
        self.cg_trajs = {}
        self.aa_traj_files = {ubq_site: [] for ubq_site in self.ubq_sites}
        self.aa_references = {ubq_site: [] for ubq_site in self.ubq_sites}
        self.cg_traj_files = {ubq_site: [] for ubq_site in self.ubq_sites}
        self.cg_references = {ubq_site: [] for ubq_site in self.ubq_sites}

        print("Loading trajs")
        for i, ubq_site in enumerate(self.ubq_sites):
            for j, dir_ in enumerate(glob.glob(f"{self.sim_dirs[0]}{ubq_site}_*") + glob.glob(f'{self.sim_dirs[1]}{ubq_site.upper()}_*')):
                # define names
                traj_file = dir_ + '/traj_nojump.xtc'
                basename = traj_file.split('/')[-2]
                if 'andrejb' in traj_file:
                    top_file = dir_ + '/start.pdb'
                else:
                    top_file = dir_ + '/init.gro'

                is_aa = True
                if 'andrejb' in traj_file:
                    if not is_aa_sim(traj_file): is_aa = False

                if is_aa:
                    self.aa_traj_files[ubq_site].append(traj_file)
                    self.aa_references[ubq_site].append(top_file)
                else:
                    self.cg_traj_files[ubq_site].append(traj_file)
                    self.cg_references[ubq_site].append(top_file)

            self.aa_trajs[ubq_site] = em.Info_all(trajs=self.aa_traj_files[ubq_site],
                                                  tops=self.aa_references[ubq_site],
                                                  basename_fn=lambda x: x.split('/')[-2],
                                                  common_str=[ubq_site, ubq_site.upper()])
            self.cg_trajs[ubq_site] = em.Info_all(trajs=self.cg_traj_files[ubq_site],
                                                  tops=self.cg_references[ubq_site],
                                                  basename_fn=lambda x: x.split('/')[-2],
                                                  common_str=[ubq_site])
            self.trajs[ubq_site] = (self.aa_trajs[ubq_site] + self.cg_trajs[ubq_site])


    def load_highd(self, overwrite=False):
        """Load sthe high-d RWMD data. If not present it constructs the data.

        The RWMD is calculated by iterating over the residues of the two subunits,
        respectively. Starting from residue MET1 of the proximal subunit, all
        distances to every residue in the distal subunit are calculated and the
        smallest distance is chosen for that residue. Then the proximal GLN2 is
        considered. The smallest distance from GLN2 to any of the distal residues
        is added to the RWMD. Next is ILE3 and so on. After that 76 distances are
        in RWMD. After that the same is done with the distal-proximal distances
        leading to 152 values per diUbi conformation.

        This method adds the following attributes to the class:

        Attrs:
            aa_dists (Dict[str, np.ndarray]): A dict mapping the ubiquitination sites
                to the numpy arrays holding the high-dimensional RWMD data of the AA trajs.
            cg_dists (Dict[str, np.ndarray]): A dict mapping the ubiquitination sites
                to the numpy arrays holding the high-dimensional RWMD data of the CG trajs.


        """
        if not hasattr(self, 'aa_dists') or overwrite:
            self.aa_dists = {}
        if not hasattr(self, 'cg_dists') or overwrite:
            self.cg_dists = {}

        if self.check_attr_all('highd') and not overwrite:
            print("Highd already in all trajs")
            return

        for i, ubq_site in enumerate(self.ubq_sites):
            aa_highd_file = os.path.join(self.analysis_dir, f'highd_rwmd_aa_{ubq_site}.npy')
            cg_highd_file = os.path.join(self.analysis_dir, f'highd_rwmd_cg_{ubq_site}.npy')
            highd_file = os.path.join(self.analysis_dir, f'highd_rwmd{ubq_site}.npy')

            if not os.path.isfile(aa_highd_file) or overwrite:
                with Capturing() as output:
                    top_aa = CustomGromacsTopFile(f'/home/andrejb/Software/custom_tools/topology_builder/topologies/gromos54a7-isop/diUBQ_{ubq_site.upper()}/system.top',
                                                  includeDir='/home/andrejb/Software/gmx_forcefields')
                top = md.Topology.from_openmm(top_aa.topology)

                CAs_prox = top.select('name CA and resid >= 76')
                CAs_dist = top.select('name CA and resid < 76')
                assert CAs_prox.shape == CAs_dist.shape

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    _ = self.aa_trajs[ubq_site].xyz
                    dists_0_1 = np.vstack([np.min(cdist(_[i, CAs_prox], _[i, CAs_dist]), axis=1) for i in range(len(_))])
                    dists_1_0 = np.vstack([np.min(cdist(_[i, CAs_dist], _[i, CAs_prox]), axis=1) for i in range(len(_))])
                self.aa_dists[ubq_site] = np.hstack([dists_0_1, dists_1_0])
                np.save(aa_highd_file, self.aa_dists[ubq_site])
            elif ubq_site not in self.aa_dists:
                self.aa_dists[ubq_site] = np.load(aa_highd_file)
            else:
                print("AA highd already loaded")

            if not os.path.isfile(cg_highd_file) or overwrite:
                BBs_prox = self.cg_trajs[ubq_site][0].top.select('name BB and resid >= 76')
                BBs_dist = self.cg_trajs[ubq_site][0].top.select('name BB and resid < 76')
                assert BBs_prox.shape == BBs_dist.shape

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    _ = self.cg_trajs[ubq_site].xyz
                    dists_0_1 = np.vstack(
                        [np.min(cdist(_[i, BBs_prox], _[i, BBs_dist]), axis=1) for i in range(len(_))])
                    dists_1_0 = np.vstack(
                        [np.min(cdist(_[i, BBs_dist], _[i, BBs_prox]), axis=1) for i in range(len(_))])
                self.cg_dists[ubq_site] = np.hstack([dists_0_1, dists_1_0])
                np.save(cg_highd_file, self.cg_dists[ubq_site])
            elif ubq_site not in self.cg_dists:
                self.cg_dists[ubq_site] = np.load(cg_highd_file)
            else:
                print("CG highd already loaded")

            if not hasattr(self.trajs[ubq_site], 'highd') or overwrite:
                self.aa_trajs[ubq_site].load_CVs(self.aa_dists[ubq_site], 'highd')
                self.cg_trajs[ubq_site].load_CVs(self.cg_dists[ubq_site], 'highd')
            else:
                print("Highd already loaded")

    def train_encodermap(self, overwrite=False):
        """Trans encodermap with the RWMD data. Does not add any attributes to self."""
        if self.check_attr_all('lowd') and not overwrite:
            print("Lowd already in all trajs")
            return

        for i, ubq_site in enumerate(self.ubq_sites):
            train_path = os.path.join(self.analysis_dir, f'runs/{ubq_site}/production_run_tf2/')
            checkpoints = glob.glob(f'{train_path}*encoder')
            if checkpoints:
                print("There are checkpoints present")
            if overwrite:
                if hasattr(self.trajs[ubq_site], 'lowd'):
                    print('Deleting directory to prepare new run.')
                    del self.trajs[ubq_site]._CVs['lowd']
                if os.path.isdir(train_path):
                    shutil.rmtree(train_path)
            if hasattr(self.trajs[ubq_site], 'lowd') and not overwrite:
                print("Lowd already loaded")
                return
            elif checkpoints and not overwrite:
                checkpoint = list(sorted(checkpoints, key=ckpt_step))[-1].replace('model_encoder', '*')
                e_map = em.EncoderMap.from_checkpoint(checkpoint)
            else:
                os.makedirs(train_path, exist_ok=True)
                parameters = em.Parameters()
                parameters.main_path = train_path
                parameters.n_neurons = [250, 250, 125, 2]
                parameters.activation_functions = ['', 'tanh', 'tanh', 'tanh', '']
                parameters.periodicity = float('inf')
                parameters.dist_sig_parameters = [5.9, 12, 4, 5.9, 2, 4]
                parameters.learning_rate = 0.00001
                print("training")
                array = np.vstack([self.aa_dists[ubq_site], self.cg_dists[ubq_site]])
                e_map = em.EncoderMap(parameters, array)
                e_map.add_images_to_tensorboard(array[::1000])
                e_map.train()
                e_map.save()

            aa_lowd = e_map.encode(self.aa_trajs[ubq_site].highd)
            cg_lowd = e_map.encode(self.cg_trajs[ubq_site].highd)
            self.aa_trajs[ubq_site].load_CVs(aa_lowd, attr_name='lowd')
            self.cg_trajs[ubq_site].load_CVs(cg_lowd, attr_name='lowd')

    def load_xplor_data(self, overwrite=False,
                        csv='conect', overwrite_normalization=False):
        if all([hasattr(self, _) for _ in ['df_obs', 'fast_exchangers', 'in_secondary', 'df_comp_norm', 'norm_factors']]) and not overwrite:
            print("XPLOR data already loaded")
            return
        self.df_comp = csv
        if overwrite or overwrite_normalization or not hasattr(self, 'df_comp_norm'):
            self.df_comp_norm, self.centers_prox, self.centers_dist = normalize_sPRE(self.df_comp, self.df_obs)

    def cluster(self, overwrite=False):
        """Uses HDBSCAN to decide cluster_membership of trajs does not add any attribute to self."""
        if self.check_attr_all('cluster_membership'):
            print("Cluster Membership already in all trajs")
            return

        for i, ubq_site in enumerate(self.ubq_sites):
            if not hasattr(self.trajs[ubq_site], 'cluster_membership') or overwrite:
                aa_cluster_file = os.path.join(self.analysis_dir, f'cluster_membership_aa_{ubq_site}.npy')
                cg_cluster_file = os.path.join(self.analysis_dir, f'cluster_membership_cg_{ubq_site}.npy')
                if not all([os.path.isfile(file) for file in [aa_cluster_file, cg_cluster_file]]) or overwrite:
                    raise Exception("Run HDBSCAN on 4GPU")
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=2500, cluster_selection_method='leaf').fit(
                        self.trajs[ubq_site].lowd)
                    if max(clusterer.labels_) > 20:
                        raise Exception(f"Too many clusters {np.unique(clusterer.labels_)}")
                    else:
                        print(f"Found {np.unique(clusterer.labels_)} clusters.")
                    aa_index = np.arange(self.aa_trajs[ubq_site].n_frames)
                    cg_index = np.arange(self.aa_trajs[ubq_site].n_frames,
                                         self.cg_trajs[ubq_site].n_frames + self.aa_trajs[ubq_site].n_frames)
                    self.aa_trajs[ubq_site].load_CVs(clusterer.labels_[aa_index], 'cluster_membership')
                    self.cg_trajs[ubq_site].load_CVs(clusterer.labels_[cg_index], 'cluster_membership')
                    np.save(aa_cluster_file, clusterer.labels_[aa_index])
                    np.save(cg_cluster_file, clusterer.labels_[cg_index])
                else:
                    self.aa_trajs[ubq_site].load_CVs(np.load(aa_cluster_file), 'cluster_membership')
                    self.cg_trajs[ubq_site].load_CVs(np.load(cg_cluster_file), 'cluster_membership')
            else:
                print('Cluster_membership already in trajs')

    def cluster_analysis(self, overwrite=False, overwrite_image=False):
        """Analyzes the clusters.

        Steps:
            * Check how much CG and AA is distributed over the clusters.
            * Find the geometric center and the closest AA structure
            * Calculate the mean sPRE in that cluster
            * compare mean sPRE and sPRE of geometric center

        """
        for i, ubq_site in enumerate(self.ubq_sites):
            linear_combination_file = os.path.join(self.analysis_dir, f'clusters_{ubq_site}_sPRE_linear_combi.npy')
            means_file = os.path.join(self.analysis_dir, f'clusters_{ubq_site}_means.npy')
            if not os.path.isfile(linear_combination_file) or overwrite:
                linear_combination, means = make_linear_combination_from_clusters(self.trajs[ubq_site], self.df_comp_norm, self.df_obs,
                                                                           self.fast_exchangers, ubq_site=ubq_site,
                                                                           return_means=True, exclusions=self.cluster_exclusions[ubq_site])
                np.save(linear_combination_file, linear_combination)
                np.save(means_file, means)
            else:
                print("Loading linear combinations")
                linear_combination = np.load(linear_combination_file)
                means = np.load(means_file)

            latex_table_data = {'cluster_num': [], 'percent of aa frames': [], 'percent of cg frames': [],
                                'percent in full ensemble': [], 'coefficient in linear combination': [],
                                'mean abs difference of cluster mean to exp values': []}
            sPRE_ind = ['sPRE' in i for i in self.df_obs.index]
            exp_values = self.df_obs[ubq_site][self.df_obs[ubq_site].index.str.contains('sPRE')].values

            for cluster_num, (linear, mean) in enumerate(zip(linear_combination, means)):
                if cluster_num == -1:
                    continue
                if cluster_num in self.cluster_exclusions[ubq_site]:
                    print(f"Cluster {cluster_num} of {ubq_site} is excluded")
                    continue
                if cluster_num not in np.unique(self.aa_trajs[ubq_site].cluster_membership):
                    print(f"Cluster {cluster_num} of {ubq_site} is not an aa cluster")
                    continue
                cluster_analysis_outdir = os.path.join(self.analysis_dir,
                                                       f'cluster_analysis/{ubq_site}/cluster_{cluster_num}')
                os.makedirs(cluster_analysis_outdir, exist_ok=True)
                image_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_summary.png')

                # Check how much AA and CG is distributed in this cluster
                aa_cluster_points_ind = np.where(self.aa_trajs[ubq_site].cluster_membership == cluster_num)[0]
                cg_cluster_points_ind = np.where(self.cg_trajs[ubq_site].cluster_membership == cluster_num)[0]
                sum_cluster = len(cg_cluster_points_ind) + len(aa_cluster_points_ind)
                aa_percent = len(aa_cluster_points_ind) / sum_cluster * 100
                cg_percent = len(cg_cluster_points_ind) / sum_cluster * 100
                coeff = str(np.format_float_scientific(linear, 2))
                raise Exception("Mean abs diff without fast exchangers")
                mean_abs_diff = np.round(np.mean(np.abs(mean - exp_values)), 2)
                percentage = sum_cluster / self.trajs[ubq_site].n_frames * 100
                print(f"Cluster {cluster_num} consists of {aa_percent:.2f}% aa structures and {cg_percent:.2f}% of cg structures."
                      f"The mean abs difference between sim and exp is {mean_abs_diff}."
                      f"The coefficient in the linear combination is {coeff}")

                # rmsd centroid
                cluster_pdb_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_linear_coefficient_{coeff}_ensemble_contribution_{percentage:.1f}_percent.pdb')
                rmsd_centroid_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_rmsd_centroid.pdb')
                rmsd_centroid_index_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_rmsd_centroid.npy')
                if not os.path.isfile(rmsd_centroid_file) or overwrite:
                    max_frames = 500
                    if not cluster_num in np.unique(self.aa_trajs[ubq_site].cluster_membership):
                        continue
                    view, dummy_traj = gen_dummy_traj(self.aa_trajs[ubq_site], cluster_num, max_frames=max_frames, superpose=True)
                    where = np.where(self.aa_trajs[ubq_site].cluster_membership == cluster_num)[0]
                    idx = np.round(np.linspace(0, len(where) - 1, max_frames)).astype(int)
                    where = where[idx]
                    index, mat, rmsd_centroid = rmsd_centroid_of_cluster(dummy_traj)
                    rmsd_centroid_index = where[index]
                    rmsd_centroid.save_pdb(rmsd_centroid_file)
                    rmsd_centroid.save_pdb(cluster_pdb_file)
                    np.save(rmsd_centroid_index_file, rmsd_centroid_index)

                # geometric centroid
                geom_centroid_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_geom_centroid.pdb')
                geom_centroid_index_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_geom_centroid.npy')
                if not os.path.isfile(geom_centroid_file) or overwrite:
                    geom_center = np.mean(self.aa_trajs[ubq_site].lowd[aa_cluster_points_ind], axis=0)
                    sums = np.sum((self.aa_trajs[ubq_site].lowd[aa_cluster_points_ind] - geom_center) ** 2, axis=1)
                    geom_centroid_index = aa_cluster_points_ind[np.argmin(sums)]
                    geom_centroid = self.aa_trajs[ubq_site].get_single_frame(geom_centroid_index).traj
                    geom_centroid.save_pdb(geom_centroid_file)
                    np.save(geom_centroid_index_file, geom_centroid_index)

                latex_table_data['cluster_num'].append(cluster_num)
                latex_table_data['percent of aa frames'].append(f'{aa_percent:.0f}% ')
                latex_table_data['percent of cg frames'].append(f'{cg_percent:.0f}%')
                latex_table_data['percent in full ensemble'].append(f'{percentage:.0f}%')
                latex_table_data['coefficient in linear combination'].append(coeff)
                latex_table_data['mean abs difference of cluster mean to exp values'].append(mean_abs_diff)

                # if not os.path.isfile(image_file) or overwrite_image:
                #     self.plot_cluster(cluster_num, ubq_site, out_file=image_file, overwrite=True)

            latex_file = os.path.join(self.analysis_dir, f'cluster_analysis_{ubq_site}.tex')
            excel_file = os.path.join(self.analysis_dir, f'cluster_analysis_{ubq_site}.xlsx')
            df_ = pd.DataFrame(latex_table_data)

            # check for bad sum in k6
            if ubq_site == 'k6':
                print(np.sum(df_['coefficient in linear combination'].astype(float)))
                df_.to_latex(latex_file, index=False)
                df_.to_excel(excel_file, index=False)
                raise Exception("STOP")

    def plot_cluster(self, cluster_num, ubq_site, out_file=None,
                     overwrite=False, nbins=100, mhz=800, reduced=True):
        """
        Layout:
            +-------------------------------------------------------+
            |sPRE linear combination factor for this cluster is     |
            |                                                       |
            |cluster occupies X% of ubq_site ensemble               |
            |                                                       |
            +-------------------------------------------------------+
            |                                                       |
            |                       proximal                        |
            |    per-residue difference between mean sPRE and       |
            |    sPRE of geometric center                           |
            |                                                       |
            +-------------------------------------------------------+
            |                                                       |
            |                       distal                          |
            |    per-residue difference between mean sPRE and       |
            |    sPRE of geometric center                           |
            |                                                       |
            +--------------------------+-+--------------------------+
            |                          | |                          |
            |   scatter cg and aa      | |   density with pyemma    |
            |   with different colors  | |   of all lowd            |
            |                          | |                          |
            +--------------------------+ +--------------------------+
            |                          | |                          |
            |    contour of cg and     | |   using seabiornighted   |
            |    contour of aa         | |                          |
            |                          | |                          |
            +--------------------------+ +--------------------------+
            |    histogram             | |                          |
            |                          | |                          |
            +---------------------+----+ |                          |
            |                     | h  | |                          |
            |     cg and aa scatte| i  | |                          |
            |     this cluster    | s  | |  vmd render              |
            |     see encodermap  | t  | |                          |
            |                     | o  | |                          |
            +---------------------+----+-+--------------------------+
            |                                                       |
            |       sPRE proximal                                   |
            |       everything and this cluster                     |
            |                                                       |
            +-------------------------------------------------------+
            |                                                       |
            |        sPRE distal                                    |
            |        everything and this cluster                    |
            |                                                       |
            +-------------------------------------------------------+
            |                                                       |
            |            available 15N data proximal                |
            |            everything and this cluster                |
            |                                                       |
            +-------------------------------------------------------+
            |                                                       |
            |           available 15N data distal                   |
            |           everything and this cluster                 |
            |                                                       |
            +-------------------------------------------------------+

        Args:
            cluster_num (int): Number of the cluster to be plotted.
            ubq_site (str): The ubiquitination site to be considered.

        Keyword Args:
            out_file (Union[str, None], optional): Where to save the file to.
                String should end in '.png'. A .pdf will be also be saved.
                If None is provided will be saved at
                self.analysis_dir/cluster_analysis/{ubq_site}/cluster_{cluster_num}
                Defaults to None
            overwrite (bool, optional): Whether to overwrite if file is already present.
                Defaults to False
            nbins (int, optional): The number of bins to be used in 2D binning.
                Defaults to 100.
            mhz (Union[int, float], optional): The MHz for the 15N relax data to be
                considered.

        """
        cluster_analysis_outdir = os.path.join(self.analysis_dir,
                                               f'cluster_analysis/{ubq_site}/cluster_{cluster_num}')
        # set file
        if out_file is None:
            out_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cluster_{cluster_num}_summary.png')
        if os.path.isfile(out_file) and not overwrite:
            print(f"Image for cluster {cluster_num} already exists.")
            return

        # get some data
        aa_cluster_points_ind = np.where(self.aa_trajs[ubq_site].cluster_membership == cluster_num)[0]
        cg_cluster_points_ind = np.where(self.cg_trajs[ubq_site].cluster_membership == cluster_num)[0]
        cluster_analysis_outdir = os.path.join(self.analysis_dir,
                                               f'cluster_analysis/{ubq_site}/cluster_{cluster_num}')
        rmsd_centroid_index_file = os.path.join(cluster_analysis_outdir,
                                                f'{ubq_site}_cluster_{cluster_num}_rmsd_centroid.npy')
        geom_centroid_index_file = os.path.join(cluster_analysis_outdir,
                                                f'{ubq_site}_cluster_{cluster_num}_geom_centroid.npy')
        rmsd_centroid_index = np.load(rmsd_centroid_index_file)
        geom_centroid_index = np.load(geom_centroid_index_file)

        plt.close('all)')

        if reduced:
            fig = plt.figure(constrained_layout=False, figsize=(45, 20))
        else:
            fig = plt.figure(constrained_layout=False, figsize=(20, 50))
        if reduced:
            spec = fig.add_gridspec(ncols=2, nrows=1, width_ratios=[1, 2])
        else:
            spec = fig.add_gridspec(ncols=2, nrows=10)

        if reduced:
            left = spec[0].subgridspec(ncols=2, nrows=3)
            right = spec[1].subgridspec(ncols=1, nrows=2)
            scatter_all_ax = fig.add_subplot(left[0, 0])
            dense_all_ax = fig.add_subplot(left[0, 1])
            cg_vs_aa_contour_ax = fig.add_subplot(left[1, 0])
            clustered_ax = fig.add_subplot(left[1, 1])
            scatter_cluster_ax = fig.add_subplot(left[2, 0])
            render_ax = fig.add_subplot(left[2, 1])
            render_ax.axis('off')
            sPRE_vs_prox_ax = fig.add_subplot(right[0, 0])
            sPRE_vs_dist_ax = fig.add_subplot(right[1, 0])
        else:
            text_ax = fig.add_subplot(spec[0, :2])
            scatter_all_ax = fig.add_subplot(spec[1, 0])
            dense_all_ax = fig.add_subplot(spec[1, 1])
            cg_vs_aa_contour_ax = fig.add_subplot(spec[2, 0])
            clustered_ax = fig.add_subplot(spec[2, 1])
            scatter_cluster_ax = fig.add_subplot(spec[3, 0])
            render_ax = fig.add_subplot(spec[3, 1])
            sPRE_prox_ax = fig.add_subplot(spec[4, :2])
            sPRE_dist_ax = fig.add_subplot(spec[5, :2])
            N15_prox_ax = fig.add_subplot(spec[6, :2])
            N15_dist_ax = fig.add_subplot(spec[7, :2])
            sPRE_vs_prox_ax = fig.add_subplot(spec[8, :2])
            sPRE_vs_dist_ax = fig.add_subplot(spec[9, :2])


        titles = [f'General Info of cluster {cluster_num}', 'Lowd scatter after Encodermap AA and CG',
                  'Density plot AA and CG', 'Contours of AA and CG simulations',
                  'Cluster assignment with HDBSCAN', f'Scatter plot of cluster {cluster_num}',
                  f'Render of cluster {cluster_num}', 'sPRE proximal', 'sPRE distal', '15N proximal',
                  '15N distal', 'sPRE proximal median vs geom/rmsd centroid', 'sPRE distal median vs geom/rmsd centroid']

        for i, (ax, title) in enumerate(zip(fig.axes, titles)):
            ax.set_title(title)
            if i > 0:
                txt = ax.text(0.02, 0.85, ascii_uppercase[i - 1], transform=ax.transAxes, fontsize=26, va='center', ha='center', color='k')
            if 'render' in title.lower() or 'general' in title.lower():
                ax.axis('off')

        # text
        sum_cluster = len(cg_cluster_points_ind) + len(aa_cluster_points_ind)
        aa_percent = len(aa_cluster_points_ind) / sum_cluster * 100
        cg_percent = len(cg_cluster_points_ind) / sum_cluster * 100
        percentage = sum_cluster / self.trajs[ubq_site].n_frames * 100
        total_time = sum([traj.time[-1] for traj in self.trajs[ubq_site]]) / 1000 / 1000
        text = f"""\
        This is the summary for cluster {cluster_num} of ubiquitination site {ubq_site}
        For these images {self.cg_trajs[ubq_site].n_trajs} coarse-grained and {self.aa_trajs[ubq_site].n_trajs} all-atom trajectories were analyzed.
        These trajs cover a total of {total_time:.2f} micro seconds of simulated time. Using encodermap every frame of these simulations was projected into a 2D
        landscape (seen in A, B, C) representing the conformational ensemble. These points were clustered in 2D using the HDBSCAN clustering algorithm (D),
        resulting in {int(np.max(self.trajs[ubq_site].cluster_membership) + 1)} clusters. This cluster (id {cluster_num}) contains {percentage:.2f}% of all
        sampled points, of which {aa_percent:.2f}% (total: {len(aa_cluster_points_ind)}) are AA structures and {cg_percent:.2f} %(total: {len(cg_cluster_points_ind)}) are CG structures.
        The clustering in the structure space yields a reasonable tight structure ensemble. In (F) 10 evenly spaced all-atom structures are rendered.
        Only the AA structures could be used for sPRE calculation, because back-mapping from CG to AA requires a minimization step which takes 10 minutes per structure.
        The results are in figs G-L show measured sPRE values vs all (AA) calculated sPRE values (G and H), measured 15N relaxation times at {mhz} MHz and the
        calculated 15 N times (I and J) and the difference between median sPRE values of clustered points vs the sPRE values of the RMSD centroid of
        this cluster and the structure closest to the geom. centroid of the cluster (K and L). Considered trajectories in this cluster are:
        {np.unique(self.trajs[ubq_site].name_arr[self.trajs[ubq_site].cluster_membership == cluster_num]).tolist()[:3]} and more.
        Trajectories with `G_2ub_{ubq_site}` are atomistic sims started from extended structures.
        `GfM` are atomistic, started from back-mapped CG trajs.
        `GfM` and `SMin` are extracted from Sketchmap minima.
        `GfM` and `SMin_rnd` are extracted from random positions in Sketchmap minima.
        Trajectories with `2ub` without `G` are CG trajectories.
        Trajectories without date-code are also all-atom from extended. These simulations were started to increase the number of all-atom structures.
        """
        if not reduced:
            text_ax.text(0.05, 0.95, text, transform=text_ax.transAxes, fontsize=14,
                    verticalalignment='top')


        # scatter
        scatter_all_ax.scatter(*self.aa_trajs[ubq_site].lowd[::10].T, s=1, c='C0', label='aa')
        scatter_all_ax.scatter(*self.cg_trajs[ubq_site].lowd[::10].T, s=1, c='C1', label='cg')
        scatter_all_ax.legend(loc=2)

        pyemma.plots.plot_free_energy(*self.trajs[ubq_site].lowd.T, cmap='turbo', cbar=False, ax=dense_all_ax, nbins=nbins)

        aa_H, xedges, yedges = np.histogram2d(*self.aa_trajs[ubq_site].lowd[:, :2].T, bins=nbins)
        xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
        ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)
        X, Y = np.meshgrid(xcenters, ycenters)
        aa_H[aa_H > 0] = 1
        out = copy.deepcopy((aa_H))
        for i in range(nbins - 1):
            for j in range(nbins - 1):
                if any([aa_H[i - 1, j], aa_H[i + 1, j], aa_H[i, j - 1], aa_H[i, j + 1]]):
                    out[i, j] = 1
        cg_vs_aa_contour_ax.contour(X, Y, out, levels=2, cmap='Blues')
        cg_H, xedges, yedges = np.histogram2d(*self.cg_trajs[ubq_site].lowd[:, :2].T, bins=nbins)
        xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
        ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)
        X, Y = np.meshgrid(xcenters, ycenters)
        cg_H[cg_H > 0] = 1
        out = copy.deepcopy((cg_H))
        for i in range(nbins - 1):
            for j in range(nbins - 1):
                if any([aa_H[i - 1, j], aa_H[i + 1, j], aa_H[i, j - 1], aa_H[i, j + 1]]):
                    out[i, j] = 1
        cg_vs_aa_contour_ax.contour(X, Y, out, levels=2, cmap='Greys')

        color_palette = sns.color_palette('deep', max(self.trajs[ubq_site].cluster_membership) + 1)
        cluster_colors = [(*color_palette[x], 1) if x >= 0
                          else (0.5, 0.5, 0.5, 0.1)
                          for x in self.trajs[ubq_site].cluster_membership]
        clustered_ax.scatter(*self.trajs[ubq_site].lowd[:, :2].T, s=1, c=cluster_colors)
        ind_ = np.where(self.trajs[ubq_site].cluster_membership == cluster_num)[0]
        lower_left = np.min(self.trajs[ubq_site].lowd[ind_], axis=0)
        upper_right = np.max(self.trajs[ubq_site].lowd[ind_], axis=0)
        width = upper_right[0] - lower_left[0]
        height = upper_right[1] - lower_left[1]
        rect = mpl.patches.Rectangle(lower_left, width, height, fill=False, ec='red', ls='--')
        clustered_ax.add_artist(rect)

        where = np.where(self.trajs[ubq_site].cluster_membership == cluster_num)[0]
        data = self.trajs[ubq_site].lowd
        ax4 = scatter_cluster_ax
        divider = make_axes_locatable(ax4)
        axHistx = divider.append_axes("top", size=1.2, pad=0.1)#, sharex=ax1)
        axHisty = divider.append_axes("right", size=1.2, pad=0.1)#, sharey=ax1)
        spines = [k for k in axHistx.spines.values()]
        spines[1].set_linewidth(0)
        spines[3].set_linewidth(0)
        axHistx.set_xticks([])
        H, edges, patches = axHistx.hist(data[:, 0][where], bins=50)
        axHistx.set_ylabel('count')
        axHistx.set_title('Scatter of Cluster')
        # y hist
        spines = [k for k in axHisty.spines.values()]
        spines[1].set_linewidth(0)
        spines[3].set_linewidth(0)
        axHisty.set_yticks([])
        H, edges, patches = axHisty.hist(data[:, 1][where], bins=50, orientation='horizontal')
        axHisty.set_xlabel('count')
        # scatter data
        ax4.scatter(x=data[where,0], y=data[where,1], s=1)
        spines = [k for k in ax4.spines.values()]
        spines[3].set_linewidth(0)
        spines[1].set_linewidth(0)
        ax4.set_xlabel('x in a.u.')
        ax4.set_ylabel('y in a.u.')

        # add geom and rmsd centroid
        scatter_cluster_ax.scatter(*self.trajs[ubq_site].lowd[rmsd_centroid_index], marker='H', c='red', label='rmsd centroid')
        scatter_cluster_ax.scatter(*self.trajs[ubq_site].lowd[geom_centroid_index], marker='D', c='k', label='geom centroid')
        scatter_cluster_ax.legend()

        try:
            view, dummy_traj = em.misc.clustering.gen_dummy_traj(self.aa_trajs[ubq_site], 0, align_string='name CA and resid > 76', superpose=self.aa_trajs[ubq_site][0][0].traj, shorten=True,)
            dummy_traj.save_pdb('/tmp/tmp.pdb')
            image = em.plot.plotting.render_vmd('/tmp/tmp.pdb', drawframes=True, scale=1.5)
        finally:
            os.remove('/tmp/tmp.pdb')
        render_ax.imshow(image)

        if reduced:
            ax_pairs = [[sPRE_vs_prox_ax, sPRE_vs_dist_ax], [None, None], [None, None]]
        else:
            ax_pairs = [[sPRE_prox_ax, sPRE_dist_ax], [N15_prox_ax, N15_dist_ax], [sPRE_vs_prox_ax, sPRE_vs_dist_ax]]

        for i, ax_pair in enumerate(ax_pairs):
            if ax_pair[0] is None and ax_pair[1] is None:
                continue
            ax1, ax2 = ax_pair
            if i == 0: # should be 0
                fake_legend_dict1 = {'line': [{'label': 'sPRE NMR experiment', 'color': 'C1'}],
                                     'hatchbar': [
                                         {'label': 'Fast exchanging residues', 'color': 'lightgrey', 'alpha': 0.3,
                                          'hatch': '//'}],
                                     'text': [
                                         {'label': 'Residue used for normalization', 'color': 'red', 'text': 'Ile3'}]}

                (ax1, ax2) = plot_line_data((ax1, ax2), self.df_obs, {'rows': 'sPRE', 'cols': ubq_site})
                (ax1, ax2) = plot_hatched_bars((ax1, ax2), self.fast_exchangers, {'cols': ubq_site}, color='k')
                (ax1, ax2), color_1 = plot_confidence_intervals((ax1, ax2), self.df_comp_norm, {'rows': 'sPRE', 'cols': ubq_site}, cbar_axis=0)
                (ax1, ax2), color_2 = plot_confidence_intervals((ax1, ax2), self.df_comp_norm, {'rows': 'sPRE', 'cols': ubq_site}, cmap='Oranges', trajs=self.aa_trajs[ubq_site], cluster_num=cluster_num, cbar_axis=1, outliers_offset=0.25)

                fake_legend_dict2 = {'envelope': [{'label': 'sPRE calculation with quantiles', 'color': 'lightblue'},
                                                  {'label': f'sPRE for cluster {cluster_num} with quantile', 'color': 'bisque'}],
                                     'line': [{'label': 'median sPRE calculation', 'color': color_1},
                                              {'label': f'median sPRE for cluster {cluster_num}', 'color': color_2}],
                                     'scatter': [{'label': 'outliers sPRE calculation', 'color': color_1, 'marker': 'o'},
                                                 {'label': f'outliers sPRE for cluster {cluster_num}', 'color': color_2, 'marker': 'o'}]}

                if reduced:
                    (ax1, ax2) = plot_single_struct_sPRE((ax1, ax2), self.trajs[ubq_site].get_single_frame(
                        int(rmsd_centroid_index)), self.norm_factors[ubq_site], ubq_site=ubq_site, color='red')
                    (ax1, ax2) = plot_single_struct_sPRE((ax1, ax2), self.trajs[ubq_site].get_single_frame(
                        int(geom_centroid_index)), self.norm_factors[ubq_site], ubq_site=ubq_site, color='k')
                    fake_legend_dict1['line'].extend([{'label': f'sPRE of rmsd center of cluster {cluster_num}', 'color': 'red'},
                                                      {'label': f'sPRE of geom. center of cluster {cluster_num}', 'color': 'black'}])

                ax1 = fake_legend(ax1, fake_legend_dict1)
                ax2 = fake_legend(ax2, fake_legend_dict2)


            if i == 1: # should be 1
                (ax1, ax2) = plot_line_data((ax1, ax2), self.df_obs, {'rows': f'15N_relax_{mhz}', 'cols': ubq_site})
                (ax1, ax2), color_1 = try_to_plot_15N((ax1, ax2), ubq_site, mhz, cbar_axis=0)
                (ax1, ax2), color_2 = try_to_plot_15N((ax1, ax2), ubq_site, mhz, cmap='Oranges',
                                                      trajs=self.aa_trajs[ubq_site], cluster_num=cluster_num,
                                                      cbar_axis=1, outliers_offset=0.25)

                fake_legend_dict = {'envelope': [{'label': f'15N {mhz} calculation with quantiles', 'color': 'lightblue'}],
                                     'line': [{'label': f'15N {mhz} NMR experiment', 'color': color_1},
                                              {'label': f'median 15N {mhz} calculation', 'color': color_2}],
                                    'scatter': [{'label': f'outliers 15N {mhz} calculation', 'color': color_1, 'marker': 'o'},
                                                {'label': f'outliers 15N {mhz} for cluster {cluster_num}', 'color': color_2,
                                                 'marker': 'o'}]}

                ax1 = fake_legend(ax1, fake_legend_dict)


            if i == 2: # should be 2
                (ax1, ax2), color = plot_confidence_intervals((ax1, ax2), self.df_comp_norm,
                                                                {'rows': 'sPRE', 'cols': ubq_site}, cmap='Oranges',
                                                                trajs=self.aa_trajs[ubq_site], cluster_num=cluster_num,)

                fake_legend_dict = {'envelope': [{'label': f'sPRE for cluster {cluster_num} with quantile',
                                                   'color': 'bisque'}],
                                     'line': [{'label': f'median sPRE for cluster {cluster_num}', 'color': color},
                                              {'label': f'sPRE of rmsd center of cluster {cluster_num}', 'color': 'red'},
                                              {'label': f'sPRE of geom. center of cluster {cluster_num}', 'color': 'black'}]}

                fake_legend_dict2 = {'scatter': [{'label': f'outliers sPRE for cluster {cluster_num}', 'color': color, 'marker': 'o'}]}

                ax1 = fake_legend(ax1, fake_legend_dict)
                ax2 = fake_legend(ax2, fake_legend_dict2)

                (ax1, ax2) = plot_single_struct_sPRE((ax1, ax2), self.trajs[ubq_site].get_single_frame(int(rmsd_centroid_index)), self.norm_factors[ubq_site], ubq_site=ubq_site, color='red')
                (ax1, ax2) = plot_single_struct_sPRE((ax1, ax2), self.trajs[ubq_site].get_single_frame(int(geom_centroid_index)), self.norm_factors[ubq_site], ubq_site=ubq_site, color='k')


        # add more fancy stuff to axes
        for i, ax_pair in enumerate(ax_pairs):
            if ax_pair[0] is None and ax_pair[1] is None:
                continue
            ax1, ax2 = ax_pair
            for ax, centers in zip([ax1, ax2], [self.centers_prox, self.centers_dist - 76]):
                ax = add_sequence_to_xaxis(ax)
                ax = color_labels(ax, positions=centers)
                if i == 0 or i == 2:
                    ax.set_ylabel(r'sPRE in $\mathrm{mM^{-1}ms^{-1}}$')

        print(f"Saving image at {out_file}")
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.savefig(out_file.replace('.png', '.pdf'))

    def fitness_assessment(self, overwrite=False, overwrite_image=False):
        print("Manually setting cluster exclusions to exclude no clusters.")
        self.cluster_exclusions = {ubq_site: [] for ubq_site in self.ubq_sites}
        json_savefile = os.path.join(self.analysis_dir, f'quality_factors_with_fixed_normalization.json')
        self.quality_factor_means = {ubq_site: [] for ubq_site in self.ubq_sites}
        if not os.path.isfile(json_savefile) or overwrite:
            self.all_quality_factors = {ubq_site: {} for ubq_site in self.ubq_sites}
        else:
            with open(json_savefile, 'r') as f:
                self.all_quality_factors = json.load(f)

        if overwrite:
            for ubq_site in self.ubq_sites:
                obs = self.df_obs[self.df_obs.index.str.contains('sPRE')][ubq_site].values

                df = self.aa_df[self.aa_df['ubq_site'] == ubq_site]
                sPRE_ind = [i for i in df.columns if 'sPRE' in i and 'norm' in i]

                # put cluster membership into df
                cluster_membership_sPRE = df['cluster_membership']

                # old stuff from when the aa_df was not working
                # for i, traj in enumerate(self.trajs[ubq_site]):
                #     frames = df[df['traj_file'] == traj.traj_file]['frame'].values
                #     cluster_membership_sPRE.append(traj.cluster_membership[frames])
                # cluster_membership_sPRE = np.hstack(cluster_membership_sPRE)
                # df['cluster_membership'] = cluster_membership_sPRE

                # calculcate the per-cluster per-residue median
                cluster_means = {}
                allowed_clusters = np.unique(cluster_membership_sPRE)[1:]
                for cluster_num in allowed_clusters:
                    if cluster_num == -1:
                        continue
                    mean = np.median(df[sPRE_ind][df['cluster_membership'] == cluster_num], axis=0)
                    cluster_means[cluster_num] = mean
                # cluster_means = np.vstack(cluster_means)

                fast_exchange = self.fast_exchangers[ubq_site].values

                # n_clust = self.trajs[ubq_site].cluster_membership.max() + 1
                n_clust = len(allowed_clusters)
                print('checking clusters for ', ubq_site, allowed_clusters, n_clust)
                for no_of_considered_clusters in range(2, n_clust + 1):
                    print(f'considering {no_of_considered_clusters} clusters')
                    combinations = itertools.combinations(allowed_clusters, no_of_considered_clusters)
                    if str(no_of_considered_clusters) in self.all_quality_factors[ubq_site] and not overwrite:
                        print(f"{no_of_considered_clusters} already in json")
                        continue
                    else:
                        self.all_quality_factors[ubq_site][str(no_of_considered_clusters)] = {}
                    for i, combination in enumerate(combinations):
                        combination = np.asarray(combination)
                        if i == 0:
                            print(f"First combination: {combination}")
                        if np.any([np.isnan(cluster_means[c]) for c in combination]):
                            print(f"Cluster in combination {combination} does not occur in aa.")
                            continue
                        if np.any([c in self.cluster_exclusions[ubq_site] for c in combination]):
                            print(f"Cluster in combination {combination} was excluded")
                            continue
                        # solv = scipy.optimize.nnls(np.vstack([cluster_means[c] for c in combination]).T[~fast_exchange], obs[~fast_exchange])[0]
                        solv = make_linear_combination_from_clusters(None, self.aa_df, self.df_obs,
                                                                     self.fast_exchangers, ubq_site, cluster_nums=combination)
                        result = np.sum(solv * np.vstack([cluster_means[c] for c in combination]).T, 1)
                        diff = float(np.mean(np.abs(result[~fast_exchange] - obs[~fast_exchange])))
                        self.all_quality_factors[ubq_site][str(no_of_considered_clusters)][', '.join([str(c) for c in combination])] = diff
                    else:
                        print(f"Last combination: {combination}")

                with open(json_savefile, 'w') as f:
                    json.dump(self.all_quality_factors, f)

        for ubq_site in self.ubq_sites:
            image_file = f'/home/kevin/projects/tobias_schneider/cluster_analysis_with_fixed_normalization/fitness_assessment_{ubq_site}.png'
            if not os.path.isfile(image_file) or overwrite or overwrite_image:
                plt.close('all')
                # data = np.array(self.quality_factor_means[ubq_site])
                fig, ax = plt.subplots()
                for key, value in self.all_quality_factors[ubq_site].items():
                    self.quality_factor_means[ubq_site].append(np.mean(list(value.values())))
                ax.boxplot([[v for v in value.values()] for value in self.all_quality_factors[ubq_site].values()],
                           positions=list(map(int, self.all_quality_factors[ubq_site].keys())))
                # ax.plot(np.arange(len(data)), data)

                ax.set_title(f"Quality factor for n clusters for {ubq_site}")
                ax.set_xlabel("n")
                ax.set_ylabel("Mean abs difference between exp and sim")
                plt.savefig(image_file)

    def where_is_best_fitting_point(self):
        frames = []
        for i, ubq_site in enumerate(self.ubq_sites):
            df = self.df_comp_norm[self.df_comp_norm['ubq_site'] == ubq_site]
            sPRE_ind = [i for i in df.columns if 'sPRE' in i and 'norm' in i]
            calc_values = df[sPRE_ind].values
            values = np.expand_dims(self.df_obs[ubq_site][self.df_obs[ubq_site].index.str.contains('sPRE')], 0)
            diff = np.abs(calc_values - values)
            mean = np.mean(diff, axis=1)
            found = np.argmin(mean)
            traj_file = df.iloc[found]['traj_file']
            frame_no = df.iloc[found]['frame']
            print(traj_file, frame_no)
            traj_ind = np.where(np.array([traj.traj_file for traj in self.aa_trajs[ubq_site]]) == traj_file)[0][0]
            frame = self.aa_trajs[ubq_site][traj_ind][frame_no]
            frames.append(frame)

        plt.close('all')
        fig, axes = plt.subplots(ncols=3, figsize=(15, 5))

        for ax, frame, ubq_site in zip(axes, frames, self.ubq_sites):
            color_palette = sns.color_palette('deep', max(self.trajs[ubq_site].cluster_membership) + 1)
            cluster_colors = [(*color_palette[x], 1) if x >= 0
                              else (0.5, 0.5, 0.5, 0.1)
                              for x in self.trajs[ubq_site].cluster_membership]
            ax.scatter(*self.trajs[ubq_site].lowd[:, :2].T, s=1, c=cluster_colors)

            ax.scatter(*frame.lowd.T, marker='h', facecolors='w', edgecolors='k')

            text = f"Best fitting conformation for {ubq_site}"
            if frame.cluster_membership == -1:
                text += " does not appear in any cluster."
            else:
                text += f" appears in cluster {frame.cluster_membership}"

            ax.set_title(text)
        plt.tight_layout()
        image_file = os.path.join(self.analysis_dir, f'best_fitting_in_cluster.png')
        plt.savefig(image_file)

    def find_lowest_diffs_in_all_quality_factors(self, overwrite=False):
        result = {}
        if overwrite or not hasattr(self, 'all_quality_factors_with_combinations'):
            self.min_cluster_combinations = {ubq_site: {} for ubq_site in self.ubq_sites}
            for i, ubq_site in enumerate(self.ubq_sites):
                for j, no_of_considered_clusters in enumerate(self.all_quality_factors[ubq_site].keys()):
                    min_ = float('inf')
                    for k, (combination, value) in enumerate(self.all_quality_factors[ubq_site][no_of_considered_clusters].items()):
                        if value < min_:
                            min_ = value
                            min_combination = set(ast.literal_eval(combination))
                    self.min_cluster_combinations[ubq_site][no_of_considered_clusters] = min_combination
                values = list(self.min_cluster_combinations[ubq_site].values())
                intersection = values[0].intersection(*values[1:])
                result[ubq_site] = intersection
        pprint(result)

        for ubq_site, (clu1, clu2) in result:
            pass

    def plot_cluster_rmsds(self):
        for i, ubq_site in enumerate(self.ubq_sites):
            labels = []
            for j, cluster_num in enumerate(np.unique(self.aa_trajs[ubq_site].cluster_membership)[1:]):
                cluster_analysis_outdir = os.path.join(self.analysis_dir,
                                                       f'cluster_analysis/{ubq_site}/cluster_{cluster_num}')
                rmsd_centroid_index_file = os.path.join(cluster_analysis_outdir,
                                                        f'{ubq_site}_cluster_{cluster_num}_rmsd_centroid.npy')
                if not os.path.isfile(rmsd_centroid_index_file):
                    continue
                if ubq_site in self.cluster_exclusions:
                    if cluster_num in self.cluster_exclusions[ubq_site]:
                        continue
                rmsd_centroid_index = np.load(rmsd_centroid_index_file).astype(int)
                index = self.aa_trajs[ubq_site].index_arr[rmsd_centroid_index]
                frame = self.aa_trajs[ubq_site][index[0]].traj[index[1]]
                if j == 0:
                    traj = copy.deepcopy(frame)
                else:
                    traj = traj.join(frame)
                labels.append(j)

            atom_indices = [a.index for a in traj.topology.atoms if a.element.symbol != 'H']
            distances = np.empty((traj.n_frames, traj.n_frames))
            for i in range(traj.n_frames):
                distances[i] = md.rmsd(traj, traj, i, atom_indices=atom_indices)

            image_file = os.path.join(self.analysis_dir, f'rmsd_matrix_between_clusters_{ubq_site}.png')

            plt.close('all')

            fig, ax = plt.subplots()
            ax.imshow(distances)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

            ax.set_xlabel("Cluster ID")
            ax.set_ylabel("Cluster ID")

            # Loop over data dimensions and create text annotations.
            for i in range(len(labels)):
                for j in range(len(labels)):
                    text = ax.text(j, i, np.round(distances[i, j], 1), fontsize=6,
                                   ha="center", va="center", color="w")

            ax.set_title("RMSD between two clusters")
            plt.tight_layout()
            plt.savefig(image_file)

    def get_surface_coverage(self, overwrite=False, overwrite_image=False):
        if not (hasattr(self, 'polar_coordinates_aa') or hasattr(self, 'polar_coordinates_cg')) or overwrite:
            self.polar_coordinates_aa = {k: [[], []] for k in self.ubq_sites}
            self.polar_coordinates_cg = {k: [[], []] for k in self.ubq_sites}
        for ubq_site in self.ubq_sites:
            if self.polar_coordinates_aa[ubq_site] == [[], []] or overwrite:
                image_file = os.path.join(self.analysis_dir, f'surface_coverage_{ubq_site}.png')
                image_file2 = f'/home/kevin/projects/tobias_schneider/cluster_analysis_with_fixed_normalization/{ubq_site}/surface_coverage_{ubq_site}.png'
                polar_coordinates_aa_file = os.path.join(self.analysis_dir, f'polar_coordinates_aa_{ubq_site}.npy')
                polar_coordinates_cg_file = os.path.join(self.analysis_dir, f'polar_coordinates_cg_{ubq_site}.npy')

                ref, idx, labels = center_ref_and_load()

                if not (os.path.isfile(polar_coordinates_aa_file) and os.path.isfile(polar_coordinates_cg_file)) or overwrite:
                    for i, traj in enumerate(self.trajs[ubq_site]):
                        if i % 10 == 0:
                            print(i)
                        if any([a.name == 'BB' for a in traj.top.atoms]):
                            traj = traj.traj.superpose(reference=ref, ref_atom_indices=idx, atom_indices=traj.top.select('name BB and resid >= 76'))
                            dist_ind = traj.top.select('resid < 76')
                            coords = appendLonLatTraj(traj.xyz)[:, dist_ind, 4:]
                            self.polar_coordinates_cg[ubq_site][0].append(coords[:, :, 0])
                            self.polar_coordinates_cg[ubq_site][1].append(coords[:, :, 1])
                        else:
                            traj = traj.traj.superpose(reference=ref, ref_atom_indices=idx, atom_indices=traj.top.select('name CA and resid >= 76'))
                            dist_ind = traj.top.select('resid < 76')
                            coords = appendLonLatTraj(traj.xyz)[:, dist_ind, 4:]
                            self.polar_coordinates_aa[ubq_site][0].append(coords[:, :, 0])
                            self.polar_coordinates_aa[ubq_site][1].append(coords[:, :, 1])

                    self.polar_coordinates_aa[ubq_site] = np.stack((np.vstack(self.polar_coordinates_aa[ubq_site][0]), np.vstack(self.polar_coordinates_aa[ubq_site][1])), axis=2)
                    self.polar_coordinates_cg[ubq_site] = np.stack((np.vstack(self.polar_coordinates_cg[ubq_site][0]), np.vstack(self.polar_coordinates_cg[ubq_site][1])), axis=2)
                    np.save(polar_coordinates_aa_file, self.polar_coordinates_aa[ubq_site])
                    np.save(polar_coordinates_cg_file, self.polar_coordinates_cg[ubq_site])
                else:
                    self.polar_coordinates_aa[ubq_site] = np.load(polar_coordinates_aa_file)
                    self.polar_coordinates_cg[ubq_site] = np.load(polar_coordinates_cg_file)
            else:
                pass


            if not os.path.isfile(image_file) or overwrite or overwrite_image:

                plt.close('all')
                fig, ax = plt.subplots(subplot_kw={'projection': ccrs.EckertIV()})
                ax = add_reference_to_map(ax)

                nbins = 200

                x = np.concatenate([self.polar_coordinates_aa[ubq_site][:, :, 0].flatten(), self.polar_coordinates_cg[ubq_site][:, :, 0].flatten()])
                y = np.concatenate([self.polar_coordinates_aa[ubq_site][:, :, 1].flatten(), self.polar_coordinates_cg[ubq_site][:, :, 1].flatten()])

                H, xedges, yedges = np.histogram2d(x=x, y=y, bins=(nbins, int(nbins / 2)), range=[[-180, 180], [-90, 90]])
                xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
                ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)
                X, Y = np.meshgrid(xcenters, ycenters)


                lon = np.linspace(-180, 180, nbins)
                lat = np.linspace(-90, 90, int(nbins / 2))
                lon2d, lat2d = np.meshgrid(lon, lat)


                cmap = plt.cm.get_cmap('viridis').copy()
                cmap.set_bad('w', 0.0)
                H = np.ma.masked_where(H < 0.1, H)

                ax.contourf(X, Y, H.T, transform=ccrs.PlateCarree())

                plt.savefig(image_file)
                plt.savefig(image_file2)

    def get_mean_tensor_of_inertia(self, overwrite=False, overwrite_image=False,
                                   with_Ixx_y_Izz_analysis=False):

        if not hasattr(self, 'inertia_tensors') or overwrite:
            self.inertia_tensors = {k: [] for k in self.ubq_sites}
            self.aa_inertia_tensors = {k: [] for k in self.ubq_sites}
        else:
            if set(self.inertia_tensors.keys()) != set(self.ubq_sites):
                raise NotImplementedError()

        for i, ubq_site in enumerate(self.ubq_sites):
            inertia_tensors_file = os.path.join(self.analysis_dir, f'inertia_distribution_{ubq_site}.npy')
            aa_inertia_tensors_file = os.path.join(self.analysis_dir, f'aa_inertia_distribution_{ubq_site}.npy')
            image_file = os.path.join(self.analysis_dir, f'inertia_distribution_{ubq_site}.png')

            ref, idx, labels = center_ref_and_load()
            ref_aa = self.aa_trajs['k6'][0][0].traj.superpose(ref,
                                                             atom_indices=self.aa_trajs[ubq_site][0][0].traj.top.select('name CA and resid >= 76'),
                                                             ref_atom_indices=ref.top.select('name CA'))

            if not os.path.isfile(inertia_tensors_file) or overwrite:
                found_aa = False
                found_cg = False
                for j, traj in enumerate(self.trajs[ubq_site]):
                    if j % 10 == 0:
                        print(j)
                    if not is_aa_sim(traj.traj_file):
                        traj = traj.traj.superpose(reference=ref, ref_atom_indices=idx,
                                                   atom_indices=traj.top.select('name BB and resid >= 76'))
                        # found_cg = True
                        # take the Ixx, Iyy, and Izz values
                        tensor = md.compute_inertia_tensor(traj)
                    else:
                        traj = traj.traj.superpose(reference=ref, ref_atom_indices=idx,
                                                   atom_indices=traj.top.select('name CA and resid >= 76'))
                        # found_aa = True
                        # take the Ixx, Iyy, and Izz values
                        tensor = md.compute_inertia_tensor(traj)
                        self.aa_inertia_tensors[ubq_site].append(tensor)
                    self.inertia_tensors[ubq_site].append(tensor)
                    if found_cg and found_aa:
                        break
                self.inertia_tensors[ubq_site] = np.vstack(self.inertia_tensors[ubq_site])
                self.aa_inertia_tensors[ubq_site] = np.vstack(self.aa_inertia_tensors[ubq_site])
                np.save(inertia_tensors_file, self.inertia_tensors[ubq_site])
                np.save(aa_inertia_tensors_file, self.aa_inertia_tensors[ubq_site])
            elif hasattr(self, 'inertia_tensors') and hasattr(self, 'aa_inertia_tensors'):
                if len(self.inertia_tensors[ubq_site]) == self.trajs[ubq_site].n_frames:
                    pass
                else:
                    raise Exception("Wrong number of tensors loaded.")
            else:
                print("Loading inertia tensors from file.")
                self.inertia_tensors[ubq_site] = np.load(inertia_tensors_file)
                self.aa_inertia_tensors[ubq_site] = np.load(aa_inertia_tensors_file)

            assert len(self.inertia_tensors[ubq_site] == self.trajs[ubq_site].n_frames)
            assert len(self.aa_inertia_tensors[ubq_site] == self.aa_trajs[ubq_site].n_frames)

            mean = np.mean(self.aa_inertia_tensors[ubq_site], axis=0)
            std = np.std(self.aa_inertia_tensors[ubq_site], axis=0)

            lower = mean - 0.75 * std
            upper = mean + 0.75 * std

            where = np.logical_and(self.aa_inertia_tensors[ubq_site] >= lower, self.aa_inertia_tensors[ubq_site] <= upper)
            where = np.all(where, axis=(1, 2))
            print(np.unique(where, return_counts=True))
            where = np.where(where)[0]
            view, traj = _gen_dummy_traj_single(self.aa_trajs[ubq_site], where, max_frames=300, superpose=ref_aa, stack_atoms=False, align_string='name CA and resid >= 76')

            tensor_analysis_outdir = os.path.join(self.analysis_dir, f'inertia_tensor_analysis/{ubq_site}/')
            os.makedirs(tensor_analysis_outdir, exist_ok=True)
            pdb_file = os.path.join(tensor_analysis_outdir, f'mean_structure_from_inertia.pdb')
            xtc_file = os.path.join(tensor_analysis_outdir, f'mean_structure_from_inertia.xtc')

            traj[0].save_pdb(pdb_file)
            traj[1:].save_xtc(xtc_file)

            if with_Ixx_y_Izz_analysis:

                all_means = []
                all_covars = []
                all_weights = []
                for ax, direction in zip([0, 1, 2], ['Ixx', 'Iyy', 'Izz']):

                    tensor_analysis_outdir = os.path.join(self.analysis_dir,
                                                          f'inertia_tensor_analysis/{ubq_site}/{direction}')
                    os.makedirs(tensor_analysis_outdir, exist_ok=True)

                    data = np.expand_dims(self.inertia_tensors[ubq_site][:, ax], 1)
                    gmm = GMM(n_components=3, covariance_type='full').fit(data)
                    # kde = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=10000).fit(data)

                    means = gmm.means_.flatten()
                    colvars = gmm.covariances_.flatten()
                    weights = gmm.weights_.flatten()

                    # filter low weights
                    high_weights = weights > 0.15
                    weights = weights[high_weights]
                    colvars = colvars[high_weights]
                    means = means[high_weights]

                    # filter similar peaks
                    tolerance = 15000
                    while np.any(np.abs(means[:-1] - means[1:]) < tolerance):
                        out = np.full(means.shape, True)
                        for idx in range(len(means)):
                            try:
                                close = np.isclose(means, means[idx], atol=tolerance)
                                close[idx] = False
                                out &= ~close
                                means = means[out]
                                colvars = colvars[out]
                                weights = weights[out]
                            except (IndexError, ValueError):
                                break

                    # iterate over the 'clusters'
                    for cluster_no, (cluster_mean, cluster_colvar) in enumerate(zip(means, colvars)):
                        std = scipy_norm.std(cluster_mean, np.sqrt(cluster_colvar))
                        std = np.array([cluster_mean - std, cluster_mean + std])
                        where = np.logical_and(self.aa_inertia_tensors[ubq_site][:, ax] >= std[0], self.aa_inertia_tensors[ubq_site][:, ax] <= std[1])
                        assert len(where) == self.aa_trajs[ubq_site].n_frames, print(len(where))
                        where = np.where(where)[0]
                        if where.size == 0:
                            continue
                        view, traj = _gen_dummy_traj_single(self.aa_trajs[ubq_site], where, max_frames=300, superpose=ref_aa, stack_atoms=False, align_string='name CA and resid >= 76')

                        pdb_file = os.path.join(tensor_analysis_outdir, f'cluster_{cluster_no}_along_{direction}_axis.pdb')
                        xtc_file = os.path.join(tensor_analysis_outdir, f'cluster_{cluster_no}_along_{direction}_axis.xtc')

                        traj[0].save_pdb(pdb_file)
                        traj[1:].save_xtc(xtc_file)

                    # append
                    all_means.append(means)
                    all_covars.append(colvars)
                    all_weights.append(weights)

                # find structures that fit into the ranges
                if not os.path.isfile(image_file) or overwrite or overwrite_image:
                    nbins = 100
                    H, (xedges, yedges, zedges) = np.histogramdd(self.inertia_tensors[ubq_site], bins=nbins)

                    xcenters = np.mean(np.vstack([xedges[0:-1], xedges[1:]]), axis=0)
                    ycenters = np.mean(np.vstack([yedges[0:-1], yedges[1:]]), axis=0)

                    X, Y = np.meshgrid(xcenters, ycenters)

                    plt.close('all')
                    fig = plt.figure()
                    ax = fig.add_subplot(221, projection='3d')
                    ax1 = fig.add_subplot(222)
                    ax2 = fig.add_subplot(223)
                    ax3 = fig.add_subplot(224)
                    # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

                    # ax.scatter(*analysis.inertia_tensors[::100].T, c='k', s=1, alpha=0.01)

                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')

                    for ct in np.linspace(0, nbins, 20, dtype=int, endpoint=False):
                        cmap = plt.cm.get_cmap('turbo').copy()
                        cmap.set_bad('w', alpha=0.0)
                        hist = np.ma.masked_where(H[:, :, ct] == 0, H[:, :, ct])
                        cs = ax.contourf(X, Y, hist.T, zdir='z',
                                         offset=zedges[ct],
                                         levels=20,
                                         cmap='turbo', alpha=0.5)
                    # divider = make_axes_locatable(ax)
                    # cax = divider.append_axes('right', size='2%', pad=0.05)
                    # plt.colorbar(cs, cax=cax)

                    ax.set_xlim(xedges[[0, -1]])
                    ax.set_ylim(yedges[[0, -1]])
                    ax.set_zlim(zedges[[0, -1]])

                    # plt.savefig('/mnt/data/kevin/xplor_analysis_files/inertia_distribution.png')
                    ax1.hist(self.inertia_tensors[ubq_site][:, 0], bins=xedges, density=True)
                    ax2.hist(self.inertia_tensors[ubq_site][:, 1], bins=yedges, density=True)
                    ax3.hist(self.inertia_tensors[ubq_site][:, 2], bins=zedges, density=True)

                    for ax, means, covars, weights in zip([ax1, ax2, ax3], all_means, all_covars, all_weights):
                        x = np.linspace(0, ax.get_xlim()[1], 10000)
                        for mean, covar, weight, color in zip(means, covars, weights, ['C1', 'C2', 'C3']):
                            ax.plot(x, weight * scipy_norm.pdf(x, mean, np.sqrt(covar)).ravel(), c=color)

                    for pos, ax in zip(['x', 'y', 'z'], [ax1, ax2, ax3]):
                        ax.set_xlabel(r"$I_{" f"{pos}{pos}"+ r"}$ in $amu \cdot nm^{2}$")
                        ax.set_ylabel("Count")
                        ax.set_title(f"Moment of inertia along {pos} axis")

                    plt.tight_layout()
                    plt.savefig(image_file)

    def distance_vs_pseudo_torsion(self, overwrite=False, overwrite_image=False):
        if self.check_attr_all('pseudo_dihedral') and not overwrite:
            print("Pseudo dihedral already in all trajs")
        else:
            for i, ubq_site in enumerate(self.ubq_sites):
                dihedrals_file = os.path.join(self.analysis_dir, f'pseudo_dihedrals_{ubq_site}.npy')
                dists_file = os.path.join(self.analysis_dir, f'cog_dists_{ubq_site}.npy')
                if not os.path.isfile(dihedrals_file) or not os.path.isfile(dists_file) or overwrite:

                    dihedrals = []
                    dists = []

                    aa_ca_indices = self.aa_trajs[ubq_site][0].top.select('(resname LYQ and name CA) or (resname GLQ and name CA)')
                    aa_subunit_prox_indices = self.aa_trajs[ubq_site][0].top.select('resid >= 76')
                    aa_subunit_dist_indices = self.aa_trajs[ubq_site][0].top.select('resid < 76')

                    cg_ca_indices = self.cg_trajs[ubq_site][0].top.select('(resname LYQ and name BB) or (resname GLQ and name BB)')
                    cg_subunit_prox_indices = self.cg_trajs[ubq_site][0].top.select('resid >= 76')
                    cg_subunit_dist_indices = self.cg_trajs[ubq_site][0].top.select('resid < 76')

                    for j, traj in enumerate(self.trajs[ubq_site]):
                        if j % 20 == 0:
                            print(f"{j} of {self.trajs[ubq_site].n_trajs}")
                        if is_aa_sim(traj.traj_file):
                            ca_indices = aa_ca_indices
                            prox_indices = aa_subunit_prox_indices
                            dist_indices = aa_subunit_dist_indices
                        else:
                            ca_indices = cg_ca_indices
                            prox_indices = cg_subunit_prox_indices
                            dist_indices = cg_subunit_dist_indices
                        pos = np.empty((traj.n_frames, 4, 3))
                        pos[:, 0] = np.mean(traj.xyz[:, prox_indices], axis=1)
                        pos[:, 1] = traj.xyz[:, ca_indices[0]]
                        pos[:, 2] = traj.xyz[:, ca_indices[1]]
                        pos[:, 3] = np.mean(traj.xyz[:, dist_indices], axis=1)

                        for p in pos:
                            dihedrals.append(new_dihedral(p))
                        dists.extend(np.linalg.norm(pos[:, 2] - pos[:, 1], axis=1))

                    dihedrals = np.asarray(dihedrals)
                    dists = np.asarray(dists)

                    np.save(dihedrals_file, dihedrals)
                    np.save(dists_file, dists)

                else:
                    dihedrals = np.load(dihedrals_file)
                    dists = np.load(dists_file)

                aa_indices = np.arange(self.aa_trajs[ubq_site].n_frames)
                cg_indices = np.arange(self.aa_trajs[ubq_site].n_frames, self.aa_trajs[ubq_site].n_frames + self.cg_trajs[ubq_site].n_frames)
                self.aa_trajs[ubq_site].load_CVs(dihedrals[aa_indices], 'pseudo_dihedrals')
                self.cg_trajs[ubq_site].load_CVs(dihedrals[cg_indices], 'pseudo_dihedrals')
                self.aa_trajs[ubq_site].load_CVs(dists[aa_indices], 'cog_dists')
                self.cg_trajs[ubq_site].load_CVs(dists[cg_indices], 'cog_dists')

        for ubq_site in self.ubq_sites:
            image_file = os.path.join(self.analysis_dir, f'cog_vs_dihe_{ubq_site}.png')
            if not os.path.isfile(image_file) or overwrite or overwrite_image:
                fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)

                for i, ax in enumerate([ax1, ax2]):
                    ax.set_ylim([-180, 180])
                    ax.set_xlabel('COG distance dist-prox in nm')
                    if i == 0:
                        ax.set_ylabel('Pseudo dihedral prox-LYQ-GLQ-dist in degree')

                ax1.scatter(self.trajs[ubq_site].cog_dists, self.trajs[ubq_site].pseudo_dihedrals, s=1)
                pyemma.plots.plot_density(self.trajs[ubq_site].cog_dists, self.trajs[ubq_site].pseudo_dihedrals, cmap='turbo', ax=ax2)
                print(f"Saving Image at {image_file}")
                plt.tight_layout()
                plt.savefig(image_file)

    def cluster_pseudo_torsion(self, overwrite=False, overwrite_image=False,
                               overwrite_struct_files=False):
        if self.check_attr_all('cluster_membership_in_pseudo_dih_space') and not overwrite:
            print("Pseudo dihedral already in all trajs")
        else:
            for i, ubq_site in enumerate(self.ubq_sites):
                print(ubq_site)
                cluster_membeship_file = os.path.join(self.analysis_dir, f'pseudo_dihedrals_cluster_membership_{ubq_site}.npy')
                if not os.path.isfile(cluster_membeship_file) or overwrite:
                    data = np.vstack([self.trajs[ubq_site].cog_dists, self.trajs[ubq_site].pseudo_dihedrals]).T
                    print(data.shape)
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=100).fit(data)
                    labels = clusterer.labels_
                    unique_labels, counts = np.unique(clusterer.labels_, return_counts=True)
                    print(unique_labels, counts)
                    most_populated_clusters = np.argsort(counts[1:])[::-1][:5]
                    print(most_populated_clusters)
                    small_clusters = np.setdiff1d(unique_labels, most_populated_clusters)
                    print(small_clusters)
                    for small_cluster in small_clusters:
                        labels[labels == small_cluster] = -1
                    print(np.unique(labels))
                    for j, cluster_num in enumerate(np.unique(labels)[1:]):
                        labels[labels == cluster_num] = j
                    print(np.unique(labels, return_counts=True))
                    np.save(cluster_membeship_file, labels)
                else:
                    labels = np.load(cluster_membeship_file)
                aa_indices = np.arange(self.aa_trajs[ubq_site].n_frames)
                cg_indices = np.arange(self.aa_trajs[ubq_site].n_frames,
                                       self.aa_trajs[ubq_site].n_frames + self.cg_trajs[ubq_site].n_frames)
                self.aa_trajs[ubq_site].load_CVs(labels[aa_indices], 'cluster_membership_in_pseudo_dih_space')
                self.cg_trajs[ubq_site].load_CVs(labels[cg_indices], 'cluster_membership_in_pseudo_dih_space')

        for ubq_site in self.ubq_sites:
            image_file = os.path.join(self.analysis_dir, f'cog_vs_dihe_{ubq_site}.png')
            if not os.path.isfile(image_file) or overwrite or overwrite_image:
                fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(15, 5))

                for i, ax in enumerate([ax1, ax2]):
                    ax.set_ylim([-180, 180])
                    ax.set_xlabel('COG distance dist-prox in nm')
                    if i == 0:
                        ax.set_ylabel('Pseudo dihedral prox-LYQ-GLQ-dist in degree')

                ax1.scatter(self.trajs[ubq_site].cog_dists, self.trajs[ubq_site].pseudo_dihedrals, s=1)
                pyemma.plots.plot_density(self.trajs[ubq_site].cog_dists, self.trajs[ubq_site].pseudo_dihedrals,
                                          cmap='turbo', ax=ax2)
                color_palette = sns.color_palette('deep', max(self.trajs[ubq_site].cluster_membership_in_pseudo_dih_space) + 1)
                cluster_colors = [(*color_palette[x], 1) if x >= 0
                                  else (0.5, 0.5, 0.5, 0.1)
                                  for x in self.trajs[ubq_site].cluster_membership_in_pseudo_dih_space]
                ax3.scatter(self.trajs[ubq_site].cog_dists, self.trajs[ubq_site].pseudo_dihedrals, s=1, c=cluster_colors)
                for cluster_num in np.unique(self.trajs[ubq_site].cluster_membership_in_pseudo_dih_space):
                    if cluster_num == -1:
                        continue
                    center_cog_dist = np.mean(self.trajs[ubq_site].cog_dists[self.trajs[ubq_site].cluster_membership_in_pseudo_dih_space == cluster_num])
                    center_dih_angl = np.mean(self.trajs[ubq_site].pseudo_dihedrals[self.trajs[ubq_site].cluster_membership_in_pseudo_dih_space == cluster_num])
                    ax3.scatter(center_cog_dist, center_dih_angl, marker='h', facecolors='w', edgecolors='k')
                    ax3.annotate(cluster_num, (center_cog_dist, center_dih_angl), xytext=(5, 5), textcoords='offset pixels')
                print(f"Saving Image at {image_file}")
                plt.tight_layout()
                plt.savefig(image_file)
            else:
                print("Not overwriting image")

        ref, idx, labels = center_ref_and_load()
        ref_aa = self.aa_trajs['k6'][0][0].traj.superpose(ref,
                                                          atom_indices=self.aa_trajs[ubq_site][0][0].traj.top.select(
                                                              'name CA and resid >= 76'),
                                                          ref_atom_indices=ref.top.select('name CA'))

        cluster_selections = {'k6': [2], 'k29': [0, 3, 4], 'k33': [3, 4]}

        if overwrite_struct_files:
            shutil.rmtree(os.path.join(self.analysis_dir, f'cluster_analysis_cog_dihe/'))

        for ubq_site in self.ubq_sites:
            for cluster_num in cluster_selections[ubq_site]:
                if cluster_num == -1:
                    continue
                cluster_analysis_outdir = os.path.join(self.analysis_dir,
                                                       f'cluster_analysis_cog_dihe/{ubq_site}/cluster_{cluster_num}')
                os.makedirs(cluster_analysis_outdir, exist_ok=True)
                struct_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cog_vs_dihe_cluster_{cluster_num}.gro')
                stacked_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cog_vs_dihe_cluster_{cluster_num}_stacked.pdb')
                traj_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_cog_vs_dihe_cluster_{cluster_num}.xtc')
                if not os.path.isfile(struct_file) or overwrite or overwrite_struct_files:
                    view, traj = _gen_dummy_traj_single(self.aa_trajs[ubq_site], cluster_num, col='cluster_membership_in_pseudo_dih_space',
                                                        superpose=ref_aa, max_frames=300, stack_atoms=False, align_string='name CA and resid >= 76')
                    index = []
                    for frame_no, frame in enumerate(traj):
                        if not one_bond_too_long(frame, 3):
                            index.append(frame_no)
                    print(f"Found {traj.n_frames - len(index)} bonds with too long bond lengths for {ubq_site}, {cluster_num}.")
                    index = np.array(index)
                    traj = traj[index]
                    traj[0].save_gro(struct_file)
                    traj[1:].save_xtc(traj_file)
                    traj = traj.atom_slice(traj.top.select('resid < 76'))
                    for frame_no, frame in enumerate(traj):
                        if frame_no == 0:
                            stacked = copy.deepcopy(frame)
                        else:
                            stacked = stacked.stack(frame)
                    stacked.save_pdb(stacked_file)
                else:
                    print("Not overwriting struct files.")

    def save_stl_convex_hull(self, overwrite=False):

        cluster_selections = {'k6': [2], 'k29': [0, 3, 4], 'k33': [3, 4]}

        for ubq_site in self.ubq_sites:
            for cluster_num in cluster_selections[ubq_site]:
                if cluster_num == -1:
                    continue
                cluster_analysis_outdir = os.path.join(self.analysis_dir,
                                                       f'cluster_analysis_conv_hull/{ubq_site}/cluster_{cluster_num}')
                os.makedirs(cluster_analysis_outdir, exist_ok=True)
                struct_file = os.path.join(cluster_analysis_outdir, f'{ubq_site}_conv_hull_cluster_{cluster_num}.stl')
                if not os.path.isfile(struct_file) or overwrite or overwrite_struct_files:
                    view, traj = _gen_dummy_traj_single(self.aa_trajs[ubq_site], cluster_num,
                                                        col='cluster_membership_in_pseudo_dih_space',
                                                        superpose=ref_aa, max_frames=300, stack_atoms=False,
                                                        align_string='name CA and resid >= 76')
                index = []
                for frame_no, frame in enumerate(traj):
                    if not one_bond_too_long(frame, 3):
                        index.append(frame_no)
                print(f"Found {traj.n_frames - len(index)} bonds with too long bond lengths.")
                index = np.array(index)
                traj = traj[index]

    def get_75_percent_convex_hull(self):
        pass

    def get_3d_histogram(self):
        pass

    def load_specific_checkpoint(self, step=None):
        for i, ubq_site in enumerate(self.ubq_sites):
            if step is not None:
                checkpoint = os.path.join(self.analysis_dir, f'runs/{ubq_site}/production_run_tf2/saved_model_{step}.model_*')
            else:
                train_path = os.path.join(self.analysis_dir, f'runs/{ubq_site}/production_run_tf2/')
                checkpoints = glob.glob(f'{train_path}*encoder')
                checkpoint = list(sorted(checkpoints, key=ckpt_step))[-1].replace('model_encoder', '*')
            e_map = em.EncoderMap.from_checkpoint(checkpoint)
            aa_lowd = e_map.encode(self.aa_trajs[ubq_site].highd)
            cg_lowd = e_map.encode(self.cg_trajs[ubq_site].highd)
            self.aa_trajs[ubq_site].load_CVs(aa_lowd, attr_name='lowd')
            self.cg_trajs[ubq_site].load_CVs(cg_lowd, attr_name='lowd')

def one_bond_too_long(traj, threshold=8):
    return np.any([one_bond_too_long_frame(frame, threshold) for frame in traj])

def one_bond_too_long_frame(frame, threshold=8):
    return np.any([get_bond_length(frame, bond) > threshold for bond in frame.top.bonds])

def get_bond_length(frame, bond):
    return np.linalg.norm(frame.xyz[0, bond.atom2.index] - frame.xyz[0, bond.atom1.index])