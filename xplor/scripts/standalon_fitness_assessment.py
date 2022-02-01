
import pandas as pd
import numpy as np
import xplor
import os
import itertools
import json
from joblib import Parallel, delayed
from multiprocessing import cpu_count as mp_cpu_count
from typing import Callable
from scipy.optimize import minimize, NonlinearConstraint, Bounds, nnls

constraint = NonlinearConstraint(np.sum, 1., 1.)
bounds = Bounds(0, 1)

def get_objective_function(cluster_means: np.ndarray, observed_values: np.ndarray) -> Callable:
    def obj_fun(x: np.ndarray) -> float:
        return np.linalg.norm(np.sum(x * cluster_means, 1) - observed_values)
    return obj_fun

def make_linear_combination_from_clusters(aa_df, exp_values, exclude_data,
                                          ubq_site, parallel=False, cluster_nums=None):
    
    only_clusters_aa_df = aa_df.loc[(aa_df[('data', '', 'count_id')].isin(cluster_nums)) & (aa_df[('data', '', 'ubq_site')] == ubq_site)]
    means = only_clusters_aa_df.groupby([('data', '', 'ubq_site'), ('data', '', 'count_id')]).mean()['normalized sPRE']

    # reshape the cluster means
    means = means.droplevel(0, axis='columns').reset_index().drop(columns=[('data', '', 'count_id'), ('data', '', 'ubq_site')])

    # get observed values and exclusions
    exp = exp_values[ubq_site]
    if not exp.index.str.contains('normalized').all(None):
        exp.index = exp.index.map(lambda _: 'normalized ' + _)
    exclude = exclude_data[ubq_site]
    if not exclude.index.str.contains('normalized').all(None):
        exclude.index = exclude.index.map(lambda _: 'normalized ' + _)

    # remove exclusions
    try:
        exp = exp[~exclude]
        means = means.loc[:, ~exclude]
    except pd.core.indexing.IndexingError:
        print('exclude\n', exclude.index, '\n\n')
        print('exp\n', exp.index, '\n\n')
        print('means\n', means.columns, '\n\n')
        raise

    # initial guess
    initial_guess = nnls(means.T, exp)[0]

    # optimize
    res = minimize(get_objective_function(means.T, exp), initial_guess,
                   constraints=[constraint], bounds=bounds)
    
    # apply
    combination = (np.expand_dims(res.x, 1) * means).sum('rows')
    
    # get mad
    mad = (combination - exp).abs().mean()

    if parallel:
        return ', '.join([str(c) for c in cluster_nums]), mad
    else:
        return mad
    
# def parallel_fitness_assesment(cluster_nums):
#     solv = make_linear_combination_from_clusters(aa_df, exp_values, cluster_nums, ubq_site, exclude_data)
#     result = np.sum(solv * np.vstack([cluster_means[c] for c in cluster_nums]).T, 1)
#     diff = float(np.mean(np.abs(result[~fast_exch] - obs[~fast_exch])))
#     return ', '.join([str(c) for c in cluster_nums]), diff
    

def main(aa_df, exp_values, exclude_data,
         overwrite=False, soft_overwrite=True, parallel=True):
    
    if not parallel:
        json_savefile = '/home/kevin/git/xplor_functions/xplor/data/quality_factors_with_fixed_normalization_pandas_native.json'
    else:
        json_savefile = '/home/kevin/git/xplor_functions/xplor/data/quality_factors_with_fixed_normalization_pandas_native_parallel.json'

    quality_factor_means = {ubq_site: [] for ubq_site in UBQ_SITES}
    if not os.path.isfile(json_savefile) or (overwrite and not soft_overwrite):
        all_quality_factors = {ubq_site: {} for ubq_site in UBQ_SITES}
    else:
        with open(json_savefile, 'r') as f:
            all_quality_factors = json.load(f)
                
    for ubq_site in UBQ_SITES:
        obs = exp_values[ubq_site]

        df = aa_df[aa_df[('data', '', 'ubq_site')] == ubq_site]
        
        # calculate the per-cluster per-residue mean
        cluster_means = df.groupby([('data', '', 'count_id')]).mean()['normalized sPRE']
        cluster_means = cluster_means.droplevel([0], 'columns')
        cluster_means.index.name = 'count_id'
        cluster_means.columns = cluster_means.columns.map(lambda _: _.replace('normalized ', ''))

        # calculcate the per-cluster per-residue median
        allowed_clusters = np.unique(cluster_means.index)[1:]

        # check for soft overwrite
        if soft_overwrite:
            if len(all_quality_factors[ubq_site].keys()) == len(list(range(2, len(allowed_clusters) + 1))):
                print(f"All clusters already in the file for ubq_site {ubq_site}")
                continue
            else:
                print("Starting fitness assessment.")
                

        exclude = ~exclude_data[ubq_site]

        # n_clust = self.trajs[ubq_site].cluster_membership.max() + 1
        n_clust = len(allowed_clusters)
        print('checking clusters for ', ubq_site, allowed_clusters, n_clust)
        for no_of_considered_clusters in range(2, n_clust + 1):
            print(f'considering {no_of_considered_clusters} clusters')
            if soft_overwrite:
                if str(no_of_considered_clusters) in all_quality_factors[ubq_site]:
                    print(f"The combinations for {no_of_considered_clusters} for {ubq_site} "
                          f"are already in the dict. Continuing.")
                    continue
            combinations = itertools.combinations(allowed_clusters, no_of_considered_clusters)
            if str(no_of_considered_clusters) in all_quality_factors[ubq_site] and not overwrite:
                print(f"{no_of_considered_clusters} already in json")
                continue
            else:
                all_quality_factors[ubq_site][str(no_of_considered_clusters)] = {}
            if not parallel:
                for i, combination in enumerate(combinations):
                    combination = np.asarray(combination)
                    if i == 0:
                        print(f"First combination: {combination}")
                    # solv = scipy.optimize.nnls(np.vstack([cluster_means[c] for c in combination]).T[~fast_exchange], obs[~fast_exchange])[0]
                    diff = make_linear_combination_from_clusters(aa_df, exp_values, exclude_data,
                                                                 ubq_site, cluster_nums=combination)
                    all_quality_factors[ubq_site][str(no_of_considered_clusters)][', '.join([str(c) for c in combination])] = diff
                else:
                    print(f"Last combination: {combination}")
            else:
                # filter the combinations with non-aa clusters and cluster exclusions
                # not_aa_filter_func = lambda x: not np.any([np.isnan(cluster_means[c]) for c in x])
                # cluster_exclusions_filter_func = lambda x: not np.any([c in self.cluster_exclusions[ubq_site] for c in x])
                results = Parallel(n_jobs=mp_cpu_count() - 4)(delayed(make_linear_combination_from_clusters)(aa_df, exp_values, exclude_data, ubq_site, True, c) for c in combinations)

                # unpack results
                try:
                    results_dict = {k: v for k, v in results}
                except TypeError:
                    print(results)
                    raise
                all_quality_factors[ubq_site][str(no_of_considered_clusters)] = results_dict
            with open(json_savefile, 'w') as f:
                print("Dumping json")
                json.dump(all_quality_factors, f)
    

if __name__ == '__main__':
    UBQ_SITES = ['k6', 'k29', 'k33']
    
    # load the fast exchangers
    exp_values = xplor.functions.parse_input_files.get_observed_df(UBQ_SITES)
    fast_exchangers = xplor.functions.parse_input_files.get_fast_exchangers(UBQ_SITES)
    fast_exchangers.index = fast_exchangers.index.str.replace('fast_exchange', 'sPRE')
    exp_values = exp_values.filter(like='sPRE', axis=0)
    not_exp_values = exp_values == 0
    exclude_data = fast_exchangers | not_exp_values
    
    aa_df = pd.read_hdf('/home/kevin/git/xplor_functions/xplor/data/all_frames_sPRE_sim.h5', 'aa_df')
    main(aa_df, exp_values, exclude_data)

# main(aa_df, exp_values, exclude_data)
