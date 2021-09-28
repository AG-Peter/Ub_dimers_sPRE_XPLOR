import numpy as np
import hdbscan
import encodermap as em
ubq_site = 'k33'
e_map = em.EncoderMap.from_checkpoint(f'/mnt/data/kevin/xplor_analysis_files/runs/{ubq_site}/production_run_tf2/saved_model_100000.model*')
highd = np.vstack([np.load(f'highd_rwmd_aa_{ubq_site}.npy'), np.load(f'highd_rwmd_cg_{ubq_site}.npy')])
lowd = e_map.encode(highd)
clusterer = hdbscan.HDBSCAN(min_cluster_size=2500, cluster_selection_method='leaf')
clusterer.fit(lowd)
np.unique(clusterer.labels_)
aa_highd = np.load(f'highd_rwmd_aa_{ubq_site}.npy')
cg_highd = np.load(f'highd_rwmd_cg_{ubq_site}.npy')
aa_indices = np.arange(len(aa_highd))
print(aa_indices.shape, aa_highd.shape)
cg_indices = np.arange(len(aa_highd), len(aa_highd) + len(cg_highd))
print(cg_indices.shape, cg_highd.shape)
aa_labels = clusterer.labels_[aa_indices]
cg_labels = clusterer.labels_[cg_indices]
np.save(f'cluster_membership_aa_{ubq_site}.npy', aa_labels)
np.save(f'cluster_membership_cg_{ubq_site}.npy', cg_labels)
