# Main TODO

- [x] sPRE confidence interval.
- Now with IQR and Q1 - IQR, Q3 + IQR and outliers.
- [x] quality factor for 2, 3, 4, .., clusters
- Quality factor is steadily decresaing
- [x] Use boxplots for the quality factor.
- This graph shows the whiskers to start at the very bottom of the page. This led to further investigations regarding the coefficients of the clusters and for K6 it was found that cluster 5 (not too sure about that) has the greatest coefficient and contains 11% of all points of all simulations, which is in great accordance of sPRE data and MD simulation.
- [ ] Finish inkscape.
- [x] Watch video.
- [x] Watch video again and check for some question regarding RMSD.
- [ ] Where does the best-fitting-structure originate from? Does it lie in a certain cluster?
- [ ] Difference between measurement, best-fitting and median of everything.

- [ ] Ensemble overview
  - [x] Make some polar plots of Ub surface coverage.
  - [x] Get mean tensor of inertia. This needs to be done after superposing the near unit.
  - [x] Get statistics over the tensor of inertia. Best outcome: N main motifs of diUBQ.
  - [ ] Make images of the centroids of these motifs.
  - [ ] For every motif do:
    - [x] Get the statistical weight of that motif (X%).
    - [x] Use the trace/determinant to make statisitcs.
    - [x] Choose the structures within std and save their pdbs.
    - [x] Use vmd to make wireframe images of these ensembles
    - [ ] These structures should be easily differentiable and there should be N main motifs.
    - [ ] What surface features are covered by these structures.

## This is not possible with tensors of inertia

A tensor describes the ineria of an object without a Basis (i.e. an underlying coordinate system). Cosnider the moment of inertia along the xx axis (I_xx). It captures how far away the distal unit from the proximal unit's x-axis lies. A diUbi structure where the distal unit is intersected by the x-vector has a high inertia along that direction. If the distal unit points away, the inertia decreases. The problem is, that it can not in what direction of the coordinate system the distal unit points. The same holds true for I_yy and I_zz. If we now move towards a tensor of inertia, there are no directions anymore. Thus, the only thing the tensor of inertia can capture is the elongation of the diUbi chain. If the subunits are further apart (let's say along any one of the three base vectors), the inertia along the other two increases. This can be captured. Again, we can't say anything about the positioning of proximal and distal subunit.

- [ ] Cluster overview
  - [ ] How many clusters do I need to explain conformational space.


## On hold:
- 15N spin relaxations.
  - Use the tensor of inertia as a weighting. Problem: Mean of tensors needs a fitting.
  - Align to coordinate systems. Fit one domain (MDTraj superpose and then tensor calculation).
  - Ausdehnung in diese Richtungen.
  - Axes of inertia. Symmetrical, or not
  - Similar to atomic Orbitals probability of aotms to be in this region and a cutoff at 75%.
  - 

## Finally
- [x] Write a large LaTeX summary:
  - [ ] General procedure. Running XPLOR.
  - [ ] Linear combinations.
  - [ ] Quality factors and CG/AA cluster all ensemble probabilities.
  - [ ] How many structures needed to build sPRE/MD ensemble.
  - [ ] Weights and probability.
  - [ ] For K6 only one cluster is needed.