# Main TODO

- [x] sPRE confidence interval.
- Now with IQR and Q1 - IQR, Q3 + IQR and outliers.
- [x] quality factor for 2, 3, 4, .., clusters
- Quality factor is steadily decresaing
- [x] Use boxplots for the quality factor.
- This graph shows the whiskers to start at the very bottom of the page. This led to further investigations regarding the coefficients of the clusters and for K6 it was found that cluster 5 (not too sure about that) has the greatest coefficient and contains 11% of all points of all simulations, which is in great accordance of sPRE data and MD simulation.
- [ ] Finish inkscape.
- [x] Watch video.
- [ ] Watch video again and check for some question regarding RMSD.
- [ ] Where does the best-fitting-structure originate from? Does it lie in a certain cluster?
- [ ] Difference between measurement, best-fitting and median of everything.

- [ ] Ensemble overview
  - [x] Make some polar plots of Ub surface coverage.
  - [ ] Get mean tensor of inertia. This needs to be done after superposing the near unit.
  - [ ] Get statistics over the tensor of inertia. Best outcome: N main motifs of diUBQ.
  - [ ] Make images of the centroids of these motifs.
  - [ ] For every motif do:
    - [ ] Get the statistical weight of that motif (X%).
    - [ ] Use the trace/determinant to make statisitcs.
    - [ ] Choose the structures within std and save their pdbs.
    - [ ] Use vmd to make wireframe images of these ensembles
    - [ ] These structures should be easily differentiable and there should be N main motifs.
    - [ ] What surface features are covered by these structures.

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
- [ ] Write a large LaTeX summary:
  - [ ] General procedure. Running XPLOR.
  - [ ] Linear combinations.
  - [ ] Quality factors and CG/AA cluster all ensemble probabilities.
  - [ ] How many structures needed to build sPRE/MD ensemble.
  - [ ] Weights and probability.
  - [ ] For K6 only one cluster is needed.