
================================================================================
          APPENDIX A: THEOREM 1 MATHEMATICAL DERIVATION (CORRECTED)
================================================================================

Theorem 1 (Structural Instability of Thresholded Graph Boundaries).
Let A_{ij} = H(|r_{ij}| - \theta) denote the binary adjacency indicator for edge (i,j) 
governed by Heaviside step function H(x). Under finite sample variance \epsilon ~ N(0, \sigma^2), 
the edge variance Var(A_{ij}) is given by:

    Var(A_{ij}) = P(|r_{ij} + \epsilon| \ge \theta) [ 1 - P(|r_{ij} + \epsilon| \ge \theta) ]

By the distributional derivative identity dH/dx = \delta(x), where \delta(x) represents 
the Dirac delta distribution, the boundary sensitivity of the adjacency matrix is maximal 
at the threshold boundary |r_{ij}| = \theta.

For any edge residing exactly at the threshold boundary where |r_{ij}| equals \theta, 
any nonzero \epsilon, regardless of how small, produces a discrete adjacency transition 
of magnitude 1. This is because the Heaviside function is discontinuous at 0 and has no 
transition zone. The sensitivity is maximal precisely at the boundary, where infinitesimal 
continuous variance produces maximal discrete structural change.
================================================================================
