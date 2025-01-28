# Core Multipers

Here, we implement the Alpha/Delaunay Core Bifiltration for use with the [multipers](https://github.com/DavidLapous/multipers) library.

## Overview

The algorithm consists of the following steps:

1. Compute the Delaunay triangulation $\text{Del}(A)$ of the input points $A\subseteq(M,d)$.
2. For each simplex $\sigma\in\text{Del}(A)$, compute the set of minimal appearances (birth set) $b(\sigma)\subset[0,\infty]\times[0,\infty]^\text{op}$.
3. Use multipers to compute the persistence diagram of the bifiltration.

