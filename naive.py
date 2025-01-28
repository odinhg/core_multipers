from multipers.data import noisy_annulus
import gudhi as gd
import multipers as mp
import numpy as np
import matplotlib.pyplot as plt

X = noisy_annulus(n1=1000, n2=1000) # 1000 points on the annulus, 500 points on the square.
plt.scatter(X[:,0], X[:,1], s=5, c='k');plt.gca().set_aspect(1);
plt.show()

# Gudhi SimplexTree
st = gd.RipsComplex(points=X, sparse=.2).create_simplex_tree()
# SimplexTreeMulti, with 2 parameters
st = mp.SimplexTreeMulti(st, num_parameters=2)

from multipers.ml.convolutions import KDE as KernelDensity
codensity = - KernelDensity(bandwidth=0.2).fit(X).score_samples(X) 
# parameter = 1 is the second parameter as python starts with 0.
st.fill_lowerstar(codensity, parameter=1) # This fills the second parameter with the co-log-density

print("Before edge collapses :", st.num_simplices)
st.collapse_edges(-2) # This should take less than 20s. -1 is for maximal "simple" collapses, -2 is for maximal "harder" collapses.
print("After edge collapses :", st.num_simplices)
st.expansion(2) # Be careful, we have to expand the dimension to 2 before computing degree 1 homology.
print("After expansion :", st.num_simplices)

mp.module_approximation(st).plot(degree=1) # Should take less than ~10 seconds depending on the seed

plt.show()
