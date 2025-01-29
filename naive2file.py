from multipers.data import noisy_annulus, three_annulus
import gudhi as gd
import multipers as mp
import multipers.plots as mpp
import multipers.slicer as mps
import numpy as np
import matplotlib.pyplot as plt
from miniball import Miniball
import math
from time import time
# KDTree for nearest neighbor search
from scipy.spatial import KDTree
from tqdm import tqdm

rivet_data = []
rivet_data.append("--datatype bifiltration\n")
rivet_data.append("--xlabel r\n")
rivet_data.append("--ylabel k\n")
rivet_data.append("# Data\n")

# TODO

# Put things in functions
# Stick to multipers, but allow for exporting bifiltration for use with RIVET

# The main bottleneck is inserting simplices into a SimplexTreeMulti object. Are there any methods to insert multiple simplices at once?
# Or maybe insert simplices first, and then assign all critical values at once for each simplex?


X = three_annulus(1000, 1000)

diameter = Miniball(X).squared_radius() ** 0.5 / 2

t0 = time()
delaunay_complex = gd.AlphaComplex(points=X, precision="safe").create_simplex_tree(default_filtration_value=True)
t1 = time()
print(f"Constructing Delaunay complex took {t1 - t0:.2f} s")
print(f"Number of simplices in the Delaunay complex: {delaunay_complex.num_simplices()}")

t0 = time()
kd_tree = KDTree(X)
t1 = time()
print(f"Constructing KDTree took {t1 - t0:.2f} s")

k_max = 100 
beta = 1.0

ks = np.linspace(1, k_max, k_max, dtype=int)

t_miniball = 0
t_knn = 0
t_critical_radii = 0
t_insert = 0

knn_distances = kd_tree.query(X, k=k_max)[0]

for simplex, _ in tqdm(delaunay_complex.get_simplices()):
    t = time()
    r_mb = math.sqrt(Miniball(X[simplex]).squared_radius()) # Minimal bounding sphere radius for the simplex
    t_miniball += time() - t

    t = time()
    max_knn_distances = np.max(knn_distances[simplex], axis=0)
    t_knn += time() - t
    
    t = time()
    critical_radii = np.maximum(r_mb, beta * max_knn_distances)
    t_critical_radii += time() - t
    
    t = time()
    rivet_data.append(" ".join(map(str, simplex)) + " ; " + " ".join(f"{r:.5f} {-k}" for r, k in zip(critical_radii, ks)) + "\n")
    t_insert += time() - t

print(f"Miniball computation took {t_miniball:.2f} s")
print(f"KNN computation took {t_knn:.2f} s")
print(f"Critical radii computation took {t_critical_radii:.2f} s")
print(f"Insertion took {t_insert:.2f} s")

rivet_data = "".join(rivet_data)
with open("rivet_data.txt", "w") as f:
    f.write(rivet_data)

plt.scatter(*X.T)
plt.axis('equal')
#plt.show(block=False)
#pers = mp.module_approximation(st, box=[[0, 0],[diameter, -1]], verbose=True)
#pers.plot(alpha=0.9)
plt.show()
