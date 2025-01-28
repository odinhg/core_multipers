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

#n = 200 
#theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
#X = np.column_stack((np.cos(theta), np.sin(theta))) + np.random.normal(0, 0.05, (n, 2))
#X = np.concatenate((X, np.random.uniform(-1, 1, (n, 2))))

X = three_annulus(500, 50)

diameter = Miniball(X).squared_radius() ** 0.5 / 2

t0 = time()
delaunay_complex = gd.AlphaComplex(points=X).create_simplex_tree(default_filtration_value=True)
t1 = time()
print(f"Constructing Delaunay complex took {t1 - t0:.2f} s")
print(f"Number of simplices in the Delaunay complex: {delaunay_complex.num_simplices()}")

t0 = time()
kd_tree = KDTree(X)
t1 = time()
print(f"Constructing KDTree took {t1 - t0:.2f} s")

t0 = time()
st = mp.SimplexTreeMulti(num_parameters=2, kcritical=True, dtype=np.float32)

k_max = 100
k_step = 1
ks = list(range(1, k_max + 1, k_step))
beta = 0.5

t_miniball = 0
t_knn = 0
t_max_knn = 0
t_critical_radii = 0
t_insert = 0

for simplex, _ in tqdm(delaunay_complex.get_simplices()):
    t2 = time()
    r_mb = math.sqrt(Miniball(X[simplex]).squared_radius()) # Minimal bounding sphere radius for the simplex
    t_miniball += time() - t2
    t2 = time()
    knn_distances, _ = kd_tree.query(X[simplex], k=ks) # Distances to k_max nearest neighbors for each point in the simplex, shape (k_max, simplex_size) 
    t_knn += time() - t2
    t2 = time()
    max_knn_distances = np.max(knn_distances, axis=0) # Maximum distance to k_max nearest neighbors for each k, shape (k_max,)
    t_max_knn += time() - t2
    t2 = time()
    critical_radii = np.maximum(r_mb, beta * max_knn_distances)
    t_critical_radii += time() - t2
    t2 = time()
    births = {r: -k/k_max for k, r in zip(ks, critical_radii)}.items()
    for birth in births:
        st.insert(simplex, np.array(birth))
    t_insert += time() - t2
t1 = time()
print(f"Computing birth sets took {t1 - t0:.2f} s\n")
print(f"Miniball computation took {t_miniball:.2f} s")
print(f"KNN computation took {t_knn:.2f} s")
print(f"Max KNN computation took {t_max_knn:.2f} s")
print(f"Critical radii computation took {t_critical_radii:.2f} s")
print(f"Insertion took {t_insert:.2f} s")

plt.scatter(*X.T)
plt.axis('equal')
plt.show(block=False)
pers = mp.module_approximation(st, box=[[0, 0],[diameter, -1]], verbose=True)
pers.plot(alpha=0.9)
plt.show()
