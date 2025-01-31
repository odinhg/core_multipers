import math
import gudhi as gd
import numpy as np
from tqdm import tqdm
import multipers as mp
from miniball import Miniball
from scipy.spatial import KDTree


def build_delanauy_complex(X, precision="safe"):
    """
    Build the Delaunay complex of a point cloud X. Returns a SimplexTree object with NaN filtration values.
    """
    return gd.AlphaComplex(points=X, precision=precision).create_simplex_tree(default_filtration_value=True)

def compute_knn_distances(X, k_max):
    """
    Compute the distances to the k_max nearest neighbors for each point in X.
    """
    return KDTree(X).query(X, k=k_max)[0]

def compute_critical_radii(simplex, X, knn_distances, beta):
    """
    Compute the critical radii for a simplex in the Delaunay complex. For each k<=k_max, the critical radius is the maximum of the minimal bounding sphere radius and beta times the maximum distance to the k nearest neighbors in the simplex.
    """
    r_mb = math.sqrt(Miniball(X[simplex]).squared_radius())
    max_knn_distances = np.max(knn_distances[simplex], axis=0)
    return np.maximum(r_mb, beta * max_knn_distances)

def compute_filtration_values(X, k_max, beta=1.0):
    st = mp.SimplexTreeMulti(num_parameters=2, kcritical=True, dtype=np.float64)
    delaunay_complex = build_delanauy_complex(X)
    knn_distances = compute_knn_distances(X, k_max)

    filtration_values = {}

    print("Computing critical filtration values for simplices in the Delaunay complex...")

    for simplex, _ in tqdm(delaunay_complex.get_simplices(), total=delaunay_complex.num_simplices()):
        critical_radii = compute_critical_radii(simplex, X, knn_distances, beta)
        filtration_values[tuple(simplex)] = critical_radii

    return filtration_values

def build_bifiltration(filtration_values):
    st = mp.SimplexTreeMulti(num_parameters=2, kcritical=True, dtype=np.float64)
    
    print("Building the multiparameter simplex tree...")

    for simplex, critical_radii in tqdm(filtration_values.items()):
        # Bottleneck: inserting simplices one by one (any other way to do this in multipers?)
        for k, r in enumerate(critical_radii, 1):
            st.insert(simplex, (r, -k))

    return st


if __name__ == "__main__":
    from multipers.data import noisy_annulus, three_annulus
    import matplotlib.pyplot as plt

    X = three_annulus(1000, 1000)

    plt.scatter(*X.T, s=5, c="black")
    plt.axis("equal")
    plt.show()

    # Parameters for computing the Delaunay Core Bifiltration
    k_max = 150 
    beta = 1.0

    filtration_values = compute_filtration_values(X, k_max, beta)
    st = build_bifiltration(filtration_values)

    diameter = Miniball(X).squared_radius() ** 0.5
    pers = mp.module_approximation(st, box=[[0, 0],[diameter / 2, -k_max]], verbose=True)
    pers.plot(degree=1, alpha=0.9)

    plt.show()

