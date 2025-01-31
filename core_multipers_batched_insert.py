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

def delaunay_core(X: np.ndarray, k_max: int | None = None, beta: float = 1.0):
    """
    Build a multi-critical simplex tree storing the Delaunay Core Bifiltration of a point cloud X with k ranging from 1 to k_max and the beta > 0 parameter.
    """

    if k_max is None:
        k_max = X.shape[0]

    print(f"Computing the Delaunay Core Bifiltration of {len(X)} points in dimension {X.shape[1]} with k_max={k_max} and beta={beta}.")

    st = mp.SimplexTreeMulti(num_parameters=2, kcritical=True, dtype=np.float64)
    delaunay_complex = build_delanauy_complex(X)
    knn_distances = compute_knn_distances(X, k_max)
    
    # Group simplices by dimension
    simplices_in_dimension = {dim: [] for dim in range(delaunay_complex.dimension() + 1)}
    for simplex, _ in tqdm(delaunay_complex.get_simplices()):
        simplices_in_dimension[len(simplex) - 1].append(simplex)

    ks = (-1) * np.arange(1, k_max + 1) # Opposite ordering

    # Compute birth sets and insert into the simplex tree
    for dim, simplices in simplices_in_dimension.items():
        num = len(simplices)

        print(f"Dimension {dim}: {num} simplices -> {num * k_max} minimal birth values.")
        vertex_array = np.empty((dim + 1, num), dtype=int)
        filtrations = np.empty((num, k_max, 2), dtype=np.float64)

        for i, simplex in enumerate(tqdm(simplices, total=num)):
            vertex_array[:, i] = simplex
            critical_radii = compute_critical_radii(simplex, X, knn_distances, beta)
            filtrations[i] = np.stack([critical_radii, ks], axis=-1)

        st.insert_batch(vertex_array, filtrations)

    return st

if __name__ == "__main__":
    from multipers.data import noisy_annulus, three_annulus
    import matplotlib.pyplot as plt

    X = three_annulus(1000, 500)

    #plt.scatter(*X.T, s=5, c="black")
    #plt.axis("equal")
    #plt.show()

    # Parameters for computing the Delaunay Core Bifiltration
    k_max = 200 
    beta = 0.5

    # Construct the Delaunay Core Bifiltration
    st = delaunay_core(X, k_max, beta)

    # Compute persistence
    pers = mp.module_approximation(st, verbose=True)
    pers.plot(degree=1, alpha=0.9)
    plt.show()
