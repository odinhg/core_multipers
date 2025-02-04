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
    return gd.AlphaComplex(points=X, precision=precision).create_simplex_tree()

def compute_knn_distances(X, k_max):
    """
    Compute the distances to the k_max nearest neighbors for each point in X.
    """
    return KDTree(X).query(X, k=k_max)[0]

def compute_critical_radii(simplex, alpha, knn_distances, beta):
    """
    Compute the critical radii for a simplex in the Delaunay complex. For each k<=k_max, the critical radius is the maximum of the minimal bounding sphere radius and beta times the maximum distance to the k nearest neighbors in the simplex.
    """
    max_knn_distances = np.max(knn_distances[simplex], axis=0)
    return np.maximum(alpha, beta * max_knn_distances)

def delaunay_core(X: np.ndarray, k_max: int | None = None, beta: float = 1.0):
    """
    Build a multi-critical simplex tree storing the Delaunay Core Bifiltration of a point cloud X with k ranging from 1 to k_max and the beta > 0 parameter.
    """

    if k_max > X.shape[0]:
        raise ValueError("k_max must be less than the number of points in the point cloud.")

    if k_max is None:
        k_max = X.shape[0]

    print(f"Computing the Delaunay Core Bifiltration of {len(X)} points in dimension {X.shape[1]} with k_max={k_max} and beta={beta}.")

    st = mp.SimplexTreeMulti(num_parameters=2, kcritical=True, dtype=np.float64)
    delaunay_complex = build_delanauy_complex(X)
    knn_distances = compute_knn_distances(X, k_max)
    
    # Group simplices by dimension
    simplices_in_dimension = {dim: [] for dim in range(delaunay_complex.dimension() + 1)}
    for simplex, alpha_sq in tqdm(delaunay_complex.get_simplices()):
        alpha = math.sqrt(alpha_sq)
        simplices_in_dimension[len(simplex) - 1].append((simplex, alpha))

    ks = (-1) * np.arange(1, k_max + 1) # Opposite ordering

    # Compute birth sets and insert into the simplex tree
    for dim, simplices in simplices_in_dimension.items():
        num = len(simplices)

        print(f"Dimension {dim}: {num} simplices -> {num * k_max} minimal birth values.")
        vertex_array = np.empty((dim + 1, num), dtype=int)
        filtrations = np.empty((num, k_max, 2), dtype=np.float64)
        filtrations[:, :, 1] = ks

        for i, (simplex, alpha) in enumerate(tqdm(simplices, total=num)):
            vertex_array[:, i] = simplex
            critical_radii = compute_critical_radii(simplex, alpha, knn_distances, beta)
            filtrations[i, :, 0] = critical_radii
        print("Inserting simplices...", end=" ")
        st.insert_batch(vertex_array, filtrations)
        print("Done.")

    return st

def get_bounding_box(pers, k_max):
    r_deaths_all = []
    for summand in pers:
        death_list = summand.get_death_list()
        r_deaths = [death[0] for death in death_list if death[0] != np.inf]
        r_deaths_all.extend(r_deaths)
    r_max = max(r_deaths_all)
    return [[0, -k_max], [r_max, 1]]

if __name__ == "__main__":
    from multipers.data import noisy_annulus, three_annulus
    import matplotlib.pyplot as plt

    # Bug in three_annulus: one point missing (division by 3)
    np.random.seed(0)
    X = three_annulus(1800, 200)

    #plt.scatter(*X.T, s=5, c="black")
    #plt.axis("equal")
    #plt.show()

    # Parameters for computing the Delaunay Core Bifiltration
    k_max = 200 
    beta = 1.0 

    # Construct the Delaunay Core Bifiltration
    st = delaunay_core(X, k_max, beta)

    # Compute persistence
    pers = mp.module_approximation(st, verbose=True)
    box = get_bounding_box(pers, k_max)
    pers.plot(degree=1, alpha=0.9, dpi=400, xlabel="r", ylabel="k", box=box, min_persistence=10e-3)
    plt.show()
