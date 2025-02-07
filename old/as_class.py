import numpy as np
import math
from scipy.spatial import KDTree
from gudhi import AlphaComplex
from multipers import SimplexTreeMulti

class DelaunayCore:
    def __init__(self, points: np.ndarray, beta:float=1.0, k_max:int|None=None, k_step:int=1, precision:str = "safe", verbose: bool=False) -> None:

        if k_max is None:
            k_max = len(points)

        assert beta > 0, f"The parameter beta must be positive, got {beta}."
        assert k_max > 0, f"The parameter k_max must be positive, got {k_max}."
        assert k_step > 0, f"The parameter k_step must be positive, got {k_step}."
        assert precision in ["safe", "exact", "fast"], f"The parameter precision must be one of ['safe', 'exact', 'fast'], got {precision}."

        self.points = points
        self.beta = beta
        self.k_max = k_max
        self.k_step = k_step
        self.precision = precision
        self.ks = np.arange(1, self.k_max + 1, self.k_step)
        self.verbose = verbose

    def create_simplex_tree(self, max_alpha_square:float = float("inf")) -> SimplexTreeMulti:
        if self.verbose:
            print(f"Computing the Delaunay Core Bifiltration of {len(self.points)} points in dimension {self.points.shape[1]} with parameters:")
            print(f"\tbeta = {self.beta}")
            print(f"\tk_max = {self.k_max}")
            print(f"\tk_step = {self.k_step} (total of {len(self.ks)} k-values)")
        
        if self.verbose:
            print("Building the alpha complex...")
        alpha_complex = AlphaComplex(points=self.points, precision=self.precision).create_simplex_tree(max_alpha_square=max_alpha_square)

        if self.verbose:
            print("Computing the k-nearest neighbors...")
        knn_distances = KDTree(self.points).query(self.points, k=self.ks)[0]

        # Group simplices by dimension
        simplices_in_dimension = {dim: [] for dim in range(alpha_complex.dimension() + 1)}
        for simplex, alpha_sq in alpha_complex.get_simplices():
            alpha = math.sqrt(alpha_sq)
            simplices_in_dimension[len(simplex) - 1].append((simplex, alpha))

        def compute_critical_radii(simplex: list[int], alpha: float) -> np.ndarray: 
            """
            Given a simplex, compute the critical radii for each k in ks. Returns a 1D array of critical radii for the simplex.
            """
            max_knn_distances = np.max(knn_distances[simplex], axis=0)
            return np.maximum(alpha, self.beta * max_knn_distances)

        simplex_tree_multi = SimplexTreeMulti(num_parameters=2, kcritical=True, dtype=np.float64)

        for dim, simplices in simplices_in_dimension.items():
            if self.verbose:
                print(f"Inserting {len(simplices)} simplices of dimension {dim} ({len(simplices) * len(self.ks)} birth values)...")
            num_simplices = len(simplices)
            vertex_array = np.empty((dim + 1, num_simplices), dtype=int)
            filtrations = np.empty((num_simplices, len(self.ks), 2), dtype=np.float64)
            filtrations[:, :, 1] = (-1) * self.ks # Opposite ordering

            for i, (simplex, alpha) in enumerate(simplices):
                vertex_array[:, i] = simplex
                filtrations[i, :, 0] = compute_critical_radii(simplex, alpha)

            simplex_tree_multi.insert_batch(vertex_array, filtrations)

        if self.verbose:
            print("done.")

        return simplex_tree_multi

    def get_point(self, vertex: int) -> np.ndarray:
        return self.points[vertex]

def get_bounding_box(pers, k_max):
    r_deaths_all = []
    for summand in pers:
        death_list = summand.get_death_list()
        r_deaths = [death[0] for death in death_list if death[0] != np.inf]
        r_deaths_all.extend(r_deaths)
    r_max = max(r_deaths_all)
    return [[0, -k_max], [r_max, 1]]

if __name__ == "__main__":
    import multipers as mp
    from multipers.data import noisy_annulus, three_annulus
    import matplotlib.pyplot as plt

    # Bug in three_annulus: one point missing (division by 3)
    np.random.seed(0)
    X = three_annulus(3000, 1000)

    #plt.scatter(*X.T, s=5, c="black")
    #plt.axis("equal")
    #plt.show()

    # Parameters for computing the Delaunay Core Bifiltration
    k_max = 500 
    k_step = 10 
    beta = 0.5 
    
    dc = DelaunayCore(points=X, beta=beta, k_max=k_max, k_step=k_step, verbose=True)
    st = dc.create_simplex_tree()

    # Compute persistence
    pers = mp.module_approximation(st, verbose=True)
    box = get_bounding_box(pers, k_max)
    pers.plot(degree=1, alpha=0.9, dpi=400, xlabel="r", ylabel="k", box=box, min_persistence=10e-3)
    plt.show()

