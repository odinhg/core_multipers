import numpy as np
import math
from scipy.spatial import KDTree
from gudhi import AlphaComplex
from multipers import SimplexTreeMulti


class DelaunayCore:
    """
    This is a class for computing the Delaunay Core Bifiltration of a point cloud from the paper "Core Bifiltration" https://arxiv.org/abs/2405.01214.
    """

    def __init__(
        self,
        points: np.ndarray,
        beta: float = 1.0,
        k_max: int | None = None,
        k_step: int = 1,
        precision: str = "safe",
        verbose: bool = False,
    ) -> None:
        """
        Constructor for the DelaunayCore class.

        Parameters:
            points (np.ndarray): The point cloud as a 2D array of shape (n, d) where n is the number of points and d is the dimension of the points.
            beta (float): The beta parameter for the Delaunay Core Bifiltration (default 1.0).
            k_max (int | None): The maximum number of nearest neighbors to consider (default None). If None, k_max is set to the number of points in the point cloud.
            k_step (int): The step size for the number of nearest neighbors (default 1).
            precision (str): The precision of the computation of the AlphaComplex, one of ['safe', 'exact', 'fast'] (default 'safe'). See the GUDHI documentation for more information.
            verbose (bool): Whether to print progress messages (default False).
        """
        if k_max is None:
            k_max = len(points)

        assert len(points) > 0, f"The point cloud must contain at least one point."
        assert points.ndim == 2, (
            f"The point cloud must be a 2D array, got {points.ndim}D."
        )
        assert beta > 0, f"The parameter beta must be positive, got {beta}."
        assert k_max > 0, f"The parameter k_max must be positive, got {k_max}."
        assert k_step > 0, f"The parameter k_step must be positive, got {k_step}."
        assert precision in ["safe", "exact", "fast"], (
            f"The parameter precision must be one of ['safe', 'exact', 'fast'], got {precision}."
        )

        self.points = points
        self.beta = beta
        self.k_max = k_max
        self.k_step = k_step
        self.precision = precision
        self.ks = np.arange(1, self.k_max + 1, self.k_step)
        self.verbose = verbose

    def create_simplex_tree(
        self, max_alpha_square: float = float("inf")
    ) -> SimplexTreeMulti:
        """
        Compute and return the Delaunay Core Bifiltration of the point cloud.

        Parameters:
            max_alpha_square (float): The maximum squared alpha value to consider when createing the alpha complex (default inf). See the GUDHI documentation for more information.

        Returns:
            SimplexTreeMulti: A multi-critical simplex tree storing the Delaunay Core Bifiltration.
        """
        if self.verbose:
            print(
                f"Computing the Delaunay Core Bifiltration of {len(self.points)} points in dimension {self.points.shape[1]} with parameters:"
            )
            print(f"\tbeta = {self.beta}")
            print(f"\tk_max = {self.k_max}")
            print(f"\tk_step = {self.k_step} (total of {len(self.ks)} k-values)")

        if self.verbose:
            print("Building the alpha complex...")
        alpha_complex = AlphaComplex(
            points=self.points, precision=self.precision
        ).create_simplex_tree(max_alpha_square=max_alpha_square)

        if self.verbose:
            print("Computing the k-nearest neighbors...")
        knn_distances = KDTree(self.points).query(self.points, k=self.ks)[0]

        # Group simplices by dimension
        simplices_in_dimension = {
            dim: [] for dim in range(alpha_complex.dimension() + 1)
        }
        for simplex, alpha_sq in alpha_complex.get_simplices():
            alpha = math.sqrt(alpha_sq)
            simplices_in_dimension[len(simplex) - 1].append((simplex, alpha))

        def compute_critical_radii(simplex: list[int], alpha: float) -> np.ndarray:
            """
            Given a simplex, compute the critical radii for each k in ks. Returns a 1D array of critical radii for the simplex.
            """
            max_knn_distances = np.max(knn_distances[simplex], axis=0)
            return np.maximum(alpha, self.beta * max_knn_distances)

        simplex_tree_multi = SimplexTreeMulti(
            num_parameters=2, kcritical=True, dtype=np.float64
        )

        for dim, simplices in simplices_in_dimension.items():
            if self.verbose:
                print(
                    f"Inserting {len(simplices)} simplices of dimension {dim} ({len(simplices) * len(self.ks)} birth values)..."
                )
            num_simplices = len(simplices)
            vertex_array = np.empty((dim + 1, num_simplices), dtype=int)
            filtrations = np.empty((num_simplices, len(self.ks), 2), dtype=np.float64)
            filtrations[:, :, 1] = (-1) * self.ks  # Opposite ordering

            for i, (simplex, alpha) in enumerate(simplices):
                vertex_array[:, i] = simplex
                filtrations[i, :, 0] = compute_critical_radii(simplex, alpha)

            simplex_tree_multi.insert_batch(vertex_array, filtrations)

        if self.verbose:
            print("Done computing the Delaunay Core Bifiltration.")

        return simplex_tree_multi

    def get_point(self, vertex: int) -> np.ndarray:
        """
        Get the point at the given vertex index.

        Parameters:
            vertex (int): The vertex index.

        Returns:
            np.ndarray: The point at the given vertex index.
        """
        return self.points[vertex]
