import numpy as np
import multipers as mp

st = mp.SimplexTreeMulti(num_parameters=2, kcritical=True, dtype=np.float64)

# Two 1-simplices -> shape (2, 2)
vertex_set = np.array([[0, 1],
                       [1, 2]], dtype=int)

# Three 2-dimensional filtration values per simplex -> shape (2, 3, 2)
filtrations = np.array([[[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
                        [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]]], dtype=float)

st.insert_batch(vertex_set, filtrations)
