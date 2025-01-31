import numpy as np
import multipers as mp

st = mp.SimplexTreeMulti(num_parameters=2, kcritical=True, dtype=np.float64)

vertex_set = np.array([[0, 0, 1, 1]], dtype=int)
filtrations = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 3.0]], dtype=float)

st.insert_batch(vertex_set, filtrations)
