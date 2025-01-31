import numpy as np
import multipers as mp

st = mp.SimplexTreeMulti(num_parameters=2, kcritical=True)

vertices = [[0,1]] # shape (dim+1, num_vertices)
vertices_filtrations = np.array([[[-1, -2]],[[-2, -1]]])
st.insert_batch(vertices,vertices_filtrations)
print(f"vertices shape: {np.array(vertices).shape}")
print(f"vertices_filtrations shape: {vertices_filtrations.shape}")

edges = np.array([[0, 1],[1, 2], [2,0]]).T
edges_filtrations = np.array([
    [[1,0],[0,1], [np.inf,np.inf]],
    [[1,0],[0,1], [np.inf,np.inf]],
    [[1,0],[0,1], [-1,3]],
])
st.insert_batch(edges, edges_filtrations)
print(f"edges shape: {edges.shape}")
print(f"edges_filtrations shape: {edges_filtrations.shape}")

triangle = np.array([[0,1,2]]).T
triangle_filration = [[[2,2]]]
st.insert_batch(triangle, triangle_filration)
print(f"triangle shape: {triangle.shape}")
print(f"triangle_filration shape: {np.array(triangle_filration).shape}")

for s, f in st:
    print(s, f)
