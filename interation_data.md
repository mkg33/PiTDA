# Iteration Data

For iteration data:
- Rows = Iterations
- Columns = Data
  - x is fixed point to make the data 2-dimensional (should fix to 0)
  - y_1, y_2, y_3 randomly generated (np.random)
  - y_1, y_2 are generated between 0 and 1450
  - y_3 is generated between 1000 to 1450
    This is to ensure one point is far enough away to generate correct
    connected components. This can change however we want to guarentee
    at least 2 connected components for some mapper parameters.

Mapper Parameters:
- Coordinate projection to y-axis
- Cubical Cover (n_intervals =3, vary overlap_frac)
- DBSCAN (multiple parameter choices) Should vary eps but fix num_points_in_cluster=1.


Future Ideas: Use other clusterers from the clustering_algorithms.py file.