import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import gtda.mapper as mp
import numpy as np
from sklearn.cluster import KMeans
import gtda.diagrams.distance
from sklearn.decomposition import PCA
import igraph as ig
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import pandas as pd


overlap_matrix_projection = []
distance_matrix = []
counter_1 = 0
while counter_1 < 1000:
    l_projection = 0
    r_projection = 1
    counter_2 = 0
    data_x = np.random.uniform(1,100,20)
    data_y = np.random.uniform(1,100,20)
    mean_dist = np.mean(data_y)


    data = np.vstack((data_x, data_y)).T

    
    while r_projection - l_projection > 1e-6:
        m_projection = (l_projection + r_projection)/2

        filter_func =  mp.Projection(columns=[1])
        cover = mp.CubicalCover(n_intervals=10, overlap_frac=m_projection) # Define cover
        clusterer = DBSCAN() # Define clusterer

        # Initialise pipeline
        pipe_projection = mp.make_mapper_pipeline(
            filter_func=filter_func,
            cover=cover,
            clusterer=clusterer
        )

        graph = pipe_projection.fit_transform(data)
        igraph_graph = graph
        num_connected_components = len(igraph_graph.connected_components())
        if num_connected_components ==1:
            r_projection = m_projection
        else:
            l_projection = m_projection
    
    overlap_matrix_projection.append(l_projection)
    distance_matrix.append(mean_dist)
    counter_1 += 1
    print(counter_1)

print(overlap_matrix_projection)
print(distance_matrix)
plt.scatter(distance_matrix, overlap_matrix_projection)
plt.title('Random Points, mean distance vs minimum overlap, number of clusters = 10')
plt.show()

saved_file = pd.DataFrame({"interval": distance_matrix, 
                           "overlap_fraction": overlap_matrix_projection})
saved_file.to_csv("random_data.csv", index = False)