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
interval_matrix = []
counter_1 = 0
while counter_1 < 1000:
    l_projection = 0
    r_projection = 1
    data_y = []
    counter_2 = 0
    interval = np.random.uniform(1,20)
    data_x = np.random.uniform(0,100,100)
    data_y = np.random.normal(0,interval,100)
    data = np.vstack((data_x, data_y)).T
    
    while r_projection - l_projection > 1e-6:
        m_projection = (l_projection + r_projection)/2

        filter_func =  mp.Projection(columns=[1])
        cover = mp.CubicalCover(n_intervals=15, overlap_frac=m_projection) # Define cover
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

    counter_1 += 1
    print(counter_1)
    overlap_matrix_projection.append(l_projection)
    interval_matrix.append(interval)

print(overlap_matrix_projection)
plt.scatter(interval_matrix, overlap_matrix_projection)
plt.title("Normally Distributed Datasets vs Minimum Overlap, cover intervals = 15")
plt.show()




saved_file = pd.DataFrame({"interval": interval_matrix, 
                           "overlap_fraction": overlap_matrix_projection})
saved_file.to_csv("normally_distributed_data.csv", index = False)