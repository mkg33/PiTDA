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
import csv








short_distance = []
long_distance = []
overlap_matrix_PCA = []
overlap_matrix_projection = []
distance_matrix = []
x_matrix = []
y1_matrix = []
y2_matrix = []
y3_matrix = []
counter_1 = 1
while counter_1 < 1000:
    l_PCA = 0.5
    r_PCA = 1
    l_projection = 0.5
    r_projection = 1
    x = np.random.uniform(0,100)
    y1 = np.random.uniform(0,10)
    y2 = np.random.uniform(0,10)
    y3 = np.random.uniform(300,1450)
    data_y = [y1, y2, y3]
    data_x = [x, x, x]
    
    data = np.vstack((data_x, data_y)).T

    while r_projection - l_projection > 1e-6:
        m_projection = (l_projection + r_projection)/2

        filter_func =  mp.Projection(columns=[1])
        cover = mp.CubicalCover(n_intervals=2, overlap_frac=m_projection) # Define cover
        clusterer = DBSCAN() # Define clusterer
        # Initialise pipeline
        pipe_projection = mp.make_mapper_pipeline(
            filter_func=filter_func,
            cover=cover,
            clusterer=clusterer
        )
        #plt.figure()
        #fig = mp.plot_static_mapper_graph(pipe_2, data, color_data=data[:,1])
        #fig.show()
        graph = pipe_projection.fit_transform(data)
        igraph_graph = graph
        num_connected_components = len(igraph_graph.connected_components())
        if num_connected_components ==1:
            r_projection = m_projection
        else:
            l_projection = m_projection
    
    overlap_matrix_projection.append(l_projection)

    distance_1 = abs(y2-y1)
    distance_2 = abs(y3-y2)
    short_distance.append(distance_1)
    long_distance.append(distance_2)
    distance_matrix.append([distance_1, distance_2])
    x_matrix.append(x)
    y1_matrix.append(y1)
    y2_matrix.append(y2)
    y3_matrix.append(y3)
    counter_1 += 1
    print(counter_1)
    
    

saved_file = pd.DataFrame({"short_distance": short_distance, 
                           "long_distance": long_distance, 
                           "x_value": x_matrix, 
                           "y1_value": y1_matrix, 
                           "y2_value": y2_matrix, 
                           "y3_value": y3_matrix, 
                           "DB_scan_overlap": overlap_matrix_projection})
saved_file.to_csv("tda_experiment_data.csv", index = False)

plt.figure()
ax = plt.axes(projection= '3d')
#ax.scatter3D(short_distance, long_distance, overlap_matrix_PCA, color = 'red')
ax.scatter3D(short_distance, long_distance, overlap_matrix_projection, color = 'blue')
ax.set_xlabel("Distance Between Close Points")
ax.set_ylabel("Distance Between Middle and Third Point")
ax.set_zlabel("Minimum Overlap Fraction")       
plt.show()