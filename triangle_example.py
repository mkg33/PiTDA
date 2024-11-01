from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import gtda.mapper as mp
import numpy as np
from sklearn.cluster import KMeans
import gtda.diagrams.distance
from sklearn.decomposition import PCA
import igraph as ig


counter_1 = 1
counter_2 = 1
connected_components_counts = []
while  counter_1 <= 1000:
    x = np.random.uniform(0,100)
    y1 = np.random.uniform(0,10)
    y2 = np.random.uniform(0,10)
    y3 = np.random.uniform(300,1450)

    data_y = [y1, y2, y3]
    data_x = [x, x, x]
    #plt.figure()
    #plt.scatter(data_x, data_y, color='blue', marker='o')
    #plt.show()
    data = np.vstack((data_x, data_y)).T

    #filter_func = np.linalg.norm # Define filter function
    filter_func = pca = PCA(n_components=1)
    cover = mp.CubicalCover(n_intervals=3, overlap_frac=0.99) # Define cover
    clusterer = DBSCAN() # Define clusterer

    # Initialise pipeline
    pipe = mp.make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover,
        clusterer=clusterer
    )
    #plt.figure()
    #fig = mp.plot_static_mapper_graph(pipe, data, color_data=data[:,1])
    #fig.show()
    graph = pipe.fit_transform(data)
    igraph_graph = graph
    num_connected_components = len(igraph_graph.connected_components())
    connected_components_counts.append(num_connected_components)
    if num_connected_components == 1:
        print(counter_2)
        print(data)
        plt.figure()
        fig = mp.plot_static_mapper_graph(pipe, data, color_data=data[:,1])
        fig.show()
        counter_2 += 1

    counter_1 += 1

print(counter_1)
print(counter_2)
