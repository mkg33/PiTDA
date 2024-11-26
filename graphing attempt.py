from sklearn import datasets
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






counter_1 = 1
counter_3 = 1

connected_components_counts = []
mesh_size = 0.2
short_distance = []
long_distance = []
overlap_matrix = []
distance_matrix = []

while counter_3 < 5:
    x = np.random.uniform(0,100)
    y1 = np.random.uniform(0,10)
    y2 = np.random.uniform(0,10)
    y3 = np.random.uniform(300,1450)
    data_y = [y1, y2, y3]
    data_x = [x, x, x]
    data = np.vstack((data_x, data_y)).T
    counter_2 = 0.9999


    while counter_1 <10:


        filter_func = pca = PCA(n_components=1)
        cover = mp.CubicalCover(n_intervals=3, overlap_frac=counter_2) # Define cover
        clusterer = DBSCAN() # Define clusterer

        # Initialise pipeline
        pipe = mp.make_mapper_pipeline(
            filter_func=filter_func,
            cover=cover,
            clusterer=clusterer
        )
        plt.figure()
        fig = mp.plot_static_mapper_graph(pipe_2, data, color_data=data[:,1])
        fig.show()
        graph = pipe.fit_transform(data)
        igraph_graph = graph
        num_connected_components = len(igraph_graph.connected_components())
        
        if num_connected_components == 1:
            counter_2 -= 0.0001
            counter_1 += 1
            print(counter_2)
        else:
            counter_1 += 1
            print(counter_2)
            print(counter_3)
        print(f"Outer Iteration {counter_3}, Inner Iteration {counter_1}")
        print(f"  Data: {data}")
        print(f"  Num Connected Components: {num_connected_components}")
        print(f"  Counter_2: {counter_2}")
        print(f"  Counter_1: {counter_1}")
        

        

    if counter_2 < 0.9999:
        counter_overlap = counter_2 + 0.0001
    else:
       coupnter_overlap = counter_2

    cover_2 = mp.CubicalCover(n_intervals=3, overlap_frac=counter_overlap) # Define cover
    pipe_2 = mp.make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover_2,
        clusterer=clusterer
    )

    
    

    #plt.figure()
    #fig = mp.plot_static_mapper_graph(pipe_2, data, color_data=data[:,1])
    #fig.show()

    distance_1 = abs(y2-y1)
    distance_2 = abs(y3-y2)
    short_distance.append(distance_1)
    long_distance.append(distance_2)
    overlap_matrix.append(counter_overlap)
    distance_matrix.append([distance_1, distance_2])

    counter_3 += 1





regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2)) #from svr documentation
regr.fit(distance_matrix, overlap_matrix)

x_min, x_max = 0, 10 #adapted from plotly 
y_min, y_max = 290, 1440
X = np.arange(x_min, x_max, mesh_size)
Y = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(X, Y)

prediction = regr.predict(np.c_[xx.ravel(), yy.ravel()])
prediction = prediction.reshape(xx.shape)

plt.figure()
ax = plt.axes(projection= '3d')
ax.scatter3D(short_distance, long_distance, overlap_matrix, color = 'red')
ax.plot_surface(xx, yy, prediction)
ax.set_zlim(0.996, 1)
plt.show()

print(regr.get_params)
#print(print(regr.predict([[9, 1500]])))
print(overlap_matrix)