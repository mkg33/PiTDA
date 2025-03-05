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






counter_1 = 1
counter_3 = 1

connected_components_counts = []
mesh_size = 0.2
short_distance = []
long_distance = []
overlap_matrix = []
distance_matrix = []

while counter_3 < 500:
    x = np.random.uniform(0,100)
    y1 = np.random.uniform(0,10)
    y2 = np.random.uniform(0,10)
    y3 = np.random.uniform(300,1450)
    data_y = [y1, y2, y3]
    data_x = [x, x, x]
    data = np.vstack((data_x, data_y)).T
    counter_2 = 0.9999
    counter_1 = 1


    while counter_1 <100:


        filter_func = pca = PCA(n_components=1)
        cover = mp.CubicalCover(n_intervals=3, overlap_frac=counter_2) # Define cover
        clusterer = DBSCAN() # Define clusterer

        # Initialise pipeline
        pipe = mp.make_mapper_pipeline(
            filter_func=filter_func,
            cover=cover,
            clusterer=clusterer
        )
        #plt.figure()
       #fig = mp.plot_static_mapper_graph(pipe_2, data, color_data=data[:,1])
        #fig.show()
        graph = pipe.fit_transform(data)
        igraph_graph = graph
        num_connected_components = len(igraph_graph.connected_components())
        
        if num_connected_components == 1:
            counter_2 -= 0.0001
            round(counter_2,4)
            counter_1 += 1
            print(counter_2)
        elif num_connected_components < 1 and counter_2 < 0.9999:
            counter_1 += 1
            counter_2 += 0.0001
            round(counter_2, 4)
            print(counter_2)
            print(counter_3)
        else:
            counter_2 += 0
            counter_1 += 1
        print(f"Outer Iteration {counter_3}, Inner Iteration {counter_1}")
        print(f"  Data: {data}")
        print(f"  Num Connected Components: {num_connected_components}")
        print(f"  Counter_2: {counter_2}")
        print(f"  Counter_1: {counter_1}")
        

        

    #if counter_2 < 0.9999:
        #counter_overlap = counter_2 + 0.0001
    #else:
       #coupnter_overlap = counter_2
    filter_func = pca = PCA(n_components=1)
    cover = mp.CubicalCover(n_intervals=3, overlap_frac=counter_2) # Define cover
    clusterer = DBSCAN() # Define clusterer

        # Initialise pipeline
    pipe = mp.make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover,
        clusterer=clusterer
    )
        #plt.figure()
       #fig = mp.plot_static_mapper_graph(pipe_2, data, color_data=data[:,1])
        #fig.show()
    graph = pipe.fit_transform(data)
    igraph_graph = graph
    num_connected_components = len(igraph_graph.connected_components())
    cover_2 = mp.CubicalCover(n_intervals=3, overlap_frac=counter_2) # Define cover
    pipe_2 = mp.make_mapper_pipeline(
        filter_func=filter_func,
        cover=cover_2,
        clusterer=clusterer
    )
    if num_connected_components==1 and counter_2 < 0.9999:
        counter_2 += 0.0001
        round(counter_2, 4)
    else:
        counter_2 += 0

    print(counter_2)


    
    

    #plt.figure()
    #fig = mp.plot_static_mapper_graph(pipe_2, data, color_data=data[:,1])
    #fig.show()

    distance_1 = abs(y2-y1)
    distance_2 = abs(y3-y2)
    short_distance.append(distance_1)
    print(short_distance)
    long_distance.append(distance_2)
    print(long_distance)
    overlap_matrix.append(counter_2)
    print(overlap_matrix)
    distance_matrix.append([distance_1, distance_2])
    print(distance_matrix)

    counter_3 += 1



scaler = StandardScaler()
X_scaled = scaler.fit_transform(distance_matrix)

model = LinearRegression()
model.fit(X_scaled, overlap_matrix)

# Print coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

new_data = [[2, 400]]
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print("Prediction for new data:", prediction)

x_min, x_max = 0, 10 #adapted from plotly 
y_min, y_max = 290, 1440
X = np.arange(x_min, x_max, mesh_size)
Y = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(X, Y)

X_plot = np.c_[xx.ravel(), yy.ravel()]
y_pred = model.predict(X_plot).reshape(xx.shape)

plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], overlap_matrix, color='red', label='Data Points')

ax.plot_surface(xx, yy, y_pred, alpha=0.5, cmap='viridis', edgecolor='k', label='Regression Plane')
plt.show()





regr = SVR(C=10, epsilon=0.01) #from svr documentation
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
ax.set_zlim(0.99, 1)
plt.show()


print(regr.predict([[9, 400]]))
print(regr.predict([[5, 1400]]))
print(overlap_matrix)