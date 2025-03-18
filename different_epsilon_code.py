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

#short_distance = [] #create matrix for the distance between the closer points
#long_distance = [] #create matrix for the distance between the second two points
#overlap_matrix_projection = [] #create a matrix for the overlap fraction for each iteration
#distance_matrix = [] #create a matrix to store both distances
#x_matrix = [] #store x values for each iteration
#y1_matrix = [] #store y1 values for each iteration
#y2_matrix = [] #store y2 values for each iteration
#y3_matrix = [] #store y3 values for each iteration
#counter_1 = 1 #initialize counter
epsilon = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] #attempt at epsilon matrix
color = ['blue', 'green', 'red', "cyan", "magenta", "yellow", ]

for i in epsilon:
    short_distance = [] #create matrix for the distance between the closer points
    long_distance = [] #create matrix for the distance between the second two points
    overlap_matrix_projection = [] #create a matrix for the overlap fraction for each iteration
    distance_matrix = [] #create a matrix to store both distances
    x_matrix = [] #store x values for each iteration
    y1_matrix = [] #store y1 values for each iteration
    y2_matrix = [] #store y2 values for each iteration
    y3_matrix = [] #store y3 values for each iteration
    counter_i = 0
    while counter_i < 1000: #start iteration
        l_projection = 0.5 #lower starting point for bisection
        r_projection = 1 #upper bound for bisection
        x = np.random.uniform(0,100) #generate a random  x point
        y1 = np.random.uniform(0,10) #generate a random y point
        y2 = np.random.uniform(0,10) #generate another random y point close to the first
        y3 = np.random.uniform(300,1450) #generate a random y point far away 
        data_y = [y1, y2, y3]
        data_x = [x, x, x]
        data = np.vstack((data_x, data_y)).T #create matrix of each point
        
        while r_projection - l_projection > 1e-6: #give a threashhold to end bisection
            m_projection = (l_projection + r_projection)/2 #start bisection

            filter_func =  mp.Projection(columns=[1]) #use projection filter function
            cover = mp.CubicalCover(n_intervals=2, overlap_frac=m_projection) # Define cover, feed new overlap into cover
            clusterer = DBSCAN(eps = i) # Define clusterer
            # Initialise pipeline
            pipe_projection = mp.make_mapper_pipeline(
                filter_func=filter_func,
                cover=cover,
                clusterer=clusterer
            )

            graph = pipe_projection.fit_transform(data)
            igraph_graph = graph
            num_connected_components = len(igraph_graph.connected_components()) #check to see if the data points make a triangle
            if num_connected_components ==1:
                r_projection = m_projection
            else:
                l_projection = m_projection
            
        overlap_matrix_projection.append(l_projection) #update overlap fraction with minimum
        distance_1 = abs(y2-y1) #calculate distance 1
        distance_2 = abs(y3-y2) #calculate distance 2
        short_distance.append(distance_1) #add distance
        long_distance.append(distance_2) # add distance
        distance_matrix.append([distance_1, distance_2]) #assign to big distance matrix
        x_matrix.append(x) #store x
        y1_matrix.append(y1) #store y1
        y2_matrix.append(y2) #store y2
        y3_matrix.append(y3) #store y3
        counter_i += 1 #update counter
        print(counter_i)

    plt.figure()
    ax = plt.axes(projection= '3d')
    #ax.scatter3D(short_distance, long_distance, overlap_matrix_PCA, color = 'red')
    ax.scatter3D(short_distance, long_distance, overlap_matrix_projection, color = 'blue')
    ax.set_xlabel("Distance Between Close Points")
    ax.set_ylabel("Distance Between Middle and Third Point")
    ax.set_zlabel("Minimum Overlap Fraction")       
    ax.set_title("plot for epsilon = i")
    plt.show()

