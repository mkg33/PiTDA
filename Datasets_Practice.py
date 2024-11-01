#data gathering methods? 

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import gtda.mapper as mp
import numpy as np
from sklearn.cluster import KMeans
import gtda.diagrams.distance
from sklearn.datasets import fetch_species_distributions
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA



#data = datasets.make_circles(n_samples=300, factor=0.4, noise=.02)[0]
#plt.figure()
#plt.scatter(*data.T)

data_2, labels = datasets.make_blobs(n_samples=300, centers=3, cluster_std=5.0, random_state=42)

# Plot the data
plt.figure()
plt.scatter(data_2[:, 0], data_2[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.show()

#filter_func = np.linalg.norm # Define filter function
filter_func = pca = PCA(n_components=1)
cover = mp.CubicalCover(n_intervals=50, overlap_frac=0.5) # Define cover
clusterer = DBSCAN() # Define clusterer

# Initialise pipeline
pipe = mp.make_mapper_pipeline(
    filter_func=filter_func,
    cover=cover,
    clusterer=clusterer
)

plt.figure()
fig = mp.plot_static_mapper_graph(pipe, data_2, color_data=data_2[:,0])
fig.show()


data_3, labels = datasets.make_classification(n_samples=300, n_features=10, n_classes=2, n_clusters_per_class=1, random_state=42)

plt.figure()
plt.scatter(data_3[:, 0], data_3[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
plt.show()

plt.figure()
fig = mp.plot_static_mapper_graph(pipe, data_3, color_data=data_3[:,0])
fig.show()

data_4 = fetch_species_distributions(download_if_missing= True)
print(data_4.keys())
lat = data_4.coverages

plt.figure()
plt.scatter(lat[:, 0], lat[:, 1])
plt.show()
#fig = mp.plot_static_mapper_graph(pipe, lat, color_data=lat[:,0])
#fig.show()

data_5 = fetch_olivetti_faces(download_if_missing = True)
print(data_5.keys())
lat_5 = data_5.data
plt.figure()
plt.scatter(lat_5[:, 0], lat_5[:, 1])
plt.show()

plt.figure()
fig = mp.plot_static_mapper_graph(pipe, lat_5, color_data=lat_5[:,0])
fig.show()