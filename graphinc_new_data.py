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

# Read the CSV file into a DataFrame
data = pd.read_csv('tda_experiment_data.csv')

# Access data in the DataFrame using column names or indexing
#print(data['y1_value'])
#print(data.iloc[0])  # Access first row
plt.scatter(data['short_distance'], data['DB_scan_overlap'], color = 'green')
plt.title('Short Distance vs Overlap Fraction')
plt.show()

plt.scatter(data['long_distance'], data['DB_scan_overlap'], color = 'blue')
plt.title('Long Distance vs Overlap Fraction')
plt.show()