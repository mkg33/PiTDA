import numpy as np
import gudhi
import networkx as nx
from sklearn.base import BaseEstimator, ClusterMixin

"""Clustering Function Class which results in trivial clustering.
Formatted to be placed as a clustering function for the gtda.mapper.make_mapper_pipeline class.

Set clusterer = trivial_clusterer()

Trivial Clustering: For subset $A$ of dataset $X$, trivial_clusterer.fit(A) = A.
"""
class trivial_clusterer(ClusterMixin, BaseEstimator):

    def fit(self, X, y=None):
        """ Fit the clustering from the fit mapper cover.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) where 

        y : ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
        
        """

        self.labels_ = np.ones(shape=(len(X),), dtype=np.int64)

        return self


"""Clustering Function Class for neighborhood graph clustering.
Formatted to be chosen as a clustering function for the gtda.mapper.make_mapper_pipeline class.

Set clusterer = nbhd_clusterer()

Neighborhood Clustering: For a subset $A$ of dataset $X$ (assumed to be a metric space) compute the 1-skeleton of the VR-complex of $A$ with maximum edge length given as `threshold` parameter $t$, denoted VR^1_t(A) with `max_edge_length = threshold`. Then the clusters of $A$ is the number of connected components of $VR^1_t(A)$.
"""
class nbhd_clusterer(ClusterMixin, BaseEstimator):

    def __init__(self, threshold = 0.4):
        self.threshold = threshold

    def fit(self, X, y=None):
        """ Fit the clustering from the fit mapper cover.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) where 

        y : ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self
        
        """
        # Compute VR-complex
        vr = gudhi.RipsComplex(points=X, max_edge_length=self.threshold)
        tree = vr.create_simplex_tree(max_dimension=1)

        # Construct Networkx graph
        'there has to be a more sophisticated way to construct G'
        G = nx.Graph()
        for simplex in tree.get_simplices():
            if len(simplex[0]) == 1:
                G.add_node(simplex[0][0])
            if len(simplex[0]) == 2:
                v1, v2 = simplex[0][0], simplex[0][1]
                G.add_edge(v1, v2, weight = simplex[1])

        # Find the connected components and assign labels to data.
        labels = np.ones(shape=(len(X),), dtype = np.int64)
        l = 1
        for cc in nx.connected_components(G):
            cc = list(cc)
            labels[cc] = l
            l += 1
        self.labels_ = labels
        return self
