import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

"""Clustering Function Class which results in trivial clustering.
Formatted to be placed as a clustering function for the gtda.mapper.make_mapper_pipeline class.

Set clusterer = trivial_clusterer()
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
