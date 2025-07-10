import copy
import heapq

import numpy as np


def euclidean_dist(u, v):
    if not isinstance(u, np.ndarray):
        u = np.array(u)
    if not isinstance(v, np.ndarray):
        v = np.array(v)
    return np.linalg.norm(u - v)

class SSDBSCAN:
    """Perform Semi Supervised DBSCAN clustering from vector array.
    For further details, see the References below.

    Parameters
    ----------
    metric : callable, default=euclidean_dist
        The metric to use when calculating distance between instances in a
        feature array. If metric is a callable, it must be one of
        the options allowed by :func:`sklearn.metrics.pairwise_distances` for
        its metric parameter.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples)
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    References
    ----------
    Lelis, L., and  J. Sander, `"Semi-supervised Density-Based Clustering" 
    <https://webdocs.cs.ualberta.ca/~santanad/papers/2009/lelisS09.pdf>` _. 
    In: Proceedings of the 2009 Ninth IEEE International Conference on Data 
    Mining (ICDM '09). IEEE Computer Society, USA, 842-847. 2009 

    Example
    --------
    >>> from ss_clustering import SSDBSCAN
    >>> import numpy as np
    >>> from sklearn.datasets import make_moons
    >>> X, _ = make(n_samples=200, noise=0.05, random_state=0)
    >>> y = np.full(X.shape[0], -1)
    >>> y[0] = 0
    >>> y[1] = 0
    >>> y[100] = 0
    >>> y[101] = 1
    >>> clustering = SSDBSCAN().fit(X, y)
    >>> clustering.labels_
    """
    def __init__(self, metric : callable = euclidean_dist):
        self.metric = metric

    def build_complete_graph_(self):
        adj = {sample_idx:[] for sample_idx in range(len(self.samples_in_))} # i:list of [cost, node]
        for sample_idx in range(len(self.samples_in_)):
            u = self.samples_in_[sample_idx]
            for sample_jdx in range(sample_idx+1, len(self.samples_in_)):
                v = self.samples_in_[sample_jdx]
                dist = self.metric(u, v)
                adj[sample_idx].append([dist, sample_jdx])
                adj[sample_jdx].append([dist, sample_idx])
        self.adjacency_ = adj

    def prims_dbscan_(self, root_sample_idx, labels, visit):
        viz = copy.deepcopy(visit)

        cluster = []
        minH = [[0, root_sample_idx]] # [cost, sample]
        while len(viz)<self.n_samples_in_ and len(minH) > 0:
            # Debug information
            if len(minH) == 0:
                print(f'{set(range(self.n_samples_in_)) - viz=}, {len(viz)=}, {self.n_samples_in_=}')
                print(f'{set([x[1] for x in self.adjacency_[sample_idx]]) - viz=}')

            # Find closest sample to the cluster
            sample_cost, sample_idx = heapq.heappop(minH)
            if sample_idx in viz:
                # Already popped the minimum version
                continue
            
            # Check if we've left the cluster and entered into a new one
            if labels[sample_idx] != -1 and labels[sample_idx] != labels[root_sample_idx]:
                # Backtrack to the boundary sample
                # Samples only upto the boundary sample are part of the cluster
                boundary_sample = max(cluster)
                boundary_sample_idx = cluster.index(boundary_sample)
                for i in range(boundary_sample_idx, len(cluster)):
                    viz.remove(cluster[i][1])
                # Assign labels to the cluster samples
                for i in range(boundary_sample_idx):
                    labels[cluster[i][1]] = labels[root_sample_idx]
                return (labels, viz)

            # Bookkeeping
            cluster.append([sample_cost, sample_idx]) # add to cluster
            viz.add(sample_idx) # mark visited

            # Add neighbors with new distances
            # we add duplicates instead of updating to keep it simple
            # minheap selects minimum we just check if we've already seen the sample
            for nei_sample_cost, nei_sample_idx in self.adjacency_[sample_idx]:
                if nei_sample_idx not in viz:
                    heapq.heappush(minH, [nei_sample_cost, nei_sample_idx])

        # Assign labels to the cluster samples
        for i in range(len(cluster)):
            labels[cluster[i][1]] = labels[root_sample_idx]
        return (labels, viz)

    def fit(self, X, y):
        """Perform SSDBSCAN clustering from features.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training instances to cluster.

        y : {array-like} of shape (n_samples)
            Labels of the input sample. Unlabeled points should be marked with -1.

        Returns
        -------
        self : object
            Returns a fitted instance of self.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        assert X.ndim == 2 , f'Expected X.ndim = 2 but got {X.ndim}'
        assert y.ndim == 1 , f'Expected y.ndim = 1 but got {y.ndim}'
        assert X.shape[0] == y.shape[0], f'Expected same number of samples and \
            labels but got {X.shape[0]} samples and {y.sshape[0]} labels'
        
        visit = set()
        labels = copy.deepcopy(y)

        self.samples_in_ = copy.deepcopy(X)
        self.n_samples_in_ = len(self.samples_in_)
        self.build_complete_graph_()
        labeled_sample_idxs = [i for i in range(len(self.samples_in_)) if labels[i] != -1]        
        for labeled_sample_idx in labeled_sample_idxs:
            labels, visit = self.prims_dbscan_(root_sample_idx=labeled_sample_idx, labels=labels, visit=visit)
        self.labels_ = labels
        return self
    
    def fit_predict(self, X, y):
        """Perform SSDBSCAN clustering from features and predict labels.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training instances to cluster.

        y : {array-like} of shape (n_samples)
            Labels of the input sample. Unlabeled points should be marked with -1.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Noisy samples are given the label -1.
        """
        self.fit(X, y)
        return self.labels_
    
    def predict(self, X):
        """Predict labels from the fitted SSDBSCAN instance.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Instances to predict clusters for.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels based on nearest neighbors copmuted from `metric`.
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        assert X.ndim == 2 , f'Expected X.ndim = 2 but got {X.ndim}'

        labels = [-1] * len(X)
        for i in range(len(X)):
            labels[i] = self.labels_[np.argmin([self.metric(X[i], self.samples_in_[j]) for j in range(self.n_samples_in_)])]
        return labels