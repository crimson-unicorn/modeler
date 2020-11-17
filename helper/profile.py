#!/usr/bin/env python

##########################################################################################
# Some of the code is adapted from:
# https://github.com/sbustreamspot/sbustreamspot-train/blob/master/create_seed_clusters.py
##########################################################################################
import numpy as np
from scipy.spatial.distance import hamming
from sklearn.metrics import silhouette_score
from medoids import _k_medoids_spawn_once


class BestClusterGroup():
    """This class creates a cluster model from a graph (represented as a series of sketches).
    A cluster represents a broader "stage" of a progressing, dynamic graph. The cluster group
    represents the overall progress of the changing graph. All clusters in one group represent
    only one changing graph. """
    def __init__(self, best_num_clusters=-1, best_cluster_labels=None, best_medoids=None):
        self.best_num_clusters = best_num_clusters
        self.best_cluster_labels = best_cluster_labels
        self.best_medoids = best_medoids
    
    def optimize(self, arrs, distance, method='hamming', max_cluster_num=6, num_trials=20, max_iterations=1000):
        """Performs the clustering of input arrays and finds the best cluster group based on the Silhouette Coefficient.

        @distance:          the distance function needed to obtain the distance between two elements in the @arr. 
        @method:            the distance metric we use. Default is Hamming. Unless we change our way of hashing,
                            this default should not be modified.
        @max_cluster_num:   the maximal number of clusters that an ideal cluster group should have. We set
                            the default to 6 to save computation time. We observe that in most cases, the ideal
                            number of clusters for our experiment is smaller than 6.
        @num_trials:        the number of times we try to cluster for a given number of clusters for best
                            performance. Default is set to 20.
        @max_iterations:    the number of iterations to compute medoids. Default is set to 1000. """
        # We use silhouette coefficient to decide the ideal number of clusters. The best Silhouette
        # Coefficient value is 1 and the worst is -1. Values near 0 indicate overlapping clusters.
        # Negative values generally indicate that a sample has been assigned to the wrong cluster.
        best_silhouette_coef = -2   # initialization, must always be overwritten.
        max_cluster_num = min(arrs.shape[0], max_cluster_num)   # max_cluster_num should never be bigger than the total number of arrays
        for num_clusters in range(2, max_cluster_num):
            for trial in range(num_trials):
                # We use _k_medoids_spawn_once from medoids.py for computation.
                _, medoids = _k_medoids_spawn_once(points=range(arrs.shape[0]),
                                                   k=num_clusters,
                                                   distance=distance,
                                                   max_iterations=max_iterations,
                                                   verbose=False)
                # We assign each array element its cluster number based on the result.
                cluster_labels = [-1] * arrs.shape[0] # initialization
                actual_num_cluster = len(medoids) # @num_clusters is the maximum possible cluster. The actually number (@actual_num_cluster) may be smaller.
                for medoid_idx, medoid in enumerate(medoids):
                    for element in medoid.elements: # medoid.elements are all the array indices that belong to a medoid's cluster.
                        cluster_labels[element] = medoid_idx
                cluster_labels = np.array(cluster_labels)
                
                if actual_num_cluster == 1: # If the ideal number of cluster is 1, we cannot use silhouette_score function, which requires at least two clusters.
                    silhouette_coef = -1 # We prefer more than 1 cluster.
                else:
                    silhouette_coef = silhouette_score(arrs, cluster_labels, metric=method)
                # We decide if this run is the best value we have seen.
                if silhouette_coef > best_silhouette_coef or (silhouette_coef == best_silhouette_coef and actual_num_cluster > self.best_num_clusters):	# We prefer larger cluster number.
                    best_silhouette_coef = silhouette_coef
                    self.best_num_clusters = actual_num_cluster
                    self.best_cluster_labels = cluster_labels
                    self.best_medoids = medoids
    
    def get_best_num_clusters(self):
        return self.best_num_clusters
    
    def get_best_cluster_labels(self):
        return self.best_cluster_labels
    
    def get_best_medoids(self):
        return self.best_medoids


class Model():
    """Each training graph constructs a (sub)model. """
    def __init__(self, name, medoids=None, members=None, mean_thresholds=None, max_thresholds=None, stds=None, evolution=None):
        """
        @name:              the name of the training graph (which is probably the training file name)
        @medoids:           the actual vector of each cluster medoid (not just its index in @arr)
        @members:           the index of each member that belongs to the corresponding medoid cluster.
        @mean_thresholds:   the mean distance of each cluster.
        @max_thresholds:    the max distance of each cluster.
        @stds:              the standard deviation of each cluster.
        @evolution:         the ordered sequence of the cluster indices."""
        self.name = name
        self.medoids = medoids
        self.members = members
        self.mean_thresholds = mean_thresholds
        self.max_thresholds = max_thresholds
        self.stds = stds
        self.evolution = evolution
    
    def construct(self, arrs, dists, best_cluster_group):
        """Constructs the model from the data (@arrs) and from @best_cluster_group.
        @dists:     the matrix that contains the distance between every two elements in @arrs. """
        best_num_clusters = best_cluster_group.get_best_num_clusters()
        best_medoids = best_cluster_group.get_best_medoids()
        # cluster_centers have the indices of the medoid of each cluster.
        cluster_centers = [-1] * best_num_clusters # initialization
        # Initialize class members
        self.medoids = [[]] * best_num_clusters
        self.members = [[]] * best_num_clusters
        self.mean_thresholds = [-1] * best_num_clusters
        self.max_thresholds = [-1] * best_num_clusters
        self.stds = [-1] * best_num_clusters
        self.evolution = list()
        
        for cluster_idx in range(best_num_clusters):
            cluster_centers[cluster_idx] = best_medoids[cluster_idx].kernel     # kernel is the index of the medoid sketch
            self.medoids[cluster_idx] = arrs[cluster_centers[cluster_idx]]      # the actually sketch array of the medoid
            self.members[cluster_idx] = best_medoids[cluster_idx].elements      # elements is a list of indices of all the members in the cluster
            # cluster_dists contains all the distances between the kernel (medoid) and each member array
            cluster_dists = [dists[cluster_centers[cluster_idx]][elem] for elem in self.members[cluster_idx] if elem != cluster_centers[cluster_idx]]
            if len(cluster_dists) == 0:	# This cluster has only one member (this is usually not good)
                self.mean_thresholds[cluster_idx] = 0.0
                self.max_thresholds[cluster_idx] = 0.0
                self.stds[cluster_idx] = 0.0
            else:
                self.mean_thresholds[cluster_idx] = np.mean(cluster_dists)
                self.max_thresholds[cluster_idx] = np.max(cluster_dists)
                self.stds[cluster_idx] = np.std(cluster_dists)
        
        # Compute the evolution of the dynamic graph based on its members and the cluster index to which they belong
        prev = -1 # The cluster index of a previous sketch
        for elem_idx in range(arrs.shape[0]): # We go through every array element to construct the evolution
            for cluster_idx in range(best_num_clusters):
                if elem_idx in self.members[cluster_idx] or elem_idx == cluster_centers[cluster_idx]:
                    # We find the cluster index of the @elem_idx array.
                    current = cluster_idx
                    # If @current is equal to @prev, then we will not record it in
                    # evolution, because the evolving graph stays in the same cluster
                    if current == prev:
                        break	# We move on to the next @elem_idx.
                    else:
                        # Otherwise, we record @current in the @evolution.
                        prev = current
                        self.evolution.append(current)

    def get_name(self):
        return self.name

    def get_medoids(self):
        return self.medoids
    
    def get_members(self):
        return self.members
    
    def get_mean_thresholds(self):
        return self.mean_thresholds
    
    def get_max_thresholds(self):
        return self.max_thresholds
    
    def get_stds(self):
        return self.stds
    
    def get_evolution(self):
        return self.evolution


def test_single_graph(arrs, models, metric, num_stds, debug_info=None):
    """Test a single graph (@arrs) against all @models.
    @metric: can either be 'mean' or 'max'.
    The thresholds of the @models will be determined
    by @metric and @num_stds. @debug_info, if is not
    None, should be populated as a dictionary to store
    any useful debugging information. """
    abnormal = True             # Flag signaling whether the test graph is abnormal.
    abnormal_point = []         # @abnormal_point is valid only if eventually @abnormal is True.
                                # Since we test all models, @abnormal_point might not be empty
                                # even for a normal graph.
    max_abnormal_point = None   # The latest stage the graph cannot be fitted
    num_fitted_model = 0 # The total number of models that can be fitted by the test graph.
    # Additional logic for debugging only
    if isinstance(debug_info,dict): # debug_info is either None (no debugging) or a dictionary
        failed_at = dict()          # failed_at maps the name of the model to the first arr_id
                                    # that signals that the graph cannot be fitted to the model
        fitted_models = list()      # fitted_models records all the models that can be fitted
    # Fit the sketch arrays in @arrs to each model in @models. 
    # As long as the test graph could fit into one of the models,
    # we will set the @abnormal flag to False. If it could not
    # fit into any of the models, the @abnormal flag remains True
    # and we will signal the user. We also record the latest stage
    # (the index of the array in @arrs) the graph cannot be fitted
    # into any of the @models.
    for model in models:
        check_next_model = False        # Flag signaling whether we should proceed 
                                        # to check the next model because the
                                        # current one does not fit.
        if not model.get_evolution():	# If the evolution is empty
            continue
        
        current_evolution_idx = 0 
        current_cluster_idx = model.get_evolution()[current_evolution_idx]
        current_medoid = model.get_medoids()[current_cluster_idx]
        if metric == 'mean':
            current_threshold = model.get_mean_thresholds()[current_cluster_idx] + num_stds * model.get_stds()[current_cluster_idx]
        elif metric == 'max':
            current_threshold = model.get_max_thresholds()[current_cluster_idx] + num_stds * model.get_stds()[current_cluster_idx]
        
        for arr_id, sketch in enumerate(arrs):
            distance_from_medoid = hamming(sketch, current_medoid) # Compute the hamming distance between the current medoid and the current test sketch.
            if distance_from_medoid > current_threshold:
                # Check maybe the evolution has evolved to the next cluster if it exsits.
                if current_evolution_idx < len(model.get_evolution()) - 1: # If there is a next cluster in evolution.
                    current_evolution_idx = current_evolution_idx + 1      # Move on to the next cluster and see if it fits.
                    current_cluster_idx = model.get_evolution()[current_evolution_idx]
                    current_medoid = model.get_medoids()[current_cluster_idx]
                    if metric == 'mean':
                        current_threshold = model.get_mean_thresholds()[current_cluster_idx] + num_stds * model.get_stds()[current_cluster_idx]
                    elif metric == 'max':
                        current_threshold = model.get_max_thresholds()[current_cluster_idx] + num_stds * model.get_stds()[current_cluster_idx]
                    distance_from_medoid = hamming(sketch, current_medoid)
                    if distance_from_medoid > current_threshold: # If it still does not fit, we consider it abnormal
                        check_next_model = True	                 # We know this graph does not fit into this model,
                                                                 # but it may fit into a different model.
                        abnormal_point.append(arr_id)            # Record at which point the graph stops being normal
                        # Debugging only
                        if isinstance(debug_info,dict):
                            failed_at[model.get_name()] = arr_id
                        break
                else:                                            # If there is not a next cluster in evolution
                    check_next_model = True                      # We consider it abnormal in this model and check next model.
                    abnormal_point.append(arr_id)
                    # Debugging only
                    if isinstance(debug_info,dict):
                        failed_at[model.get_name()] = arr_id
                    break
        if not check_next_model:
            abnormal = False
            # If we don't need to check with the next model,
            # we know this test graph fits in this model, so
            # we are done. However, we would like to see how
            # many models our test graph could fit, so we
            # will test all the models.
            num_fitted_model = num_fitted_model + 1
            # Debugging only: record fitted model names
            if isinstance(debug_info,dict):
                fitted_models.append(model.get_name())
    if abnormal:
        max_abnormal_point = max(abnormal_point)

    # Additional logic for debugging only
    if isinstance(debug_info,dict): # debug_info is either None (no debugging) or a dictionary
        for model in models:
            sketch_info = dict()
            # @sketch_info maps for each test sketch,
            # the distance between itself and all
            # the clusters in the model
            for arr_id, sketch in enumerate(arrs):
                sketch_info[arr_id] = list()
                for cluster in model.get_evolution():
                    medoid = model.get_medoids()[cluster]
                    distance_from_medoid = hamming(sketch, medoid)
                    if metric == 'mean':
                        threshold = model.get_mean_thresholds()[cluster] + num_stds * model.get_stds()[cluster]
                    elif metric == 'max':
                        threshold = model.get_max_thresholds()[cluster] + num_stds * model.get_stds()[cluster]
                    sketch_info[arr_id].append((cluster, distance_from_medoid - threshold))
            debug_info[model.get_name()] = sketch_info
        debug_info["fitted Models"] = fitted_models
        debug_info["Failed Arr"] = failed_at

    return abnormal, max_abnormal_point, num_fitted_model

