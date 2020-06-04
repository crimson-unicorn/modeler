#!/usr/bin/env python

##########################################################################################
# Partial credit to:
# Emaad Manzoor
# Some of the code is adapted from:
# https://github.com/sbustreamspot/sbustreamspot-train/blob/master/create_seed_clusters.py
##########################################################################################

import numpy as np

from scipy.spatial.distance import pdist, squareform, hamming
from sklearn.metrics import silhouette_score, silhouette_samples
from medoids import _k_medoids_spawn_once
from collections import OrderedDict

class BestClusterGroup():
	"""
	This class instantiates a cluster group of a graph.
	A cluster represents a broader "stage" of a progressing, dynamic graph.
	The cluster group represents overall progress of the changing graph.
	All clusters in one group represent only one changing graph.
	"""
	def __init__(self, best_num_clusters=-1, best_cluster_labels=None, best_medoids=None):
		self.best_num_clusters = best_num_clusters
		self.best_cluster_labels = best_cluster_labels
		self.best_medoids = best_medoids

	def optimize(self, arr, distance, method='hamming', max_cluster_num=6, num_trials=20, max_iterations=1000):
		"""
		This function performs the clustering of the input arrays and finds the best cluster group based on the Silhouette Coefficient.
		
		@distance is the distance function needed to obtain distance between two elements in the @arr. 
		@method is the distance metric we use. Default is Hamming. Unless we change our way of hashing, this default should not be modified.
		@max_cluster_num is the maximal number of clusters that an ideal cluster group should have. We set the default to 6 to save computation time. 
		We observe that in most cases, the ideal number of clusters for our experiment datasets is smaller than 6.
		@num_trials is the number of times we try to cluster for a given number of cluster for best performance. Default is set to 20.
		@max_iterations is the number of iterations to compute medoids. Default is set to 1000.
		"""
		# We use this coefficient to decide the ideal number of clusters.
		# The best Silhouette Coefficient value is 1 and the worst value is -1. 
		# Values near 0 indicate overlapping clusters. 
		# Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.
		best_silhouette_coef = -2 # initialization, must always be overwritten.

		max_cluster_num = min(arr.shape[0], max_cluster_num)	# max_cluster_num should not be bigger than the total number of elements availale to be clustered.
		for num_clusters in range(2, max_cluster_num):
			for trial in range(num_trials):
				# We use @_k_medoids_spawn_once from @medoids.py for computation.
				_, medoids = _k_medoids_spawn_once(points=range(arr.shape[0]), k=num_clusters, distance=distance, max_iterations=max_iterations, verbose=False)
				# Now we assign each array element its cluster number based on the result of the previous computation.
				cluster_labels = [-1] * arr.shape[0] # initialization
				actual_cluster_num = len(medoids)	# @num_clusters represents the maximum possible cluster. The actually number of cluster (@actual_cluster_num) may be smaller.
				for medoid_idx, medoid in enumerate(medoids):
					elements = medoid.elements	# @elements contains all array indices (based on its row position i in @arr) that beling to this medoid's cluster.
					for element in elements:
						cluster_labels[element] = medoid_idx
				cluster_labels = np.array(cluster_labels)

				if actual_cluster_num == 1:	# if the ideal number of cluster is 1, we cannot use silhouette_score function, which requires at least two clusters.
					silhouette_coef = -1 	# we prefer more than 1 cluster.
				else:
					silhouette_coef = silhouette_score(arr, cluster_labels, metric=method)

				# Now we decide if this run is the best value seen so far.
				if silhouette_coef > best_silhouette_coef or (silhouette_coef == best_silhouette_coef and actual_cluster_num > self.best_num_clusters):	# We prefer larger cluster number.
					best_silhouette_coef = silhouette_coef
					self.best_num_clusters = actual_cluster_num
					self.best_cluster_labels = cluster_labels
					self.best_medoids = medoids

	def get_best_num_clusters(self):
		return self.best_num_clusters

	def get_best_cluster_labels(self):
		return self.best_cluster_labels

	def get_best_medoids(self):
		return self.best_medoids

class Model():
	"""
	Each training graph constructs a model, which may be merged with other models if possible.
	"""
	def __init__(self, medoids=None, members=None, mean_thresholds=None, max_thresholds=None, stds=None, evolution=None):
		"""
		@medoids contains the actual vector of each cluster medoid (not just its index in @arr)
		@members contains the index of each member that belongs to the corresponding medoid cluster.
		@mean_thresholds contains the mean distance of each cluster.
		@max_thresholds contains the max distance of each cluster.
		@stds contains the standard deviation of each cluster.
		@evolution contains the ordered sequence of the cluster indices.
		"""
		self.medoids = medoids
		self.members = members
		self.mean_thresholds = mean_thresholds
		self.max_thresholds = max_thresholds
		self.stds = stds
		self.evolution = evolution

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

	def print_mean_thresholds(self):
		for mt in self.mean_thresholds:
			print mt

	def print_evolution(self):
		print self.evolution

	def construct_model(self, arr, dists, best_cluster_group):
		"""
		The function construct the model from the data (@arr) and from @best_cluster_group.
		@dists is the matrix that contains the distance between every two elements in @arr.
		"""
		best_num_clusters = best_cluster_group.get_best_num_clusters()
		best_medoids = best_cluster_group.get_best_medoids()
		# @cluster_center contains the index of the medoid of each cluster.
		cluster_center = [-1] * best_num_clusters
		# initialize class members
		self.medoids = [[]] * best_num_clusters
		self.members = [[]] * best_num_clusters
		self.mean_thresholds = [-1] * best_num_clusters
		self.max_thresholds = [-1] * best_num_clusters
		self.stds = [-1] * best_num_clusters
		self.evolution = []

		for cluster_idx in range(best_num_clusters):
			cluster_center[cluster_idx] = best_medoids[cluster_idx].kernel	# @kernel is the index of the sketch that is considered the medoid.
			self.medoids[cluster_idx] = arr[cluster_center[cluster_idx]]
			cluster_elements = best_medoids[cluster_idx].elements	# @elements is a list that contains the indices of all the members in the cluster.
			self.members[cluster_idx] = cluster_elements
			# @cluster_dists contains all distances between the kernel and each element in @elements
			cluster_dists = [dists[cluster_center[cluster_idx]][elem] for elem in cluster_elements if elem != cluster_center[cluster_idx]]
			if len(cluster_dists) == 0:	# This cluster has only one member.
				self.mean_thresholds[cluster_idx] = 0.0
				self.max_thresholds[cluster_idx] = 0.0
				self.stds[cluster_idx] = 0.0
			else:
				self.mean_thresholds[cluster_idx] = np.mean(cluster_dists)
				self.max_thresholds[cluster_idx] = np.max(cluster_dists)
				self.stds[cluster_idx] = np.std(cluster_dists)
				
		# The last step is to compute the evolution of the graph based on its members and the cluster index to which they belong.
		prev = -1 	# Check what cluster index a previous sketch is in.
		for elem_idx in range(arr.shape[0]):	# We go through every array element to summarize the evolution.
			for cluster_idx in range(best_num_clusters):
				if elem_idx in self.members[cluster_idx] or elem_idx == cluster_center[cluster_idx]:
					# We find what cluster index the @elem_idx array belongs to.
					current = cluster_idx
					# If @current is equal to @prev, then we will not record it in evolution, since the evolving graph stays in the same cluster.
					if current == prev:
						break	# We simply move on to the next @elem_idx.
					else:
						# Otherwise, we record @current in the @evolution.
						prev = current
						self.evolution.append(current)

                for cluster_idx in range(best_num_clusters):
                        print("{}->{}".format(cluster_idx, self.members[cluster_idx]))


def load_sketch(file_handle):
	"""
	Load sketches in a file (from @file_handle) to memory as numpy arrays.
	"""
	sketches = []	# The sketch on row i is the ith stage of the changing graph.
	# We read all the sketches in the file and save it in memory in @sketches
	for line in file_handle:
		sketch_vector = map(long, line.strip().split(" "))
		sketches.append(sketch_vector)

	sketches = np.array(sketches)
	return sketches

def pairwise_distance(arr, method='hamming'):
	"""
	Wrapper function that calculates the pairwise distance between every two elements within the @arr.
	The metric (@method) is default as hamming.

	@squareform function makes it a matrix for easy indexing and accessing.
	"""
	return squareform(pdist(arr, metric=method))

def test_cluster(arr, models, metric, num_stds):
        for model_id, model in enumerate(models):
                arr_cluster = OrderedDict()
                for cluster in model.get_evolution():
                        medoid = model.get_medoids()[cluster]
                        if metric == 'mean':
                                threshold = model.get_mean_thresholds()[cluster] + num_stds * model.get_stds()[cluster]
                        elif metric == 'max':
                                threshold = model.get_max_thresholds()[cluster] + num_stds * model.get_stds()[cluster]
                        for arr_id, sketch in enumerate(arr):
                                if not arr_id in arr_cluster:
                                        arr_cluster[arr_id] = OrderedDict()
                                distance_from_medoid = hamming(sketch, medoid)
                                arr_cluster[arr_id][cluster] = distance_from_medoid - threshold
                #print(arr_cluster)


def test_single_graph(arr, models, metric, num_stds):
	"""
	This function test a single graph (@arr) against all @models.
	@metric can either be 'mean' or 'max'.
	The thresholds of the @models will be determined by @metric and @num_stds.
	"""
	abnormal = True # Flag signalling whether the test graph is abnormal.
	abnormal_point = [] # @abnormal_point is valid only if eventually @abnormal is True. Since we test all models, @abnormal_point might not be empty even for a normal graph.
	num_fitted_model = 0	# Calculate the total number of models that can be fitted by the test graph.
	# We now fit the vectors in @arr to each model in @models. 
	# As long as the test graph could fit into one of the models, we will set the @abnormal flag to False.
	# If it could not fit into any of the models, the @abnormal flag remains True and we will signal the user.
	# We also record at which stage (the index of the arr in @arr) the graph cannot be fitted into any of the @models.
	for model_id, model in enumerate(models):
		check_next_model = False	# Flag signalling whether we should proceed to check with the next model because the current one does not fit.
		if not model.get_evolution():	# If the evolution is empty
			check_next_model = True
			break
		current_evolution_idx = 0 
		current_cluster_idx = model.get_evolution()[current_evolution_idx]
		current_medoid = model.get_medoids()[current_cluster_idx]	# Get the medoid of the current cluster.
		if metric == 'mean':
			current_threshold = model.get_mean_thresholds()[current_cluster_idx] + num_stds * model.get_stds()[current_cluster_idx]
		elif metric == 'max':
			current_threshold = model.get_max_thresholds()[current_cluster_idx] + num_stds * model.get_stds()[current_cluster_idx]
		for arr_id, sketch in enumerate(arr):
			distance_from_medoid = hamming(sketch, current_medoid)	# Compute the hamming distance between the current medoid and the current test vector.
			if distance_from_medoid > current_threshold:
				# We check maybe the evolution has evolved to the next cluster if it exsits.
				if current_evolution_idx < len(model.get_evolution()) - 1:	# If there is actually a next cluster in evolution.
					current_evolution_idx = current_evolution_idx + 1 # Let's move on to the next cluster and see if it fits.
					current_cluster_idx = model.get_evolution()[current_evolution_idx]
					current_medoid = model.get_medoids()[current_cluster_idx]
					if metric == 'mean':
						current_threshold = model.get_mean_thresholds()[current_cluster_idx] + num_stds * model.get_stds()[current_cluster_idx]
					elif metric == 'max':
						current_threshold = model.get_max_thresholds()[current_cluster_idx] + num_stds * model.get_stds()[current_cluster_idx]
					distance_from_medoid = hamming(sketch, current_medoid)
					if distance_from_medoid > current_threshold:	# if it still does not fit, we consider it abnormal
						check_next_model = True	# So we know this graph does not fit into this model, but it may fit into a different model.
						# Record at which point the graph stops being normal
						abnormal_point.append(arr_id)
						break
				else:	# If there is not a next cluster in evolution
					check_next_model = True	# We consider it abnormal in this model and check next model.
					abnormal_point.append(arr_id)
					break	# TODO: we have not yet coded recurrent modelling, which could happen.
		if not check_next_model:
			abnormal = False	
			# If we don't need to check with the next model, we know this test graph fits in this model, so we are done.
			# break
			# However, we would like to see how many models our test graph could fit, so we will test all the models.
			num_fitted_model = num_fitted_model + 1
	if abnormal:
		max_abnormal_point = max(abnormal_point)
	else:
		max_abnormal_point = None
	return abnormal, max_abnormal_point, num_fitted_model




