#!/usr/bin/env python

##########################################################################################
# Partial credit to:
# Emaad Manzoor
# Some of the code is adapted from:
# https://github.com/sbustreamspot/sbustreamspot-train/blob/master/create_seed_clusters.py
##########################################################################################

import argparse
import numpy as np
import random
import os, sys
from helper.medoids import _k_medoids_spawn_once
from scipy.spatial.distance import pdist, squareform, hamming
from sklearn.metrics import silhouette_score, silhouette_samples
from copy import deepcopy

class Model():
	"""
	Each training graph constructs a model, which may be merged with other models if possible.
	A model contains the following components:
	1. A list of medoids, e.g., [M_a, M_b, M_c]
	2. Parameter of each cluster correspond to the medoids. Currently, it is the mean of cluster distances between cluster member and the medoid. e.g., [A_a, A_b, A_c].
	3. Thresholds of each cluster correspond to the medoids (for all configuration), e.g., [[T_a, T_b, T_c], ...]
	4. A list of lists of members belong to each cluster, e.g., [[E_a_1, E_a_2, ...], [E_b_1, E_b_2, ...], [E_c_1, E_c_2, ...]]
	5. Confidence vector of the model
	6. The evolution of the graph based on cluster indices, e.g., We have a total three clusters, [0, 1, 2, 1, 2, ...]
	"""
	def __init__(self, medoids, params, thresholds, members, evolution):
		self.medoids = medoids
		self.params = params
		self.thresholds = thresholds
		self.members = members
		self.evolution = evolution

	def print_thresholds(self):
		for ts in self.thresholds:
			print ts

	def print_evolution(self):
		print self.evolution


def model(train_files, train_dir_name, num_trials, threshold_metrics, nums_stds):
	# Now we will open every file and read the sketch vectors in the file for modeling.
	# We will create a model for each file and then merge the models if necessary.
	# @models contains a list of models from each file.
	models = []

	for model_num, input_train_file in enumerate(train_files):
		with open(os.path.join(train_dir_name, input_train_file), 'r') as f:
			sketches = []	# The sketch on row i is the ith stage of the changing graph.

			# We read all the sketches in the file and save it in memory in @sketches
			for line in f:
				sketch_vector = map(long, line.strip().split(" "))
				sketches.append(sketch_vector)

			sketches = np.array(sketches)
			# @dists now contains pairwise Hamming distance (using @pdist) between any two sketches in @sketches.
			# @squareform function makes it a matrix for easy indexing and accessing.
			dists = squareform(pdist(sketches, metric='hamming'))
			# We define a @distance function to use in @_k_medoids_spawn_once.
			def distance(x, y):
				return dists[x][y]

			# Now we use mean Silhouette Coefficient to determine the best number of clusters.
			# In our context, a cluster represents a broader "stage" of the progressing, dynamic graph.
			# Clusters show the progress of the changing graph, and all clusters represents one changing graph.
			best_num_clusters = -1	# The best number of clusters. We favor larger numbers of clusters.
			# The best Silhouette Coefficient value is 1 and the worst value is -1. 
			# Values near 0 indicate overlapping clusters. 
			# Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.
			best_silhouette_coef = -1
			best_cluster_labels = None	# A vector contains labels for each sketch.
			best_medoids = None	# A vector contains all medoids of clusters and their members (see @Medoid class in @medoids.py).

			# for num_clusters in range(2, sketches.shape[0]):	# At least 2 cluster and at most every sketch belongs to a different cluster.
			for num_clusters in range(2, 6):	# Try 2 - 5 for now.
				for trial in range(num_trials):	# We run many trials for a given number of clusters for best performance.
					# We use @_k_medoids_spawn_once from @medoids.py for computation.
					_, medoids = _k_medoids_spawn_once(points=range(sketches.shape[0]), k=num_clusters, distance=distance, max_iterations=1000, verbose=False)
					# Now we assign each sketch its cluster number based on the result of the previous computation.
					cluster_labels = [-1] * sketches.shape[0]
					real_cluster_num = len(medoids)	# @num_cluster represents the maximum possible cluster. The actually number of cluster may be smaller.
					for medoid_idx, medoid in enumerate(medoids):
						elements = medoid.elements	# @elements contains all sketch indices (in @sketches based on its row position i) that beling to this medoid's cluster.
						for element in elements:
							cluster_labels[element] = medoid_idx
					cluster_labels = np.array(cluster_labels)

					# TODO: The following should not happen.
					#######################################
					set_labels = set(cluster_labels)
					if (len(set_labels) == 1):
						continue
					#######################################

					# Once we know each sketch belongs to which cluster, we can calculate the Silhouette Coefficient
					silhouette_coef = silhouette_score(sketches, cluster_labels, metric='hamming')

					# Now we decide if this run is the best value seen so far.
					if silhouette_coef > best_silhouette_coef or (silhouette_coef == best_silhouette_coef and real_cluster_num > best_num_clusters):	# We favor larger cluster number.
						best_silhouette_coef = silhouette_coef
						best_num_clusters = real_cluster_num
						best_cluster_labels = cluster_labels
						best_medoids = medoids

			# Now that we have determined the best medoids, we calculate some statistics for modeling.
			# @cluster_medoids contains the sketch vector of each cluster medoid (not just its index in @sketches)
			cluster_medoids = [[]] * best_num_clusters
			# @cluster_members contains the index of each member that belongs to the corresponding medoid.
			cluster_members = [[]] * best_num_clusters
			# @cluster_center contains the index of the medoid of each cluster.
			cluster_center = [-1] * best_num_clusters
			# @cluster_param contains the density of each cluster used for normalization.
			cluster_param = [-1] * best_num_clusters
			# @all_cluster_dists contains cluster distances of all clusters.
			all_cluster_dists = [[]] * best_num_clusters
			for cluster_idx in range(best_num_clusters):
				cluster_center[cluster_idx] = best_medoids[cluster_idx].kernel	# @kernel is the index of the sketch that is considered the centroid.
				cluster_medoids[cluster_idx] = sketches[cluster_center[cluster_idx]]
				cluster_sketches = best_medoids[cluster_idx].elements	# @elements is a list that contains the indices of all the member sketches in the cluster.
				# @cluster_dists contains all distances between the kernel and each element in @elements
				cluster_members[cluster_idx] = cluster_sketches
				
				cluster_dists = [dists[cluster_center[cluster_idx]][skt] for skt in cluster_sketches if skt != cluster_center[cluster_idx]]
				if len(cluster_dists) == 0:	# This cluster has only one member.
					cluster_param[cluster_idx] = 0.0
				else:
					cluster_param[cluster_idx] = np.mean(cluster_dists)
				
				all_cluster_dists[cluster_idx] = cluster_dists

			# Now we can calculate the threshold based on @param_dist and @std_dist for all configurations.
			cluster_thresholds = []
			for threshold_metric in threshold_metrics:
				for num_stds in nums_stds:
					# @cluster_threshold contains a threshold for each cluster.
					# For each cluster, we calulate the threshold based on the mean/max distances of each member of the cluster from the center, and standard deviations.
					cluster_threshold = [-1] * best_num_clusters
					for cluster_idx in range(best_num_clusters):
						cluster_dists = all_cluster_dists[cluster_idx]
						if len(cluster_dists) == 0: # This cluster has only one member.
							threshold_base = 0.0
							std_dist = 0.0
						else:
							if threshold_metric == 'mean':
								threshold_base = np.mean(cluster_dists)
							elif threshold_metric == 'max':
								threshold_base = np.max(cluster_dists)
							else:
								print "Input threshold metric is currently not supported. We will the default metric (mean) instead."
								threshold_base = np.mean(cluster_dists)
							std_dist = np.std(cluster_dists)
						cluster_threshold[cluster_idx] = threshold_base + num_stds * std_dist
					# Add @cluster_threshold to cluster_thresholds
					cluster_thresholds.append(cluster_threshold)
			# The last step of generating a model from the training graph is to compute the evolution of the graph based on its members and the cluster index to which they belong.
			evolution = []
			prev = -1 	# Check what cluster index a previous sketch is in.
			for elem_idx in range(sketches.shape[0]):	# We go through every sketch to summarize the evolution.
				for cluster_idx in range(best_num_clusters):
					if elem_idx in cluster_members[cluster_idx] or elem_idx == cluster_center[cluster_idx]:
						# We find what cluster index the @elem_idx sketch belongs to.
						current = cluster_idx
						# If @current is equal to @prev, then we will not record it in evolution, since the evolving graph stays in the same cluster.
						if current == prev:
							break	# We simply move on to the next @elem_idx.
						else:
							# Otherwise, we record @current in the @evolution.
							prev = current
							evolution.append(current)

			# Now that we have @evolution, we have all the information we need for our model. We create the model and save it in @models.
			new_model = Model(cluster_medoids, cluster_param, cluster_thresholds, cluster_members, evolution)
			
			print "Model " + str(model_num) + " is done!"
			new_model.print_thresholds()
			new_model.print_evolution()
			
			models.append(new_model)

		# We are done with this training file. Close the file and proceed to the next file.
		f.close()
	return models

# TODO: We can merge similar models in @models here.

def confidence(train_sketches, train_cluster_labels, best_num_clusters, l):
	'''
	Calculate confidence of the model.
	'''
	confidence = [0] * len(train_cluster_labels)
	for n in range(len(train_cluster_labels)):
		shift_sil = [0] * best_num_clusters
		for m in range(best_num_clusters):
			cluster_labels_copy = deepcopy(train_cluster_labels)
			cluster_labels_copy[n] = m
			shift_sil[m] = pow(silhouette_samples(train_sketches, cluster_labels_copy, "hamming")[n] + 1, l)
		confidence[n] = max(shift_sil) / sum(shift_sil)
	return np.array(confidence)

def test(test_files, test_dir_name, models, index):
	# Validation/Testing code starts here.
	total_graphs = 0.0
	predict_correct = 0.0
	printout = ""
	for input_test_file in test_files:
		with open(os.path.join(test_dir_name, input_test_file), 'r') as f:
			sketches = []	# The sketch on row i is the ith stage of the changing graph.

			# We read all the sketches in the file and save it in memory in @sketches
			for line in f:
				sketch_vector = map(long, line.strip().split(" "))
				sketches.append(sketch_vector)

			sketches = np.array(sketches)

			num_fitted_model = 0	# Calculate the total number of models that can be fitted with the test graph.
			abnormal = True # Flag signalling whether the test graph is abnormal.
			# We now fit the sketch vectors in @sketches to each model in @models. 
			# As long as the test graph could fit into one of the models, we will set the @abnormal flag to False.
			# If it could not fit into any of the models, the @abnormal flag remains True and we will signal the user.
			for model in models:
				check_next_model = False	# Flag signalling whether we should proceed to check with the next model because the current one does not fit.
				current_evolution_idx = 0 
				current_cluster_idx = model.evolution[current_evolution_idx]
				current_medoid = model.medoids[current_cluster_idx]	# Get the medoid of the current cluster.
				current_threshold = model.thresholds[index][current_cluster_idx]	# Get the threshold of the current cluster.
				for sketch in sketches:
					distance_from_medoid = hamming(sketch, current_medoid)	# Compute the hamming distance between the current medoid and the current test vector.
					if distance_from_medoid > current_threshold:
						# We check maybe the evolution has evolved to the next cluster if it exsits.
						if current_evolution_idx < len(model.evolution) - 1:	# If there is actually a next cluster in evolution.
							current_evolution_idx = current_evolution_idx + 1 # Let's move on to the next cluster and see if it fits.
							current_cluster_idx = model.evolution[current_evolution_idx]
							current_medoid = model.medoids[current_cluster_idx]
							current_threshold = model.thresholds[index][current_cluster_idx]
							current_param = model.params[current_cluster_idx]
							distance_from_medoid = hamming(sketch, current_medoid)
							if distance_from_medoid > current_threshold:	# if it still does not fit, we consider it abnormal
								check_next_model = True	# So we know this graph does not fit into this model, but it can probably fit into a different model.
								break
						else:	# If there is not a next cluster in evolution
							check_next_model = True	# We consider it abnormal in this model and check next model.
							break	# TODO: we have not yet coded recurrent modelling, which could happen.
				if not check_next_model:
					abnormal = False	
					# If we don't need to check with the next model, we know this test graph fits in this model, so we are done.
					# break
					# However, we would like to see how many models our test graph could fit, so we will test all the models.
					num_fitted_model = num_fitted_model + 1
					# print the confidence stats of the model that fits
					# print "This model fits. Stats: " + str(np.mean(model.confidence))
				
		f.close()
		total_graphs = total_graphs + 1
		if not abnormal:	# We have decided that the graph is not abnormal
			printout += "This graph: " + input_test_file + " is considered NORMAL (" + str(num_fitted_model) + "/" + str(len(models)) + ").\n"
			if "attack" not in input_test_file:
				predict_correct = predict_correct + 1
		else:
			printout += "This graph: " + input_test_file + " is considered ABNORMAL.\n"
			if "attack" in input_test_file:
				predict_correct = predict_correct + 1
	accuracy = predict_correct / total_graphs
	return accuracy, printout



if __name__ == "__main__":

	# Marcos that are fixed every time.
	NUM_TRIALS = 20
	SEED = 42
	random.seed(SEED)
	np.random.seed(SEED)

	# Parse arguments from the user who must provide the following information:
	# '--train_dir <directory_path>': the path to the directory that contains data files of all training graphs.
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_dir', help='Absolute path to the directory that contains all training vectors', required=True)
	# '--validate_dir <directory_path>': the path to the directory that contains data files of all validation graphs.
	# parser.add_argument('--validate_dir', help='Absolute path to the directory that contains all validation vectors', required=True)
	# '--test_dir <directory_path>': the path to the directory that contains data files of all testing graphs.
	parser.add_argument('--test_dir', help='Absolute path to the directory that contains all testing vectors', required=True)
	# '--threshold_metric <mean/max>': whether the threshold uses mean or max of the cluster distances between cluster members and the medoid.
	parser.add_argument('--threshold_metric', help='options: mean/max', required=False)
	# '--num_stds <number>': the number of standard deviations a threshold should tolerate.
	parser.add_argument('--num_stds', help='Input a number of standard deviations the threshold should tolerate when testing', type=float, required=False)
	args = vars(parser.parse_args())

	train_dir_name = args['train_dir']	# The directory absolute path name from the user input of training vectors.
	train_files = os.listdir(train_dir_name)	# The training file names within that directory.
	# Note that we will read every file within the directory @train_dir_name.
	# We do not do error checking here. Therefore, make sure every file in @train_dir_name is valid.
	# We do the same for validation/testing files.
	# validate_dir_name = args['validate_dir']	# The directory absolute path name from the user input of validation vectors.
	# validate_files = os.listdir(validate_dir_name)	# The validation file names within that directory.
	test_dir_name = args['test_dir']	# The directory absolute path name from the user input of testing vectors.
	test_files = os.listdir(test_dir_name)	# The testing file names within that directory.

	threshold_metric = args['threshold_metric']
	if threshold_metric is None:	# If this argument is not supplied by the user, we try all possible configurations.
		threshold_metric_config = ['mean', 'max']
	else:
		threshold_metric_config = [threshold_metric]

	num_stds = args['num_stds']
	if num_stds is None:	# If this argument is not supplied by the user, we try all possible configurations.
		num_stds_config = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
	else:
		num_stds_config = [num_stds]

	# Modeling (training)
	models = model(train_files, train_dir_name, NUM_TRIALS, threshold_metric_config, num_stds_config)

	print "We will attempt multiple cluster threshold configurations for the best results."
	print "Trying: mean/max distances with 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0 standard deviation(s)..."
	print "Best Configuration: "
        best_accuracy = 0.0
        final_printout = ""
        best_metric = None
        best_std = 0.0
	for tm_num, tm in enumerate(threshold_metric_config):
		for ns_num, ns in enumerate(num_stds_config):
			# Validation/Testing
			index = tm_num * len(num_stds_config) + ns_num
			test_accuracy, printout = test(test_files, test_dir_name, models, index)
			if test_accuracy > best_accuracy:
				best_accuracy = test_accuracy
				final_printout = printout
				best_metric = tm
				best_std = ns
			elif test_accuracy == best_accuracy:
				if best_metric == 'max' and tm == 'mean':	# same accuracy, prefer mean than max
					best_metric = tm
					best_std = ns
					final_printout = printout
				elif best_metric == tm:				# same accuracy and same metric, prefer lower number of std
					if ns < best_std:
						best_std = ns
						final_printout = printout

	print "Threshold metric: " + best_metric
	print "Number of standard deviations: " + str(best_std)
	print "Test accuracy: " + str(best_accuracy)
	print "Results: "
	print final_printout









