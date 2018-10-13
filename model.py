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
from helper.profile import *
from scipy.spatial.distance import pdist, squareform, hamming
from sklearn.metrics import silhouette_score, silhouette_samples
from copy import deepcopy

def model_all_training_graphs(train_files, train_dir_name, max_cluster_num=6, num_trials=20, max_iterations=1000):
	# Now we will open every file and read the sketch vectors in the file for modeling.
	# We will create a model for each file and then merge the models if necessary (#TODO).
	
	# @models contains a list of models from each file.
	models = []
	for model_num, input_train_file in enumerate(train_files):
		with open(os.path.join(train_dir_name, input_train_file), 'r') as f:
			sketches = load_sketch(f)
			# @dists now contains pairwise Hamming distance (using @pdist) between any two sketches in @sketches.
			dists = pairwise_distance(sketches)
			# We define a @distance function to use in @optimize.
			def distance(x, y):
				return dists[x][y]
			best_cluster_group = BestClusterGroup()
			best_cluster_group.optimize(arr=sketches, distance=distance, max_cluster_num=max_cluster_num, num_trials=num_trials, max_iterations=max_iterations)
			# Now that we have determined the best medoids, we calculate some statistics for modeling.
			model = Model()
			model.construct_model(sketches, dists, best_cluster_group)
						
			print "Model " + str(model_num) + " is done!"
			# model.print_mean_thresholds()
			# model.print_evolution()
			
			models.append(model)
		# We are done with this training file. Close the file and proceed to the next file.
		f.close()
	return models

# TODO: We can merge similar models in @models here.

def test_all_testing_graphs(test_files, test_dir_name, models, metric, num_stds):
	# Validation/Testing code starts here.
	total_graphs = 0.0
	tp = 0.0	# true positive (intrusion and alarmed)
	tn = 0.0	# true negative (not intrusion and not alarmed)
	fp = 0.0	# false positive (not intrusion but alarmed)
	fn = 0.0	# false negative (intrusion but not alarmed)

	printout = ""
	for input_test_file in test_files:
		with open(os.path.join(test_dir_name, input_test_file), 'r') as f:
			sketches = load_sketch(f)
			abnormal, max_abnormal_point, num_fitted_model = test_single_graph(sketches, models, metric, num_stds)
		f.close()
		total_graphs = total_graphs + 1

		if not abnormal:	# We have decided that the graph is not abnormal
			printout += "This graph: " + input_test_file + " is considered NORMAL (" + str(num_fitted_model) + "/" + str(len(models)) + ").\n"
			if "attack" not in input_test_file:
				tn = tn + 1
			else:
				fn = fn + 1
		else:
			printout += "This graph: " + input_test_file + " is considered ABNORMAL at " + str(max_abnormal_point) + "\n"
			if "attack" in input_test_file:
				tp = tp + 1
			else:
				fp = fp + 1
	if (tp + fp) == 0:
		precision = None
	else:
		precision = tp / (tp + fp)
	if (tp + fn) == 0:
		print "[ERROR] No attack cases provided. This should not have happened. Check dataset."
		sys.exit(1)
	recall = tp / (tp + fn)
	accuracy = (tp + tn) / (tp + tn + fp + fn)
	if (precision + recall) == 0:
		f_measure = None
	else:
		f_measure = 2 * (precision * recall) / (precision + recall)
	return tp, fp, tn, fn, precision, recall, accuracy, f_measure, printout

if __name__ == "__main__":

	# Marcos that are fixed every time.
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
	models = model_all_training_graphs(train_files, train_dir_name)

	# To save stats data
	stats = open("stats.csv", "a+")
	for tm in threshold_metric_config:
		for ns in num_stds_config:
			# Validation/Testing
			test_tp, test_fp, test_tn, test_fn, test_precision, test_recall, test_accuracy, test_f_measure, printout = test_all_testing_graphs(test_files, test_dir_name, models, tm, ns)
			stats.write(tm + ',' + str(ns) + ',' + str(test_tp) + ',' + str(test_fp) + ',' + str(test_tn) + ',' + str(test_fn) + ',' + str(test_precision) + ',' + str(test_recall) + ',' + str(test_accuracy) + ',' + str(test_f_measure) + '\n')









