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
from sklearn.model_selection import KFold, ShuffleSplit
from copy import deepcopy

def save_model(model, model_num, fh):
	"""Save a model with model number @model_num to a file with handle @fh.
	"""
	fh.write("MODEL " + str(model_num) + "\n")

	num_cluster = len(model.medoids)
	fh.write(str(num_cluster) + "\n")

	for medoid in model.medoids:
		for elem in medoid:
			fh.write(str(int(float(elem))) + " ")
		fh.write("\n")

	for met in model.mean_thresholds:
		fh.write(str(float(met)) + " ")
	fh.write("\n")

	for mat in model.max_thresholds:
		fh.write(str(float(mat)) + " ")
	fh.write("\n")

	for std in model.stds:
		fh.write(str(float(std)) + " ")
	fh.write("\n")

	for e in model.evolution:
		fh.write(str(e) + " ")
	fh.write("\n")

def load_sketches(file_names, dir_name, size_check):
	included_sketches = []
	included_targets = []
	for num, input_file in enumerate(file_names):
		with open(os.path.join(dir_name, input_file), 'r') as f:
			sketches = load_sketch(f, size_check)
			if sketches.size == 0:
				f.close()
				continue
			else:
				included_sketches.append(sketches)
				included_targets.append(file_names[num])
				f.close()
	return np.asarray(included_sketches), np.asarray(included_targets)

def model_all_training_graphs(train_sketches, train_names, size_check, max_cluster_num=6, num_trials=20, max_iterations=1000):
	# Now we will open every file and read the sketch vectors in the file for modeling.
	# We will create a model for each file and then merge the models if necessary (#TODO).
	
	# @models contains a list of models from each file.
	models = []
	# @savefile saves all the models
	savefile = open('models.txt', 'a+')
	for model_num, sketches in enumerate(train_sketches):
		# @dists now contains pairwise Hamming distance (using @pdist) between any two sketches in @sketches.
		try:
			dists = pairwise_distance(sketches)
		except Exception as e:
			print "Exception in model file " + str(train_names[model_num]) + ": " + str(e)
			raise RuntimeError("Model cannot be built properly: " + str(e))
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

		# saving model to the file
		print "Saving model " + str(train_names[model_num]) + "..."
		save_model(model, model_num, savefile)

		models.append(model)
	savefile.close()
	return models

# TODO: We can merge similar models in @models here.

def test_all_testing_graphs(test_sketches, test_targets, size_check, models, metric, num_stds):
	# Validation/Testing code starts here.
	total_graphs = 0.0
	tp = 0.0	# true positive (intrusion and alarmed)
	tn = 0.0	# true negative (not intrusion and not alarmed)
	fp = 0.0	# false positive (not intrusion but alarmed)
	fn = 0.0	# false negative (intrusion but not alarmed)

	printout = ""
	for num, sketches in enumerate(test_sketches):
		if sketches.size == 0:
			continue
		else:
			abnormal, max_abnormal_point, num_fitted_model = test_single_graph(sketches, models, metric, num_stds)
		total_graphs = total_graphs + 1
		if not abnormal:	# We have decided that the graph is not abnormal
			printout += "This graph: " + test_targets[num] + " is considered NORMAL (" + str(num_fitted_model) + "/" + str(len(models)) + ").\n"
			if "attack" not in test_targets[num]:
				tn = tn + 1
			else:
				fn = fn + 1
		else:
			printout += "This graph: " + test_targets[num] + " is considered ABNORMAL at " + str(max_abnormal_point) + "\n"
			if "attack" in test_targets[num]:
				tp = tp + 1
			else:
				fp = fp + 1
	if (tp + fp) == 0:
		precision = None
	else:
		precision = tp / (tp + fp)
	if (tp + fn) == 0:
		print "[ERROR] This should not have happened. Check your dataset."
		sys.exit(1)
	recall = tp / (tp + fn)
	accuracy = (tp + tn) / (tp + tn + fp + fn)
	if precision == None or (precision + recall) == 0:
		f_measure = None
	else:
		f_measure = 2 * (precision * recall) / (precision + recall)
	return precision, recall, accuracy, f_measure, printout

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
	parser.add_argument('--size', help='The expected size of a single sketch', type=int, default=2000, required=False)
	# '--threshold_metric <mean/max>': whether the threshold uses mean or max of the cluster distances between cluster members and the medoid.
	parser.add_argument('--threshold_metric', help='options: mean/max', required=False)
	# '--num_stds <number>': the number of standard deviations a threshold should tolerate.
	parser.add_argument('--num_stds', help='Input a number of standard deviations the threshold should tolerate when testing', type=float, required=False)
	args = vars(parser.parse_args())

	train_dir_name = args['train_dir']	# The directory absolute path name from the user input of training vectors.
	train_files = sortfilenames(os.listdir(train_dir_name))	# The training file names within that directory.
	train_sketches, train_targets = load_sketches(train_files, train_dir_name, args['size'])
	# Note that we will read every file within the directory @train_dir_name.
	# We do not do error checking here. Therefore, make sure every file in @train_dir_name is valid.
	# We do the same for validation/testing files.
	# validate_dir_name = args['validate_dir']	# The directory absolute path name from the user input of validation vectors.
	# validate_files = os.listdir(validate_dir_name)	# The validation file names within that directory.
	test_dir_name = args['test_dir']	# The directory absolute path name from the user input of testing vectors.
	test_files = os.listdir(test_dir_name)	# The testing file names within that directory.
	test_sketches, test_targets = load_sketches(test_files, test_dir_name, args['size'])

	threshold_metric = args['threshold_metric']
	if threshold_metric is None:	# If this argument is not supplied by the user, we try all possible configurations.
		threshold_metric_config = ['mean', 'max']
	else:
		threshold_metric_config = [threshold_metric]

	num_stds = args['num_stds']
	if num_stds is None:	# If this argument is not supplied by the user, we try all possible configurations.
		num_stds_config = [1.0, 1,1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
	else:
		num_stds_config = [num_stds]

	# only need to generate models once for all CVs
	all_models = model_all_training_graphs(train_sketches, train_targets, args['size'])

	num_cross_validation = 5
	# kf = KFold(n_splits=num_cross_validation)
	kf = ShuffleSplit(n_splits=num_cross_validation, test_size=0.2, random_state=0)
	print "We will perform " + str(num_cross_validation) + "-fold cross validation..."
	for benign_train, benign_validate in kf.split(train_targets):
		benign_validate_sketches, benign_validate_names = train_sketches[benign_validate], train_targets[benign_validate]
		kf_test_sketches = np.concatenate((test_sketches, benign_validate_sketches), axis=0)
		kf_test_targets = np.concatenate((test_targets, benign_validate_names), axis=0)

		# Modeling (training)
		models = []
		for index in benign_train:
			models.append(all_models[index])

		print "We will attempt multiple cluster threshold configurations for the best results."
		print "Trying: mean/max distances with 1.0, 1,1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0 standard deviation(s)..."
		print "Best Configuration: "
	        best_accuracy = 0.0
	        final_printout = ""
	        final_precision = None
	        final_recall = None
	        final_f = None
	        best_metric = None
	        best_std = 0.0
		for tm in threshold_metric_config:
			for ns in num_stds_config:
				# Validation/Testing
				test_precision, test_recall, test_accuracy, test_f_measure, printout = test_all_testing_graphs(kf_test_sketches, kf_test_targets, args['size'], models, tm, ns)
				print "Threshold metric: " + tm
				print "Number of standard deviations: " + str(ns)
				print "Test accuracy: " + str(test_accuracy)
				print "Test Precision: " + str(test_precision)
				print "Test Recall: " + str(test_recall)
				print "Test F-1 Score: " + str(test_f_measure)
				print "Results: "
				print printout
		# 		if test_accuracy > best_accuracy:
		# 			best_accuracy = test_accuracy
		# 			final_precision = test_precision
		# 			final_recall = test_recall
		# 			final_f = test_f_measure
		# 			final_printout = printout
		# 			best_metric = tm
		# 			best_std = ns
		# 		elif test_accuracy == best_accuracy:
		# 			if best_metric == 'max' and tm == 'mean':	# same accuracy, prefer mean than max
		# 				best_metric = tm
		# 				best_std = ns
		# 				final_precision = test_precision
		# 				final_recall = test_recall
		# 				final_f = test_f_measure
		# 				final_printout = printout
		# 			elif best_metric == tm:				# same accuracy and same metric, prefer lower number of std
		# 				if ns < best_std:
		# 					best_std = ns
		# 					final_precision = test_precision
		# 					final_recall = test_recall
		# 					final_f = test_f_measure
		# 					final_printout = printout

		# print "Threshold metric: " + best_metric
		# print "Number of standard deviations: " + str(best_std)
		# print "Test accuracy: " + str(best_accuracy)
		# print "Test Precision: " + str(final_precision)
		# print "Test Recall: " + str(final_recall)
		# print "Test F-1 Score: " + str(final_f)
		# print "Results: "
		# print final_printout









