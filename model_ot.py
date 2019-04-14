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
import os, sys, shutil, re
from helper.medoids import _k_medoids_spawn_once
from helper.profile import *
from model import load_sketches, model_all_training_graphs, test_all_testing_graphs
from scipy.spatial.distance import pdist, squareform, hamming
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import KFold, ShuffleSplit
from copy import deepcopy

import opentuner
from opentuner.search.manipulator import ConfigurationManipulator
from opentuner.search.manipulator import IntegerParameter
from opentuner.search.manipulator import FloatParameter
from opentuner.search.manipulator import EnumParameter
from opentuner.measurement import MeasurementInterface
from opentuner.search.objective import MaximizeAccuracy
from opentuner.resultsdb.models import Result
from opentuner.measurement.inputmanager import FixedInputManager

from RegularStepSearch import *

# Marcos
SEED = 42
NUM_CROSS_VALIDATION = 5
random.seed(SEED)
np.random.seed(SEED)

# Parse arguments from the user who must provide the following information:
parser = argparse.ArgumentParser(parents=opentuner.argparsers())
parser.add_argument('--base_folder_train', help='Path to the directory that contains edge list files of base part of the training graphs', required=True)
parser.add_argument('--stream_folder_train', help='Path to the directory that contains edge list files of streaming part of the training graphs', required=True)
parser.add_argument('--sketch_folder_train', help='Path to the directory that saves the training graph sketches', required=True)
parser.add_argument('--base_folder_test', help='Path to the directory that contains edge list files of base part of the test graphs', required=True)
parser.add_argument('--stream_folder_test', help='Path to the directory that contains edge list files of streaming part of the test graphs', required=True)
parser.add_argument('--sketch_folder_test', help='Path to the directory that saves the test graph sketches', required=True)
parser.add_argument('--window', help='Tune window hyperparameter', type=int, nargs='?', required=False, const='3000')
parser.add_argument('--interval', help='Tune interval hyperparameter', type=int, nargs='?', required=False, const='6000')
parser.add_argument('--sketch_size', help='Tune sketch size hyperparameter', type=int, nargs='?', required=False, const='2000')
parser.add_argument('--k_hops', help='Tune hop hyperparameter', type=int, nargs='?', required=False, const='3')
parser.add_argument('--chunk_size', help='Tune chunk size hyperparameter', type=int, nargs='?', required=False, const='5')
parser.add_argument('--lambda_param', help='Tune lambda hyperparameter', type=float, nargs='?', required=False, const='0.02')

class Unicorn(MeasurementInterface):
	'''
	Use OpenTuner to turn hyperparameters used by Unicorn System.
	'''
	def __init__(self, args):
		super(Unicorn, self).__init__(args,
			input_manager=FixedInputManager(),
			objective=MaximizeAccuracy())

	def manipulator(self):
		'''
		Define the search space by creating a ConfigurationManipulator
		'''
		manipulator = ConfigurationManipulator()
		if self.args.window is None:
			manipulator.add_parameter(IntegerParameter('window', 1000, 5000))
		if self.args.interval is None:
			manipulator.add_parameter(IntegerParameter('interval', 2000, 10000))
		if self.args.sketch_size is None:
			manipulator.add_parameter(IntegerParameter('sketch-size', 1000, 5000))
		if self.args.k_hops is None:
			manipulator.add_parameter(IntegerParameter('k-hops', 1, 5))
		if self.args.chunk_size is None:
			manipulator.add_parameter(IntegerParameter('chunk-size', 5, 20))
		if self.args.lambda_param is None:
			manipulator.add_parameter(FloatParameter('lambda-param', 0.0, 1.0))
		# manipulator.add_parameter(EnumParameter('threshold-metric', ['mean', 'max']))
		# manipulator.add_parameter(FloatParameter('num-stds', 3.5, 6.0))
		return manipulator

	def run(self, desired_result, input, limit):
		cfg = desired_result.configuration.data

		if self.args.window is None:
			window_param = cfg['window']
		else:
			window_param = self.args.window
		if self.args.interval is None:
			interval_param = cfg['interval']
		else:
			interval_param = self.args.interval
		if self.args.sketch_size is None:
			sketch_size_param = cfg['sketch-size']
		else:
			sketch_size_param = self.args.sketch_size
		if self.args.k_hops is None:
			k_hops_param = cfg['k-hops']
		else:
			k_hops_param = self.args.k_hops
		if self.args.chunk_size is None:
			chunk_size_param = cfg['chunk-size']
		else:
			chunk_size_param = self.args.chunk_size
		if self.args.lambda_param is None:
			lambda_param = cfg['lambda-param']
		else:
			lambda_param = self.args.lambda_param

		print "Configuration: "
		print "\t\t Window: " + str(window_param)
		print "\t\t Interval: " + str(interval_param)
		print "\t\t Sketch Size: " + str(sketch_size_param)
		print "\t\t K Hops: " + str(k_hops_param)
		print "\t\t Chunk Size: " + str(chunk_size_param)
		print "\t\t Lambda: " + str(lambda_param)

		# Compile GraphChi with different flags.
		gcc_cmd = 'g++ -std=c++11 -g -O3 -I/usr/local/include/ -I../graphchi-cpp/src/ -fopenmp -Wall -Wno-strict-aliasing -lpthread'
		gcc_cmd += ' -DSKETCH_SIZE=' + str(sketch_size_param)
		gcc_cmd += ' -DK_HOPS=' + str(k_hops_param)
		gcc_cmd += ' -DDEBUG -DPREGEN=10000 -DMEMORY=1 -g -I../graphchi-cpp/streaming/ ../graphchi-cpp/streaming/main.cpp -o ../graphchi-cpp/bin/streaming/main -lz'

		compile_result = self.call_program(gcc_cmd)
		assert compile_result['returncode'] == 0

		prog = re.compile("\.txt[\._]")

		# Run every training and test graph of the same experiment with the same hyperparameter
		train_base_dir_name = self.args.base_folder_train	# The directory absolute path name from the user input of base training graphs.
		train_base_files = sortfilenames(os.listdir(train_base_dir_name))
		train_stream_dir_name = self.args.stream_folder_train	# The directory absolute path name from the user input of streaming part of the training graphs.
		train_stream_files = sortfilenames(os.listdir(train_stream_dir_name))
		train_sketch_dir_name = self.args.sketch_folder_train	# The directory absolute path name to save the training graph sketch

		test_base_dir_name = self.args.base_folder_test	# The directory absolute path name from the user input of base test graphs.
		test_base_files = sortfilenames(os.listdir(test_base_dir_name))
		test_stream_dir_name = self.args.stream_folder_test	# The directory absolute path name from the user input of streaming part of the test graphs.
		test_stream_files = sortfilenames(os.listdir(test_stream_dir_name))
		test_sketch_dir_name = self.args.sketch_folder_test
		

		for i in range(len(train_base_files)):
			train_base_file_name = os.path.join(train_base_dir_name, train_base_files[i])
			train_stream_file_name = os.path.join(train_stream_dir_name, train_stream_files[i])
			train_sketch_file = 'sketch-' + str(i) + '.txt'
			train_sketch_file_name = os.path.join(train_sketch_dir_name, train_sketch_file)

			run_cmd = '../graphchi-cpp/bin/streaming/main filetype edgelist'
			run_cmd += ' file ' + train_base_file_name
			run_cmd += ' niters 100000'
			run_cmd += ' stream_file ' + train_stream_file_name
			run_cmd += ' decay 500'
			run_cmd += ' lambda ' + str(lambda_param)
			run_cmd += ' window ' + str(window_param)
			run_cmd += ' interval ' + str(interval_param)
			run_cmd += ' sketch_file ' + train_sketch_file_name
			run_cmd += ' chunkify 1 '
			run_cmd += ' chunk_size ' + str(chunk_size_param)

			print run_cmd
			run_result = self.call_program(run_cmd)
			assert run_result['returncode'] == 0

			# clean up after every training graph is run
			for file_name in os.listdir(train_base_dir_name):
				file_path = os.path.join(train_base_dir_name, file_name)
				if re.search(prog, file_path):
					try:
						if os.path.isfile(file_path):
							os.unlink(file_path)
						elif os.path.isdir(file_path):
							shutil.rmtree(file_path)
					except Exception as e:
						print(e)

		for i in range(len(test_base_files)):
			test_base_file_name = os.path.join(test_base_dir_name, test_base_files[i])
			test_stream_file_name = os.path.join(test_stream_dir_name, test_stream_files[i])
			if "attack" in test_base_file_name:
				test_sketch_file = 'sketch-attack-' + str(i) + '.txt'
			else:
				test_sketch_file = 'sketch-' + str(i) + '.txt'
			test_sketch_file_name = os.path.join(test_sketch_dir_name, test_sketch_file)

			run_cmd = '../graphchi-cpp/bin/streaming/main filetype edgelist'
			run_cmd += ' file ' + test_base_file_name
			run_cmd += ' niters 100000'
			run_cmd += ' stream_file ' + test_stream_file_name
			run_cmd += ' decay 500'
			run_cmd += ' lambda ' + str(lambda_param)
			run_cmd += ' window ' + str(window_param)
			run_cmd += ' interval ' + str(interval_param)
			run_cmd += ' sketch_file ' + test_sketch_file_name
			run_cmd += ' chunkify 1 '
			run_cmd += ' chunk_size ' + str(chunk_size_param)

			print run_cmd
			run_result = self.call_program(run_cmd)
			assert run_result['returncode'] == 0

			# clean up after every test graph is run
			for file_name in os.listdir(test_base_dir_name):
				file_path = os.path.join(test_base_dir_name, file_name)
				if re.search(prog, file_path):
					try:
						if os.path.isfile(file_path):
							os.unlink(file_path)
						elif os.path.isdir(file_path):
							shutil.rmtree(file_path)
					except Exception as e:
						print(e)

		# Note that we will read every file within the directory @train_dir_name.
		# We do not do error checking here. Therefore, make sure every file in @train_dir_name is valid.
		sketch_train_files = sortfilenames(os.listdir(train_sketch_dir_name))
		train_sketches, train_targets = load_sketches(sketch_train_files, train_sketch_dir_name, sketch_size_param)
		sketch_test_files = sortfilenames(os.listdir(test_sketch_dir_name))
		test_sketches, test_targets = load_sketches(sketch_test_files, test_sketch_dir_name, sketch_size_param)
		# We generate models once for all CVs
		all_models = model_all_training_graphs(train_sketches, train_targets, sketch_size_param)

		print "We will perform " + str(NUM_CROSS_VALIDATION) + "-fold cross validation..."
		threshold_metric_config = ['mean', 'max']
		num_stds_config = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
		# We record the average results
		# We use (true negatives)/(total validation datasets) as our accuracy metric because we want to minimize that now.
		average_accuracy = 0.0
		# final_printout = ""
		# final_precision = None
		# final_recall = None
		# final_f = None

		# kf = KFold(n_splits=NUM_CROSS_VALIDATION)
		kf = ShuffleSplit(n_splits=NUM_CROSS_VALIDATION, test_size=0.2, random_state=0)
		for benign_train, benign_validate in kf.split(train_targets):
			benign_validate_sketches, benign_validate_names = train_sketches[benign_validate], train_targets[benign_validate]
			kf_test_sketches = np.concatenate((test_sketches, benign_validate_sketches), axis=0)
			kf_test_targets = np.concatenate((test_targets, benign_validate_names), axis=0)
		
			# Modeling (training)
			models = []
			for index in benign_train:
				models.append(all_models[index])

			# Testing
			print "We will attempt multiple cluster threshold configurations for the best results."
			print "Trying: mean/max distances with 1.0, 1,1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0 standard deviation(s)..."
			for tm in threshold_metric_config:
				for ns in num_stds_config:
					tn, tp, fn, fp, total_normal_graphs, total_graphs, recall, precision, accuracy, f_measure, printout  = test_all_graphs(kf_test_sketches, kf_test_targets, sketch_size_param, models, tm, ns)
					print "Threshold metric: " + tm
					print "Number of standard deviations: " + str(ns)
					print "TP: " + str(tp) + "\t TN: " + str(tn) + "\t FP: " + str(fp) + "\t FN: " + str(fn)
					print "Test accuracy: " + str(accuracy)
					print "Test Precision: " + str(precision)
					print "Test Recall: " + str(recall)
					print "Test F-1 Score: " + str(f_measure)
					print "Results: "
					print printout

			# This value currently does not have any significance. We create just so that OpenTuner has an oracle.
			test_accuracy = tn / total_normal_graphs	#TODO: Currently we are concerned only of FPs. 
			average_accuracy = average_accuracy + test_accuracy
		average_accuracy = average_accuracy / NUM_CROSS_VALIDATION

		# For next experiment, remove sketch files from this experiment
		for sketch_train_file in sketch_train_files:
			file_to_remove = os.path.join(train_sketch_dir_name, sketch_train_file)
			try:
				if os.path.isfile(file_to_remove):
					os.unlink(file_to_remove)
			except Exception as e:
				print(e)
		for sketch_test_file in sketch_test_files:
			file_to_remove = os.path.join(test_sketch_dir_name, sketch_test_file)
			try:
				if os.path.isfile(file_to_remove):
					os.unlink(file_to_remove)
			except Exception as e:
				print(e)

		return Result(time=1.0, accuracy=average_accuracy)

	def save_final_config(self, configuration):
		"""called at the end of tuning"""
		print "Saving Optimal Configuration to a File..."
		self.manipulator().save_to_file(configuration.data, 'final_config.json')

def test_all_graphs(test_sketches, test_targets, size_check, models, metric, num_stds):
	# Validation/Testing code starts here.
	total_graphs = 0.0
	total_normal_graphs = 0.0
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
				total_normal_graphs = total_normal_graphs + 1
			else:
				fn = fn + 1
		else:
			printout += "This graph: " + test_targets[num] + " is considered ABNORMAL at " + str(max_abnormal_point) + "\n"
			if "attack" in test_targets[num]:
				tp = tp + 1
			else:
				fp = fp + 1
				total_normal_graphs = total_normal_graphs + 1
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
	return tn, tp, fn, fp, total_normal_graphs, total_graphs, recall, precision, accuracy, f_measure, printout

if __name__ == "__main__":
	args = parser.parse_args()
	Unicorn.main(args)












