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
from medoids import _k_medoids_spawn_once
from scipy.spatial.distance import pdist, squareform, hamming
from sklearn.metrics import silhouette_score, silhouette_samples
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

# Marcos.
NUM_TRIALS = 20
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Parse arguments from the user who must provide the following information:
parser = argparse.ArgumentParser(parents=opentuner.argparsers())
parser.add_argument('--base_folder_train', help='Path to the directory that contains adjacency list files of base part of the training graphs', required=True)
parser.add_argument('--stream_folder_train', help='Path to the directory that contains adjacency list files of streaming part of the training graphs', required=True)
parser.add_argument('--sketch_folder_train', help='Path to the directory that saves the training graph sketches', required=True)
parser.add_argument('--base_folder_test', help='Path to the directory that contains adjacency list files of base part of the test graphs', required=True)
parser.add_argument('--stream_folder_test', help='Path to the directory that contains adjacency list files of streaming part of the test graphs', required=True)
parser.add_argument('--sketch_folder_test', help='Path to the directory that saves the test graph sketches', required=True)
# '--train-dir <directory_path>': the path to the directory that contains data files of all training graphs.
# parser.add_argument('--train-dir', help='Absolute path to the directory that contains all training vectors', required=True)
# '--test-dir <directory_path>': the path to the directory that contains data files of all testing graphs.
# parser.add_argument('--test-dir', help='Absolute path to the directory that contains all testing vectors', required=True)
# '--threshold-metric <mean/max>': whether the threshold uses mean or max of the cluster distances between cluster members and the medoid.
# parser.add_argument('--threshold-metric', help='options: mean/max', required=False)
# '--num-stds <number>': the number of standard deviations a threshold should tolerate.
# parser.add_argument('--num-stds', help='Input a number of standard deviations the threshold should tolerate when testing', type=float, required=False)


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
		manipulator.add_parameter(IntegerParameter('decay', 100, 150))
		manipulator.add_parameter(IntegerParameter('interval', 1500, 2000))
		manipulator.add_parameter(IntegerParameter('chunk-size', 15, 20))
		manipulator.add_parameter(FloatParameter('lambda', 0.05, 0.3))
		manipulator.add_parameter(EnumParameter('threshold-metric', ['mean', 'max']))
		manipulator.add_parameter(FloatParameter('num-stds', 3.5, 6.0))
		manipulator.add_parameter(IntegerParameter('sketch-size', 2000, 2500))
		manipulator.add_parameter(IntegerParameter('k-hops', 3, 4))
		return manipulator

	def run(self, desired_result, input, limit):
		cfg = desired_result.configuration.data

		print "Configuration: " + cfg['threshold-metric'] + " with " + str(cfg['num-stds'])
		print "\t\t Decay: " + str(cfg['decay'])
		print "\t\t Interval: " + str(cfg['interval'])
		print "\t\t Lambda: " + str(cfg['lambda'])
		print "\t\t Chunk Size: " + str(cfg['chunk-size'])

		# Compile GraphChi with different flags.
		gcc_cmd = 'g++-4.9 -std=c++11 -g -O3 -I/usr/local/include/ -I./src/  -fopenmp -Wall -Wno-strict-aliasing -lpthread'
		gcc_cmd += ' -DSKETCH_SIZE=' + str(cfg['sketch-size'])
		gcc_cmd += ' -DK_HOPS=' + str(cfg['k-hops'])
		gcc_cmd += ' -DDEBUG -g -Istreaming/ streaming/main.cpp -o bin/streaming/main -lz'

		compile_result = self.call_program(gcc_cmd)
		assert compile_result['returncode'] == 0

		prog = re.compile("\.txt[\._]")

		# Run every training and test graph of the same experiment with the same hyperparameter
		train_base_dir_name = self.args.base_folder_train	# The directory absolute path name from the user input of base training graphs.
		train_base_files = sorted(os.listdir(train_base_dir_name))
		train_stream_dir_name = self.args.stream_folder_train	# The directory absolute path name from the user input of streaming part of the training graphs.
		train_stream_files = sorted(os.listdir(train_stream_dir_name))
		train_sketch_dir_name = self.args.sketch_folder_train	# The directory absolute path name to save the training graph sketch

		test_base_dir_name = self.args.base_folder_test	# The directory absolute path name from the user input of base test graphs.
		test_base_files = sorted(os.listdir(test_base_dir_name))
		test_stream_dir_name = self.args.stream_folder_test	# The directory absolute path name from the user input of streaming part of the test graphs.
		test_stream_files = sorted(os.listdir(test_stream_dir_name))
		test_sketch_dir_name = self.args.sketch_folder_test
		

		for i in range(len(train_base_files)):
			train_base_file_name = os.path.join(train_base_dir_name, train_base_files[i])
			train_stream_file_name = os.path.join(train_stream_dir_name, train_stream_files[i])
			train_sketch_file = 'sketch_' + str(i) + '.txt'
			train_sketch_file_name = os.path.join(train_sketch_dir_name, train_sketch_file)


			run_cmd = 'bin/streaming/main filetype edgelist'
			run_cmd += ' file ' + train_base_file_name
			run_cmd += ' niters 100000'
			run_cmd += ' stream_file ' + train_stream_file_name
			run_cmd += ' decay ' + str(cfg['decay'])
			run_cmd += ' lambda ' + str(cfg['lambda'])
			run_cmd += ' interval ' + str(cfg['interval'])
			run_cmd += ' sketch_file ' + train_sketch_file_name
			run_cmd += ' chunkify 1 '
			run_cmd += ' chunk_size ' + str(cfg['chunk-size'])

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
				test_sketch_file = 'sketch_attack_' + str(i) + '.txt'
			else:
				test_sketch_file = 'sketch_' + str(i) + '.txt'
			test_sketch_file_name = os.path.join(test_sketch_dir_name, test_sketch_file)


			run_cmd = 'bin/streaming/main filetype edgelist'
			run_cmd += ' file ' + test_base_file_name
			run_cmd += ' niters 100000'
			run_cmd += ' stream_file ' + test_stream_file_name
			run_cmd += ' decay ' + str(cfg['decay'])
			run_cmd += ' lambda ' + str(cfg['lambda'])
			run_cmd += ' interval ' + str(cfg['interval'])
			run_cmd += ' sketch_file ' + test_sketch_file_name
			run_cmd += ' chunkify 1 '
			run_cmd += ' chunk_size ' + str(cfg['chunk-size'])

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

		# train_dir_name = self.args['train-dir']	# The directory absolute path name from the user input of training vectors.
		# train_files = os.listdir(train_dir_name)	# The training file names within that directory.
		# Note that we will read every file within the directory @train_dir_name.
		# We do not do error checking here. Therefore, make sure every file in @train_dir_name is valid.
		# We do the same for validation/testing files.
		# test_dir_name = self.args['test-dir']	# The directory absolute path name from the user input of testing vectors.
		# test_files = os.listdir(test_dir_name)	# The testing file names within that directory.
		sketch_train_files = sorted(os.listdir(train_sketch_dir_name))
		sketch_test_files = sorted(os.listdir(test_sketch_dir_name))

		# Modeling (training)
		models = model(sketch_train_files, train_sketch_dir_name, NUM_TRIALS)
		# Testing
		test_accuracy = test(sketch_test_files, test_sketch_dir_name, models, cfg['threshold-metric'], cfg['num-stds'])
		print "Test Accuracy: " + str(test_accuracy)
	
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

		# remove Unicorn DB for this configuration
		try:
			if os.path.isfile("/local/data/unicorn.db"):
				os.unlink("/local/data/unicorn.db")
		except Exception as e:
			print(e)

		return Result(time=1.0, accuracy=test_accuracy)

	def save_final_config(self, configuration):
		"""called at the end of tuning"""
		print "Saving Optimal Configuration to a File..."
		self.manipulator().save_to_file(configuration.data, 'final_config.json')

class Model():
	"""
	Each training graph constructs a model, which may be merged with other models if possible.
	A model contains the following components:
	1. A list of medoids, e.g., [M_a, M_b, M_c]
	2. Parameter of each cluster correspond to the medoids. Currently, it is the mean of cluster distances between cluster member and the medoid. e.g., [A_a, A_b, A_c].
	3. Thresholds of each cluster correspond to the medoids
	4. A list of lists of members belong to each cluster, e.g., [[E_a_1, E_a_2, ...], [E_b_1, E_b_2, ...], [E_c_1, E_c_2, ...]]
	5. Confidence vector of the model
	6. The evolution of the graph based on cluster indices, e.g., We have a total three clusters, [0, 1, 2, 1, 2, ...]
	"""
	def __init__(self, medoids, mean_thresholds, max_thresholds, stds, members, evolution):
		self.medoids = medoids
		self.mean_thresholds = mean_thresholds
		self.max_thresholds = max_thresholds
		self.stds = stds
		self.members = members
		self.evolution = evolution

	def print_evolution(self):
		print self.evolution


def model(train_files, train_dir_name, num_trials):
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
			max_cluster_num = min(sketches.shape[0], 6)	# Some parameter may create very small number of possible clusters.
			for num_clusters in range(2, max_cluster_num):	# Try 2 - 5 for now.
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
			# @cluster_mean_thresholds contains the mean of each cluster used for normalization.
			cluster_mean_thresholds = [-1] * best_num_clusters
			# @cluster_max_thresholds contains the max of each cluster used for normalization.
			cluster_max_thresholds = [-1] * best_num_clusters
			# @cluster_stds contains the standard deviation of each cluster used for normalization.
			cluster_stds = [-1] * best_num_clusters
			for cluster_idx in range(best_num_clusters):
				cluster_center[cluster_idx] = best_medoids[cluster_idx].kernel	# @kernel is the index of the sketch that is considered the centroid.
				cluster_medoids[cluster_idx] = sketches[cluster_center[cluster_idx]]
				cluster_sketches = best_medoids[cluster_idx].elements	# @elements is a list that contains the indices of all the member sketches in the cluster.
				# @cluster_dists contains all distances between the kernel and each element in @elements
				cluster_members[cluster_idx] = cluster_sketches
				
				cluster_dists = [dists[cluster_center[cluster_idx]][skt] for skt in cluster_sketches if skt != cluster_center[cluster_idx]]
				if len(cluster_dists) == 0:	# This cluster has only one member.
					cluster_mean_thresholds[cluster_idx] = 0.0
					cluster_max_thresholds[cluster_idx] = 0.0
				else:
					cluster_mean_thresholds[cluster_idx] = np.mean(cluster_dists)
					cluster_max_thresholds[cluster_idx] = np.max(cluster_dists)
				cluster_stds[cluster_idx] = np.std(cluster_dists)
				
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
			new_model = Model(cluster_medoids, cluster_mean_thresholds, cluster_max_thresholds, cluster_stds, cluster_members, evolution)
			
			print "Model " + str(model_num) + " is done!"
			new_model.print_evolution()
			
			models.append(new_model)

		# We are done with this training file. Close the file and proceed to the next file.
		f.close()
	return models

# TODO: We can merge similar models in @models here.

def test(test_files, test_dir_name, models, threshold_metric, num_std):
	# Validation/Testing code starts here.
	total_graphs = 0.0
	predict_correct = 0.0
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
				if not model.evolution:	# If the evolution is empty
					check_next_model = True
					break
				current_evolution_idx = 0 
				current_cluster_idx = model.evolution[current_evolution_idx]
				current_medoid = model.medoids[current_cluster_idx]	# Get the medoid of the current cluster.
				if threshold_metric == 'mean':
					current_threshold = model.mean_thresholds[current_cluster_idx] + num_std * model.stds[current_cluster_idx]
				elif threshold_metric == 'max':
					current_threshold = model.max_thresholds[current_cluster_idx] + num_std * model.stds[current_cluster_idx]
				for sketch in sketches:
					distance_from_medoid = hamming(sketch, current_medoid)	# Compute the hamming distance between the current medoid and the current test vector.
					if distance_from_medoid > current_threshold:
						# We check maybe the evolution has evolved to the next cluster if it exsits.
						if current_evolution_idx < len(model.evolution) - 1:	# If there is actually a next cluster in evolution.
							current_evolution_idx = current_evolution_idx + 1 # Let's move on to the next cluster and see if it fits.
							current_cluster_idx = model.evolution[current_evolution_idx]
							current_medoid = model.medoids[current_cluster_idx]
							if threshold_metric == 'mean':
								current_threshold = model.mean_thresholds[current_cluster_idx] + num_std * model.stds[current_cluster_idx]
							elif threshold_metric == 'max':
								current_threshold = model.max_thresholds[current_cluster_idx] + num_std * model.stds[current_cluster_idx]
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
					
		f.close()
		total_graphs = total_graphs + 1
		if not abnormal:	# We have decided that the graph is not abnormal
			print "This graph: " + input_test_file + " is considered NORMAL (" + str(num_fitted_model) + "/" + str(len(models)) + ")."
			if "attack" not in input_test_file:
				predict_correct = predict_correct + 1
		else:
			print "This graph: " + input_test_file + " is considered ABNORMAL."
			if "attack" in input_test_file:
				predict_correct = predict_correct + 1
	accuracy = predict_correct / total_graphs
	return accuracy


if __name__ == "__main__":
	args = parser.parse_args()
	Unicorn.main(args)











