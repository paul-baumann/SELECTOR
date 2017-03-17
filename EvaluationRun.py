#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class is a data structure containing all 
# relevant information of an experiment.
#
# copyright Paul Baumann
#############################################

import numpy
from MetricResults import MetricResults
from UserData import UserData

class EvaluationRun:
    
    task_next_slot_place = 'NextSlotPlace'
    task_next_slot_transition = 'NextSlotTransition'
    task_next_place = 'NextPlace'
    
    task_next_place_daily = 'DailyNextPlace'
    
    metric_accuracy = 'accuracy'
    metric_fscore = 'fscore'
    metric_MCC = 'MCC'
    
    alg_logistic_regression = 'logistic_regression'
    alg_knn = 'knn'
    alg_knn_dyn = 'knn_dyn'
    alg_perceptron = 'perceptron'
    alg_decision_tree = 'decision_tree'
    alg_gradient_boost = 'gradient_boost'
    alg_svm = 'svm'
    alg_naivebayes = 'naive_bayes'
    alg_stupid = 'stupid'
    
    ## baselines
    alg_random = 'random'
    alg_majority = 'majority'
    alg_histogram = 'histogram'
    
    algorithms = [alg_knn_dyn, alg_perceptron, alg_decision_tree, alg_svm];
    
    metrics_next_place = [metric_accuracy, metric_fscore, metric_MCC]
    
    def __init__ (self):
        self.task = None;
        self.task_object = None;
        
        self.run_id = 0;
        self.userData = None;
        
        self.available_features = [];
        self.selected_features = [];
        
        self.training_set = None;
        self.test_set = None;
        
        self.start_time = 1;
        self.end_time = 96;
        self.when_to_predict_scope = None;
        self.demo_group = None;
        
        self.selected_metric = None;
        self.selected_algorithm = None;
        self.selected_feature_code = None;
        self.prediction_area = None;
        
        self.prediction_probabilities = [];
        self.prediction = [];
        self.data_ids_for_prediction = [];
        self.ground_truth = [];
        
        self.metric_results = MetricResults();
        
        self.random_prediction = [];
        self.dominating_class_prediction = [];
        self.histogram_class_prediction = [];
        
        self.is_network = True;
        self.is_temporal = True;
        self.is_spatial = True;
        self.is_context = True;
        

    def copy(self, evaluation_run):
        self.task = evaluation_run.task;
        self.task_object = evaluation_run.task_object;
        
        self.run_id = evaluation_run.run_id;
        self.userData = UserData()
        self.userData.copy(evaluation_run.userData);
        
        self.available_features = numpy.copy(evaluation_run.available_features);
        self.selected_features = numpy.copy(evaluation_run.selected_features);
        
        self.training_set = evaluation_run.training_set;
        self.test_set = evaluation_run.test_set;
               
        self.start_time = evaluation_run.start_time;
        self.end_time = evaluation_run.end_time;
        self.when_to_predict_scope = evaluation_run.when_to_predict_scope;
        self.demo_group = evaluation_run.demo_group;
        
        self.selected_metric = evaluation_run.selected_metric;
        self.selected_algorithm = evaluation_run.selected_algorithm;
        self.selected_feature_code = evaluation_run.selected_feature_code;
        self.prediction_area = evaluation_run.prediction_area;
        
        self.prediction_probabilities = evaluation_run.prediction_probabilities;
        self.prediction = evaluation_run.prediction;
        self.data_ids_for_prediction = evaluation_run.data_ids_for_prediction;
        self.ground_truth = evaluation_run.ground_truth;
        
        self.metric_results = evaluation_run.metric_results;
        
        self.random_prediction = evaluation_run.random_prediction;
        self.dominating_class_prediction = evaluation_run.dominating_class_prediction;
        self.histogram_class_prediction = evaluation_run.histogram_class_prediction;
        
        self.is_network = evaluation_run.is_network;
        self.is_temporal = evaluation_run.is_temporal;
        self.is_spatial = evaluation_run.is_spatial;
        self.is_context = evaluation_run.is_context;
        