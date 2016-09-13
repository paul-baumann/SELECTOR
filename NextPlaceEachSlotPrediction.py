#!/usr/bin/python
# -*- coding: utf-8 -*-

from multiprocessing import Process
import os
import thread
import threading
from time import time
import MySQLdb
import Database_Handler
import datetime
import math
import array
import sys
import UserData
import UserDataSet
import Mobility_Features_Prediction
from EvaluationRun import EvaluationRun
import SFFS
import ResultAnalysis
from PredictorsPipeline import PredictorsPipeline
import UserDataAssemble
import NextResidenceTimePredictionTask
import NextPlaceOrSlotPredictionTask
from datetime import datetime
from operator import mul
import numpy
import scipy.io
import sklearn.linear_model as linMod
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import tree
from sklearn.decomposition import PCA
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import gaussian_process
from sklearn.hmm import MultinomialHMM
from sklearn.ensemble import GradientBoostingClassifier

from pylab import *

DEBUG_LEVEL = 6
THREAD_LEVEL = 3

def Save_End_Evaluation_Run_To_DB(evaluation_run, is_prediction):
    
    correct_prediction = (ravel(evaluation_run.prediction) == ravel(evaluation_run.ground_truth)).astype(int);
    
    # store prediction run details
    values = []
    values.append(evaluation_run.userData.userId)
    values.append(evaluation_run.run_id)
    values.append(datetime.datetime.now().strftime('%d-%m-%Y-%H:%M:%S'))
    values.append(evaluation_run.is_network)
    values.append(evaluation_run.is_temporal)
    values.append(evaluation_run.is_spatial)
    values.append(evaluation_run.is_context)
    
    values.append(evaluation_run.selected_feature_code)
    if is_prediction == True:
        values.append(', '.join(str(x) for x in list(ravel(evaluation_run.selected_features))))
    else:
        values.append('No feature selection done!')
    
    values.append(evaluation_run.selected_algorithm)
    values.append(evaluation_run.selected_metric)    
    values.append(evaluation_run.prediction_area)
    
    if evaluation_run.task == EvaluationRun.task_next_residence_time:
        values.append(evaluation_run.metric_results.relative_prediction_error)
        values.append(evaluation_run.metric_results.absolute_prediction_error)
    else:
        values.append(evaluation_run.metric_results.accuracy)
        values.append(evaluation_run.metric_results.precision)
        values.append(evaluation_run.metric_results.recall)
        values.append(evaluation_run.metric_results.fscore)
        values.append(evaluation_run.metric_results.MCC)
        values.append(evaluation_run.metric_results.frequency_of_top_class)
        values.append(evaluation_run.metric_results.kappa)
        
    values.append(evaluation_run.metric_results.kappa_random)
    values.append(evaluation_run.metric_results.kappa_histogram)
    values.append(evaluation_run.metric_results.kappa_dominating)
    values.append(', '.join(str(x) for x in list(ravel(evaluation_run.prediction))))
    values.append(', '.join(str(x) for x in list(ravel(evaluation_run.data_ids_for_prediction))))
    values.append(', '.join(str(x) for x in list(ravel(correct_prediction))))
    values.append(', '.join(str(x) for x in list(ravel(evaluation_run.prediction_probabilities))))
    values.append(', '.join(str(x) for x in list(ravel(evaluation_run.ground_truth))))
    
    dbHandler = Get_DB_Handler()

    run_fields = ['user_id', 'feature_selection_id', 'timestamp',
                  'is_network', 'is_temporal', 'is_spatial', 'is_context', 'feature_code',
                  'selected_features', 'selected_algorithm', 'selected_metric', 'prediction_area', 'accuracy',
                   'precis', 'recall', 'fscore', 'mcc', 'frequency_of_top_class','kappa',
                    'baseline_random', 'baseline_histogram', 'baseline_majority', 'predictions',
                    'prediction_set_ids', 'is_prediction_correct', 'prediction_probabilities', 'ground_truth']
    
    dbHandler.insert("PostSelection_NextPlaceEachSlot_Prediction", run_fields, values)



def Get_DB_Handler():
    
    # return Database_Handler.Database_Handler("127.0.0.1", 3306, "pbaumann", "paul1234", "Nokia_DB")
    # return Database_Handler.Database_Handler("127.0.0.1", 12345, "root", "Heimdall4", "PaperUbiComp2014")
    return Database_Handler.Database_Handler("127.0.0.1", 8889, "root", "root", "PaperUbiComp2014")
    # return Database_Handler.Database_Handler("130.83.198.188", 3306, "root", "Heimdall4", "PaperUbiComp2014")



def Run_Main_Loop():
    
    start = time()
    
    ## PREPARE TO RUN THE LOOP
    
    # algorithms = [EvaluationRun.alg_linear_regression, EvaluationRun.alg_knn, alg_perceptron, EvaluationRun.alg_decision_tree, EvaluationRun.alg_gradient_boost, EvaluationRun.alg_svm];
    algorithms = [EvaluationRun.alg_knn, EvaluationRun.alg_perceptron, EvaluationRun.alg_decision_tree, EvaluationRun.alg_svm];
    # algorithms = [EvaluationRun.alg_svm];
    
    # metrics_next_place = [EvaluationRun.metric_fscore]
    metrics_next_place = [EvaluationRun.metric_accuracy, EvaluationRun.metric_fscore, EvaluationRun.metric_MCC]
    metrics_residence_time = [EvaluationRun.metric_absolute_prediction_error]
    
    list_of_metrics = [metrics_next_place, metrics_next_place, metrics_next_place, metrics_residence_time]
    tasks = [EvaluationRun.task_next_slot_place]
    task_objects = [NextPlaceOrSlotPredictionTask]
    
    feature_group_combinations = [[False, True, True, True, False]]; # , [False, True, True, True, False], [False, False, False, False, True][True, False, False, False, False],  
    # 
    prediction_areas = ['begin', 'middle', 'end'];
    
    # read user list
    text_file = open("userids.txt", "r")
    userids = text_file.read().split('\n')
    text_file.close()
    
    feature_code_id = int(sys.argv[1])
    start_used_id = int(sys.argv[2]) - 1
    end_user_id = int(sys.argv[3])
    userids = userids[start_used_id:end_user_id]
    
    feature_codes = ['feature_selection', 'temporal_only', 'spatial_only', 'temporal_and_spatial', 'global_set'];
#    selected_feature_code = feature_codes[feature_code_id];
    
    for selected_feature_code in feature_codes:
        ## RUN THE LOOP
        threads = []
        for user in userids:
            task_id = -1
            
            for current_task in tasks:
                task_id = task_id + 1 
                
                userData = UserData.UserData()
                userData.userId = int(user)  
                
                evaluation_run = EvaluationRun()
                evaluation_run.task = current_task
                evaluation_run.task_object = task_objects[task_id]
                evaluation_run.selected_feature_code = selected_feature_code
                
                # get data
                if DEBUG_LEVEL > 0:
                    print "Loading... USER: %s -- after: %s seconds" % (user, time() - start)
                evaluation_run.userData = userData
                user_data_assemble = UserDataAssemble.UserDataAssemble(evaluation_run)
                evaluation_run = user_data_assemble.Get_User_Data()
                
                complete_ground_truth = ravel(userData.complete_feature_matrix[:,7:8].astype(float))
                
                dbHandler = Get_DB_Handler()
                query = "SELECT optimization_array, training_array, test_array FROM NextPlace_Prediction_Run WHERE user_id = %i LIMIT 1" % (evaluation_run.userData.userId)
                existing_data = dbHandler.select(query)
                existing_data = numpy.array(existing_data)[0]
                
                training_idx = numpy.fromstring(existing_data[1], sep=', ').astype(int)
                test_idx = numpy.fromstring(existing_data[2], sep=', ').astype(int)
                data_masks = [training_idx, test_idx]
                
                for prediction_area in prediction_areas:
                    current_evaluation_run = EvaluationRun()
                    current_evaluation_run.copy(evaluation_run)
                    current_evaluation_run.prediction_area = prediction_area
                    
                    datasets = [current_evaluation_run.userData.training_set, current_evaluation_run.userData.test_set]
                    for dataset_id in range(len(datasets)):
                        dataset = datasets[dataset_id]
                        data_mask = data_masks[dataset_id]
                        #ground_truth = list(dataset.ground_truth)
                        ground_truth = list(complete_ground_truth)
                        
                        departure_idx = numpy.array(list(ravel(numpy.where(diff(ground_truth) != 0)) + 1))
                        departure_idx = departure_idx[0:len(departure_idx)-1]
                        departure_idx = departure_idx[data_mask]
                        
                        if prediction_area == 'begin':
                            idx = list([0]) + list(ravel(numpy.where(diff(ground_truth) != 0)) + 2)
                            idx = idx[0:len(idx)-1];
                            idx = numpy.array(idx);
                            if idx[len(idx) - 1] > len(ground_truth):
                                idx[len(idx) - 1] = len(ground_truth);
                            idx = idx[data_mask]
                        
                        if prediction_area == 'middle':
                            arrival_idx = numpy.array(list([0]) + list(ravel(numpy.where(diff(ground_truth) != 0)) + 2))
                            arrival_idx = arrival_idx[0:len(arrival_idx)-1];
                            arrival_idx = numpy.array(arrival_idx);
                            if arrival_idx[len(arrival_idx) - 1] > len(ground_truth):
                                arrival_idx[len(arrival_idx) - 1] = len(ground_truth);
                            
                            arrival_idx = arrival_idx[data_mask]
                            idx = numpy.array(arrival_idx + ceil((departure_idx - arrival_idx) / 2)).astype(int);
                        
                        if prediction_area == 'end':
                            arrival_idx = numpy.array(list([0]) + list(ravel(numpy.where(diff(ground_truth) != 0)) + 2))
                            arrival_idx = arrival_idx[0:len(arrival_idx)-1];
                            arrival_idx = numpy.array(arrival_idx);
                            if arrival_idx[len(arrival_idx) - 1] > len(ground_truth):
                                arrival_idx[len(arrival_idx) - 1] = len(ground_truth);
                            arrival_idx = arrival_idx[data_mask]
                                
                            idx = departure_idx - 1;
                            idx = numpy.maximum(arrival_idx, idx);
                        
                        
                        dataset.ground_truth = userData.complete_feature_matrix[departure_idx,7:8].astype(float);
                        dataset.feature_matrix = userData.complete_feature_matrix[idx,8:userData.complete_feature_matrix.shape[1]];
    #                     dataset.rows_mask = dataset.rows_mask[idx];
    #                     dataset.timestamps = dataset.timestamps[idx];
    #                     dataset.day_string = dataset.day_string[idx];
    #                     dataset.time_string = dataset.time_string[idx];
                                
                    if DEBUG_LEVEL > 0:
                        print "Loading DONE -- USER: %s -- after: %s seconds" % (user, time() - start)
                    
                    # run threads
                    if THREAD_LEVEL > 3:
                        task_thread = threading.Thread( target=Thread_Task, args=(current_evaluation_run, task_id, algorithms, 
                                                                                  list_of_metrics, start, feature_group_combinations,) )
                        threads.append(task_thread)
                        task_thread.start()
                    else:
                        Thread_Task(current_evaluation_run, task_id, algorithms, list_of_metrics, start, feature_group_combinations)
                
        if THREAD_LEVEL > 3:
            for thread in threads:
                thread.join()
      
        print "FINISH after : %s seconds" % (time() - start)
        


def Thread_Task(evaluation_run, task_id, algorithms, list_of_metrics, start, feature_group_combinations):
    
    threads = []
    
    for current_algorithm in algorithms:
        metrics = list_of_metrics[task_id]
        current_evaluation_run = EvaluationRun()
        current_evaluation_run.copy(evaluation_run)
        current_evaluation_run.selected_algorithm = current_algorithm
        
        if THREAD_LEVEL > 2:
            algorithm_thread = threading.Thread( target=Thread_Algorithm, args=(current_evaluation_run, metrics, start, feature_group_combinations,) )
            threads.append(algorithm_thread)
            algorithm_thread.start()
        else:
            Thread_Algorithm(current_evaluation_run, metrics, start, feature_group_combinations)
        
    if THREAD_LEVEL > 2:
        for thread in threads:
            thread.join()
        
    current_task = evaluation_run.task
    user = evaluation_run.userData.userId
    
    if DEBUG_LEVEL > 1:
        print "Done with Feature Group -- task: %s, user: %s -- after: %s seconds" % (current_task, user, time() - start)
        print "######################################################"        

        
def Thread_Algorithm(evaluation_run, metrics, start, feature_group_combinations):
    
    current_algorithm = evaluation_run.selected_algorithm
    current_task = evaluation_run.task
    user = evaluation_run.userData.userId
    
    threads = []
    for current_metric in metrics:
        current_evaluation_run = EvaluationRun()
        current_evaluation_run.copy(evaluation_run)
        current_evaluation_run.selected_metric = current_metric  
        
        if THREAD_LEVEL > 1:
            metric_thread = threading.Thread( target=Thread_Metric, args=(current_evaluation_run, start, feature_group_combinations,) )
            threads.append(metric_thread)
            metric_thread.start()
        else:   
            Thread_Metric(current_evaluation_run, start, feature_group_combinations) 
        
    if THREAD_LEVEL > 1:
        for thread in threads:
            thread.join()
    
    if DEBUG_LEVEL > 2:
        print "Done with ALGORITHM: %s, task: %s, user: %s -- after: %s seconds" % (current_algorithm, current_task, user, time() - start)
        print "######################################################"
            
                    
def Thread_Metric(evaluation_run, start, feature_group_combinations):  
    
    current_metric = evaluation_run.selected_metric
    current_algorithm = evaluation_run.selected_algorithm
    current_task = evaluation_run.task
    user = evaluation_run.userData.userId
    
    threads = []
    for current_feature_group in feature_group_combinations:        
        current_evaluation_run = EvaluationRun()
        current_evaluation_run.copy(evaluation_run)
        
        if THREAD_LEVEL > 0:
            feature_group_thread = threading.Thread( target=Thread_Feature_Group, args=(current_evaluation_run, start, current_feature_group,) )
            threads.append(feature_group_thread)
            feature_group_thread.start()
        else:
            Thread_Feature_Group(current_evaluation_run, start, current_feature_group)
            
    if THREAD_LEVEL > 0:
        for thread in threads:
            thread.join()
    
    if DEBUG_LEVEL > 4:
        print "Done with METRIC: %s, algorithm: %s, task: %s, run_id: %s, user: %s -- after: %s seconds" % (current_metric, current_algorithm, current_task, evaluation_run.run_id, user, time() - start)
        print "######################################################"   

    

def Thread_Feature_Group(evaluation_run, start, current_feature_group):
    
    current_metric = evaluation_run.selected_metric
    current_algorithm = evaluation_run.selected_algorithm
    current_task = evaluation_run.task
    user = evaluation_run.userData.userId
    
    if evaluation_run.selected_feature_code == 'spatial_only':
        evaluation_run.selected_features = [21, 22]
        
    if evaluation_run.selected_feature_code == 'global_set':
        evaluation_run.selected_features = [5, 21, 23]
        
    if evaluation_run.selected_feature_code == 'temporal_only':
        evaluation_run.selected_features = [11, 13, 14, 16]
        
    if evaluation_run.selected_feature_code == 'temporal_and_spatial':
        evaluation_run.selected_features = [14, 16, 21] #13, 14, 16, 21, 22
    
    if evaluation_run.selected_feature_code == 'feature_selection':
        evaluation_run.is_network = current_feature_group[1]
        evaluation_run.is_temporal = current_feature_group[2]
        evaluation_run.is_spatial = current_feature_group[3]
        evaluation_run.is_context = current_feature_group[4]
        
        fixed_task = 'NextPlace';
        
        dbHandler = Mobility_Features_Prediction.Get_DB_Handler()
        query = ("select %s_Prediction_Run.id, %s_Prediction_Result_Analysis.feature_combination FROM (%s_Prediction_Result_Analysis left join %s_Prediction_Run on %s_Prediction_Result_Analysis.run_id = %s_Prediction_Run.id) WHERE %s_Prediction_Run.user_id = %s AND %s_Prediction_Result_Analysis.final = 1 AND %s_Prediction_Run.selected_algorithm = '%s' AND %s_Prediction_Run.selected_metric = '%s' AND %s_Prediction_Run.is_network = %s AND %s_Prediction_Run.is_temporal = %s AND %s_Prediction_Run.is_spatial = %s AND %s_Prediction_Run.is_context = %s order by %s_Prediction_Run.id DESC LIMIT 1") % ( 
                 fixed_task, fixed_task, fixed_task, fixed_task, 
                 fixed_task, fixed_task, fixed_task, 
                 evaluation_run.userData.userId, fixed_task, fixed_task, evaluation_run.selected_algorithm,
                 fixed_task, evaluation_run.selected_metric, fixed_task, evaluation_run.is_network,
                 fixed_task, evaluation_run.is_temporal, fixed_task, evaluation_run.is_spatial, 
                 fixed_task, evaluation_run.is_context, fixed_task)
        
        
        feature_combinations = dbHandler.select(query)
        for row in feature_combinations:
            evaluation_run.run_id = int(row[0])
            evaluation_run.selected_features = ravel(numpy.fromstring(row[1], sep=', ').astype(int))
            break;
        

    # prepare data
    if len(evaluation_run.selected_features) > 0:
        evaluation_run.training_set = evaluation_run.userData.training_set
        evaluation_run.test_set = evaluation_run.userData.test_set
        
        predictors_pipeline = PredictorsPipeline(evaluation_run)
        evaluation_run = predictors_pipeline.Run_Predictions()
        evaluation_run.metric_results = evaluation_run.task_object.Run_Analysis(evaluation_run)
        evaluation_run.metric_results.selected_features = evaluation_run.selected_features
    
        # save results
        Save_End_Evaluation_Run_To_DB(evaluation_run, True)
    else:
        Save_End_Evaluation_Run_To_DB(evaluation_run, False)
    
    if DEBUG_LEVEL > 5:
        print "Done with FEATURE COMBINATION -- metric: %s, algorithm: %s, task: %s, user: %s -- after: %s seconds" % (current_metric, current_algorithm, current_task, user, time() - start)
        #print "######################################################"


if __name__ == "__main__":
    
    Run_Main_Loop()

    
    