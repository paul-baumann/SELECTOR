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
import itertools
import UserDataSet
import Mobility_Features_Prediction
from EvaluationRun import EvaluationRun
import SFFS
import ResultAnalysis
from PredictorsPipeline import PredictorsPipeline
import UserDataAssemble
import NextResidenceTimePredictionTask
import NextPlaceOrSlotPredictionTask
import PredictWithOtherUsersFeatureSet
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
THREAD_LEVEL = 0

def Save_End_Evaluation_Run_To_DB(evaluation_run, is_prediction):
    
    correct_prediction = (ravel(evaluation_run.prediction) == ravel(evaluation_run.ground_truth)).astype(int);
    
    # store prediction run details
    values = []
    values.append(evaluation_run.userData.userId)
    values.append(datetime.datetime.now().strftime('%d-%m-%Y-%H:%M:%S'))
    values.append(evaluation_run.feature_scope)
    values.append(len(evaluation_run.selected_features))
    values.append(', '.join(str(x) for x in list(ravel(evaluation_run.selected_features))))
    values.append(evaluation_run.selected_algorithm)
    values.append(evaluation_run.selected_metric)
    values.append(evaluation_run.task)

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
    #values.append(', '.join(str(x) for x in list(ravel(evaluation_run.prediction))))
    #values.append(', '.join(str(x) for x in list(ravel(evaluation_run.data_ids_for_prediction))))
    #values.append(', '.join(str(x) for x in list(ravel(correct_prediction))))
    
    values.append(evaluation_run.when_to_predict_scope)
    values.append(evaluation_run.demo_group)
    
    dbHandler = Mobility_Features_Prediction.Get_DB_Handler()
    
    run_fields = ['user_id',  'timestamp', 'feature_scope', 'number_of_features',
                  'selected_features', 'selected_algorithm', 'selected_metric', 'prediction_task', 'accuracy',
                   'precis', 'recall', 'fscore', 'mcc', 'frequency_of_top_class','kappa',
                    'baseline_random', 'baseline_histogram', 'baseline_majority', 'when_to_predict',  'demo_group'] #'predictions','prediction_set_ids', 'is_prediction_correct',
    
    dbHandler.insert("GlobalSet_DailyPeriod_Prediction", run_fields, values)



def Run_Main_Loop():
    
    start = time()
    
    ## PREPARE TO RUN THE LOOP
    
    # algorithms = [EvaluationRun.alg_linear_regression, EvaluationRun.alg_knn, alg_perceptron, EvaluationRun.alg_decision_tree, EvaluationRun.alg_gradient_boost, EvaluationRun.alg_svm];
    algorithms = [EvaluationRun.alg_knn, EvaluationRun.alg_perceptron, EvaluationRun.alg_decision_tree, EvaluationRun.alg_svm];
    
    # metrics_next_place = [EvaluationRun.metric_fscore]
    metrics_next_place = [EvaluationRun.metric_accuracy, EvaluationRun.metric_fscore, EvaluationRun.metric_MCC] #EvaluationRun.metric_accuracy, 
    
    metrics_residence_time = [EvaluationRun.metric_absolute_prediction_error]
    
    list_of_metrics = [metrics_next_place]
    tasks = [EvaluationRun.task_next_place_daily]
    task_objects = [NextPlaceOrSlotPredictionTask]
    
    # 
    # read user list
    text_file = open("userids.txt", "r")
    userids = text_file.read().split('\n')
    text_file.close()
    
    start_task = int(sys.argv[1]) - 1
    end_task = int(sys.argv[2])
    start_used_id = int(sys.argv[3]) - 1
    end_user_id = int(sys.argv[4])
    
    # update arrays according to the user's parameters
    list_of_metrics = list_of_metrics[start_task:end_task] 
    tasks = tasks[start_task:end_task]
    task_objects = task_objects[start_task:end_task]
    userids = userids[start_used_id:end_user_id]
    
    daily_periods = ['time_1_48', 'time_49_68', 'time_69_96'];
    start_periods = [1, 49, 69];
    end_periods = [48, 68, 96];
    ## RUN THE LOOP
    threads = []
    for user in userids:
        task_id = -1
        
        for current_task in tasks:
            task_id = task_id + 1 
            
            for time_index in range(len(daily_periods)):
                daily_period = daily_periods[time_index]
                
                userData = UserData.UserData()
                userData.userId = int(user)  
                
                evaluation_run = EvaluationRun()
                evaluation_run.task = current_task
                evaluation_run.task_object = task_objects[task_id]
                
                evaluation_run.when_to_predict_scope = daily_period;
                evaluation_run.start_time = start_periods[time_index]
                evaluation_run.end_time = end_periods[time_index]
                
                # get data
                if DEBUG_LEVEL > 0:
                    print "Loading... USER: %s -- after: %s seconds" % (user, time() - start)
                evaluation_run.userData = userData
                user_data_assemble = UserDataAssemble.UserDataAssemble(evaluation_run)
                evaluation_run = user_data_assemble.Get_User_Data()
                if DEBUG_LEVEL > 0:
                    print "Loading DONE -- USER: %s -- after: %s seconds" % (user, time() - start)
                
                # run threads
                if THREAD_LEVEL > 3:
                    task_thread = threading.Thread( target=Thread_Task, args=(evaluation_run, task_id, algorithms, 
                                                                              list_of_metrics, start,) )
                    threads.append(task_thread)
                    task_thread.start()
                else:
                    Thread_Task(evaluation_run, task_id, algorithms, list_of_metrics, start)
                
        if THREAD_LEVEL > 3:
            for thread in threads:
                thread.join()
  
    print "FINISH after : %s seconds" % (time() - start)
        


def Thread_Task(evaluation_run, task_id, algorithms, list_of_metrics, start):
    
    threads = []
    
    for current_algorithm in algorithms:
        metrics = list_of_metrics[task_id]
        current_evaluation_run = EvaluationRun()
        current_evaluation_run.copy(evaluation_run)
        current_evaluation_run.selected_algorithm = current_algorithm
        
        if THREAD_LEVEL > 2:
            algorithm_thread = threading.Thread( target=Thread_Algorithm, args=(current_evaluation_run, metrics, start,) )
            threads.append(algorithm_thread)
            algorithm_thread.start()
        else:
            Thread_Algorithm(current_evaluation_run, metrics, start)
        
    if THREAD_LEVEL > 2:
        for thread in threads:
            thread.join()
        
    current_task = evaluation_run.task
    user = evaluation_run.userData.userId
    
    if DEBUG_LEVEL > 1:
        print "Done with Feature Group -- task: %s, user: %s -- after: %s seconds" % (current_task, user, time() - start)
        print "######################################################"        

        
def Thread_Algorithm(evaluation_run, metrics, start):
    
    current_algorithm = evaluation_run.selected_algorithm
    current_task = evaluation_run.task
    user = evaluation_run.userData.userId
    
    threads = []
    for current_metric in metrics:
        current_evaluation_run = EvaluationRun()
        current_evaluation_run.copy(evaluation_run)
        current_evaluation_run.selected_metric = current_metric  
        
        if THREAD_LEVEL > 1:
            metric_thread = threading.Thread( target=Thread_Metric, args=(current_evaluation_run, start,) )
            threads.append(metric_thread)
            metric_thread.start()
        else:   
            Thread_Metric(current_evaluation_run, start) 
        
    if THREAD_LEVEL > 1:
        for thread in threads:
            thread.join()
    
    if DEBUG_LEVEL > 2:
        print "Done with ALGORITHM: %s, task: %s, user: %s -- after: %s seconds" % (current_algorithm, current_task, user, time() - start)
        print "######################################################"
            
                    
def Thread_Metric(evaluation_run, start):  
       
    current_metric = evaluation_run.selected_metric
    current_algorithm = evaluation_run.selected_algorithm
    current_task = evaluation_run.task
    user = evaluation_run.userData.userId
    
    demo_groups = ['all','male','female','working','study','age_group_16_21','age_group_22_27',
                  'age_group_28_33','age_group_34_38','age_group_39_44','no_children_all','with_children_all',
                  'with_children_male','with_children_female','single','family'];
                  
                  
    feature_scopes = ['per_algorithm']; #, 'per_metric', 'per_task', 'global_set'
    feature_set_sizes = 5;
    when_to_predict_scope = 'arrival';
    
    for demo_group in demo_groups:
        
        ## check whether the user belongs to the demographic group
        if PredictWithOtherUsersFeatureSet.userBelongToDemoGroup(user, demo_group) == False:
            continue;
            
        evaluation_run.demo_group = demo_group;
          
        for feature_scope in feature_scopes:
            evaluation_run.feature_scope = feature_scope;
            
            dbHandler = Mobility_Features_Prediction.Get_DB_Handler()
            query = ("select %s_best FROM PopulationFeatures_Daily where feature_scope = '%s' and prediction_task = 'NextPlace' and demo_group = '%s' and when_to_predict = '%s' and daily_period = '%s' order by id") % ( 
                     evaluation_run.selected_metric, feature_scope, demo_group, when_to_predict_scope, evaluation_run.when_to_predict_scope)
            feature_combinations = dbHandler.select(query)
            feature_combinations = ravel(feature_combinations).astype(int);
            
            feature_combinations = feature_combinations[0:feature_set_sizes];
            
            for L in range(1, len(feature_combinations)+1):
                for subset in itertools.combinations(feature_combinations, L):
                    
                    evaluation_run.selected_features = numpy.array(subset);
                    evaluation_run.training_set = evaluation_run.userData.training_set
                    evaluation_run.test_set = evaluation_run.userData.test_set
                    
                    predictors_pipeline = PredictorsPipeline(evaluation_run)
                    evaluation_run = predictors_pipeline.Run_Predictions()
                    evaluation_run.metric_results = evaluation_run.task_object.Run_Analysis(evaluation_run)
                    evaluation_run.metric_results.selected_features = evaluation_run.selected_features
                
                    # save results
                    Save_End_Evaluation_Run_To_DB(evaluation_run, True)
        
        print "Done with DEMO Daily: %s, -- scope: %s, when: %s, metric: %s, algorithm: %s, task: %s, user: %s -- after: %s seconds" % (evaluation_run.when_to_predict_scope, feature_scope, when_to_predict_scope, current_metric, current_algorithm, current_task, user, time() - start)
        
    if DEBUG_LEVEL > 5:
        print "Done with FEATURE COMBINATION -- metric: %s, algorithm: %s, task: %s, user: %s -- after: %s seconds" % (current_metric, current_algorithm, current_task, user, time() - start)
        #print "######################################################"


if __name__ == "__main__":
    
    Run_Main_Loop()

    
    