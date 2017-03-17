#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class allows measuring prediction 
# performance after features have been 
# selected or by including all features.
#
# copyright Paul Baumann
#############################################

import thread
import threading
from time import time
import UserData
from EvaluationRun import EvaluationRun
from PredictorsPipeline import PredictorsPipeline
import UserDataAssemble
import NextPlaceOrSlotPredictionTask
import numpy

import Database_Handler

from pylab import *

DEBUG_LEVEL = 6
THREAD_LEVEL = 0

BASELINE = False
IS_PER_DAY_PERIOD = False

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
    if is_prediction == True:
        values.append(', '.join(str(x) for x in list(ravel(evaluation_run.selected_features))))
    else:
        values.append('No feature selection done!')
    values.append(evaluation_run.selected_algorithm)
    values.append(evaluation_run.selected_metric)
    
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
    
    values.append(evaluation_run.start_time)
    values.append(evaluation_run.end_time)
    
    dbHandler = Database_Handler.Get_DB_Handler()
    
    run_fields = ['user_id', 'feature_selection_id', 'timestamp',
                  'is_network', 'is_temporal', 'is_spatial', 'is_context', 
                  'selected_features', 'selected_algorithm', 'selected_metric', 'accuracy',
                   'precis', 'recall', 'fscore', 'mcc', 'frequency_of_top_class','kappa',
                    'baseline_random', 'baseline_histogram', 'baseline_majority', 'predictions',
                    'prediction_set_ids', 'is_prediction_correct', 'prediction_probabilities',
                   'start_time', 'end_time']
    dbHandler.insert("PostSelection_%s_Prediction" % (evaluation_run.task), run_fields, values)


def Run_Main_Loop():
    
    start = time()
    
    ## PREPARE TO RUN THE LOOP
    list_of_metrics = [EvaluationRun.metrics_next_place, EvaluationRun.metrics_next_place, EvaluationRun.metrics_next_place]
    algorithms = [EvaluationRun.alg_knn_dyn, EvaluationRun.alg_perceptron, EvaluationRun.alg_decision_tree, EvaluationRun.alg_svm];
        
    if IS_PER_DAY_PERIOD:
        start_periods = [1, 49, 69];
        end_periods = [48, 68, 96];
        tasks = [EvaluationRun.task_next_place_daily]
        task_objects = [NextPlaceOrSlotPredictionTask]
        feature_group_combinations = [[False, True, True, True, False]];
    else:
        start_periods = [1];
        end_periods = [96];
        tasks = [EvaluationRun.task_next_slot_place, EvaluationRun.task_next_slot_transition, EvaluationRun.task_next_place]
        task_objects = [NextPlaceOrSlotPredictionTask, NextPlaceOrSlotPredictionTask, NextPlaceOrSlotPredictionTask]
        ## first argument: true = no feature selection; false = feature selection
        feature_group_combinations = [[True, True, True, True, True], 
                                      [True, True, True, True, False], 
                                      [False, True, True, True, True], 
                                      [False, True, True, True, False]]; 
        
    if BASELINE:
        algorithms = [EvaluationRun.alg_random, EvaluationRun.alg_majority, EvaluationRun.alg_histogram];
        feature_group_combinations = [[True, True, True, True, True]]; 
    
    # read user list
    text_file = open("userids.txt", "r")
    userids = text_file.read().split('\n')
    text_file.close()
    
    start_task = int(sys.argv[1]) - 1
    end_task = int(sys.argv[2])
    start_used_id = int(sys.argv[3])
    end_user_id = int(sys.argv[4])
    
    # update arrays according to the user's parameters
    list_of_metrics = list_of_metrics[start_task:end_task] 
    tasks = tasks[start_task:end_task]
    task_objects = task_objects[start_task:end_task]
    userids = userids[start_used_id:end_user_id]
    
    ## RUN THE LOOP
    threads = []
    for user in userids:
        task_id = -1
        
        for current_task in tasks:
            task_id = task_id + 1 
            
            for time_index in range(len(start_periods)):
            
                userData = UserData.UserData()
                userData.userId = int(user)  
                
                evaluation_run = EvaluationRun()
                evaluation_run.task = current_task
                evaluation_run.task_object = task_objects[task_id]
                
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
                                                                              list_of_metrics, start, feature_group_combinations,) )
                    threads.append(task_thread)
                    task_thread.start()
                else:
                    Thread_Task(evaluation_run, task_id, algorithms, list_of_metrics, start, feature_group_combinations)
            
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
        print "Done with METRIC: %s, algorithm: %s, task: %s, user: %s, run_id: %s -- after: %s seconds" % (current_metric, current_algorithm, current_task, user, evaluation_run.run_id, time() - start)
        print "######################################################"   

    

def Thread_Feature_Group(evaluation_run, start, current_feature_group):
    
    current_metric = evaluation_run.selected_metric
    current_algorithm = evaluation_run.selected_algorithm
    current_task = evaluation_run.task
    user = evaluation_run.userData.userId
    
    evaluation_run.is_network = current_feature_group[1]
    evaluation_run.is_temporal = current_feature_group[2]
    evaluation_run.is_spatial = current_feature_group[3]
    evaluation_run.is_context = current_feature_group[4]
    
    if current_feature_group[0] == True:
        if current_feature_group[4] == True:
            evaluation_run.selected_features = evaluation_run.available_features
        else:
            evaluation_run.selected_features = evaluation_run.available_features[:23]
    else:
        dbHandler = Database_Handler.Get_DB_Handler()
        query = ("select %s_Prediction_Run.id, %s_Prediction_Result_Analysis.feature_combination FROM "
                 "(%s_Prediction_Result_Analysis left join %s_Prediction_Run on %s_Prediction_Result_Analysis.run_id = %s_Prediction_Run.id) "
                 "WHERE %s_Prediction_Run.user_id = %s AND %s_Prediction_Result_Analysis.final = 1 "
                 "AND %s_Prediction_Run.selected_algorithm = '%s' AND %s_Prediction_Run.selected_metric = '%s' "
                 "AND %s_Prediction_Run.is_network = %s AND %s_Prediction_Run.is_temporal = %s AND %s_Prediction_Run.is_spatial = %s "
                 "AND %s_Prediction_Run.is_context = %s AND %s_Prediction_Run.start_time = %s "
                 "AND %s_Prediction_Run.end_time = %s order by %s_Prediction_Run.id DESC LIMIT 1") % ( 
                 evaluation_run.task, evaluation_run.task, evaluation_run.task, evaluation_run.task, 
                 evaluation_run.task, evaluation_run.task, evaluation_run.task, 
                 evaluation_run.userData.userId, evaluation_run.task, evaluation_run.task, evaluation_run.selected_algorithm,
                 evaluation_run.task, evaluation_run.selected_metric, evaluation_run.task, evaluation_run.is_network,
                 evaluation_run.task, evaluation_run.is_temporal, evaluation_run.task, evaluation_run.is_spatial, 
                 evaluation_run.task, evaluation_run.is_context, evaluation_run.task, evaluation_run.start_time, 
                 evaluation_run.task, evaluation_run.end_time, evaluation_run.task)
        
        
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

    
    