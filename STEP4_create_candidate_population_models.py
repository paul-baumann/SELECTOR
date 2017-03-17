#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class creates candidate 
# population models.
#
# copyright Paul Baumann
#############################################

import thread
import threading
from time import time
import Database_Handler
import UserData
import itertools
from EvaluationRun import EvaluationRun
from PredictorsPipeline import PredictorsPipeline
import UserDataAssemble
import NextPlaceOrSlotPredictionTask
import numpy

import Util

from pylab import *

DEBUG_LEVEL = 6
THREAD_LEVEL = 0

IS_PER_DAY_PERIOD = False

def Save_End_Evaluation_Run_To_DB(evaluation_run, is_prediction):
    
    # store prediction run details
    values = []
    values.append(evaluation_run.userData.userId)
    values.append(datetime.datetime.now().strftime('%d-%m-%Y-%H:%M:%S'))
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
    values.append(len(evaluation_run.prediction))
    values.append(evaluation_run.demo_group)
    values.append(evaluation_run.start_time)
    values.append(evaluation_run.end_time)
    
    dbHandler = Database_Handler.Get_DB_Handler()
    
    run_fields = ['user_id',  'timestamp', 'number_of_features',
                  'selected_features', 'selected_algorithm', 'selected_metric', 
                  'prediction_task', 'accuracy', 'precis', 'recall', 'fscore', 
                  'mcc', 'frequency_of_top_class', 
                   'number_of_predictions', 'demo_group',
                   'start_time', 'end_time']
    
    if IS_PER_DAY_PERIOD:
        dbHandler.insert("Candidate_Population_Models_Daily", run_fields, values)
    else:
        dbHandler.insert("Candidate_Population_Models", run_fields, values)



def Run_Main_Loop():
    
    start = time()
    
    if IS_PER_DAY_PERIOD:
        start_periods = [1, 49, 69];
        end_periods = [48, 68, 96];
        tasks = [EvaluationRun.task_next_place_daily]
        task_objects = [NextPlaceOrSlotPredictionTask]
    else:
        start_periods = [1];
        end_periods = [96];
        tasks = [EvaluationRun.task_next_slot_place, EvaluationRun.task_next_slot_transition, EvaluationRun.task_next_place]
        task_objects = [NextPlaceOrSlotPredictionTask, NextPlaceOrSlotPredictionTask, NextPlaceOrSlotPredictionTask]
    
    ## PREPARE TO RUN THE LOOP
    algorithms = [EvaluationRun.alg_knn_dyn, EvaluationRun.alg_perceptron, EvaluationRun.alg_decision_tree, EvaluationRun.alg_svm]; # 
    metrics_next_place = [EvaluationRun.metric_accuracy, EvaluationRun.metric_fscore, EvaluationRun.metric_MCC] #EvaluationRun.metric_accuracy, 
    
    list_of_metrics = [metrics_next_place, metrics_next_place, metrics_next_place]
    
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
            
            # Execute for each day period of time
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
    
    feature_set_size = 5 
    
    for demo_group in Util.demo_groups:
        
        ## check whether the user belongs to the demographic group
        if Util.userBelongToDemoGroup(user, demo_group) == False:
            continue;
        
        evaluation_run.demo_group = demo_group;
          
        dbHandler = Database_Handler.Get_DB_Handler()
        if IS_PER_DAY_PERIOD:
            table = 'FeatureRanksDaily'
        else:
            table = 'FeatureRanks'
        query = ("select feature_id FROM %s where metric = '%s' and prediction_task = '%s' "
                 "and demo_group = '%s' and start_time = %s and end_time = %s order by id") % ( 
                 table, evaluation_run.selected_metric, evaluation_run.task, demo_group, 
                 evaluation_run.start_time, evaluation_run.end_time)
        
        feature_combinations = dbHandler.select(query)
        feature_combinations = ravel(feature_combinations).astype(int);
        feature_combinations = feature_combinations[0:feature_set_size];
        
        # brute-force execution
        for L in range(1, len(feature_combinations)+1):
            for subset in itertools.combinations(feature_combinations, L):
        
                evaluation_run.selected_features = numpy.array(subset) 
                evaluation_run.training_set = evaluation_run.userData.training_set
                evaluation_run.test_set = evaluation_run.userData.test_set
                
                # Predict mobility
                predictors_pipeline = PredictorsPipeline(evaluation_run)
                evaluation_run = predictors_pipeline.Run_Predictions()
                
                # Measure performance
                evaluation_run.metric_results = evaluation_run.task_object.Run_Analysis(evaluation_run)
                evaluation_run.metric_results.selected_features = evaluation_run.selected_features
            
                # save results
                Save_End_Evaluation_Run_To_DB(evaluation_run, True)
        
        print "Done with DEMO GROUP: %s, metric: %s, algorithm: %s, task: %s, user: %s -- after: %s seconds" % (demo_group, current_metric, current_algorithm, current_task, user, time() - start)
        
    if DEBUG_LEVEL > 5:
        print ("Done with FEATURE COMBINATION -- metric: %s, algorithm: %s, "
               "task: %s, user: %s, day period: %s-%s -- after: %s seconds"
               "") % (current_metric, current_algorithm, current_task, user, 
                      evaluation_run.start_time, evaluation_run.end_time, time() - start)

if __name__ == "__main__":
           
    Run_Main_Loop()

    
    