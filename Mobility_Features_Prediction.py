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
from EvaluationRun import EvaluationRun
import SFFS
import ResultAnalysis
import PredictorsPipeline
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

import warnings
warnings.filterwarnings('ignore')

from pylab import *

DEBUG_LEVEL = 6
THREAD_LEVEL = 0

def Save_Start_Evaluation_Run_To_DB(evaluation_run):
    
    # store prediction run details
    values = []
    values.append(evaluation_run.userData.userId)
    values.append(datetime.datetime.now().strftime('%d-%m-%Y-%H:%M:%S'))
    #values.append(numpy.array_str(ravel(evaluation_run.selected_features), max_line_width=1000000))
    values.append(evaluation_run.selected_algorithm)
    values.append(evaluation_run.selected_metric)
    values.append(evaluation_run.userData.optimization_set.ground_truth.shape[0])
    values.append(evaluation_run.userData.training_set.ground_truth.shape[0])
    values.append(evaluation_run.userData.test_set.ground_truth.shape[0])
    values.append(evaluation_run.userData.optimization_set.ground_truth.shape[0] + evaluation_run.userData.training_set.ground_truth.shape[0] + evaluation_run.userData.test_set.ground_truth.shape[0])
    
    values.append(', '.join(str(x) for x in list(evaluation_run.userData.optimization_set.rows_mask)))
    values.append(', '.join(str(x) for x in list(evaluation_run.userData.training_set.rows_mask)))
    values.append(', '.join(str(x) for x in list(evaluation_run.userData.test_set.rows_mask)))
    
    values.append(evaluation_run.is_network)
    values.append(evaluation_run.is_temporal)
    values.append(evaluation_run.is_spatial)
    values.append(evaluation_run.is_context)
    
    dbHandler = Get_DB_Handler()
    run_fields = ['user_id', 'start_timestamp', 'selected_algorithm', 'selected_metric', 'number_of_optimization_data',
                   'number_of_training_data', 'number_of_test_data', 'number_of_total_data', 
                   'optimization_array', 'training_array', 'test_array', 'is_network', 'is_temporal', 'is_spatial', 'is_context']
    insert_id = dbHandler.insert("%s_Prediction_Run" % (evaluation_run.task), run_fields, values)
    evaluation_run.run_id = insert_id
    
    return evaluation_run
### RECOVER FROM DB -- DO NOT DELETE
#     a = numpy.array([1, 2, 3])
#     b = numpy.array_str(ravel(a), max_line_width=1000000)
#     b = b[1:len(b)-1]
#     c = numpy.fromstring(b, sep=' ')
### RECOVER FROM DB -- DO NOT DELETE


def Save_End_Evaluation_Run_To_DB(evaluation_run):
    
    # store prediction run details
    timestamp = datetime.datetime.now().strftime('%d-%m-%Y-%H:%M:%S')
    
    dbHandler = Get_DB_Handler()
    query = "UPDATE %s_Prediction_Run SET end_timestamp = '%s' WHERE id = %i" % (evaluation_run.task, timestamp, evaluation_run.run_id)
    dbHandler.update(query)



def Get_DB_Handler():
    
    #return Database_Handler.Database_Handler("127.0.0.1", 12345, "root", "PW4mak!010", "PaperUbiComp2014")
    #return Database_Handler.Database_Handler("127.0.0.1", 3306, "root", "PW4mak!010", "PaperUbiComp2014")
    return Database_Handler.Database_Handler("127.0.0.1", 3306, "root", "PW4mak!010", "TMC2015")
    #return Database_Handler.Database_Handler("127.0.0.1", 8889, "root", "root", "TMC2015")
    #return Database_Handler.Database_Handler("127.0.0.1", 8889, "root", "root", "PaperUbiComp2014")
    # return Database_Handler.Database_Handler("epiwork.hcii.cs.cmu.edu", 3306, "pbaumann", "paul1234", "Nokia_DB")
    # return Database_Handler.Database_Handler("127.0.0.1", 3306, "pbaumann", "paul1234", "Nokia_DB")
    # return Database_Handler.Database_Handler("130.83.198.188", 3306, "root", "Heimdall4", "PaperUbiComp2014")



def Run_Main_Loop():
    
    start = time()
    
    ## PREPARE TO RUN THE LOOP
    list_of_metrics = [EvaluationRun.metrics_next_place, EvaluationRun.metrics_next_place, EvaluationRun.metrics_next_place, EvaluationRun.metrics_residence_time]
    tasks = [EvaluationRun.task_next_slot_place, EvaluationRun.task_next_slot_transition, EvaluationRun.task_next_place, EvaluationRun.task_next_residence_time]
    task_objects = [NextPlaceOrSlotPredictionTask, NextPlaceOrSlotPredictionTask, NextPlaceOrSlotPredictionTask, NextResidenceTimePredictionTask]
    algorithms = EvaluationRun.algorithms
    
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
    
    ## RUN THE LOOP
    for user in userids:
        task_id = -1
        threads = []
        
        for current_task in tasks:
            task_id = task_id + 1 
            
            userData = UserData.UserData()
            userData.userId = int(user)  
        
            evaluation_run = EvaluationRun()
            evaluation_run.task = current_task
            evaluation_run.task_object = task_objects[task_id]
            
            # feature group selection
            evaluation_run.is_network = True;
            evaluation_run.is_temporal = True;
            evaluation_run.is_spatial = True;
            evaluation_run.is_context = True;
            
            # get data
            if DEBUG_LEVEL > 0:
                #print "Loading... USER: %s -- after: %s seconds" % (user, time() - start)
                print("Loading... USER: %s -- after: %s seconds" % (user, time() - start))
            evaluation_run.userData = userData
            user_data_assemble = UserDataAssemble.UserDataAssemble(evaluation_run)
            evaluation_run = user_data_assemble.Get_User_Data()
            if DEBUG_LEVEL > 0:
                print("Loading DONE -- USER: %s -- after: %s seconds" % (user, time() - start))
            
            # run threads
            if THREAD_LEVEL > 2: # usually 2
                task_thread = threading.Thread( target=Thread_Task, args=(task_id, evaluation_run, algorithms, list_of_metrics, userData, start,) )
                threads.append(task_thread)
                task_thread.start()
            else:
                Thread_Task(task_id, evaluation_run, algorithms, list_of_metrics, userData, start)
            
        if THREAD_LEVEL > 2:
            for thread in threads:
                thread.join()
  
    print ("FINISH after : %s seconds" % (time() - start))
        

def Thread_Task(task_id, evaluation_run, algorithms, list_of_metrics, userData, start):
    
    threads = []
    
    for current_algorithm in algorithms:
        metrics = list_of_metrics[task_id]
        current_evaluation_run = EvaluationRun()
        current_evaluation_run.copy(evaluation_run)
        current_evaluation_run.selected_algorithm = current_algorithm
        
        if THREAD_LEVEL > 1:
            algorithm_thread = threading.Thread( target=Thread_Algorithm, args=(current_evaluation_run, metrics, start,) )
            threads.append(algorithm_thread)
            algorithm_thread.start()
        else:
            Thread_Algorithm(current_evaluation_run, metrics, start)
        
    if THREAD_LEVEL > 1:
        for thread in threads:
            thread.join()
        
    current_task = evaluation_run.task
    user = evaluation_run.userData.userId
    
    if DEBUG_LEVEL > 1:
        print("Done with TASK: %s, user: %s -- after: %s seconds" % (current_task, user, time() - start))
        print("######################################################")
        
        
def Thread_Algorithm(evaluation_run, metrics, start):
    
    current_algorithm = evaluation_run.selected_algorithm
    current_task = evaluation_run.task
    user = evaluation_run.userData.userId
    
    threads = []
    for current_metric in metrics:
        current_evaluation_run = EvaluationRun()
        current_evaluation_run.copy(evaluation_run)
        current_evaluation_run.selected_metric = current_metric  
        
        if THREAD_LEVEL > 0:
            metric_thread = threading.Thread( target=Thread_Metric, args=(current_evaluation_run, start,) )
            threads.append(metric_thread)
            metric_thread.start()
        else:   
            Thread_Metric(current_evaluation_run, start) 
        
    if THREAD_LEVEL > 0:
        for thread in threads:
            thread.join()
    
    if DEBUG_LEVEL > 2:
        print("Done with ALGORITHM: %s, task: %s, user: %s -- after: %s seconds" % (current_algorithm, current_task, user, time() - start))
        print("######################################################")
            
                    
def Thread_Metric(evaluation_run, start):  
    current_metric = evaluation_run.selected_metric
    current_algorithm = evaluation_run.selected_algorithm
    current_task = evaluation_run.task
    user = evaluation_run.userData.userId
    
    if DEBUG_LEVEL > 4:
        print("Starting with metric: %s, algorithm: %s, task: %s, user: %s" % (current_metric, current_algorithm, current_task, user))
    ##### save data to database
    evaluation_run = Save_Start_Evaluation_Run_To_DB(evaluation_run)           
    
    # prepare data
    evaluation_run.training_set = evaluation_run.userData.optimization_set
    evaluation_run.test_set = evaluation_run.userData.training_set
    
    # run SFFS
    sffs = SFFS.SFFS(evaluation_run, 10, start)
    sffs.Run_SFFS()
    
    Save_End_Evaluation_Run_To_DB(evaluation_run)
    
    if DEBUG_LEVEL > 4:
        print("Done with METRIC: %s, algorithm: %s, task: %s, user: %s, run_id: %s -- after: %s seconds" % (current_metric, current_algorithm, current_task, user, evaluation_run.run_id, time() - start))
        print("######################################################")   
    


if __name__ == "__main__":
    
    Run_Main_Loop()

    
    