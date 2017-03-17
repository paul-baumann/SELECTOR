#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class prints the performance achieved
# by the candidate individual models.
#
# copyright Paul Baumann
#############################################

from EvaluationRun import EvaluationRun
import Database_Handler
import numpy
from math import sqrt
import sys
import Util

from pylab import datetime
from pylab import ravel

IS_PER_DAY_PERIOD = False

def compareCandidatePopulationModels():
    
    if IS_PER_DAY_PERIOD:
        start_periods = [1, 49, 69];
        end_periods = [48, 68, 96];
        tasks = [EvaluationRun.task_next_place_daily]
    else:
        start_periods = [1];
        end_periods = [96];
        tasks = [EvaluationRun.task_next_place, EvaluationRun.task_next_slot_place, EvaluationRun.task_next_slot_transition]
    
    list_of_metrics = [EvaluationRun.metric_accuracy, EvaluationRun.metric_fscore, EvaluationRun.metric_MCC]
    
    algorithms = [EvaluationRun.alg_knn_dyn, EvaluationRun.alg_perceptron, EvaluationRun.alg_decision_tree, EvaluationRun.alg_svm]; #
    
    start_demo_group = int(sys.argv[1])
    end_demo_group = int(sys.argv[2])
    demo_groups = Util.demo_groups[start_demo_group:end_demo_group]
    
    for demo_group in demo_groups:
        
        # read user list
        text_file = open("userids.txt", "r")
        user_ids = text_file.read().split('\n')
        text_file.close()
        
        # Identify which users belong to the current demo group
        mask_users_belong_to_demo_group = Util.areUsersBelongToDemoGroup(user_ids, demo_group)
    
        for task in tasks:
            for metric in list_of_metrics:
                for time_index in range(len(start_periods)):
                
                    # Get individual models
                    dbHandler = Database_Handler.Get_DB_Handler()
                    query = ("select a.%s from "
                             "(select user_id, %s, selected_algorithm, selected_features from PostSelection_%s_Prediction "
                             "where is_context = 0 and selected_metric = '%s' and feature_selection_id > 0 and "
                             "(selected_algorithm = 'knn_dyn' or selected_algorithm = 'svm' or "
                             "selected_algorithm = 'perceptron' or selected_algorithm = 'decision_tree') "
                             "AND start_time = %s and end_time = %s "
                             "order by %s DESC) as a group by a.user_id order by a.user_id;"
                             "") % (metric, metric, task, metric, start_periods[time_index], end_periods[time_index], metric)
                    
                    query_individual_models = numpy.array(dbHandler.select(query))
                    individual_models_performance = query_individual_models[:,0]
                    
                    # Remove user results that do not belong to the current demo group
                    individual_models_performance = individual_models_performance[mask_users_belong_to_demo_group]
        
                    # Get candidate population model configs
                    if IS_PER_DAY_PERIOD:
                        table = 'Candidate_Population_Models_Daily'
                    else:
                        table = 'Candidate_Population_Models'
                        
                    dbHandler = Database_Handler.Get_DB_Handler()
                    query = ("select distinct(selected_features) from %s where demo_group = '%s' "
                             "and prediction_task = '%s' and selected_metric = '%s' AND start_time = %s and end_time = %s;"
                             "") % (table, demo_group, task, metric, start_periods[time_index], end_periods[time_index])
                    
                    query_results_selected_features = numpy.array(dbHandler.select(query))
                    
                    # Iterate over each feature subset
                    for selected_features in query_results_selected_features[:,0]:
                        number_of_features = len(ravel(numpy.fromstring(selected_features, sep=', ').astype(int)))
                        
                        # For each predictor
                        for algorithm in algorithms:
                            dbHandler = Database_Handler.Get_DB_Handler()
                            query = ("select %s from %s where demo_group = '%s' "
                                     "and prediction_task = '%s' and selected_metric = '%s' and selected_algorithm = '%s' "
                                     "and selected_features = '%s' and start_time = %s and end_time = %s order by user_id;"
                                     "") % (metric, table, demo_group, task, metric, algorithm, selected_features, 
                                            start_periods[time_index], end_periods[time_index])
                            
                            query_results_performance = numpy.array(dbHandler.select(query))
                            candidate_population_model_performance = query_results_performance[:,0]
                            
                            # adjust performance if mcc (due to potential negative values
                            if metric == 'mcc':
                                individual_models_performance += 1.0
                                candidate_population_model_performance += 1.0
                            
                            diff_performance = individual_models_performance - candidate_population_model_performance
                            mask_individual_model_better = diff_performance > 0
                            
                            # the higher, the better is individual model
                            if sum(mask_individual_model_better) > 0:
                                RMSE_individual_models = sqrt(sum(pow(diff_performance[mask_individual_model_better], 2)) / sum(mask_individual_model_better))
                            else:
                                RMSE_individual_models = 0.0
                            # the lower, the better is population model
                            if sum(~mask_individual_model_better) > 0:
                                RMSE_population_models = -sqrt(sum(pow(diff_performance[~mask_individual_model_better], 2)) / sum(~mask_individual_model_better))
                            else:
                                RMSE_population_models = 0.0
                            
                            RMSE_total = (RMSE_individual_models * sum(mask_individual_model_better) + RMSE_population_models * sum(~mask_individual_model_better)) / len(mask_individual_model_better)
                            
                            # save results
                            ## ASSEMBLE OUTPUT RESULT
                            result = []
                            result.append(datetime.datetime.now().strftime('%d-%m-%Y-%H:%M:%S'))
                            result.append(task)
                            result.append(metric)
                            result.append(selected_features)
                            result.append(number_of_features)
                            result.append(algorithm)
                            result.append(demo_group)
                            result.append(RMSE_individual_models)
                            result.append(sum(mask_individual_model_better))
                            result.append(RMSE_population_models)
                            result.append(sum(~mask_individual_model_better))
                            result.append(RMSE_total)
                            result.append(start_periods[time_index])
                            result.append(end_periods[time_index])
                            
                            dbHandler = Database_Handler.Get_DB_Handler()
                    
                            run_fields = ['timestamp', 'prediction_task', 'selected_metric', 'selected_features', 
                                          'number_of_features', 'selected_algorithm', 'demo_group', 
                                          'RMSE_ind', 'RMSE_ind_fraction','RMSE_pop', 'RMSE_pop_fraction', 'RMSE_total',
                                          'start_time', 'end_time']
                            if IS_PER_DAY_PERIOD:
                                dbHandler.insert("Candidate_Population_Models_Performance_Daily", run_fields, result)
                            else:
                                dbHandler.insert("Candidate_Population_Models_Performance", run_fields, result)
                            
                            print ("DONE: Task: %s | Metric: %s | Algorithm: %s | "
                                   "Demo: %s | Period: %s-%s") % (task, metric, algorithm, demo_group,
                                                                  start_periods[time_index], 
                                                                  end_periods[time_index])
## 
# Entry point of the script
##
if __name__ == "__main__":
    
    compareCandidatePopulationModels()