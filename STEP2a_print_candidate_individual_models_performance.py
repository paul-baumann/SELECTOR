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

def printPerformance():
    
    IS_BASELINE = False
    
    list_of_metrics = [EvaluationRun.metric_accuracy, EvaluationRun.metric_fscore, EvaluationRun.metric_MCC]
    tasks = [EvaluationRun.task_next_place, EvaluationRun.task_next_slot_place, EvaluationRun.task_next_slot_transition]

    if IS_BASELINE:
        algorithms = [EvaluationRun.alg_random, EvaluationRun.alg_histogram, EvaluationRun.alg_majority]
        is_feature_selection_array = ['=']
        is_phone_context_array = [1]
    else:
        algorithms = [EvaluationRun.alg_knn_dyn, EvaluationRun.alg_perceptron, EvaluationRun.alg_decision_tree, EvaluationRun.alg_svm]
        is_feature_selection_array = ['=', '>']
        is_phone_context_array = [1,0]
    
    for task in tasks:
        for metric in list_of_metrics:
            for algorithm in algorithms:
                for is_feature_selection in is_feature_selection_array:
                    for is_phone_context in is_phone_context_array: 
                
                        dbHandler = Database_Handler.Get_DB_Handler()
                        query = ("select ROUND(100 * avg(%s)), count(*) from PostSelection_%s_Prediction where selected_algorithm = '%s' and is_network = 1 "
                                 "and selected_metric = '%s' and is_context = %s and feature_selection_id %s 0 and "
                                  "selected_features NOT LIKE '%%No%%';") % (metric, task, algorithm, metric, is_phone_context, is_feature_selection)
                        
                        performance = dbHandler.select(query)
                        if performance:
                            print ("Task: %s | Metric: %s | Algo: %s | FS: %s 0 | PC: %s | Per: %s") % (task,
                                                                                                                   metric,
                                                                                                                   algorithm,
                                                                                                                   is_feature_selection,
                                                                                                                   is_phone_context,
                                                                                                                   performance[0])
                
    
## 
# Entry point of the script
##
if __name__ == "__main__":
    
    printPerformance()