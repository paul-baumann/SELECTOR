#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class computes feature ranks out of 
# all individual models.
#
# copyright Paul Baumann
#############################################

from EvaluationRun import EvaluationRun
import Database_Handler
import Util
import numpy

from pylab import ravel

IS_PER_DAY_PERIOD = False

def createFeatureRanks():
    
    list_of_metrics = [EvaluationRun.metric_accuracy, EvaluationRun.metric_fscore, EvaluationRun.metric_MCC]
    
    if IS_PER_DAY_PERIOD:
        start_periods = [1, 49, 69];
        end_periods = [48, 68, 96];
        tasks = [EvaluationRun.task_next_place_daily]
    else:
        start_periods = [1];
        end_periods = [96];
        tasks = [EvaluationRun.task_next_place, EvaluationRun.task_next_slot_place, EvaluationRun.task_next_slot_transition]
    
    NUMBER_OF_FEATURES = 54
    
    for task in tasks:
        for metric in list_of_metrics:
            
            for time_index in range(len(start_periods)):
                
                dbHandler = Database_Handler.Get_DB_Handler()
                query = ("select a.user_id, a.%s, a.selected_algorithm, a.selected_features from "
                         "(select user_id, %s, selected_algorithm, selected_features from PostSelection_%s_Prediction "
                         "where is_context = 0 and selected_metric = '%s' and feature_selection_id > 0 and "
                         "(selected_algorithm = 'knn_dyn' or selected_algorithm = 'svm' or "
                         "selected_algorithm = 'perceptron' or selected_algorithm = 'decision_tree') "
                         "AND start_time = %s and end_time = %s "
                         "order by %s DESC) as a group by a.user_id order by a.user_id;"
                         "") % (metric, metric, task, metric, start_periods[time_index], end_periods[time_index], metric)
                
                query_results = numpy.array(dbHandler.select(query))
                
                for demo_key in get_demo_groups_dict().keys():
                    
                    feature_ids = numpy.arange(0,NUMBER_OF_FEATURES,1)
                    feature_occurrence = numpy.zeros((NUMBER_OF_FEATURES,))
                    total_number_of_models = 0
                    
                    # For each user
                    for row in query_results:
                        user_id = int(row[0])
                        
                        # Apply the following steps only if the user belongs to the current demographic group
                        if Util.userBelongToDemoGroup(user_id, demo_key):
                            selected_features = ravel(numpy.fromstring(row[3], sep=', ').astype(int))
                            feature_occurrence[selected_features] += 1
                            total_number_of_models += 1
                    
                    ## ASSEMBLE OUTPUT RESULT
                    result = numpy.empty((NUMBER_OF_FEATURES, 9),dtype='|S50')
                    result[:,0] = task
                    result[:,1] = metric
                    
                    sort_idx = numpy.argsort(feature_occurrence)[::-1]
                    result[:,2] = feature_ids[sort_idx]
                    result[:,3] = numpy.arange(1, NUMBER_OF_FEATURES + 1, 1)
                    result[:,4] = feature_occurrence[sort_idx]
                    
                    result[:,5] = total_number_of_models
                    result[:,6] = demo_key
                    result[:,7] = start_periods[time_index]
                    result[:,8] = end_periods[time_index]    
                    
                    dbHandler = Database_Handler.Get_DB_Handler()
            
                    run_fields = ['prediction_task', 'metric', 'feature_id', 'feature_rank', 'number_of_individual_models', 
                                  'total_number_of_models', 'demo_group', 'start_time', 'end_time']
                    if IS_PER_DAY_PERIOD:
                        dbHandler.insertMany("FeatureRanksDaily", run_fields, result)
                    else:
                        dbHandler.insertMany("FeatureRanks", run_fields, result)
                    
                    print ("DONE: Task: %s | Metric: %s | Demo: %s") % (task, metric, demo_key)



def get_demo_groups_dict():
    
    demo_groups = {'all': 0, 'female': 1, 'male': 2, 'working': 3, 'study': 4,
                   'age_group_16_21': 5, 'age_group_22_27': 6, 'age_group_28_33': 7,
                    'age_group_34_38': 8, 'age_group_39_44': 9, 'no_children_all': 10, 
                    'with_children_all': 11, 'with_children_female': 12, 'with_children_male': 13, 
                    'single': 14, 'family': 15}
    
    return demo_groups    
        
## 
# Entry point of the script
##
if __name__ == "__main__":
    
    createFeatureRanks()