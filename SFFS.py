#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class is an implementation of the 
# SFFS algorithm to select features.
#
# copyright Paul Baumann
#############################################

import thread
import threading
import STEP1_feature_selection
from EvaluationRun import EvaluationRun
from PredictorsPipeline import PredictorsPipeline
from MetricResults import MetricResults
from time import time

from pylab import *

class SFFS:
    
    def __init__ (self, evaluation_run, max_number_of_features, start):
        
        self.max_number_of_features = max_number_of_features
        self.evaluation_run = evaluation_run;
        self.start = start

        
    def Run_SFFS(self):
        
        BETTER_CLASS_THRESHOLD = 0.005
        counter = 0
        selected_features = []
        remaining_features = list(self.evaluation_run.available_features)
        
        over_all_best_performance = 0
        over_all_best_metric_results = MetricResults()
        
        while len(selected_features) < self.max_number_of_features and counter < self.max_number_of_features + 5 and len(remaining_features) > 0:
            number_of_remaining_features = len(remaining_features)
            all_metric_results = [MetricResults()] * number_of_remaining_features
            metric_result_per_feature = [0] * number_of_remaining_features
            threads = []
            
            best_metric_results = None
            counter = counter + 1
            
            for current_index in range(number_of_remaining_features):                
                feature = remaining_features[current_index]
                # create a copy
                evaluation_run = EvaluationRun()
                evaluation_run.copy(self.evaluation_run)
                # prepare feature subset
                evaluation_run.selected_features = list(selected_features)
                evaluation_run.selected_features.append(feature)
                
                if STEP1_feature_selection.THREAD_LEVEL > 3:
                    eval_thread = threading.Thread( target=self.Thread_Eval, args=(evaluation_run, 
                                                                                   metric_result_per_feature, 
                                                                                   all_metric_results, 
                                                                                   current_index, ) )
                    threads.append(eval_thread)
                    eval_thread.start()
                else:
                    self.Thread_Eval(evaluation_run, metric_result_per_feature, all_metric_results, current_index)
                                                    
            if STEP1_feature_selection.THREAD_LEVEL > 3:
                for thread in threads:
                    thread.join()
            
            # select and append the best performed feature
            best_performance = max(metric_result_per_feature)
            index_best_feature = metric_result_per_feature.index(best_performance)
            best_feature = remaining_features[index_best_feature]
            best_metric_results = all_metric_results[index_best_feature]
            
            if best_performance > over_all_best_performance + BETTER_CLASS_THRESHOLD: 
                over_all_best_metric_results = best_metric_results
                over_all_best_performance = best_performance
            
            # change feature subsets
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            
            # run backward selection only if at least three features were selected
            search_backward = True
            
            while len(selected_features) > 2 and search_backward:
                number_of_remaining_features = len(selected_features)
                all_metric_results = [MetricResults()] * number_of_remaining_features
                metric_result_per_feature = [0] * number_of_remaining_features
                threads = []
                
                for current_index in range(number_of_remaining_features):
                    feature = selected_features[current_index]
                    # skip recently added feature
                    if feature == best_feature:
                        metric_result_per_feature[current_index] = best_performance
                        all_metric_results[current_index] = best_metric_results
                        continue
                    
                    # create a copy
                    evaluation_run = EvaluationRun()
                    evaluation_run.copy(self.evaluation_run)
                    # assemble feature subset
                    evaluation_run.selected_features = list(selected_features)
                    evaluation_run.selected_features.remove(feature)
                    # run prediction and evaluation
                    
                    if STEP1_feature_selection.THREAD_LEVEL > 3:
                        eval_thread = threading.Thread( target=self.Thread_Eval, args=(evaluation_run, 
                                                                                   metric_result_per_feature, 
                                                                                   all_metric_results, 
                                                                                   current_index, ) )
                        threads.append(eval_thread)
                        eval_thread.start()  
                    else:
                        self.Thread_Eval(evaluation_run, metric_result_per_feature, all_metric_results, current_index)                  
                
                if STEP1_feature_selection.THREAD_LEVEL > 3:
                    for thread in threads:
                        thread.join()
                    
                new_best_performance = max(metric_result_per_feature)
                if new_best_performance > best_performance:
                    index_worst_feature = metric_result_per_feature.index(new_best_performance)
                    worst_feature = selected_features[index_worst_feature]
                    selected_features.remove(worst_feature)
                    remaining_features.append(worst_feature)
                    best_metric_results = all_metric_results[index_worst_feature]
                    
                    if new_best_performance > over_all_best_performance + BETTER_CLASS_THRESHOLD: 
                        over_all_best_metric_results = best_metric_results    
                        over_all_best_performance = new_best_performance
                    if len(selected_features) == 1:
                        search_backward = False
                else:
                    search_backward = False
        
            ## DEBUG
            if STEP1_feature_selection.DEBUG_LEVEL > 3:
                current_metric = evaluation_run.selected_metric
                current_algorithm = evaluation_run.selected_algorithm
                current_task = evaluation_run.task
                user = evaluation_run.userData.userId
                print("Run %i DONE with metric: %s, algorithm: %s, task: %s, user: %s, day period: %s-%s, after : %s seconds" % (counter, 
                                                                                                                              current_metric, 
                                                                                                                              current_algorithm, 
                                                                                                                              current_task, 
                                                                                                                              user, 
                                                                                                                              evaluation_run.start_time,
                                                                                                                              evaluation_run.end_time,
                                                                                                                              time() - self.start))
            
            self.evaluation_run.selected_features = selected_features
            self.evaluation_run.metric_results = best_metric_results
            self.evaluation_run.task_object.Save_To_DB(self.evaluation_run, False)
        
        self.evaluation_run.selected_features = over_all_best_metric_results.selected_features
        self.evaluation_run.metric_results = over_all_best_metric_results
        self.evaluation_run.task_object.Save_To_DB(self.evaluation_run, True)

            
    def Thread_Eval(self, evaluation_run, metric_result_per_feature, all_metric_results, index):
        # run prediction and evaluation
        predictors_pipeline = PredictorsPipeline(evaluation_run)
        evaluation_run = predictors_pipeline.Run_Predictions()
        
        # run performance evaluation
        metric_results = self.evaluation_run.task_object.Run_Analysis(evaluation_run)
        # save performance metric result

        metric_result_per_feature[index] = self.Get_Metric_Result(evaluation_run, metric_results)
        all_metric_results[index] = metric_results    
        
    
    
    def Get_Metric_Result(self, evaluation_run, metric_results):
        
        if evaluation_run.selected_metric == EvaluationRun.metric_accuracy:
            return metric_results.accuracy
        if evaluation_run.selected_metric == EvaluationRun.metric_fscore:
            return metric_results.fscore
        if evaluation_run.selected_metric == EvaluationRun.metric_MCC:
            return metric_results.MCC
        