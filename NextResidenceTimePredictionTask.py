#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy
import scipy.io
import datetime
import math
import array
import sys
from MetricResults import MetricResults
import Mobility_Features_Prediction
import NextPlaceOrSlotPredictionTask
from MyConfusionMatrix import MyConfusionMatrix
from EvaluationRun import EvaluationRun
from time import time
from collections import Counter
from datetime import datetime
from operator import mul
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from pylab import *

#class NextResidenceTimePredictionTask:
MINUTES_PER_SLOT = 15
    
def Run_Analysis(evaluation_run): 
    
    ground_truth = evaluation_run.ground_truth
    prediction = evaluation_run.prediction
    random_prediction = evaluation_run.random_prediction
    histogram_class_prediction = evaluation_run.histogram_class_prediction
    dominating_class_prediction = evaluation_run.dominating_class_prediction
    
    metric_results = MetricResults()
    
    if len(ground_truth) > 0:
        # predictors' results
        relative_prediction_error = sum(prediction / ground_truth) / len(ground_truth)
        absolute_prediction_error = sum(abs(prediction - ground_truth) * MINUTES_PER_SLOT) / len(ground_truth)
        random_relative_prediction_error = sum(random_prediction / ground_truth) / len(ground_truth)
        random_absolute_prediction_error = sum(abs(random_prediction - ground_truth) * MINUTES_PER_SLOT) / len(ground_truth)
        histogram_relative_prediction_error = sum(histogram_class_prediction / ground_truth) / len(ground_truth)
        histogram_absolute_prediction_error = sum(abs(histogram_class_prediction - ground_truth) * MINUTES_PER_SLOT) / len(ground_truth)
        dominating_relative_prediction_error = sum(dominating_class_prediction / ground_truth) / len(ground_truth)
        dominating_absolute_prediction_error = sum(abs(dominating_class_prediction - ground_truth) * MINUTES_PER_SLOT) / len(ground_truth)
        
        # assemble result
        metric_results.selected_features = evaluation_run.selected_features
        metric_results.relative_prediction_error = relative_prediction_error
        metric_results.absolute_prediction_error = absolute_prediction_error  * -1
        
        ## kappa -- absolute_prediction_error
        if evaluation_run.selected_metric == EvaluationRun.metric_absolute_prediction_error:
            kappa_random = (random_absolute_prediction_error - absolute_prediction_error) / (random_absolute_prediction_error)
            kappa_histogram = (histogram_absolute_prediction_error - absolute_prediction_error) / (histogram_absolute_prediction_error)
            kappa_dominating = (dominating_absolute_prediction_error - absolute_prediction_error) / (dominating_absolute_prediction_error)
        ## kappa -- relative_prediction_error
        if evaluation_run.selected_metric == EvaluationRun.metric_relative_prediction_error:
            kappa_random = (random_relative_prediction_error - relative_prediction_error) / (random_relative_prediction_error)
            kappa_histogram = (histogram_relative_prediction_error - relative_prediction_error) / (histogram_relative_prediction_error)
            kappa_dominating = (dominating_relative_prediction_error - relative_prediction_error) / (dominating_relative_prediction_error)
        
        metric_results.kappa_random = kappa_random
        metric_results.kappa_histogram = kappa_histogram
        metric_results.kappa_dominating = kappa_dominating
    else:
        metric_results.absolute_prediction_error = -9999999
        metric_results.relative_prediction_error = -9999999
            
    metric_results = NextPlaceOrSlotPredictionTask.Check_Metric_Values(metric_results) 
    
    return metric_results


def Save_To_DB(evaluation_run, is_final):
    # save result
    fields = ['run_id','feature_combination','final','mean_relative_prediction_error','mean_absolute_prediction_error','kappa_random','kappa_median_predictor', 'kappa_dominating']
    metric_results = evaluation_run.metric_results
    
    values = []
    values.append(evaluation_run.run_id)
    values.append(', '.join(str(x) for x in list(metric_results.selected_features)))
    values.append(is_final)
    values.append(metric_results.relative_prediction_error)
    values.append(metric_results.absolute_prediction_error * -1)
    values.append(metric_results.kappa_random)
    values.append(metric_results.kappa_histogram)
    values.append(metric_results.kappa_dominating)
    
    dbHandler = Mobility_Features_Prediction.Get_DB_Handler() 
    dbHandler.insert("%s_Prediction_Result_Analysis" % (evaluation_run.task), fields, values)