#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy
import scipy.io
import datetime
import math
import array
import sys
import Mobility_Features_Prediction
from MetricResults import MetricResults
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

def Run_Analysis(evaluation_run):        

    ground_truth = evaluation_run.ground_truth
    
    # predictions
    prediction = evaluation_run.prediction
    random_prediction = evaluation_run.random_prediction
    histogram_class_prediction = evaluation_run.histogram_class_prediction
    dominating_class_prediction = evaluation_run.dominating_class_prediction
    
    metric_results = MetricResults()
    
    if len(prediction) > 0:
        # frequency of the most frequent element
        for most_common_number, most_common_frequency in Counter(ravel(ground_truth)).most_common(1):
            count_of_most_common_element = most_common_frequency
            most_common_element = most_common_number
    
        # calculate confusion matrix
        confusion_matrix = Compute_Custom_Confusion_Matrix(ground_truth, prediction)
        random_confusion_matrix = Compute_Custom_Confusion_Matrix(ground_truth, random_prediction)
        histogram_class_confusion_matrix = Compute_Custom_Confusion_Matrix(ground_truth, histogram_class_prediction)
        dominating_class_confusion_matrix = Compute_Custom_Confusion_Matrix(ground_truth, dominating_class_prediction)
        
        # get metric results
        if evaluation_run.task == EvaluationRun.task_next_slot_transition or evaluation_run.task == EvaluationRun.task_next_slot_transition_daily:
            metric_results = Compute_Metric_for_Specific_Class(1, confusion_matrix, evaluation_run)
            random_metric_results = Compute_Metric_for_Specific_Class(1, random_confusion_matrix, evaluation_run)
            histogram_class_metric_results = Compute_Metric_for_Specific_Class(1, histogram_class_confusion_matrix, evaluation_run)
            dominating_class_metric_results = Compute_Metric_for_Specific_Class(1, dominating_class_confusion_matrix, evaluation_run)
        else:
            metric_results = Compute_Metric_Aggregated(confusion_matrix, evaluation_run)
            random_metric_results = Compute_Metric_Aggregated(random_confusion_matrix, evaluation_run)
            histogram_class_metric_results = Compute_Metric_Aggregated(histogram_class_confusion_matrix, evaluation_run)
            dominating_class_metric_results = Compute_Metric_Aggregated(dominating_class_confusion_matrix, evaluation_run)
        
        metric_results.frequency_of_top_class = (float(count_of_most_common_element) / confusion_matrix.number_of_predictions)
        metric_results.kappa_metric = evaluation_run.selected_metric
        
        metric_results = Check_Metric_Values(metric_results)
        histogram_class_metric_results = Check_Metric_Values(histogram_class_metric_results)
        dominating_class_metric_results = Check_Metric_Values(dominating_class_metric_results)
      
        ## kappa -- accuracy
        if evaluation_run.selected_metric == EvaluationRun.metric_accuracy:
            kappa_random = (metric_results.accuracy - random_metric_results.accuracy) / (1 - random_metric_results.accuracy)
            kappa_histogram = (metric_results.accuracy - histogram_class_metric_results.accuracy) / (1 - histogram_class_metric_results.accuracy)
            kappa_dominating = (metric_results.accuracy - dominating_class_metric_results.accuracy) / (1 - dominating_class_metric_results.accuracy)
        ## kappa -- fscore
        if evaluation_run.selected_metric == EvaluationRun.metric_fscore:
            kappa_random = (metric_results.fscore - random_metric_results.fscore) / (1 - random_metric_results.fscore)
            kappa_histogram = (metric_results.fscore - histogram_class_metric_results.fscore) / (1 - histogram_class_metric_results.fscore)
            kappa_dominating = (metric_results.fscore - dominating_class_metric_results.fscore) / (1 - dominating_class_metric_results.fscore)
        ## kappa -- MCC
        if evaluation_run.selected_metric == EvaluationRun.metric_MCC:
            kappa_random = (metric_results.MCC - random_metric_results.MCC) / (1 - random_metric_results.MCC)
            kappa_histogram = (metric_results.MCC - histogram_class_metric_results.MCC) / (1 - histogram_class_metric_results.MCC)
            kappa_dominating = (metric_results.MCC - dominating_class_metric_results.MCC) / (1 - dominating_class_metric_results.MCC)
        
        metric_results.kappa_random = kappa_random
        metric_results.kappa_histogram = kappa_histogram
        metric_results.kappa_dominating = kappa_dominating
               
    metric_results = Check_Metric_Values(metric_results)                    
    
    return metric_results


def Compute_Custom_Confusion_Matrix(ground_truth, prediction):

    ground_truth = ravel(ground_truth)
    prediction = ravel(prediction)
    
    # compute confusion matrix
    conf_matrix = confusion_matrix(ground_truth, prediction)
    number_of_classes = conf_matrix.shape[0]
    number_of_done_predictions = conf_matrix.sum()
    
    # metrics
    true_positives = numpy.zeros((number_of_classes, 1))
    true_negatives = numpy.zeros((number_of_classes, 1))
    false_positives = numpy.zeros((number_of_classes, 1))
    false_negatives = numpy.zeros((number_of_classes, 1))
    accuracy = numpy.zeros((number_of_classes, 1))
    precision = numpy.zeros((number_of_classes, 1))
    recall = numpy.zeros((number_of_classes, 1))
    f1 = numpy.zeros((number_of_classes, 1))
    mcc = numpy.zeros((number_of_classes, 1))
    kappa = numpy.zeros((number_of_classes, 1))
      
    for i in range(number_of_classes):
        true_positives[i] = conf_matrix[i,i]
        false_positives[i] = sum(conf_matrix[:,i]) - conf_matrix[i,i]
        false_negatives[i] = sum(conf_matrix[i,:]) - conf_matrix[i,i]
        true_negatives[i] = number_of_done_predictions - (true_positives[i] + false_positives[i] + false_negatives[i])
          
        accuracy[i] = ((true_positives[i] + true_negatives[i]) / (true_positives[i] + true_negatives[i] + false_positives[i] + false_negatives[i]))
        precision[i] = (true_positives[i] / (true_positives[i] + false_positives[i]))
        recall[i] = (true_positives[i] / (true_positives[i] + false_negatives[i]))
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        mcc[i] = Compute_MCC_Metric(true_positives[i], true_negatives[i], false_positives[i], false_positives[i])
        
#         Pr_e = ((sum(conf_matrix[i,:]) / number_of_done_predictions) * (sum(conf_matrix[:,i]) / number_of_done_predictions) 
#                 + (1 - (sum(conf_matrix[i,:]) / number_of_done_predictions)) * (1 - (sum(conf_matrix[:,i]) / number_of_done_predictions)))
        Pr_e = (((true_negatives[i] + false_positives[i]) * (true_negatives[i] + false_negatives[i]) 
                + (false_negatives[i] + true_positives[i]) * (false_positives[i] + true_positives[i]))
                / (number_of_done_predictions * number_of_done_predictions))                 
        
        
        kappa[i] = (accuracy[i] - Pr_e) / (1 - Pr_e)

    counts = conf_matrix.sum(axis=1)
    class_distribution = counts / float(number_of_done_predictions);
    
    total_true_positive = true_positives.sum()
    total_true_negative = true_negatives.sum()
    total_false_positive = false_positives.sum()
    total_false_negative = false_negatives.sum()
    
    # assemble 
    my_confusion_matrix = MyConfusionMatrix()
    my_confusion_matrix.TP = true_positives
    my_confusion_matrix.TN = true_negatives
    my_confusion_matrix.FP = false_positives
    my_confusion_matrix.FN = false_negatives
    
    my_confusion_matrix.accuracy = accuracy
    my_confusion_matrix.precision = precision
    my_confusion_matrix.recall = recall
    my_confusion_matrix.fscore = f1
    my_confusion_matrix.kappa = kappa
    
    
    REPLACE_VALUE = 1
    new_precision = ravel(precision)
    mask_nans_precision = isnan(new_precision)
    new_precision[mask_nans_precision] = REPLACE_VALUE 
     
    new_recall = ravel(recall)
    mask_nans_recall = isnan(new_recall)
    new_recall[mask_nans_recall] = REPLACE_VALUE
     
    new_fscore = ravel(f1)
    mask_nans_fscore = isnan(new_fscore)
    new_fscore[mask_nans_fscore] = REPLACE_VALUE
     
    my_confusion_matrix.total_accuracy = sum(ravel(accuracy) * class_distribution)
    my_confusion_matrix.total_precision = sum(new_precision * class_distribution)
    my_confusion_matrix.total_recall = sum(new_recall * class_distribution)
    my_confusion_matrix.total_fscore = 2 * (my_confusion_matrix.total_precision * my_confusion_matrix.total_recall) / (my_confusion_matrix.total_precision + my_confusion_matrix.total_recall)
    my_confusion_matrix.total_kappa = sum(ravel(kappa) * class_distribution)

    new_mcc = ravel(mcc)
    mask_nans_mcc = isnan(new_mcc)
    new_mcc[mask_nans_mcc] = 0

    # numpy accuracy
    numpy_accuracy = accuracy_score(ground_truth, prediction)
    my_confusion_matrix.total_accuracy = numpy_accuracy
    if my_confusion_matrix.total_accuracy == 0 or ~numpy.isfinite(my_confusion_matrix.total_accuracy):
        my_confusion_matrix.total_precision = 0
        my_confusion_matrix.total_recall = 0
        my_confusion_matrix.total_fscore = 0
        
    # numpy precision, recall, fscore
#     numpy_metrics = precision_recall_fscore_support(ground_truth, prediction, average='weighted')
#     my_confusion_matrix.total_precision = numpy_metrics[0]
#     my_confusion_matrix.total_recall = numpy_metrics[1]
#     my_confusion_matrix.total_fscore = numpy_metrics[2]
    my_confusion_matrix.total_MCC = sum(new_mcc * class_distribution)
    
    my_confusion_matrix.number_of_classes = number_of_classes
    my_confusion_matrix.number_of_predictions = number_of_done_predictions
    
    # print "%s / %s / %s -- %s / %s -- %s / %s -- %s / %s" % (a, numpy_accuracy, my_confusion_matrix.total_accuracy, numpy_metrics[0], my_confusion_matrix.total_precision, numpy_metrics[1], my_confusion_matrix.total_recall, numpy_metrics[2], my_confusion_matrix.total_fscore) 
    
    return my_confusion_matrix


def Compute_Metric_Aggregated(confusion_matrix, evaluation_run):    
    # assemble result
    metric_results = MetricResults()
    
    metric_results.selected_features = evaluation_run.selected_features
    metric_results.accuracy = confusion_matrix.total_accuracy
    metric_results.precision = confusion_matrix.total_precision
    metric_results.recall = confusion_matrix.total_recall
    metric_results.fscore = confusion_matrix.total_fscore
    metric_results.MCC = confusion_matrix.total_MCC
    metric_results.kappa = confusion_matrix.total_kappa
    
    return metric_results


    
def Compute_Metric_for_Specific_Class(class_id, confusion_matrix, evaluation_run):

    metric_results = MetricResults()
    metric_results.selected_features = evaluation_run.selected_features
    metric_results.accuracy = confusion_matrix.accuracy[class_id][0]
    metric_results.precision = confusion_matrix.precision[class_id][0]
    metric_results.recall = confusion_matrix.recall[class_id][0]
    metric_results.fscore = confusion_matrix.fscore[class_id][0]
    metric_results.kappa = confusion_matrix.kappa[class_id][0]
    
    metric_results.accuracy = ((confusion_matrix.TP[class_id] + confusion_matrix.TN[class_id]) / (confusion_matrix.TP[class_id] + confusion_matrix.TN[class_id] + confusion_matrix.FP[class_id] + confusion_matrix.FN[class_id]))[0]
    metric_results.precision = (confusion_matrix.TP[class_id] / (confusion_matrix.TP[class_id] + confusion_matrix.FP[class_id]))[0]
    metric_results.recall = (confusion_matrix.TP[class_id] / (confusion_matrix.TP[class_id] + confusion_matrix.FN[class_id]))[0]
    metric_results.fscore = (2 * metric_results.precision * metric_results.recall / (metric_results.precision + metric_results.recall))
    
    metric_results.MCC = Compute_MCC_Metric(confusion_matrix.TP[class_id][0], confusion_matrix.TN[class_id][0], confusion_matrix.FP[class_id][0], confusion_matrix.FN[class_id][0])    
    
    return metric_results

        
def Check_Metric_Values(metric_results):
    
    if ~numpy.isfinite(metric_results.accuracy):
        metric_results.accuracy = 0
    
    if ~numpy.isfinite(metric_results.precision):
        metric_results.precision = 0
    
    if ~numpy.isfinite(metric_results.recall):
        metric_results.recall = 0
    
    if ~numpy.isfinite(metric_results.fscore):
        metric_results.fscore = 0
    
    if ~numpy.isfinite(metric_results.kappa):
        metric_results.kappa = 0
    else:
        metric_results.kappa = min(metric_results.kappa, 1);
        
    if ~numpy.isfinite(metric_results.kappa_random):
        metric_results.kappa_random = 0
    
    if ~numpy.isfinite(metric_results.kappa_histogram):
        metric_results.kappa_histogram = 0
        
    if ~numpy.isfinite(metric_results.kappa_dominating):
        metric_results.kappa_dominating = 0
    
    if ~numpy.isfinite(metric_results.MCC):
        metric_results.MCC = 0
        
    if metric_results.relative_prediction_error != None and ~numpy.isfinite(metric_results.relative_prediction_error):
        metric_results.relative_prediction_error = 0
        
    if metric_results.absolute_prediction_error != None and ~numpy.isfinite(metric_results.absolute_prediction_error):
        metric_results.absolute_prediction_error = 0
    
    return metric_results


def Compute_MCC_Metric(TP, TN, FP, FN):
    
    return (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))


def Compute_Kappa_Metric(your_system_metric, base_line_metric):
    
    return (your_system_metric - base_line_metric) / (1 - base_line_metric)


def Save_To_DB(evaluation_run, is_final):
    # save result
    fields = ['run_id','feature_combination','final','accuracy','precis','recall','fscore','kappa_random','kappa_histogram', 'kappa_dominating', 'mcc','frequency_of_top_class']
    metric_results = evaluation_run.metric_results
    
    values = []
    values.append(evaluation_run.run_id)
    values.append(', '.join(str(x) for x in list(metric_results.selected_features)))
    values.append(is_final)
    values.append(metric_results.accuracy)
    values.append(metric_results.precision)
    values.append(metric_results.recall)
    values.append(metric_results.fscore)
    values.append(metric_results.kappa_random)
    values.append(metric_results.kappa_histogram)
    values.append(metric_results.kappa_dominating)
    values.append(metric_results.MCC)
    values.append(metric_results.frequency_of_top_class)
    
    dbHandler = Mobility_Features_Prediction.Get_DB_Handler() 
    dbHandler.insert("%s_Prediction_Result_Analysis" % (evaluation_run.task), fields, values)
      