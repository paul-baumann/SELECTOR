#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class prints the performance achieved
# by the candidate individual models.
#
# copyright Paul Baumann
#############################################

import matplotlib as mlt
SHOW = False
SAVE = True

if SAVE:
    mlt.use('Agg')
    
import seaborn as sns
from pylab import *

from EvaluationRun import EvaluationRun
import Database_Handler
import numpy
import Util

def visualize_population_model_performance():
    
    list_of_metrics = [EvaluationRun.metric_accuracy, EvaluationRun.metric_fscore, EvaluationRun.metric_MCC]
    tasks = [EvaluationRun.task_next_slot_place, EvaluationRun.task_next_slot_transition, EvaluationRun.task_next_place]
    
    boxplot_data = numpy.zeros((141,len(list_of_metrics) * len(list_of_metrics)))
    
    idx = 0;
    
    for task in tasks:
        for metric in list_of_metrics:
            
            # Get population models
            dbHandler = Database_Handler.Get_DB_Handler()
            query = ("select selected_features, selected_algorithm from Candidate_Population_Models_Performance "
                     "where demo_group = 'all' and prediction_task = '%s' and selected_metric = '%s' "
                     "order by RMSE_total, number_of_features limit 1;"
                     "") % (task, metric)
            
            query_population_model = numpy.array(dbHandler.select(query))
            selected_features = query_population_model[0,0]
            algorithm = query_population_model[0,1]
            
            # Get population model performance
            dbHandler = Database_Handler.Get_DB_Handler()
            query = ("select %s from Candidate_Population_Models where demo_group = 'all' "
                     "and prediction_task = '%s' and selected_metric = '%s' and selected_features = '%s' "
                     "and selected_algorithm = '%s' order by user_id;"
                     "") % (metric, task, metric, selected_features, algorithm)
            
            query_results_population_performance = numpy.array(dbHandler.select(query))
            population_models_performance = query_results_population_performance[:,0]
            
            boxplot_data[:,idx] = population_models_performance
            idx += 1
            
            print ("DONE: Task: %s | Metric: %s") % (task, metric)
    
    boxplot_data = numpy.fliplr(boxplot_data)
    
    #===================================================================
    # VISUALIZE
    #===================================================================
    sns.set(font_scale = 2.5)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(13, 4))
    axis([0, 1, 0, len(list_of_metrics) * len(list_of_metrics)])

    bp = boxplot(boxplot_data, vert=0, whis=[5,95]) 
    Util.adjust_boxplot(bp)
    
    ax.set_xlabel('Performance result')
    ax.set_xticks(numpy.arange(0,1.01,.1))
    ax.set_xticklabels(['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'])
    ax.set_yticklabels(Util.TASK_METRIC_LABELS[::-1])
    
    subplots_adjust(bottom=0.2, left=0.19, top=0.97, right=0.96)
    
    if SHOW:
        show() 
    
    if SAVE:
        path = ('plots/PopulationModel_Performance_boxplot_brute_force.pdf')
        plt.savefig(path) 
## 
# Entry point of the script
##
if __name__ == "__main__":
    
    visualize_population_model_performance()