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

def visualize_demo_daily_population_models_performance():
    
    list_of_metrics = [EvaluationRun.metric_accuracy, EvaluationRun.metric_fscore, EvaluationRun.metric_MCC]
    tasks = [EvaluationRun.task_next_place_daily, EvaluationRun.task_next_place_daily, EvaluationRun.task_next_place_daily]
    
    start_periods = [1, 49, 69];
    end_periods = [48, 68, 96];
        
    data_array = []
#     data_array.append(numpy.zeros((3, len(Util.demo_groups))))
#     data_array.append(numpy.zeros((3, len(Util.demo_groups))))
#     data_array.append(numpy.zeros((3, len(Util.demo_groups))))
    
    for run_idx, task in enumerate(tasks):
        metric = list_of_metrics[run_idx]
                
        # Matrix to store the results
        data = numpy.zeros((3, len(Util.demo_groups)))
            
        ## Compare to each demographic group
        for group_idx, demo_group in enumerate(Util.demo_groups): #
              
            # Get population models -- all demographics
            dbHandler = Database_Handler.Get_DB_Handler()
            query = ("select selected_features, selected_algorithm from Candidate_Population_Models_Performance "
                     "where demo_group = '%s' and prediction_task = '%s' and selected_metric = '%s' "
                     "order by RMSE_total, number_of_features limit 1;"
                     "") % (demo_group, task[5:], metric)
                
            query_population_model = numpy.array(dbHandler.select(query))
            selected_features = query_population_model[0,0]
            algorithm = query_population_model[0,1]
                
            # Get population model performance -- all demographics
            dbHandler = Database_Handler.Get_DB_Handler()
            query = ("select %s from Candidate_Population_Models where demo_group = '%s' "
                     "and prediction_task = '%s' and selected_metric = '%s' and selected_features = '%s' "
                     "and selected_algorithm = '%s' order by user_id;"
                     "") % (metric, demo_group, task[5:], metric, selected_features, algorithm)
                
            query_results_population_performance = numpy.array(dbHandler.select(query))
            population_models_performance = query_results_population_performance[:,0]
              
            day_periods_performance = numpy.zeros((len(population_models_performance),))
            day_periods_amount_of_data = numpy.zeros((len(population_models_performance),))
              
            # Get population models for each day period  
            for time_index in range(len(start_periods)):                  
                  
                # Get population DAY PERIOD models -- selected demo group
                dbHandler = Database_Handler.Get_DB_Handler()
                query = ("select selected_features, selected_algorithm from Candidate_Population_Models_Performance_Daily "
                         "where demo_group = '%s' and prediction_task = '%s' and selected_metric = '%s' "
                         "and start_time = %s and end_time = %s "
                         "order by RMSE_total, number_of_features limit 1;"
                         "") % (demo_group, task, metric, start_periods[time_index], end_periods[time_index])
                    
                query_population_model = numpy.array(dbHandler.select(query))
                selected_features = query_population_model[0,0]
                algorithm = query_population_model[0,1]
                    
                # Get population DAY PERIOD model performance -- selected demo group
                dbHandler = Database_Handler.Get_DB_Handler()
                query = ("select %s, number_of_predictions from Candidate_Population_Models_Daily where demo_group = '%s' "
                         "and prediction_task = '%s' and selected_metric = '%s' and selected_features = '%s' "
                         "and start_time = %s and end_time = %s "
                         "and selected_algorithm = '%s' order by user_id;"
                         "") % (metric, demo_group, task, metric, selected_features, 
                                start_periods[time_index], end_periods[time_index], algorithm)
                    
                query_results_population_performance = numpy.array(dbHandler.select(query))
                  
                day_periods_performance = day_periods_performance + (query_results_population_performance[:,0] * query_results_population_performance[:,1])
                day_periods_amount_of_data += query_results_population_performance[:,1]
              
            # Aggregate performance of all day periods
            demo_group_daily_population_models_performance = day_periods_performance / day_periods_amount_of_data
                    
            # Compute performance gain
            performance_gain = demo_group_daily_population_models_performance - population_models_performance
            data[0, group_idx] = numpy.median(performance_gain)
            data[1, group_idx] = data[0, group_idx] - np.percentile(performance_gain, 25)
            data[2, group_idx] = np.percentile(performance_gain, 75) - data[0, group_idx]
                
            print ("DONE: Task: %s | Metric: %s | Demo: %s") % (task, metric, demo_group)
                    
        data_array.append(data)
            
        print ("############################")
    
    
    #===================================================================
    # VISUALIZE
    #===================================================================
    sns.set(font_scale = 2.)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(20, 4.5))
    axis([-0.5, 15.5, -.06, .09])

    bar_position = numpy.arange(len(Util.DEMO_GROUP_LABELS)) - 0.5
    width = (1. - 0.1) / (len(data_array))
    
    colors = cm.summer(numpy.linspace(0, 1, len(data_array)))
    
    labels = ['Accuracy', 'F1', 'MCC']
    
    for idx in range(len(data_array)):    
        data = data_array[idx] #
        ax.bar(0.05+bar_position+width*idx+0.025 , data[0,:], width-0.05, color=colors[idx,:], label=labels[idx]) # 
        (_, caps, _) = ax.errorbar(0.05+bar_position+width*idx+width/2, data[0,:], yerr=[data[1,:], data[2,:]], 
                                    color=colors[idx,:], fmt='o', ecolor='black', capsize=10, elinewidth=5) # 
        for cap in caps:
            cap.set_markeredgewidth(5)
    
    legend(loc='upper center', prop={'size':20}, scatterpoints=1,fancybox=True, shadow=True, ncol=4, frameon=True, bbox_to_anchor=(0.3, 1.05))
    
    ax.set_xticklabels(Util.DEMO_GROUP_LABELS, rotation=45)
    ax.set_xticks(numpy.arange(0, len(Util.DEMO_GROUP_LABELS), 1))

    ax.set_yticks([-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08])
    ax.set_yticklabels(['-6%','-4%','-2%','0%','2%','4%', '6%', '8%'])
    ax.set_ylabel('Performance\ngain')
    
    subplots_adjust(bottom=0.26, left=0.07, top=0.98, right=0.99)
    
    if SHOW:
        show() 
    
    if SAVE:
        path = ('plots/daily_vs_all_population_model_improvements.pdf')
        plt.savefig(path) 
## 
# Entry point of the script
##
if __name__ == "__main__":
    
    visualize_demo_daily_population_models_performance()