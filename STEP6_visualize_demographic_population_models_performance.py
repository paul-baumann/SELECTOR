#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class visualizes performance 
# differences between population models 
# derived for the entire population and those 
# derived for demographic groups.
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

def visualize_demographic_population_models_performance():
    
    list_of_metrics = [EvaluationRun.metric_fscore, EvaluationRun.metric_MCC, EvaluationRun.metric_accuracy]
    tasks = [EvaluationRun.task_next_slot_transition, EvaluationRun.task_next_slot_transition, EvaluationRun.task_next_place]
    
    data_array = []
    
    for run_idx, task in enumerate(tasks):
        metric = list_of_metrics[run_idx]
              
        # Matrix to store the results
        data = numpy.zeros((3, len(Util.demo_groups[1:])))
          
        # Get population models -- all demographics
        dbHandler = Database_Handler.Get_DB_Handler()
        query = ("select selected_features, selected_algorithm from Candidate_Population_Models_Performance "
                 "where demo_group = 'all' and prediction_task = '%s' and selected_metric = '%s' "
                 "order by RMSE_total, number_of_features limit 1;"
                 "") % (task, metric)
          
        query_population_model = numpy.array(dbHandler.select(query))
        selected_features = query_population_model[0,0]
        algorithm = query_population_model[0,1]
          
        # Get population model performance -- all demographics
        dbHandler = Database_Handler.Get_DB_Handler()
        query = ("select %s from Candidate_Population_Models where demo_group = 'all' "
                 "and prediction_task = '%s' and selected_metric = '%s' and selected_features = '%s' "
                 "and selected_algorithm = '%s' order by user_id;"
                 "") % (metric, task, metric, selected_features, algorithm)
          
        query_results_population_performance = numpy.array(dbHandler.select(query))
        population_models_performance = query_results_population_performance[:,0]
          
        ## Compare to each demographic group
        for group_idx, demo_group in enumerate(Util.demo_groups[1:]): # 
              
            # read user list
            text_file = open("userids.txt", "r")
            user_ids = text_file.read().split('\n')
            text_file.close()
              
            # Identify which users belong to the current demo group
            mask_users_belong_to_demo_group = Util.areUsersBelongToDemoGroup(user_ids, demo_group)
            all_demo_groups_population_models_performance = population_models_performance[mask_users_belong_to_demo_group]
              
            # Get population models -- selected demo group
            dbHandler = Database_Handler.Get_DB_Handler()
            query = ("select selected_features, selected_algorithm from Candidate_Population_Models_Performance "
                     "where demo_group = '%s' and prediction_task = '%s' and selected_metric = '%s' "
                     "order by RMSE_total, number_of_features limit 1;"
                     "") % (demo_group, task, metric)
              
            query_population_model = numpy.array(dbHandler.select(query))
            selected_features = query_population_model[0,0]
            algorithm = query_population_model[0,1]
              
            # Get population model performance -- selected demo group
            dbHandler = Database_Handler.Get_DB_Handler()
            query = ("select %s from Candidate_Population_Models where demo_group = '%s' "
                     "and prediction_task = '%s' and selected_metric = '%s' and selected_features = '%s' "
                     "and selected_algorithm = '%s' order by user_id;"
                     "") % (metric, demo_group, task, metric, selected_features, algorithm)
              
            query_results_population_performance = numpy.array(dbHandler.select(query))
            demo_group_only_population_models_performance = query_results_population_performance[:,0]
              
            # Compute performance gain
            performance_gain = demo_group_only_population_models_performance - all_demo_groups_population_models_performance
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
    fig, ax = plt.subplots(figsize=(20, 4))
    axis([-0.5, 14.5, -.05, .05])

    bar_position = numpy.arange(len(Util.DEMO_GROUP_LABELS[1:])) - 0.5
    width = (1. - 0.1) / (len(data_array))
    
    colors = cm.summer(numpy.linspace(0, 1, len(data_array)))
    
    labels = ['NST F1', 'NST MCC', 'NP Accuracy']
    
    for idx in range(len(data_array)):    
        data = data_array[idx] #
        ax.bar(0.05+bar_position+width*idx+0.025 , data[0,:], width-0.05, color=colors[idx,:], label=labels[idx]) # 
        (_, caps, _) = ax.errorbar(0.05+bar_position+width*idx+width/2, data[0,:], yerr=[data[1,:], data[2,:]], 
                                    color=colors[idx,:], fmt='o', ecolor='black', capsize=10, elinewidth=5) # 
        for cap in caps:
            cap.set_markeredgewidth(5)
    
    legend(loc='upper center', prop={'size':20}, scatterpoints=1,fancybox=True, shadow=True, ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.2))
    
    ax.set_xticklabels(Util.DEMO_GROUP_LABELS, rotation=45)
    ax.set_xticks(numpy.arange(0, len(Util.DEMO_GROUP_LABELS[1:]), 1))

    ax.set_yticks([-0.04, -0.02, 0, 0.02, 0.04])
    ax.set_yticklabels(['-4%','-2%','0%','2%','4%'])
    ax.set_ylabel('Performance\ngain')
    
    subplots_adjust(bottom=0.33, left=0.07, top=0.91, right=0.99)
    
    if SHOW:
        show() 
    
    if SAVE:
        path = ('plots/demographic_model_improvements.pdf')
        plt.savefig(path) 
## 
# Entry point of the script
##
if __name__ == "__main__":
    
    visualize_demographic_population_models_performance()