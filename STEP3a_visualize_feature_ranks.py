#!/usr/bin/python
# -*- coding: utf-8 -*-


import matplotlib as mlt
SHOW = False
SAVE = True

if SAVE:
    mlt.use('Agg')
    
import seaborn as sns

from EvaluationRun import EvaluationRun
import Database_Handler

from pylab import *
import numpy
import Util
    
MARKERS = ['o', 'v', '<', '>', 's', 'p', '*', 'D', '8']
LINES = ['-', '--', ':', '-.']
    
def main_visualize():
    
    tasks = [EvaluationRun.task_next_slot_place, EvaluationRun.task_next_slot_transition, EvaluationRun.task_next_place]
    list_of_metrics = [EvaluationRun.metric_accuracy, EvaluationRun.metric_fscore, EvaluationRun.metric_MCC]
    
    NUMBER_OF_FEATURES = 54
    num_columns = len(tasks) * len(list_of_metrics)
    
    ranks_matrix = numpy.zeros((NUMBER_OF_FEATURES, num_columns))
    idx = 0

    for task in tasks:
        for metric in list_of_metrics:
            dbHandler = Database_Handler.Get_DB_Handler()
            query = ("select number_of_individual_models / total_number_of_models "
                     "from FeatureRanks where prediction_task = '%s' and metric = '%s' "
                     "and demo_group = 'all' and daily_period = 'all' "
                     "order by feature_rank;") % (task, metric)
            
            ranks_matrix[:,idx] = ravel(numpy.array(dbHandler.select(query), dtype=float))
            idx += 1
    
    
    #===================================================================
    # VISUALIZE
    #===================================================================
    sns.set(font_scale = 2.5)
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(13, 6))
    axis([0, 25, 0, 1])
    
    for idx in range(num_columns):
        plot(numpy.arange(1,NUMBER_OF_FEATURES+1,1), ranks_matrix[:,idx], linewidth = 3, marker = MARKERS[idx], label=Util.TASK_METRIC_LABELS[idx], markersize = 14, linestyle = LINES[idx % len(LINES)])
    
    ax.annotate('', xy=(4, 0.45), xytext=(5.5, 0.6), arrowprops=dict(facecolor='black', shrink=0.1),)
        
    ax.set_xlabel('Feature rank r')
    ax.set_xticks(numpy.arange(1,25,1))
    ax.set_xticklabels(numpy.arange(1,25,1))
    ax.set_ylabel('Frequency of occurrence f')
    ax.set_yticks(numpy.arange(0,1.01,0.1))
    ax.set_yticklabels(['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%'])
    
    sns.set_style({'axes.grid': True})
    subplots_adjust(bottom=0.13, left=0.12, top=0.95, right=0.99)
    
    legend(loc='upper right', prop={'size':20}, scatterpoints=1,fancybox=True, shadow=True, ncol=1, frameon=True)
        
    if SHOW:
        show() 
    
    if SAVE:
        path = ('plots/feature_ranks.pdf')
        plt.savefig(path) 
    
if __name__ == "__main__":
    
    main_visualize()