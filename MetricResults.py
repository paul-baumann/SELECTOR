#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class is a data structure allowing 
# to store evaluation results for 
# the several metrics.
#
# copyright Paul Baumann
#############################################

class MetricResults:
    
    def __init__ (self):
        self.selected_features = [];
        
        self.accuracy = 0;
        self.precision = 0;
        self.recall = 0;
        self.fscore = 0;
        self.MCC = 0;
        
        self.kappa = 0;
        self.kappa_metric = 0;
        self.kappa_random = 0;
        self.kappa_histogram = 0;
        self.kappa_dominating = 0;
        
        self.frequency_of_top_class = 0;
        
        self.relative_prediction_error = 0;
        self.absolute_prediction_error = 0;