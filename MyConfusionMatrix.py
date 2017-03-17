#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class is a data structure allowing 
# to store evaluation results as 
# a confusion matrix.
#
# copyright Paul Baumann
#############################################


class MyConfusionMatrix:
    
    def __init__ (self):
        self.TP = None;
        self.TN = None;
        self.FP = None;
        self.FN = None;
        
        self.accuracy = None;
        self.precision = None;
        self.recall = None;
        self.fscore = None;
        self.MCC = None;
        self.kappa = None;
        
        self.total_accuracy = None;
        self.total_precision = None;
        self.total_recall = None;
        self.total_fscore = None;
        self.total_MCC = None;
        self.total_kappa = None;
        
        self.number_of_classes = None;
        self.number_of_predictions = None;