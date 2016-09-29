#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class is a data structure allowing 
# to store evaluation results as 
# a confusion matrix.
#
# copyright Paul Baumann
#############################################

import numpy
import scipy.io
import datetime
import math
import array
import sys
import Mobility_Features_Prediction
from time import time
from datetime import datetime
from operator import mul
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from pylab import *

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