#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy
import scipy.io
import datetime
import math
import array
import sys
from time import time
from datetime import datetime
from operator import mul
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from pylab import *

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