#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class is a data structure containing 
# data of a specific subset of a user.
#
# copyright Paul Baumann
#############################################

import numpy

class UserDataSet:
    
    def __init__ (self):
        self.timestamps = None;
        self.day_string = None;
        self.time_string = None;
        
        self.ground_truth = None;
        self.feature_matrix = None;
        
        self.rows_mask = None; 
        
    def copy (self, userDataSet):
        self.timestamps = numpy.copy(userDataSet.timestamps);
        self.day_string = numpy.copy(userDataSet.day_string);
        self.time_string = numpy.copy(userDataSet.time_string);
        
        self.ground_truth = numpy.copy(userDataSet.ground_truth);
        self.feature_matrix = numpy.copy(userDataSet.feature_matrix);
        
        self.rows_mask = numpy.copy(userDataSet.rows_mask); 