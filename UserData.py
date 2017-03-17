#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class is a data structure 
# containing user data subsets.
#
# copyright Paul Baumann
#############################################

import numpy
from UserDataSet import UserDataSet

class UserData:
    
    def __init__ (self):
        self.userId = None;
        self.pre_feature_combination = None;
        self.complete_feature_matrix = None;
        
        self.optimization_set = None;   
        self.training_set = None;
        self.test_set = None;     
    
    def copy (self, userData):
        self.userId = userData.userId;
        self.pre_feature_combination = numpy.copy(userData.pre_feature_combination);
        self.complete_feature_matrix = numpy.copy(userData.complete_feature_matrix);
        
        self.optimization_set = UserDataSet();
        self.optimization_set.copy(userData.optimization_set);
           
        self.training_set = UserDataSet();
        self.training_set.copy(userData.training_set);
        
        self.test_set = UserDataSet();
        self.test_set.copy(userData.test_set);    