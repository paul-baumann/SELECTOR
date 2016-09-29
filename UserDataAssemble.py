#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class is a helper to access user data 
# in the database and to assemble it for SELECTOR.
#
# copyright Paul Baumann
#############################################

import MySQLdb
import Database_Handler
import datetime
import math
import array
import sys
import UserData
import UserDataSet
from EvaluationRun import EvaluationRun
import Mobility_Features_Prediction
from time import time
from datetime import datetime
from operator import mul
import numpy
import scipy.io
import sklearn.linear_model as linMod
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import tree
from sklearn.decomposition import PCA
import sklearn
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import gaussian_process
from sklearn.hmm import MultinomialHMM
from sklearn.ensemble import GradientBoostingClassifier

from pylab import *

class UserDataAssemble:
    
    def __init__ (self, evaluation_run):
        self.evaluation_run = evaluation_run;

        
    def Get_User_Data(self): 
    
        evaluation_run = self.evaluation_run
        evaluation_run = self.Get_User_Data_From_Database(evaluation_run)
        evaluation_run = self.Apply_Feature_Mask(evaluation_run)
        
        feature_matrix = evaluation_run.userData.complete_feature_matrix
        
        number_of_rows = feature_matrix.shape[0]
        
        # create data mask
        day_strings = feature_matrix[:, 3:4]
        unique_days = unique(day_strings)
        number_of_days = len(unique_days) 
        
        # c = numpy.fromstring(b, sep=', ')
        dbHandler = Mobility_Features_Prediction.Get_DB_Handler()
        if evaluation_run.task == EvaluationRun.task_next_slot_transition_daily or evaluation_run.task == EvaluationRun.task_next_place_daily:
            query = "SELECT optimization_array, training_array, test_array FROM %s_Prediction_Run WHERE user_id = %i and start_time = %i and end_time = %i LIMIT 1" % (evaluation_run.task, evaluation_run.userData.userId, evaluation_run.start_time, evaluation_run.end_time)
        else:
            query = "SELECT optimization_array, training_array, test_array FROM %s_Prediction_Run WHERE user_id = %i  LIMIT 1" % (evaluation_run.task, evaluation_run.userData.userId)
        existing_data = dbHandler.select(query)
        existing_data = numpy.array(existing_data)
        
        if existing_data.shape[0] > 0:
            optimization_idx = numpy.fromstring(existing_data[0,0], sep=', ').astype(int)
            training_idx = numpy.fromstring(existing_data[0,1], sep=', ').astype(int)
            test_idx = numpy.fromstring(existing_data[0,2], sep=', ').astype(int)
        else:
            optimization_set_size = self.Get_Optimization_Set_Size(number_of_days);
            training_set_size = self.Get_Training_Set_Size(number_of_days - optimization_set_size);
    
            # select sub sets of data    
            indices = numpy.random.permutation(unique_days)
            optimization_idx, training_idx, test_idx = indices[:optimization_set_size], indices[optimization_set_size:optimization_set_size+training_set_size], indices[optimization_set_size+training_set_size:]
            
            optimization_membership = np.array([i in optimization_idx for i in day_strings])
            optimization_idx = numpy.where(optimization_membership == True)[0]
    
            training_membership = np.array([i in training_idx for i in day_strings])
            training_idx = numpy.where(training_membership == True)[0]
            
            test_membership = np.array([i in test_idx for i in day_strings])
            test_idx = numpy.where(test_membership == True)[0]
                
        # check for the available features
        evaluation_run.available_features = numpy.where(evaluation_run.userData.pre_feature_combination==True)[0]
        evaluation_run.available_features = evaluation_run.available_features.tolist()
        
        # optimization set
        optimization_set = UserDataSet.UserDataSet()
        optimization_set.timestamps = ravel(feature_matrix[optimization_idx, 2:3])
        optimization_set.day_string = ravel(feature_matrix[optimization_idx, 3:4])
        optimization_set.time_string = ravel(feature_matrix[optimization_idx, 4:5])
        
        optimization_set.ground_truth = ravel(feature_matrix[optimization_idx, 7:8]).astype(float)
        optimization_set.feature_matrix = feature_matrix[optimization_idx, 8:feature_matrix.shape[1]]
        optimization_set.rows_mask = optimization_idx
        
        # training set
        training_set = UserDataSet.UserDataSet()
        training_set.timestamps = ravel(feature_matrix[training_idx, 2:3])
        training_set.day_string = ravel(feature_matrix[training_idx, 3:4])
        training_set.time_string = ravel(feature_matrix[training_idx, 4:5])
        
        training_set.ground_truth = ravel(feature_matrix[training_idx, 7:8]).astype(float)
        training_set.feature_matrix = feature_matrix[training_idx, 8:feature_matrix.shape[1]]
        training_set.rows_mask = training_idx
        
        # test set
        test_set = UserDataSet.UserDataSet()
        test_set.timestamps = ravel(feature_matrix[test_idx, 2:3])
        test_set.day_string = ravel(feature_matrix[test_idx, 3:4])
        test_set.time_string = ravel(feature_matrix[test_idx, 4:5])
        
        test_set.ground_truth = ravel(feature_matrix[test_idx, 7:8]).astype(float)
        test_set.feature_matrix = feature_matrix[test_idx, 8:feature_matrix.shape[1]]
        test_set.rows_mask = test_idx
        
        
        # create user object
        evaluation_run.userData.optimization_set = optimization_set;
        evaluation_run.userData.training_set = training_set;
        evaluation_run.userData.test_set = test_set;
        
        return evaluation_run
    
        
    def Get_User_Data_From_Database(self, evaluation_run):
        
        dbHandler = Mobility_Features_Prediction.Get_DB_Handler()
        if evaluation_run.task == EvaluationRun.task_next_residence_time: #  OR (SI1 = 0 AND ground_truth <= 8)
            query = "SELECT * FROM %s_Feature_Matrix WHERE userId = %i AND (SI1 > 0)" % (evaluation_run.task, evaluation_run.userData.userId)
        else:
            if evaluation_run.task == EvaluationRun.task_next_place_daily:
                 query = "SELECT * FROM NextPlace_Feature_Matrix WHERE userId = %i and TI9 >= %i AND TI9 <= %i" % (evaluation_run.userData.userId, evaluation_run.start_time, evaluation_run.end_time)
            else: 
                if evaluation_run.task == EvaluationRun.task_next_slot_transition_daily:
                    query = "SELECT * FROM NextSlotTransition_Feature_Matrix WHERE userId = %i and TI9 >= %i AND TI9 <= %i" % (evaluation_run.userData.userId, evaluation_run.start_time, evaluation_run.end_time)
                else: 
                    query = "SELECT * FROM %s_Feature_Matrix WHERE userId = %i" % (evaluation_run.task, evaluation_run.userData.userId)
        feature_matrix = dbHandler.select(query)
        feature_matrix = numpy.array(feature_matrix)
        
        if evaluation_run.task == EvaluationRun.task_next_place_daily:
            selected_features = dbHandler.select("SELECT * FROM NextPlace_Pre_Selected_Features WHERE userId = %i" % (evaluation_run.userData.userId))
        else:
            if evaluation_run.task == EvaluationRun.task_next_slot_transition_daily:
                selected_features = dbHandler.select("SELECT * FROM NextSlotTransition_Pre_Selected_Features WHERE userId = %i" % (evaluation_run.userData.userId))
            else:
                selected_features = dbHandler.select("SELECT * FROM %s_Pre_Selected_Features WHERE userId = %i" % (evaluation_run.task, evaluation_run.userData.userId))
        selected_features = numpy.array(selected_features);
        selected_features = selected_features[:, 2:selected_features.shape[1]];
        evaluation_run.userData.pre_feature_combination = ravel(selected_features > 0);
        
        evaluation_run.userData.complete_feature_matrix = feature_matrix;
        
        return evaluation_run

    
    def Apply_Feature_Mask(self, evaluation_run):
        if evaluation_run.is_network == False:
            evaluation_run.userData.pre_feature_combination[0:6] = False;
        if evaluation_run.is_temporal == False:
            evaluation_run.userData.pre_feature_combination[6:21] = False;
        if evaluation_run.is_spatial == False:
            evaluation_run.userData.pre_feature_combination[21:26] = False;
        if evaluation_run.is_context == False:
            evaluation_run.userData.pre_feature_combination[26:54] = False;
            
        return evaluation_run
        
    
    def Get_Optimization_Set_Size(self, number_of_rows):
        return numpy.min([numpy.divide(number_of_rows, 3), 30]);

    def Get_Training_Set_Size(self, remaining_number_of_rows):
        return numpy.min([numpy.divide(remaining_number_of_rows, 2), 90]);