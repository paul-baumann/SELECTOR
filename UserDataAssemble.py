#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class is a helper to access user data 
# in the database and to assemble it for SELECTOR.
#
# copyright Paul Baumann
#############################################

import UserDataSet
from EvaluationRun import EvaluationRun
import numpy

from pylab import *

import Database_Handler

class UserDataAssemble:
    
    def __init__ (self, evaluation_run):
        self.evaluation_run = evaluation_run;

        
    def Get_User_Data(self): 
    
        evaluation_run = self.evaluation_run
        evaluation_run = self.Get_User_Data_From_Database(evaluation_run)
        evaluation_run = self.Apply_Feature_Mask(evaluation_run)
        
        feature_matrix = evaluation_run.userData.complete_feature_matrix
        
        # create data mask
        day_strings = feature_matrix[:, 3:4]
        unique_days = unique(day_strings)
        number_of_days = len(unique_days) 
        
        if evaluation_run.task == EvaluationRun.task_next_place_daily:
            task = evaluation_run.task[5:]
        else:
            task = evaluation_run.task
        query = "SELECT optimization_array, training_array, test_array FROM %s_Prediction_Run WHERE user_id = %i  LIMIT 1" % (task, evaluation_run.userData.userId)
        
        dbHandler = Database_Handler.Get_DB_Handler()
        existing_data = dbHandler.select(query)
        existing_data = numpy.array(existing_data)

        ## A partition exists
        if existing_data.shape[0] > 0:
            optimization_idx = numpy.fromstring(existing_data[0,0], sep=', ').astype(int)
            training_idx = numpy.fromstring(existing_data[0,1], sep=', ').astype(int)
            test_idx = numpy.fromstring(existing_data[0,2], sep=', ').astype(int)
        else: ## No partition found --> create a new partition of data into three subsets
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
        
        day_of_time_feature_array = feature_matrix[:,22].astype(int)
        day_period_mask = (day_of_time_feature_array >= evaluation_run.start_time) & (day_of_time_feature_array <= evaluation_run.end_time) 
        
        # optimization set
        optimization_set = UserDataSet.UserDataSet()
        optimization_set.timestamps = ravel(feature_matrix[optimization_idx, 2:3])[day_period_mask[optimization_idx]]
        optimization_set.day_string = ravel(feature_matrix[optimization_idx, 3:4])[day_period_mask[optimization_idx]]
        optimization_set.time_string = ravel(feature_matrix[optimization_idx, 4:5])[day_period_mask[optimization_idx]]
        
        optimization_set.ground_truth = (ravel(feature_matrix[optimization_idx, 7:8]).astype(float))[day_period_mask[optimization_idx]]
        optimization_set.feature_matrix = (feature_matrix[optimization_idx, 8:feature_matrix.shape[1]])[day_period_mask[optimization_idx],:]
        optimization_set.rows_mask = optimization_idx
        
        # training set
        training_set = UserDataSet.UserDataSet()
        training_set.timestamps = ravel(feature_matrix[training_idx, 2:3])[day_period_mask[training_idx]]
        training_set.day_string = ravel(feature_matrix[training_idx, 3:4])[day_period_mask[training_idx]]
        training_set.time_string = ravel(feature_matrix[training_idx, 4:5])[day_period_mask[training_idx]]
        
        training_set.ground_truth = (ravel(feature_matrix[training_idx, 7:8]).astype(float))[day_period_mask[training_idx]]
        training_set.feature_matrix = (feature_matrix[training_idx, 8:feature_matrix.shape[1]])[day_period_mask[training_idx],:]
        training_set.rows_mask = training_idx
        
        # test set
        test_set = UserDataSet.UserDataSet()
        test_set.timestamps = ravel(feature_matrix[test_idx, 2:3])[day_period_mask[test_idx]]
        test_set.day_string = ravel(feature_matrix[test_idx, 3:4])[day_period_mask[test_idx]]
        test_set.time_string = ravel(feature_matrix[test_idx, 4:5])[day_period_mask[test_idx]]
        
        test_set.ground_truth = (ravel(feature_matrix[test_idx, 7:8]).astype(float))[day_period_mask[test_idx]]
        test_set.feature_matrix = (feature_matrix[test_idx, 8:feature_matrix.shape[1]])[day_period_mask[test_idx],:]
        test_set.rows_mask = test_idx
        
        
        # create user object
        evaluation_run.userData.optimization_set = optimization_set;
        evaluation_run.userData.training_set = training_set;
        evaluation_run.userData.test_set = test_set;
        
        return evaluation_run
    
        
    def Get_User_Data_From_Database(self, evaluation_run):
        
        dbHandler = Database_Handler.Get_DB_Handler()
#             query = "SELECT * FROM NextPlace_Feature_Matrix WHERE userId = %i and TI9 >= %i AND TI9 <= %i" % (evaluation_run.userData.userId, evaluation_run.start_time, evaluation_run.end_time)
        if evaluation_run.task == EvaluationRun.task_next_place_daily:
            task = evaluation_run.task[5:]
        else:
            task = evaluation_run.task 
        
        query = "SELECT * FROM %s_Feature_Matrix WHERE userId = %i" % (task, evaluation_run.userData.userId)
        feature_matrix = dbHandler.select(query)
        feature_matrix = numpy.array(feature_matrix)
        
        if evaluation_run.task == EvaluationRun.task_next_place_daily:
            task = evaluation_run.task[5:]
        else:
            task = evaluation_run.task
        selected_features = dbHandler.select("SELECT * FROM %s_Pre_Selected_Features WHERE userId = %i" % (task, evaluation_run.userData.userId))
            
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