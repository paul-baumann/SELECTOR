#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################
# This class is a helper and provides methods 
# to execute a set of machine learning algorithms.
#
# copyright Paul Baumann
#############################################

from collections import Counter
from EvaluationRun import EvaluationRun
import numpy
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier

from pylab import *

class PredictorsPipeline:
    
    def __init__ (self, evaluation_run):
        self.evaluation_run = evaluation_run;

        
    def Run_Predictions(self):  
    
        evaluation_run = self.evaluation_run
        training_set = evaluation_run.training_set;
        test_set = evaluation_run.test_set;
        
        # select features according to the mask
        training_feature_matrix = training_set.feature_matrix[:,evaluation_run.selected_features]
        test_feature_matrix = test_set.feature_matrix[:,evaluation_run.selected_features]
        
        # create mask for training rows that do not contain NaNs
        number_of_rows = training_feature_matrix.shape[0]
        number_of_columns = training_feature_matrix.shape[1]
        
        training_mask = training_feature_matrix #.astype(float)
        training_mask = training_mask.reshape(number_of_rows * number_of_columns, 1)
        training_mask = training_mask.astype(float)
        training_mask = training_mask.reshape(number_of_rows, number_of_columns)
        training_mask = numpy.sum(training_mask, axis=1)
        training_mask = numpy.where(~isnan(training_mask))
        
        # create mask for test rows that do not contain NaNs
        number_of_rows = test_feature_matrix.shape[0]
        number_of_columns = test_feature_matrix.shape[1]
        
        test_mask = test_feature_matrix
        test_mask = test_mask.reshape(number_of_rows * number_of_columns, 1)
        test_mask = test_mask.astype(float)
        test_mask = test_mask.reshape(number_of_rows, number_of_columns)
        test_mask = numpy.sum(test_mask, axis=1)
        test_mask = numpy.where(~isnan(test_mask))
        evaluation_run.data_ids_for_prediction = test_mask[0];
        
        # exclude rows with NaNs
        training_feature_matrix = training_feature_matrix[training_mask,:][0]
        training_ground_truth = evaluation_run.training_set.ground_truth[training_mask]
        training_ground_truth = training_ground_truth.reshape(training_feature_matrix.shape[0], 1)
        test_feature_matrix = test_feature_matrix[test_mask,:][0]
        test_ground_truth = evaluation_run.test_set.ground_truth[test_mask]
        test_ground_truth = test_ground_truth.reshape(test_feature_matrix.shape[0], 1)
        evaluation_run.ground_truth = test_ground_truth
        
        if len(training_ground_truth) == 0 or len(unique(training_ground_truth)) < 2 or len(test_feature_matrix) == 0:
            evaluation_run.ground_truth = []
            evaluation_run.prediction = []
            evaluation_run.data_ids_for_prediction = [];
            evaluation_run.random_prediction = []
            evaluation_run.dominating_class_prediction = []
            evaluation_run.histogram_class_prediction = []
            return evaluation_run
        
        try:
            # Naive bayes
            if evaluation_run.selected_algorithm == EvaluationRun.alg_naivebayes:
                evaluation_run.prediction_probabilities, evaluation_run.prediction = self.Run_NBPredictor(training_feature_matrix, test_feature_matrix, training_ground_truth)
                
            # Logistic regression
            if evaluation_run.selected_algorithm == EvaluationRun.alg_logistic_regression:
                evaluation_run.prediction_probabilities, evaluation_run.prediction = self.Run_LRPredictor(training_feature_matrix, test_feature_matrix, training_ground_truth)
            
            # kNN DYN-K  
            if evaluation_run.selected_algorithm == EvaluationRun.alg_knn_dyn:
                evaluation_run.prediction_probabilities, evaluation_run.prediction = self.Run_KNN_DYN_Predictor(training_feature_matrix, test_feature_matrix, training_ground_truth) 
                
            # kNN    
            if evaluation_run.selected_algorithm == EvaluationRun.alg_knn:
                evaluation_run.prediction_probabilities, evaluation_run.prediction = self.Run_KNNPredictor(training_feature_matrix, test_feature_matrix, training_ground_truth) 
            
            # Perceptron
            if evaluation_run.selected_algorithm == EvaluationRun.alg_perceptron:
                evaluation_run.prediction_probabilities, evaluation_run.prediction = self.Run_PerceptronPredictor(training_feature_matrix, test_feature_matrix, training_ground_truth)
            
            # Decision tree
            if evaluation_run.selected_algorithm == EvaluationRun.alg_decision_tree:
                evaluation_run.prediction_probabilities, evaluation_run.prediction = self.Run_DecisionTreePredictor(training_feature_matrix, test_feature_matrix, training_ground_truth) 
            
            # Gradient boost
            if evaluation_run.selected_algorithm == EvaluationRun.alg_gradient_boost:
                evaluation_run.prediction_probabilities, evaluation_run.prediction = self.Run_GradientBoostPredictor(training_feature_matrix, test_feature_matrix, training_ground_truth)
            
            # SVM
            if evaluation_run.selected_algorithm == EvaluationRun.alg_svm:
                evaluation_run.prediction_probabilities, evaluation_run.prediction = self.Run_SVMPredictor(training_feature_matrix, test_feature_matrix, training_ground_truth)
            
            # Stupid
            if evaluation_run.selected_algorithm == EvaluationRun.alg_stupid:
                evaluation_run.prediction_probabilities, evaluation_run.prediction = self.Run_StupidPredictor(training_feature_matrix, test_feature_matrix, training_ground_truth)
            
            # Random
            if evaluation_run.selected_algorithm == EvaluationRun.alg_random:
                evaluation_run.prediction_probabilities, evaluation_run.prediction = self.Run_RandomPredictor(test_feature_matrix, training_ground_truth)
            
            # Histogram
            if evaluation_run.selected_algorithm == EvaluationRun.alg_histogram:
                evaluation_run.prediction_probabilities, evaluation_run.prediction = self.Run_HistogramClassPredictor(test_feature_matrix, training_ground_truth)
                
            # Majority
            if evaluation_run.selected_algorithm == EvaluationRun.alg_majority:
                evaluation_run.prediction_probabilities, evaluation_run.prediction = self.Run_DominatingClassPredictor(test_feature_matrix, training_ground_truth)
                
            # compute kappa 
            _, evaluation_run.random_prediction = self.Run_RandomPredictor(test_feature_matrix, training_ground_truth)
            _, evaluation_run.histogram_class_prediction = self.Run_HistogramClassPredictor(test_feature_matrix, training_ground_truth)
            _, evaluation_run.dominating_class_prediction = self.Run_DominatingClassPredictor(test_feature_matrix, training_ground_truth)
            
        except MemoryError:
            print ">>>>>> MEMORY ERROR -- COUND NOT COMPLETE: %s" % (evaluation_run.selected_algorithm)
            evaluation_run.ground_truth = []
            evaluation_run.prediction = []
            evaluation_run.data_ids_for_prediction = [];
            evaluation_run.random_prediction = []
            evaluation_run.dominating_class_prediction = []
            evaluation_run.histogram_class_prediction = []
            
        return evaluation_run
    
    
    def Run_NBPredictor(self, training_feature_matrix, test_feature_matrix, training_ground_truth):
        
        training_ground_truth = ravel(training_ground_truth)
        NB_model = naive_bayes.GaussianNB()
        NB_model.fit(training_feature_matrix, training_ground_truth) 
        NB_predictions = NB_model.predict(test_feature_matrix)
        probabilities = NB_model.predict_proba(test_feature_matrix)
        probabilities = amax(probabilities, axis=1)
        
        return ravel(probabilities), ravel(NB_predictions.reshape(NB_predictions.shape[0], 1))
    
    
    def Run_LRPredictor(self, training_feature_matrix, test_feature_matrix, training_ground_truth):
        
        training_ground_truth = ravel(training_ground_truth)
        lr_model = linear_model.LogisticRegression(n_jobs = -1)
        lr_model.fit(training_feature_matrix, training_ground_truth)
        lr_predictions = lr_model.predict(test_feature_matrix)
        probabilities = lr_model.predict_proba(test_feature_matrix)
        probabilities = amax(probabilities, axis=1)
        
        return ravel(probabilities), lr_predictions.reshape(lr_predictions.shape[0], 1)
    
    
    def Run_KNN_DYN_Predictor(self, training_feature_matrix, test_feature_matrix, training_ground_truth):
        
        training_ground_truth = ravel(training_ground_truth)
        number_of_instances = len(training_ground_truth)
        optimal_k = max(int(sqrt(number_of_instances)), 1)

        knn_model = neighbors.KNeighborsClassifier(n_neighbors=optimal_k, n_jobs = -1) # , metric='minkowski'
        knn_model.fit(training_feature_matrix, training_ground_truth)
        knn_predictions = knn_model.predict(test_feature_matrix)
        probabilities = knn_model.predict_proba(test_feature_matrix)
        probabilities = amax(probabilities, axis=1)
        
        return ravel(probabilities), knn_predictions.reshape(knn_predictions.shape[0], 1)
        
    def Run_KNNPredictor(self, training_feature_matrix, test_feature_matrix, training_ground_truth):
        
        training_ground_truth = ravel(training_ground_truth)
        knn_model = neighbors.KNeighborsClassifier(n_neighbors=3, n_jobs = -1) # , metric='minkowski'
        knn_model.fit(training_feature_matrix, training_ground_truth)
        knn_predictions = knn_model.predict(test_feature_matrix)
        probabilities = knn_model.predict_proba(test_feature_matrix)
        probabilities = amax(probabilities, axis=1)
        
        return ravel(probabilities), knn_predictions.reshape(knn_predictions.shape[0], 1)
    
        
    def Run_PerceptronPredictor(self, training_feature_matrix, test_feature_matrix, training_ground_truth):
        
        training_ground_truth = ravel(training_ground_truth)
        perceptron_model = linear_model.Perceptron(shuffle=False, n_jobs=-1)
        perceptron_model.fit(training_feature_matrix, training_ground_truth)
        perceptron_predictions = perceptron_model.predict(test_feature_matrix)
        #probabilities = perceptron_model.predict_proba(test_feature_matrix)
        
        return [], perceptron_predictions.reshape(perceptron_predictions.shape[0], 1)
    
        
    def Run_DecisionTreePredictor(self, training_feature_matrix, test_feature_matrix, training_ground_truth):
        
        training_ground_truth = ravel(training_ground_truth)
        dt_model = tree.DecisionTreeClassifier()
        dt_model.fit(training_feature_matrix, training_ground_truth)
        dt_predictions = dt_model.predict(test_feature_matrix)
        probabilities = dt_model.predict_proba(test_feature_matrix)
        probabilities = amax(probabilities, axis=1)
        
        return ravel(probabilities), dt_predictions.reshape(dt_predictions.shape[0], 1)
    
    
    def Run_GradientBoostPredictor(self, training_feature_matrix, test_feature_matrix, training_ground_truth):
        
        training_ground_truth = ravel(training_ground_truth)
        gradientBoost_model = GradientBoostingClassifier()
        gradientBoost_model.fit(training_feature_matrix, training_ground_truth)
        gradientBoost_predictions = gradientBoost_model.predict(test_feature_matrix)
        
        return [], gradientBoost_predictions.reshape(gradientBoost_predictions.shape[0], 1)
    
        
    def Run_SVMPredictor(self, training_feature_matrix, test_feature_matrix, training_ground_truth):
        
        training_ground_truth = ravel(training_ground_truth)
        SVM_model = svm.SVC(probability=True)
        SVM_model.fit(training_feature_matrix, training_ground_truth) 
        SVM_predictions = SVM_model.predict(test_feature_matrix)
        probabilities = SVM_model.predict_proba(test_feature_matrix)
        probabilities = amax(probabilities, axis=1)
        
        return ravel(probabilities), SVM_predictions.reshape(SVM_predictions.shape[0], 1)
    
    
    def Run_StupidPredictor(self, training_feature_matrix, test_feature_matrix, training_ground_truth):
        
        stupid_predictions = test_feature_matrix.feature_matrix[:,21]
        return [], stupid_predictions.reshape(stupid_predictions.shape[0], 1)
    
    
    def Run_RandomPredictor(self, test_feature_matrix, training_ground_truth):
        
        number_of_output_elements = test_feature_matrix.shape[0]
        unique_ground_truth = numpy.unique(training_ground_truth)        
        random_prediction = unique_ground_truth[numpy.random.randint(len(unique_ground_truth), size=number_of_output_elements)]
         
        return [], random_prediction
    
    def Run_DominatingClassPredictor(self, test_feature_matrix, training_ground_truth):
        
        number_of_output_elements = test_feature_matrix.shape[0]
        for most_common_number, most_common_frequency in Counter(ravel(training_ground_truth)).most_common(1):
            count_of_most_common_element = most_common_frequency
            most_common_element = most_common_number
        
        dominating_class_prediction = [most_common_element] * number_of_output_elements;
        
        return [], dominating_class_prediction
    
    def Run_HistogramClassPredictor(self, test_feature_matrix, training_ground_truth): 
        
        number_of_output_elements = test_feature_matrix.shape[0]
        unique_values = numpy.unique(training_ground_truth)
        my_bins = numpy.array(unique_values)
        my_bins = numpy.append(my_bins, 10000)
        probabilities = numpy.histogram(training_ground_truth, bins=my_bins, density=True)[0]
        
        bins = numpy.cumsum(probabilities)
        bins[len(bins) - 1] = 1.1
        histogram_class_prediction = unique_values[np.digitize(random_sample(number_of_output_elements), bins)]
        
        return [], histogram_class_prediction
    
    def Run_MedianPredictor(self, test_feature_matrix, training_ground_truth):
        
        number_of_output_elements = test_feature_matrix.shape[0]
        mean_value = numpy.median(training_ground_truth)
        mean_prediction = numpy.array([mean_value]) * number_of_output_elements
        
        return mean_prediction