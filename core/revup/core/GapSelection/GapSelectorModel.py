# -*- coding: utf-8 -*-
"""
Created on Tue Dec 02 10:33:02 2014

@author: giris_000
"""
import pickle
import heapq
import logging
import random

import numpy
import xgboost

from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier

LOGGER = logging.getLogger(__name__)


class GapSelectorModel(object):
    """ 
    Handles the selection of good gap(s) from candidates chosen by SentenceFeatureComputer 
    and the training of the relevant classifier to achieve that
    """
    def __init__(self, gap_rating_tuples, classifier_type='svm'):
        self.classifier_type = classifier_type
        LOGGER.info("Creating datatset")
        input_x = numpy.array([g[0].vectorize() for g in gap_rating_tuples])
        labels = [g[1] for g in gap_rating_tuples]
        LOGGER.debug("Number of training samples: %d" % len(input_x))
        # scaling by mean and stdev
        self.scaler = preprocessing.StandardScaler().fit(input_x)
        input_x = self.scaler.transform(input_x)
        # selecting best features
        lsvc = svm.LinearSVC(C=0.08, penalty="l2", dual=False).fit(input_x, labels)
        self.feature_selector = SelectFromModel(lsvc, prefit=True)
        input_x = self.feature_selector.transform(input_x)
        self.input_x = input_x
        # training
        self.classifier = self.train(input_x, labels, classifier_type)
        self.labels = labels
    
    def train(self, train_x=None, train_y=None, classifier_type='svm'):
        """ 
        Trains a classifier with training_data x and labels y 
        Possible classifier_types: 
        'xgb' : XG Boost; 'svm': Support Vector Machine; 
        """
        if train_x is None or train_y is None:
            train_x = self.input_x
            train_y = self.labels

        if classifier_type not in ['svm', 'xgb', 'mlp']:
            LOGGER.error("Classfier type not understood. Terminating...")
            return
        classifier = None
        if classifier_type == 'svm':  # svm
            classifier = svm.SVC(C=1, kernel='rbf', degree=5, gamma=0.0045, 
                                 random_state=20, probability=True)
        elif classifier_type == 'xgb':
            classifier = xgboost.XGBClassifier()
        elif classifier_type == 'mlp':
            classifier = MLPClassifier()
        classifier.fit(train_x, train_y)
        return classifier

    def select_gap(self, sent, multi_gap_thres=None):
        """ 
        Given a sent (sentencefeaturecomputer object), selects one or 
        two good gaps from the sent's candidates
        """
        gap_vectors = self.scaler.transform([g.vectorize() for g in sent.gaps])
        #apply feature selection
        gap_vectors = self.feature_selector.transform(gap_vectors)
        label_probs = self.classifier.predict_proba(gap_vectors).tolist()
        best_ratings = heapq.nsmallest(2, label_probs)
        if len(best_ratings) < 1: 
            return None
        
        if (len(best_ratings) > 1 and multi_gap_thres and 
                best_ratings[0][1] - best_ratings[1][1] < multi_gap_thres and 
                best_ratings[1][1] >= 0.6): 
            #retrieve the gaps
            chosen_gaps = [sent.gaps[label_probs.index(br)] for br in best_ratings]
            #sort by position in sentence as well
            return sorted(chosen_gaps, key=lambda x: x.span.start) 

        if best_ratings[0][1] < 0.6:
            # return no gap if none of the gaps have a rating of >= 0.5
            return None
        else:
            # if not return gap with highest rating
            return [sent.gaps[label_probs.index(best_ratings[0])]]

    def predict(self, gap):
        """
        Given a gap, returns the probability that it is good (p) and bad (1-p)
        """
        gap_vector = self.scaler.transform(gap.vectorize())
        return self.classifier.predict_proba(gap_vector.reshape(1, -1))[0][1]

    def evaluate(self, gap_rating_tuples):
        """
        Evaluate accuracy of trained classifier on provided test set.
        """
        input_x = [g[0].vectorize() for g in gap_rating_tuples]
        labels = [g[1] for g in gap_rating_tuples]
        input_x = self.scaler.transform(input_x)
        # apply feature selection.
        input_x = self.feature_selector.transform(input_x)
        LOGGER.info("Accuracy: %f", self.classifier.score(input_x, labels))


    @staticmethod
    def cross_validate(gap_rating_tuples, num_folds=10, classifier_type='svm'):
        """
        Logs key classifier metrics by performing k-fold cross validation
        where num_folds is k
        """
        random.shuffle(gap_rating_tuples)
        input_x = [g[0].vectorize() for g in gap_rating_tuples]
        labels = [g[1] for g in gap_rating_tuples]
        subset_size = len(input_x)//num_folds
        metrics = {'accuracy':[], 'f1':[], 'precision':[], 'recall':[],
                   'true_positive': [], 'false_positive': []}

        for i in range(0, num_folds):
            train_x = numpy.array(input_x[:i*subset_size] + 
                                         input_x[(i+1)*subset_size:])
            train_y = numpy.array(labels[:i*subset_size] +
                                         labels[(i+1)*subset_size:])
            scaler = preprocessing.StandardScaler().fit(train_x)
            train_x = scaler.transform(train_x)
            # selecting best features
            lsvc = svm.LinearSVC(C=0.08, penalty="l2", dual=False).fit(train_x, train_y)
            feature_selector = SelectFromModel(lsvc, prefit=True)
            train_x = feature_selector.transform(train_x)

            test_x = input_x[i*subset_size: (i+1) * subset_size]
            test_x = scaler.transform(test_x)
            test_x = feature_selector.transform(test_x)
            test_true = labels[i*subset_size: (i+1) * subset_size]
            LOGGER.info("Training Set: " + str(i+1))
            if classifier_type == 'svm':  # svm
                classifier = svm.SVC(C=1, kernel='rbf', degree=5, gamma=0.0045, 
                                     random_state=20, probability=True)
            elif classifier_type == 'xgb':
                classifier = xgboost.XGBClassifier()
            elif classifier_type == 'mlp':
                classifier = MLPClassifier()
            classifier.fit(train_x, train_y)
            LOGGER.info("Testing Set: " + str(i+1))
            test_pred = [1 if val[1] >= 0.5 else 0 for val in classifier.predict_proba(test_x)]
            metrics['accuracy'].append(accuracy_score(test_true, test_pred))
            metrics['f1'].append(f1_score(test_true, test_pred))
            metrics['precision'].append(precision_score(test_true, test_pred))
            metrics['recall'].append(recall_score(test_true, test_pred))

            conf_mat = confusion_matrix(test_true, test_pred)
            true_positive = conf_mat[1][1]
            false_negative = conf_mat[1][0]
            false_positive = conf_mat[0][1]
            true_negative = conf_mat[0][0]
            metrics['true_positive'].append(float(true_positive)/(true_positive + false_negative))
            metrics['false_positive'].append(float(false_positive)/(false_positive + true_negative))

            LOGGER.info("Accuracy: " + str(numpy.mean(metrics['accuracy'])))
            LOGGER.info("Accuracy Standard Dev: "+ str(numpy.std(metrics['accuracy'])))
            LOGGER.info("F1 Score: " + str(numpy.mean(metrics['f1'])))
            LOGGER.info("F1 Score Standard Dev: " + str(numpy.std(metrics['f1'])))
            LOGGER.info("Precision: " + str(numpy.mean(metrics['precision'])))
            LOGGER.info("Precision Standard Dev: " + str(numpy.std(metrics['precision'])))
            LOGGER.info("Recall: " + str(numpy.mean(metrics['recall'])))
            LOGGER.info("Recall Standard Dev: " + str(numpy.std(metrics['recall'])))
            LOGGER.info("Mean True Positive Rate: " + str(numpy.mean(metrics['true_positive'])))
            LOGGER.info("TPR Standard Deviation: " + str(numpy.std(metrics['true_positive'])))
            LOGGER.info("Mean False Positive Rate: " + str(numpy.mean(metrics['false_positive'])))
            LOGGER.info("FPR Standard Deviation: " + str(numpy.std(metrics['false_positive'])))

    def save(self, filepath):
        """ 
        Pickles and saves the file to the designated file path
        """
        with open(filepath, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filepath):
        """ 
        Loads model from the selected filepath
        """
        gsm = None
        with open(filepath, 'rb') as model:
            gsm = pickle.load(model, encoding='latin1') #hacky 2to3
            # gsm = pickle.load(model)
        return gsm
