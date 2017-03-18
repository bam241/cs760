#! /usr/bin/env python

from __future__ import print_function
from __future__ import division

import sys
import arff
import numpy as np
import pandas as pd
from tools import *
import csv

def train_nn(num_epochs, num_folds, learning_rate, w_ij, w_jk, data):
    for epoch in range(0, num_epochs):
        strat_sets, classes = strat_split(num_folds, data)
        tests = {}
        for f in range(1, num_folds+1):
            # set aside a testing set used during prediciton phase
            test = strat_sets.pop(f)
            tests[f] = test
            for fold, fold_set in strat_sets.iteritems():
                for instance, row in fold_set.iterrows():
                ####################
                ### feed forward ###
                ####################
                    # input layer output: o_i
                    y_d, o_i = get_nn_input(row, instance, classes)
                    #net_ij = np.dot(np.transpose(o_i), w_ij).tolist()
                    net_ij = np.dot(o_i, w_ij).tolist()
                    # hidden layer output: o_j
                    o_j = get_nn_hidden(sigmoid(net_ij), instance)
                    net_jk = np.dot(o_j, w_jk)
                    # output node: o_k
                    o_k = sigmoid(net_jk)
                #####################
                ### backpropogate ###
                #####################
                    # update hidden weights
                    d_k = (y_d - o_k) * o_k * (1 - o_k)
                    delta_jk = learning_rate * d_k * np.array([o_j])
                    w_jk = np.transpose(delta_jk) + w_jk
                    # update input weights
                    d_j = d_k * np.sum(w_jk, 1) * (1 - np.array([o_j])) * np.array([o_j])
                    delta_ij = np.dot(np.transpose([o_i]), learning_rate * d_j)
                    delta_ij = np.delete(delta_ij, -1, 1) # hidden bias doesn't connect to input layer
                    w_ij = delta_ij + w_ij
    return w_ij, w_jk, tests, classes


def predict(w_ij, w_jk, tests, classes):
    to_print = []
    for fold, test_set in tests.iteritems():
        for instance, row in test_set.iterrows():
            y_d, o_i = get_nn_input(row, instance, classes)
            net_ij = np.dot(o_i, w_ij).tolist()
            o_j = get_nn_hidden(sigmoid(net_ij), instance)
            net_jk = np.dot(o_j, w_jk)
            o_k = sigmoid(net_jk)[0]
            if o_k >= 0.5:
                o_d = 1
            else:
                o_d = 0
            line = (instance+1, fold, o_d, y_d, o_k)
            to_print.append(line)
    return to_print

def main():
    # Get input: training data
    train = arff.load(open(sys.argv[1], 'rb')) 
    num_folds = int(sys.argv[2])
    learning_rate = float(sys.argv[3])
    num_epochs = int(sys.argv[4])
    # data and attributes
    attr = pd.DataFrame(train['attributes'], columns=['attr', 'vars'])
    data = pd.DataFrame(train['data'], columns=attr['attr'].tolist())
    
    in_size = len(attr) - 1
    hid_size = in_size
    
    w_ij = rand_weights(in_size+1, hid_size)
    w_jk = rand_weights(hid_size+1, 1)
    w_ij, w_jk, tests, classes = train_nn(num_epochs, num_folds, learning_rate, w_ij, w_jk, data)
    predicted = predict(w_ij, w_jk, tests, classes)
        
    # Print to command line
    ordered = pd.DataFrame(predicted, columns=['Instance', 'FoldNumber', 'PredictedOutput', 'ActualOutput', 'Confidence'])
    ordered = ordered.sort_values(by='Instance')
    #del ordered['Instance']
    toprint = ordered.to_string()
    file_name = str(num_folds) +  "_" + str(learning_rate) + "_" + str(num_epochs) + ".csv"
    ordered.to_csv(file_name, sep=',', index=False)
    print(''.join([''.join(v) for v in toprint]))
    
    return

if __name__ == "__main__":
    main()
