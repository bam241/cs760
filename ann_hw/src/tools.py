
from __future__ import print_function
from __future__ import division

import sys
import arff
import numpy as np
import pandas as pd

def sigmoid(x):
    """
    Computes output values (sigmoid function) of each element in np array
    """
    exp = np.exp(-1 * np.array(x))
    sigmoid = np.add(1, exp)
    sigmoid = np.divide(1, sigmoid)
    return sigmoid

def get_nn_input(row, instance, classes):
    """
    For each row in training data, this provides a list of the input layer 
    (with a bias node of 1) as well as the instance's actual output value.
    """
    layer = pd.DataFrame(row)
    c = layer.get_value('Class', instance)
    if c == classes[1]:
        y = 1
    else: # c == class0
        y = 0
    layer = layer.drop('Class')
    layer = layer.as_matrix()
    layer = np.append(layer, [1]).tolist()
    return y, layer

def get_nn_hidden(layer, instance):
    """
    Takes an np array representing a hidden layer, adds a bias node of 1, 
    and returns as a list
    """
    layer = layer.tolist()
    layer = np.append(layer, [1]).tolist()
    return layer

def strat_split(num_folds, data):
    """ 
    Takes entire training set and splits all instances into stratified sets, 
    the number of which is defined by num_folds
    """
    temp = data
    counts = temp['Class'].value_counts()
    class0 = counts.index.tolist()[0]
    class1 = counts.index.tolist()[1]
    classes = [class0, class1]
    num0 = counts[0]
    num1 = counts[1]
    # Unsuccessful troubleshooting for instance re-use
    #num0_instances = int(round(num0 / num_folds))
    #num1_instances = int(round(num1 / num_folds))
    num0_instances = int(num0 / num_folds)
    num1_instances = int(num1 / num_folds)
    strat_data = {}
    for i in range(1, num_folds):
        set0 = temp.sample(n=num0_instances)
        set1 = temp.sample(n=num1_instances)
        i_set = set0.append(set1)
        strat_data[i] = i_set
        temp = temp.drop(i_set.index)
    strat_data[num_folds] = temp
    return strat_data, classes

def rand_weights(size1, size2):
    """
    Randomly initialize weights between two layers 
    (e.g., input layer to hidden layer)
    """
    w = np.random.uniform(-0.1, 0.1, (size1, size2)).tolist()
    return w
