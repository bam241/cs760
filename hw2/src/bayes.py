#! /usr/bin/env python

import arff
import numpy as np
import sys

class TreeNode:
    def __init__(self, col='', value=None, results=None, true_n=None, false_n=None):
        self.col = col # name (string) of column being tested
        self.value = value # true value
        self.results = results # list of results for a branch for leaf nodes only
        self.true_n = true_n # true node count
        self.false_n = false_n # false node count

#def print_bn(bn):
#    """
#    Prints the results to the command line
#    """
#   % if tree.results!=None:
#   %     print(str(tree.results))
#   % else:
#   %     print('Column ' + str(tree.col)+' : '+str(tree.value)+'? ')
#
#    return

def entropy(data):
    """
    Calculates entropy for a given data set
    """
            
    # Dict for tracking + and - train_values
    tval_freq = {}
    # Calculate the frequency of each of the train_values for total set
    for row in data:
        if (tval_freq.has_key(row['class'])):
            tval_freq[row['class']] += 1.0
        else:
            tval_freq[row['class']]  = 1.0
    
    # Entropy calculation
    entropy = 0.0
    for freq in tval_freq.keys():
        p = freq/len(data)
        entropy += -p * log(p, 2)

    return entropy

def info_gain(data, split):
    """
    Calculates the information gain for a given split
    """
    
    # Dict for tracking attribute values
    attr_freq = {}
    # Calculate the frequency of each of the train_values for split
    for row in data:
        if (attr_freq.has_key(row['split'])):
            attr_freq[row['split']] += 1.0
        else:
            attr_freq[row['split']]  = 1.0

    # Calculate the entropy of the data set
    entropy = 0.0
    for freq in attr_freq.values():
        p = freq/len(data)
        entropy += -p * log(p, 2)

    # Sum the entropy for each subset in split
    for attr_val in attr_freq.keys():
        attr_prob = attr_freq[attr_val] / sum(attr_freq.values())
        subset = [row for row in data if row['split'] == attr_val]
        subset_entropy += attr_prob * entropy(subset)

    # Calculate info gain from 
    gain = entropy(data) - subset_entropy

    return gain

def find_best_split(data, attributes, splits):
    """
    Finds the split with the most info gain and returns the attr to split on
    """
    max_gain= -1000.0
    best_split_attr = None
    
    # Calculates best split
    for split in splits:
        gain = info_gain(data, split)
        if gain > max_gain:
            max_gain = gain
            best_split_attr = split
    
    return best_split_attr

def bayesian_network(data, attributes, alg):
    """
    Builds bayesian network by determining conditional independencies.
    """
    # returns an array of CP values
    # cond_probs = get_CPT(data, attributes)
    # calculate max info gain, i.e. the largest P(y|x)
    # for each y
    #   calc P(y)
    #   


def main():
    # Get input: training and testing data
    train_arff = arff.load(open(sys.argv[1], 'rb')) 
    test_arff = arff.load(open(sys.argv[2], 'rb')) 
    alg = sys.argv[3]
    # NP arrays of data and attributes
    data_train = np.array(train_arff['data'])
    data_test = np.array(test_arff['data'])
    attr_train = np.array(train_arff['attributes'])
    attr_test = np.array(test_arff['attributes'])
    
    # Call recursive decision tree building function
    bn = bayesian_network(data_train, attr_train, alg)

    # Print the BN to command line
#    print_bn(bn)
    
    return

if __name__ == "__main__":
    main()
