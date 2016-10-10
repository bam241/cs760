#! /usr/bin/env python

import scipy.io.arff as arff
import sys
from math import log

class TreeNode:
    def __init__(self, col='', value=None, results=None, true_n=None, false_n=None):
        self.col = col # name (string) of column being tested
        self.value = value # true value
        self.results = results # list of results for a branch for leaf nodes only
        self.true_n = true_n # true node count
        self.false_n = false_n # false node count

#def print_tree(tree):
#    """
#    Prints the tree to shell according to TA's output.
#    """
#    if tree.results!=None:
#        print(str(tree.results))
#    else:
#        print('Column ' + str(tree.col)+' : '+str(tree.value)+'? ')
#
#
#    return

def split_set():
        # for each outcome in train_val, divide on that value
        for value in column_values:
            set1, set2 = divideset(rows, col, value)
    return split_set

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

def determine_candidate_splits():
    """
    Returns list of candidate attributes and where to split on if numeric
    """
    
    return splits

def decision_tree(data, meta, attributes, train_vals, m):
    """
    Builds decision tree by recursively making subtrees.
    """
    # Returns list of potential attributes for splitting
    splits = determine_candidate_splits(data, attributes)

    # Get majority value for default
    tvals = {}
    for row in data:
        if (tvals.has_key(row['class'])):
            tvals[row['class']] += 1.0
        else:
            tvals[row['class']]  = 1.0
    default = max(tvals, key=tvals.get)

    # Stopping Criteria 1: Empty data set or no more attributes to split
    if not data or (len(attributes) - 1) <= 0:
        return default
    # Stopping Criteria 2: m or less data points for a split 
    elif splits.count() <= m:
        # subtree = TreeNode()
        # determine class/label properties
        return TreeNode(LEAF)
    # Stopping Criteria 3: All data has same classification
    elif train_vals.count(train_vals[0]) == len(train_vals):
        return train_vals[0]
    else:
        split_set = find_best_split(data, splits)
        subtree = TreeNode(NEW NODE)
        for branch in split_set:
            # Performs split into sets for categorical or numerical data
            ...
            subset = instances where D gives Outcome and each Outcome of Node calls 
            subtree = TreeNode(...)


# Create a new decision tree/node with the best attribute and an empty
# dictionary object--we'll fill that up next.
tree = {best:{}}


# Create a new decision tree/sub-node for each of the values in the best attribute field
for val in get_values(data, best):
    # Create a subtree for the current value under the "best" field
    subtree = create_decision_tree(get_examples(data, best, val), [attr for attr in attributes if attr != best], target_attr, fitness_func)
    # Add the new subtree to the empty dictionary object in our new tree/node we just created.
    tree[best][val] = subtree

    return tree


def main():
    # Get input: training and testing data
    train_arff = arff.loadarff(open(sys.argv[1], 'rb')) 
    #test_arff = arff.loadarff(open(sys.argv[2], 'rb')) 
    m = sys.argv[3]
    (data_train, meta_train) = train_arff
    #(data_test, meta_test) = test_arff
    
    
    # Attribute list incl class (i.e., result)
    attributes = meta_train.names()
    # Class values - results column in data
    train_vals = data_train[attributes[-1]]
    # Call recursive decision tree building function
    tree = decision_tree(data_train, meta_train, attributes, train_vals, m)

    # Print the tree
#    print_tree(tree)
#    print(results)
    
    return

if __name__ == "__main__":
    main()
