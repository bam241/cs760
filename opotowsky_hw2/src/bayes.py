#! /usr/bin/env python

from __future__ import print_function
from __future__ import division

import sys
import arff
import numpy as np
import pandas as pd

#def predict_tan(cond_prob, data, attr):
#    """
#    """
#    preds = []
#    correct = 0
#
#    return preds, correct


#def pairwise_conditional_probability(data, attr, counts):
#    """
#    """
#    
#
#    return cond_prob, cmi_sorted


#def tan_bn(data, attributes):
#    """
#    Builds tree-augmented naive bayesian network by looping through all pairs of 
#    attributes possible and structures with maximum-likely predictive connections
#    """
#    # probability calculations 
#    counts = counts_table(data, attributes)
#    cond_probs, cmi_sorted = pairwise_conditional_probability(data, attributes, counts)
#    # graph construction
#    
#    
#    bn = []
#    return bn, cond_probs


def predict_bn(cp, prior0, prior1, data, attr):
    """
    Applies the test set to the bayesian network, calculates posterior 
    probabilites for each class, compares them and predicts a class per test
    instance. This returns an array with 1 line per test entry: predicted class, 
    actual class, and post prob of the predicted class (+ # of correct predictions)
    """
    # gets class names for dataframe manipulation
    classes = attr.tail(1)['vars'].tolist()
    classlist = [classes[0][0], classes[0][1]]
    class0 = classlist[0]
    class1 = classlist[1]
    # loops through test data and calculates a posterior probability for
    # each class
    attrs = attr['attr'].drop(attr.index[-1]).tolist()
    preds = []
    correct = 0
    for index, row in data.iterrows():
        actual_class = row['class']
        pp0 = 1.0
        pp1 = 1.0
        i = 0
        for a in attrs:
            attr_val = row[a]
            sub = cp[cp['attr']==a]
            sub = sub[sub['var']==attr_val]
            pp0 = pp0 * sub.get_value(i, class0) 
            pp1 = pp1 * sub.get_value(i, class1) 
            i = i + 1
        pp0 = (pp0 * prior0) 
        pp1 = (pp1 * prior1) 
        # prediction comparison
        predict = np.log(pp0) - np.log(pp1)
        if predict > 0:
            predicted_class = class0
            post_prob = pp0 / (pp0 + pp1)
        else:
            predicted_class = class1
            post_prob = pp1 / (pp0 + pp1)
        line = [predicted_class, actual_class, "{:.12f}".format(post_prob)]
        preds.append(line)
        if actual_class == predicted_class:
            correct = correct + 1
    
    return preds, correct


def conditional_probability(data, attr, cp_table):
    """
    Calculates the conditional probabilities table for naive bayes. Also 
    calculates the prior probabilities for both classes.
    """
    # gets class names for dataframe manipulation
    classes = attr.tail(1)['vars'].tolist()
    classlist = [classes[0][0], classes[0][1]]
    class0 = classlist[0]
    class1 = classlist[1]
    # number of instances beloning to each class
    nclass0 = cp_table.loc[0, class0].sum()
    nclass1 = cp_table.loc[0, class1].sum()
    total = nclass0 + nclass1
    # all probabilities include a laplace est of 1
    prior0 = (nclass0 + 1) / (total + 2)
    prior1 = (nclass1 + 1) / (total + 2)
    list0 = []
    list1 = []
    for index, row in cp_table.iterrows():
        numattr = len(attr.loc[index, 'vars'])
        numer0 = row[class0] + 1
        numer1 = row[class1] + 1
        denom0 = nclass0 + (1 * numattr)
        denom1 = nclass1 + (1 * numattr)
        cp0 = numer0 / denom0
        cp1 = numer1 / denom1
        list0.append(cp0)
        list1.append(cp1)
    # replacing columns in previous table with cond probs
    del cp_table[class0]
    del cp_table[class1]
    cp_table[class0] = list0
    cp_table[class1] = list1
    
    return cp_table, prior0, prior1


def counts_table(data, attr):
    """
    Tabulates the counts in a table for the data set so that probabilities
    can be calculated. Only made for data sets with nominal variables. 
    """
    pd.options.mode.chained_assignment = None  # default='warn'
    # gets class names for dataframe manipulation
    classes = attr.tail(1)['vars'].tolist()
    classlist = [classes[0][0], classes[0][1]]
    # expanding a table to have all variable options in a column with their 
    # parent attribute
    allvariables = attr.apply(lambda x: pd.Series(x['vars']),axis=1).stack().reset_index(level=1, drop=True)
    allvariables.name='var'
    allvariables = attr.drop('vars', axis=1).join(allvariables)
    av = allvariables.drop(attr.index[-1])
    # populate the table with counts
    for c in classlist:
        clist = []
        count = 0
        for i, row in av.iterrows():
            att = row['attr']
            var = row['var']
            sub = data[[att,'class']]
            sub = sub[sub[att]==var]
            if not sub.empty:
                ssub = sub[sub['class']==c]
                if not ssub.empty:
                    count = len(ssub)
                else:
                    count = 0
            clist.append(count)
        av[c] = clist

    return av


def naive_bn(data, attributes):
    """
    Builds naive bayesian network by assuming all attributes
    are conditionally independent.
    """
    bn = []
    attr = attributes['attr'].tolist()
    # each attribute is only dependent on the class node
    i = 0
    while (i < len(attr)-1):
        row = [attr[i], attr[-1]]
        bn.append(row)
        i= i + 1
    # frequency table    
    freq = counts_table(data, attributes)
    # conditional probabilities and prior probabilities
    cond_probs, prior0, prior1 = conditional_probability(data, attributes, freq)

    return bn, cond_probs, prior0, prior1

def main():
    # Get input: training and testing data
    train_arff = arff.load(open(sys.argv[1], 'rb')) 
    test_arff = arff.load(open(sys.argv[2], 'rb')) 
    alg = sys.argv[3]
    # data and attributes
    attr_train = pd.DataFrame(train_arff['attributes'], columns=['attr', 'vars'])
    data_train = pd.DataFrame(train_arff['data'], columns=attr_train['attr'].tolist())
    attr_test = pd.DataFrame(test_arff['attributes'], columns=['attr', 'vars'])
    data_test = pd.DataFrame(test_arff['data'], columns=attr_test['attr'].tolist())
    
    # bayesian network: returns array with 1 line per attr including its parents
    # predict: applies the test set to the bayesian network and
    # returns an array with 1 line per test entry: predicted class, actual 
    # class, and posterior probability of the predicted class 
    # (as well as # correct predictions)
    if alg == 'n':
        bn, cond_probs, prior0, prior1 = naive_bn(data_train, attr_train)
        preds, correct = predict_bn(cond_probs, prior0, prior1, data_test, attr_test)
    elif alg == 't':
        bn, cond_probs = tan_bn(data_train, attr_train)
        preds, correct = predict_tan(cond_probs, data_test, attr_test)
    else:
        print('\n ~~~~~~~~~~\n 3rd arg should be n or t\n ~~~~~~~~~~\n')
    
    # Print the bn and pp's to command line
    print('\n'.join([' '.join(v) for v in bn]))
    print('\n', end='')
    print('\n'.join([' '.join(v) for v in preds]))
    print('\n', end='')
    print(correct)
    return

if __name__ == "__main__":
    main()
