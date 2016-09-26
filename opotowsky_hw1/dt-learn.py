#! /usr/bin/env python

import scipy.io.arff as arff
import sys



def main():
    # Get training and testing data
    train_arff = arff.loadarff(open(sys.argv[1], 'rb')) 
    test_arff = arff.loadarff(open(sys.argv[2], 'rb')) 
    m = sys.argv[3]
    (training_data, metadata_train) = train_arff
    (testing_data, metadata_test) = test_arff
    
    # Function 
    

    return

if __name__ == "__main__":
    main()
