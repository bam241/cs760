======
README
======

This is a homework assignment for my machine learning class (cs760) on creating a neural net.
For those that care, it is a neural network with one hidden layer that uses backpropagation 
to train a model. The activation function is a sigmoid for a binary classification task, 
and the training implements a stochastic gradient descent.

-----------------------
Files in PR of Interest
-----------------------

My coding is contained in three files:
- neuralnet (for calling on command line with args)
- src/neuralnet.py (main neural net algorithm)
- src/tools.py (helper functions)

The ARFF file is a common style of training data in the machine learning world. I've
also included an ARFF file reading (and writing) python module, liac-arff.

-----------
Run Program
-----------

You can run the program on your own by typing the following into the command line:

.. code:: bash
	neuralnet trainfile num_folds learning_rate num_epochs

The script comes with an example input file (sonar.arff), so you can use that as follows:

.. code:: bash
	./neuralnet.py src/sonar.arff 10 0.1 50

----------
More Deets
----------

HW Assignment on course page: http://pages.cs.wisc.edu/~dpage/cs760/nn.html