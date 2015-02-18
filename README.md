# Task: Classification of hand-written digits

## Some statistics

### Get the distribution of 0-9 samples

`awk -F "," '{print $65}' optdigits.tra | sort -n | uniq -c`

> 376 0

> 389 1

> 380 2

> 389 3

> 387 4

> 376 5

> 377 6

> 387 7

> 380 8

> 382 9

It looks good, the distribution of 0-9 digits looks like a uniform distribution.

For hand-written digits classification, the most common way is **Logistic Regression** and **SVM**.

## Some inital thoughts (Methodology)

### How to choose model

This dataset is quite small, so I will not try to use some **heavy** classifiers such as **Deep Neural Network**, which could cause over-fitting and cannot perform well result on test set. But, the traditional **Shallow Neural Network** is a good idea, say, a network with 3 layers of neurals.

Instead of NN, I will try to use **Logistic Regression** and **SVM** first, to see whether can I get a good result. For the sake of low dimensionality, I might use some **dimension reduction** method to filter the dataset, e.g. **PCA**.

If time permits, I will try to use some **uncommon** methods, such as **random forest** and **k nearest neighbor**.

### How to do multi-classification

There are two methods to solve multi-classification problem:

1. 1 vs (k-1) classification, namely, transforming a k-classification problem into k binary classification problem.

2. 1 vs 1 k-classification classifier, such as **softmax regression** instead of **logistic regression**.

3. Error-correcting output-codes, which is an uncommon way to solve multi-classification problem.

### How to do ETL of the dataset

In order to use k-fold cross validation, I will let open source tool do it. Scikit-learn is a good choice.

### How to choose open source tools

- Spark/MLlib is the most familiar tool of me, but it is too heavy and no necessary in the scenario;

- Scikit-learn seems the most suitable tool, I will try to use ETL part and classification part of it;

- LibSVM and libLinear are much more fast than scikit-learn, but for a demo project, I prefer python, because the scale-out and scale-up capalibities are not my first consideration.

## Details

### Install scikit-learn

`sudo apt-get install python-sklearn`

### Install MDP

`sudo aptitude install python-mdp`

To prevent from the warning of the following

> >>> import mdp

> /usr/lib/python2.7/dist-packages/sklearn/pls.py:7: DeprecationWarning: This module has been moved to cross_decomposition and will be removed in 0.16 "removed in 0.16", DeprecationWarning)

We need to force import mdp itself, other than the add-in package of scikit-learn.

`export MDP_DISABLE_SKLEARN=yes`

or 

```python
import os
os.environ['MDP_DISABLE_SKLEARN']='yes'
```

### Logistic Regression

### SVM

## References

1. [Comparing Classification Algorithms for Handwritten Digits](http://blog.quantitations.com/machine%20learning/2013/02/27/comparing-classification-algorithms-for-handwritten-digits/)

2. [Example: Handwritten Digit Classification](http://pythonhosted.org/bob.learn.boosting/example.html)

3. [Classification of handwritten digits using a SVM](http://nbviewer.ipython.org/url/www.hdm-stuttgart.de/~maucher/ipnotebooks/MachineLearning/svmDigitRecognition.ipynb)

4. [Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html)

5. [Recognizing hand-written digits](http://scikit-learn.org/stable/auto_examples/plot_digits_classification.html)

6. [The MNIST Database of Handwritten Digits](http://yann.lecun.com/exdb/mnist/)

7. [Modular Toolkit for Data Processing](http://mdp-toolkit.sourceforge.net/documentation.html)

8. [Scikit-learn document](http://scikit-learn.org/stable/)
