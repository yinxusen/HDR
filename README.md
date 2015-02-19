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

### How to do grid search for hyper-parameters

### How to choose open source tools

- Spark/MLlib is the most familiar tool of me, but it is too heavy and no necessary in the scenario;

- Scikit-learn seems the most suitable tool, I will try to use ETL part and classification part of it;

- LibSVM and libLinear are much more fast than scikit-learn, but for a demo project, I prefer python, because the scale-out and scale-up capalibities are not my first consideration.

## Details

### Install scikit-learn

`sudo apt-get install python-sklearn`

I try to use sklearn, with its svm and logistic regression, and get good results.

For svm, I get

> \>\>\> print(metrics.classification_report(expected, predicted))

>             precision    recall  f1-score   support

>         0.0       1.00      1.00      1.00       190

>         1.0       0.98      0.99      0.99       194

>         2.0       0.99      1.00      0.99       186

>         3.0       0.99      0.96      0.97       192

>         4.0       0.99      0.99      0.99       202

>         5.0       0.98      0.99      0.99       194

>         6.0       0.99      0.99      0.99       184

>         7.0       0.99      0.99      0.99       188

>         8.0       0.99      0.99      0.99       201

>         9.0       0.97      0.98      0.98       181

> avg / total       0.99      0.99      0.99      1912

For logistic regression, I get

> \>\>\> print(metrics.classification_report(expected, lrpredicted))

>              precision    recall  f1-score   support

>         0.0       0.99      1.00      0.99       190

>         1.0       0.93      0.95      0.94       194

>         2.0       0.98      0.97      0.98       186

>         3.0       0.98      0.93      0.96       192

>         4.0       0.98      0.97      0.97       202

>         5.0       0.97      0.97      0.97       194

>         6.0       0.98      0.99      0.98       184

>         7.0       0.99      0.99      0.99       188

>         8.0       0.93      0.93      0.93       201

>         9.0       0.91      0.94      0.93       181

> avg / total       0.96      0.96      0.96      1912

Code snippts as follows:

```python
import numpy
from sklearn import datasets, svm, metrics
from sklearn.linear_regression import LogisticRegression

digits = numpy.loadtxt(fname="~/data/rubikloud/optdigits.tra", delimiter=',')
n_samples = len(digits)

data = digits[:,:-1]
target = digits[:,-1]

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# Create a logistic regression classifier
lr = LogisticRegression()

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples / 2], target[:n_samples / 2])

# learn with lr
lr.fit(data[:n_samples / 2], target[:n_samples / 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])
lrpredicted = lr.predict(data[n_samples / 2:])

print(metrics.classification_report(expected, predicted)))
print(metrics.classification_report(expected, lrpredicted)))
```

### Install MDP

`sudo aptitude install python-mdp`

To prevent from the warning of the following

> \>\>\> import mdp

> /usr/lib/python2.7/dist-packages/sklearn/pls.py:7: DeprecationWarning: This module has been moved to cross_decomposition and will be removed in 0.16 "removed in 0.16", DeprecationWarning)

We need to force import mdp itself, other than the add-in package of scikit-learn.

`export MDP_DISABLE_SKLEARN=yes`

or 

```python
import os
os.environ['MDP_DISABLE_SKLEARN']='yes'
```

So, let's try something of the **flow**, I love this kind of **pipeline**. Here is the result:

>           precision    recall  f1-score   support

>         0.0       1.00      0.99      1.00       130

>         1.0       0.99      0.98      0.98       130

>         2.0       1.00      0.99      1.00       119

>         3.0       0.98      1.00      0.99       129

>         4.0       0.99      0.98      0.99       130

>         5.0       0.99      1.00      1.00       128

>         6.0       0.99      1.00      1.00       124

>         7.0       0.99      0.98      0.99       126

>         8.0       0.97      0.99      0.98       139

>         9.0       0.98      0.98      0.98       120

> avg / total       0.99      0.99      0.99      1275

We can see that it is even better than the former SVM result.

Here is the code snippt:

```python
import mdp
import numpy
from sklearn import metrics

digits = numpy.loadtxt(fname="/home/lan/data/rubikloud/optdigits.tra", delimiter=',')
n_samples = len(digits)

data = digits[:,:-1]
target = digits[:,-1]

n_trains = n_samples / 3 * 2

train_data = [data[:n_trains, :]]
train_data_with_labels = [(data[:n_trains, :], target[:n_trains])]

test_data = data[n_trains:, :]
test_labels = target[n_trains]

flow = mdp.Flow([mdp.nodes.PCANode(output_dim=25, dtype='f'),
    mdp.nodes.PolynomialExpansionNode(3),
    mdp.nodes.PCANode(output_dim=0.99),
    mdp.nodes.FDANode(output_dim=9),
    mdp.nodes.SVCScikitsLearnNode(kernel='rbf')], verbose=True)

flow.train([train_data, None, train_data, train_data_with_labels, train_data_with_labels])

flow[-1].execute = flow[-1].label

prediction = flow(test_data)

print metrics.classification_report(test_labels, prediction)
```

To testify my assumption, I substitude `SVCScikitLearnNode` with `LogisticRegressionScikitLearnNode`, and get similar result:

>             precision    recall  f1-score   support

>         0.0       1.00      0.99      1.00       130

>         1.0       0.98      0.98      0.98       130

>         2.0       1.00      1.00      1.00       119

>         3.0       0.98      0.99      0.99       129

>         4.0       0.99      0.98      0.99       130

>         5.0       0.98      1.00      0.99       128

>         6.0       0.99      1.00      1.00       124

>         7.0       0.99      0.98      0.99       126

>         8.0       0.99      0.98      0.98       139

>         9.0       0.99      0.98      0.99       120

> avg / total       0.99      0.99      0.99      1275

So, in the hand-written digits recognition scenario, logistic regression with some feature expansion and transformation can compete SVM.

### Logistic Regression

Add k-fold cross validation and grid search in lr. Result:

> Best score: 0.964

> Best parameters set:

> {'C': 0.1, 'intercept_scaling': 1, 'fit_intercept': True, 'penalty': 'l2', 'random_state': None, 'dual': False, 'tol': 0.0001, 'class_weight': None}

>              precision    recall  f1-score   support

>         0.0       1.00      1.00      1.00       130

>         1.0       0.93      0.95      0.94       130

>         2.0       0.99      0.93      0.96       119

>         3.0       0.96      0.97      0.97       129

>         4.0       0.98      0.95      0.96       130

>         5.0       0.97      0.99      0.98       128

>         6.0       0.99      0.99      0.99       124

>         7.0       0.98      0.98      0.98       126

>         8.0       0.91      0.92      0.91       139

>         9.0       0.94      0.94      0.94       120

> avg / total       0.96      0.96      0.96      1275

It looks better than the previous LR result.

Code snippt:

```python
mport numpy
from sklearn import datasets, svm, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

digits = numpy.loadtxt(fname="/home/lan/data/rubikloud/optdigits.tra", delimiter=',')
n_samples = len(digits)

data = digits[:,:-1]
target = digits[:,-1]

param_grid = [
    {'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10.0, 100.0]}
]

# Create a classifier: a support vector classifier
classifier = LogisticRegression()

grid_search = GridSearchCV(classifier, param_grid, n_jobs = -1, verbose = 1, cv = 5)

n_trains = n_samples / 3 * 2

# We learn the digits on the first half of the digits
grid_search.fit(data[:n_trains], target[:n_trains])

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
print best_parameters

# Now predict the value of the digit on the second half:
expected = target[n_trains:]
predicted = grid_search.best_estimator_.predict(data[n_trains:])

print(metrics.classification_report(expected, predicted))mport numpy
from sklearn import datasets, svm, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

digits = numpy.loadtxt(fname="/home/lan/data/rubikloud/optdigits.tra", delimiter=',')
n_samples = len(digits)

data = digits[:,:-1]
target = digits[:,-1]

param_grid = [
    {'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10.0, 100.0]}
]

# Create a classifier: a support vector classifier
classifier = LogisticRegression()

grid_search = GridSearchCV(classifier, param_grid, n_jobs = -1, verbose = 1, cv = 5)

n_trains = n_samples / 3 * 2

# We learn the digits on the first half of the digits
grid_search.fit(data[:n_trains], target[:n_trains])

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
print best_parameters

# Now predict the value of the digit on the second half:
expected = target[n_trains:]
predicted = grid_search.best_estimator_.predict(data[n_trains:])

print(metrics.classification_report(expected, predicted))mport numpy
from sklearn import datasets, svm, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

digits = numpy.loadtxt(fname="/home/lan/data/rubikloud/optdigits.tra", delimiter=',')
n_samples = len(digits)

data = digits[:,:-1]
target = digits[:,-1]

param_grid = [
    {'penalty': ['l1', 'l2'], 'C': [0.1, 1.0, 10.0, 100.0]}
]

# Create a classifier: a support vector classifier
classifier = LogisticRegression()

grid_search = GridSearchCV(classifier, param_grid, n_jobs = -1, verbose = 1, cv = 5)

n_trains = n_samples / 3 * 2

# We learn the digits on the first half of the digits
grid_search.fit(data[:n_trains], target[:n_trains])

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
print best_parameters

# Now predict the value of the digit on the second half:
expected = target[n_trains:]
predicted = grid_search.best_estimator_.predict(data[n_trains:])

print(metrics.classification_report(expected, predicted))
```

### SVM

Let's add k-fold cross validation and grid search in SVM. Here is the result:

> Best score: 0.990

> Best parameters set:

> {'kernel': 'rbf', 'C': 10, 'verbose': False, 'probability': False, 'degree': 3, 'shrinking': True, 'max_iter': -1, 'random_state': None, 'tol': 0.001, 'cache_size': 200, 'coef0': 0.0, 'gamma': 0.001, 'class_weight': None}

>              precision    recall  f1-score   support

>         0.0       1.00      1.00      1.00       130

>         1.0       0.98      0.98      0.98       130

>         2.0       1.00      1.00      1.00       119

>         3.0       0.99      0.98      0.99       129

>         4.0       0.98      0.99      0.99       130

>         5.0       0.99      1.00      1.00       128

>         6.0       0.99      0.99      0.99       124

>         7.0       0.99      0.98      0.99       126

>         8.0       0.99      0.99      0.99       139

>         9.0       0.98      0.99      0.99       120

> avg / total       0.99      0.99      0.99      1275

Code snippt:

```python
import numpy
from sklearn import datasets, svm, metrics
from sklearn.grid_search import GridSearchCV

digits = numpy.loadtxt(fname="/home/lan/data/rubikloud/optdigits.tra", delimiter=',')
n_samples = len(digits)

data = digits[:,:-1]
target = digits[:,-1]

param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

# Create a classifier: a support vector classifier
classifier = svm.SVC()

grid_search = GridSearchCV(classifier, param_grid, n_jobs = -1, verbose = 1, cv = 5)

n_trains = n_samples / 3 * 2

# We learn the digits on the first half of the digits
grid_search.fit(data[:n_trains], target[:n_trains])

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
print best_parameters

# Now predict the value of the digit on the second half:
expected = target[n_trains:]
predicted = grid_search.best_estimator_.predict(data[n_trains:])

print(metrics.classification_report(expected, predicted))
```

### SVM / LR + GridSearch + CV + Feature Engineering pipeline

### Neural network

With the help of RBM and grid cv, I can get the following result on LR:

> Best score: 0.955

> Best parameters set:

> {'rbm1__batch_size': 10, 'lr__dual': False, 'rbm1__verbose': False, 'rbm1__n_iter': 10, 'rbm1': BernoulliRBM(batch_size=10, learning_rate=0.1, n_components=36, n_iter=10,

>        random_state=None, verbose=False), 'rbm1__n_components': 36, 'lr__tol': 0.0001, 'lr__class_weight': None, 'lr': LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,

>           intercept_scaling=1, penalty='l1', random_state=None, tol=0.0001), 'rbm1__learning_rate': 0.1, 'rbm1__random_state': None, 'lr__fit_intercept': True, 'lr__penalty': 'l1', 'lr__random_state': None, 'lr__intercept_scaling': 1, 'lr__C': 100}

>              precision    recall  f1-score   support

>         0.0       0.98      1.00      0.99       130

>         1.0       0.98      0.96      0.97       130

>         2.0       0.99      0.95      0.97       119

>         3.0       0.91      0.94      0.92       129

>         4.0       0.99      0.97      0.98       130

>         5.0       0.93      0.96      0.95       128

>         6.0       0.98      0.97      0.97       124

>         7.0       0.92      0.95      0.93       126

>         8.0       0.90      0.93      0.91       139

>         9.0       0.90      0.83      0.87       120

> avg / total       0.95      0.95      0.95      1275

It is not very exciting, but it is a good solution. How about we scale up the network an use different grid of parameters? It is worth to try.

Code snippt:

```python
param_grid = {
    'rbm1__n_components': [36, 25, 16],
    'lr__penalty': ['l2', 'l1'],
    'lr__C': [1, 10, 100]
}

steps = [
    ('rbm1', BernoulliRBM()), 
    ('lr', LogisticRegression())
]

pipeline = Pipeline(steps)

grid_search = GridSearchCV(pipeline, param_grid, n_jobs = -1, verbose = 1, cv = 3)

n_trains = n_samples / 3 * 2

# We learn the digits on the first half of the digits
grid_search.fit(data[:n_trains], target[:n_trains])
```

### Random forest

It seems that the '9' is always hard to tell than '0'. How about random forest?

From random forest, I can get my best score here:

> Best score: 0.972

> Best parameters set:

> {'rf__bootstrap': True, 'rf__max_depth': None, 'rf__n_estimators': 90, 'rf__verbose': 0, 'rf__criterion': 'gini', 'rf__min_density': None, 'rf__min_samples_split': 2, 'rf__compute_importances': None, 'rf': RandomForestClassifier(bootstrap=True, compute_importances=None,

>             criterion='gini', max_depth=None, max_features='auto',

>             min_density=None, min_samples_leaf=1, min_samples_split=2,

>             n_estimators=90, n_jobs=1, oob_score=False, random_state=None,

>             verbose=0), 'rf__max_features': 'auto', 'rf__n_jobs': 1, 'rf__random_state': None, 'rf__oob_score': False, 'rf__min_samples_leaf': 1}

>              precision    recall  f1-score   support

>         0.0       0.99      0.99      0.99       130

>         1.0       0.98      0.98      0.98       130

>         2.0       1.00      0.98      0.99       119

>         3.0       0.95      0.97      0.96       129

>         4.0       0.98      0.99      0.99       130

>         5.0       0.98      0.99      0.98       128

>         6.0       0.98      0.99      0.99       124

>         7.0       0.98      0.98      0.98       126

>         8.0       0.99      0.97      0.98       139

>         9.0       0.96      0.93      0.94       120

> avg / total       0.98      0.98      0.98      1275


### K Nearest Neighbors



## References

1. [Comparing Classification Algorithms for Handwritten Digits](http://blog.quantitations.com/machine%20learning/2013/02/27/comparing-classification-algorithms-for-handwritten-digits/)

2. [Example: Handwritten Digit Classification](http://pythonhosted.org/bob.learn.boosting/example.html)

3. [Classification of handwritten digits using a SVM](http://nbviewer.ipython.org/url/www.hdm-stuttgart.de/~maucher/ipnotebooks/MachineLearning/svmDigitRecognition.ipynb)

4. [Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html)

5. [Recognizing hand-written digits](http://scikit-learn.org/stable/auto_examples/plot_digits_classification.html)

6. [The MNIST Database of Handwritten Digits](http://yann.lecun.com/exdb/mnist/)

7. [Modular Toolkit for Data Processing](http://mdp-toolkit.sourceforge.net/documentation.html)

8. [Scikit-learn document](http://scikit-learn.org/stable/)

9. [Handwritten digits classification with MDP and scikits.learn](http://mdp-toolkit.sourceforge.net/examples/scikits_learn/digit_classification.html)
