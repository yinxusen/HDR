import numpy
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
