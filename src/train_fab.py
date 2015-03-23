import numpy as np
import pandas as pd
from sklearn import datasets, svm, metrics
from sklearn.grid_search import GridSearchCV

fab = pd.read_csv('/Users/panda/data/ATM_Example/para_raw2.txt')
fab.loc[:, fab.dtypes == np.float64]
data = fab.select_dtypes(include=[np.float64, np.int64])

cols = [col for col in data.columns if col not in ['error_code', 'sdk_unit']]
nonnan = data[cols].fillna(method='pad').dropna(how="any")

cols = [col for col in nonnan.columns if col not in ['para_flag']]
features = nonnan[cols].values
targets = nonnan['para_flag'].values

n_samples = len(targets)

param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

# Create a classifier: a support vector classifier
classifier = svm.SVC()

grid_search = GridSearchCV(classifier, param_grid, n_jobs = -1, verbose = 1, cv = 3)

n_trains = n_samples / 3 * 2

# We learn the digits on the first half of the digits
grid_search.fit(features[:n_trains], targets[:n_trains])

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
print best_parameters

# Now predict the value of the digit on the second half:
expected = targets[n_trains:]
predicted = grid_search.best_estimator_.predict(features[n_trains:])

print(metrics.classification_report(expected, predicted))
