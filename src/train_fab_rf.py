import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import datasets, svm, metrics
from sklearn.grid_search import GridSearchCV

kernel = None
if len(sys.argv) == 1:
    kernel = None
else:
    kernel = sys.argv[1]

C = 1
gamma = 0.001
param_grid = {'C': [C], 'kernel': ['linear']}

if kernel == 'linear':
    C = float(sys.argv[2])
    param_grid = {'C': [C], 'kernel': [kernel]}
elif kernel == 'rbf':
    C = float(sys.argv[2])
    gamma = float(sys.argv[3])
    param_grid = {'C': [C], 'gamma': [gamma], 'kernel': [kernel]}
else:
    pass


fab = pd.read_csv('/home/sen/data/ATM_Example/para_raw2.txt')
fab.loc[:, fab.dtypes == np.float64]
data = fab.select_dtypes(include=[np.float64, np.int64])

cols = [col for col in data.columns if col not in ['error_code', 'sdk_unit']]
nonnan = data[cols].fillna(method='pad').dropna(how="any")

cols = [col for col in nonnan.columns if col not in ['para_flag']]
features = nonnan[cols].values
targets = nonnan['para_flag'].values

n_samples = len(targets)

param_grid = {
    'rf__n_estimators': [40, 50, 60, 70, 80, 90]
}

steps = [
    ('rf', RandomForestClassifier())
]

pipeline = Pipeline(steps)

grid_search = GridSearchCV(pipeline, param_grid, n_jobs = -1, verbose = 1, cv = 3)

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
