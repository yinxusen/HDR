import sys
import numpy as np
import pandas as pd
from sklearn import datasets, svm, metrics
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.lda import LDA

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
    'pca1__n_components': [16],
    'poly__degree': [3],
    'pca2__n_components': [0.99],
    'lda__n_components': [25],
    'lr__penalty': ['l2'],
    'lr__C': [0.1, 1]
}

steps = [('pca1', PCA()), 
    ('poly', PolynomialFeatures()),
    ('pca2', PCA()), 
    ('lda', LDA()),
    ('lr', LogisticRegression())]

# Create a classifier: a support vector classifier
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
