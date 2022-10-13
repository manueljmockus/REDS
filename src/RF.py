
from utils.read_data import read_data_SG
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import pylab

file, predictors, target, users = read_data_SG()

users_count = [(users == i).sum() for i in range(1,9)]
target_count = [(target == i).sum() for i in range(1,25)]
## preprocess 
normalized_predictors = (predictors - np.mean(predictors, axis = 0))/ np.std(predictors, axis = 0)
final_predictors = normalized_predictors[3:].transpose()
target = target.transpose()
print(final_predictors.shape)
print(target.shape)

seed = 1234567
X_train, X_split, y_train, y_split = train_test_split(final_predictors, target, test_size=0.7,  random_state=seed, shuffle=True)
X_validation, X_test, y_validation, y_test = train_test_split(X_split, y_split, test_size=0.5,  random_state=seed, shuffle=True)


rf_clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None)

rf_clf.fit(X_train, y_train)

y_pred = rf_clf.predict(X_validation)
print(y_pred)
print(y_validation - y_pred)