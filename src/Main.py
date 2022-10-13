from cProfile import label
from utils.read_data import read_data_SG
import numpy as np
import matplotlib.pyplot as plt

file, predictors, target, users = read_data_SG()

users_count = [(users == i).sum() for i in range(1,9)]
target_count = [(target == i).sum() for i in range(1,25)]
## preprocess 
print(np.mean(predictors, axis = 0))
normalized_predictors = (predictors - np.mean(predictors, axis = 0))/ np.std(predictors, axis = 0)


prod = np.random.choice(np.arange(2400), 200)
pred = predictors[:, prod]
normalized_pred = normalized_predictors[:, prod]

min = np.min(predictors, axis = 1)
max = np.max(predictors, axis = 1)
mean = np.mean(predictors, axis = 1)
plt.subplot(1, 2, 1)
plt.scatter(np.arange(29), min, label = 'min')
plt.scatter(np.arange(29), max, label = 'max')

plt.ylabel("value")
plt.xlabel("feature")
plt.title("Min/max Values Before Normalization")
min_normal = np.min(normalized_pred, axis = 1)
max_normal = np.max(normalized_pred, axis = 1)
plt.subplot(1, 2, 2)
plt.scatter(np.arange(29), min_normal, label = 'min')
plt.scatter(np.arange(29), max_normal, label = 'max')
plt.title("Min/max Values After Normalization")

plt.ylabel("value")
plt.xlabel("feature")

for i in range(29):
    print(i, min[i], max[i])
plt.tight_layout()

plt.show()

import pylab

data_01 = [1,2,3,4,5,6,7,8,9]
data_02 = [15,16,17,18,19,20,21,22,23,24,25]
data_03 = [5,6,7,8,9,10,11,12,13]

BoxName = ['s_1','s_2','s_3']

for i in range (22):
    BoxName.append
data = [data_01,data_02,data_03]

plt.boxplot(predictors)

plt.ylim(0,30)

pylab.xticks(np.arange(29), BoxName)

plt.savefig('MultipleBoxPlot02.png')
plt.show()

