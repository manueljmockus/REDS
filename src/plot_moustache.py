from cProfile import label
from utils.read_data import read_data_SG
import numpy as np
import matplotlib.pyplot as plt

import pylab

file, predictors, target, users = read_data_SG()

users_count = [(users == i).sum() for i in range(1,9)]
target_count = [(target == i).sum() for i in range(1,25)]
## preprocess 
normalized_predictors = (predictors - np.mean(predictors, axis = 0))/ np.std(predictors, axis = 0)


prod = np.random.choice(np.arange(2400), 200)
pred = predictors[:, prod]
normalized_pred = normalized_predictors[:, prod]

print(normalized_pred[3:].shape)

BoxName = ['','s_1','s_2','s_3', 'o_1', 'o_2', 'o_3', 'o_4',]
bname2 = ['', 'o_1', 'o_2', 'o_3', 'o_4',]
for i in range (1,23):
    BoxName.append('g_'+str(i))
    bname2.append('g_'+str(i))

plt.boxplot(np.transpose(normalized_pred[3:]), sym = "")
plt.xlabel("feature")
plt.ylabel("values")
plt.title("Caracteristiques relevantes apres normalisation")

pylab.xticks(np.arange(27), bname2)

plt.savefig('MultipleBoxPlot02.png')
plt.show()

