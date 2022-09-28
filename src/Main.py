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
fig, axis = plt.subplots(1,2)
for i in range(29):
    axis[0].scatter(np.arange(200), pred[i,:])
    axis[1].scatter(np.arange(200), normalized_pred[i,:])
plt.tight_layout()
plt.show()

