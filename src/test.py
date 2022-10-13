import matplotlib.pyplot as plt
import pylab

data_01 = [1,2,3,4,5,6,7,8,9]
data_02 = [15,16,17,18,19,20,21,22,23,24,25]
data_03 = [5,6,7,8,9,10,11,12,13]

BoxName = ['data 01','data 02','data 03']

data = [data_01,data_02,data_03]

plt.boxplot(data)

plt.ylim(0,30)

pylab.xticks([1,2,3], BoxName)

plt.savefig('MultipleBoxPlot02.png')
plt.show()