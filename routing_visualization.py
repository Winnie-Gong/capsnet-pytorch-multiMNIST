import numpy as np
import csv
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import random_projection
import torch.nn.functional as F

random_proj = random_projection.SparseRandomProjection(2)
output = genfromtxt('routing_data/routing_out.csv', delimiter=',')
output_reduced = random_proj.fit_transform(output)
for i in range(10):
	X = genfromtxt('routing_data/routing_in_'+str(i)+'.csv', delimiter=',')
	X = random_proj.fit_transform(X)
	plt.scatter(X[:,0], X[:,1], color="blue")
	plt.scatter(output_reduced[i][0], output_reduced[i][1], color="green")
	plt.show()