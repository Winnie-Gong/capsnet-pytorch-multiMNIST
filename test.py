import numpy as np
import csv
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import random_projection
import torch.nn.functional as F

# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x), axis=0)

# def squash(v):
# 	norm_v = np.linalg.norm(v)
# 	v = np.square(norm_v)/(1+np.square(norm_v))/norm_v*v
# 	return v

# def routing(u):
# 	i = u.shape[0]
# 	b = np.zeros((i,1))
# 	iterations = 3
# 	for r in range(iterations):
# 		print("----------",r)
# 		c =  softmax(b)
# 		print(c)
# 		s = np.sum(np.multiply(u, c), axis=0)
# 		print(s)
# 		v = squash(s)
# 		print(v)
# 		print(np.sum(np.multiply(u,np.transpose(v)), axis=1).reshape(i,1))
# 		b = b + np.sum(np.multiply(u,np.transpose(v)), axis=1).reshape(i,1)
# 	return v

# pca = PCA(n_components=2)
random_proj = random_projection.SparseRandomProjection(2)
output = genfromtxt('routing_data/routing_out.csv', delimiter=',')
# output_reduced = pca.fit_transform(output)
output_reduced = random_proj.fit_transform(output)
# print(output.shape)
for i in range(10):
	X = genfromtxt('routing_data/routing_in_'+str(i)+'.csv', delimiter=',')
	# my_routing = routing(X)
	# print("-------")
	# print(my_routing)
	# print(output[i])
	# # my_routing = pca.fit_transform(my_routing)
	X = random_proj.fit_transform(X)
	plt.scatter(X[:,0], X[:,1], color="blue")
	plt.scatter(output_reduced[i][0], output_reduced[i][1], color="green")
	# plt.scatter(my_routing[i][0], my_routing[i][1], color="red")
	plt.show()