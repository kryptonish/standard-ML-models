import numpy as np 
import sklearn as skl 
import csv
from numpy import array
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn import svm
import random
from sklearn import datasets


# def criterion1():


# def criterion2():

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
numclusters = 3
datarr = []
datarr = X
Y = iris.target
# print Y 
# numclusters = 2
# datarr = []
# with open('arcene_train.csv', 'rb') as csvfile:
# 	spamreader = csv.reader(csvfile, delimiter=' ')
# 	i = 0
# 	for row in spamreader:
# 		row = row[:-9991]
# 		datarr.append(np.array(row).astype(float))
# print "this is the iris dataset"
datarr = np.array(datarr)
clust = {}
for i in range(len(datarr)):
	arr = []
	[ arr.append(val) for val in datarr[i] ]
	# clust[i] =np.array(datarr[i])
	clust[i] = []
	clust[i].append(arr)

# mergeclust = {}
# index = [0,1]
# while len(clust) > numclusters:
# 	print len(clust)
# 	minn = 100000000
# 	for val1 in clust:
# 		for val2 in clust:
# 			if val1!=val2:
# 				minw = 10000000
# 				for item1 in clust[val1]:
# 					for item2 in clust[val2]:
# 						norm = np.linalg.norm(np.array(item1)-np.array(item2))
# 						if norm < minw:
# 							minw = norm

# 				if minw<minn:
# 					index = [val1,val2]
# 					minn = minw
# 	for y in range(len(clust[index[1]])):
# 		clust[index[0]].append(clust[index[1]][y])
# 	del clust[index[1]]


# for val in clust:
# 	print len(clust[val])

mergeclust = {}
index = [0,1]
while len(clust) > numclusters:
	print len(clust)
	maxx = 0
	for val1 in clust:
		for val2 in clust:
			if val1!=val2:
				maxw = 0
				for item1 in clust[val1]:
					for item2 in clust[val2]:
						norm = np.linalg.norm(np.array(item1)-np.array(item2))
						if norm > maxw:
							maxw = norm

				if maxw>maxx:
					index = [val1,val2]
					maxx = maxw
	for y in range(len(clust[index[1]])):
		clust[index[0]].append(clust[index[1]][y])
	del clust[index[1]]


for val in clust:
	print len(clust[val])






