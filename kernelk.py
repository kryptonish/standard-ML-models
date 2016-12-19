import numpy as np 
import sklearn as skl 
import csv
from numpy import array
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn import svm
import random
from sklearn import datasets
from copy import deepcopy

def distrbf(datarr,point,gamma):
		# norm1 = np.linalg.norm(point - point)
		# norm = norm1*norm1
		inter1 = np.exp(0)
		summ = 0
		for val in datarr:
			norm1 = np.linalg.norm(val - point)
			norm = norm1*norm1
			summ = summ + np.exp(-gamma*norm)
		summ = 2*summ
		if len(datarr) == 0:
			kami = 0.01
		else:
			kami = len(datarr)
		inter2 = -summ/kami
		summ1 = 0
		for val1 in datarr:
			for val2 in datarr:
				norm1 = np.linalg.norm(val1 - val2)
				norm = norm1*norm1
				summ1 = summ1 + np.exp(-gamma*norm)
		inter3 = summ1/(kami*kami)
		return inter1 + inter2 + inter3

def distpoly(datarr,point,c):
		inter1 = np.dot(point,point) + c
		summ = 0
		for val in datarr:
			norm = np.dot(point,val) + c
			# print np.shape(val),"nisahnt raifhi"
			summ = summ + norm
		summ = 2*summ
		if len(datarr) == 0:
			kami = 0.000000000001
		else:
			kami = len(datarr)
		inter2 = -summ/kami
		summ1 = 0
		for val1 in datarr:
			for val2 in datarr:
				norm = np.dot(val2,val2) + c
				norm = norm*norm
				summ1 = summ1 + norm
		inter3 = summ1/(kami*kami)
		return inter1 + inter2 + inter3




# datarr = []
# with open('arcene_train.csv', 'rb') as csvfile:
# 	spamreader = csv.reader(csvfile, delimiter=' ')
# 	i = 0
# 	for row in spamreader:
# 		row = row[:-9901]
# 		datarr.append(np.array(row).astype(float))


iris = datasets.load_iris()
dims = 2
X = iris.data[:, :dims] 
X = X[1:71,:]
Y = iris.target
Y = Y[1:71]

numclusters = 2
datarr = []
datarr = X

gamma = 0.00000001
initkmeans = np.random.random((numclusters,dims))
datarr = np.array(datarr)
clusterdict = {}


for i in range(1,numclusters+1):
	clusterdict[i] = []

for val1 in datarr:
	minn = 1000000000
	i = 1
	for val2 in initkmeans:
		norm = np.linalg.norm(val1 - val2)
		if norm < minn:
			ind = i
			minn = norm
		i = i + 1
	clusterdict[ind].append(val1)


revdict = {}
for i in range(1,numclusters+1):
	 revdict[i] = []

# for val in clusterdict:
# 	print len(clusterdict[val])

numiterations = 5
for x in range(numiterations):
	for val1 in datarr:
		minn = 1000000000
		for val2 in clusterdict:
			dist = distrbf(clusterdict[val2],val1,gamma)
			if dist < minn:
				minn = dist
				ind = val2
		revdict[ind].append(val1)
	for val in revdict:
		print len(revdict[val]),
	finaldict = deepcopy(revdict)
	print("\n")
	clusterdict = revdict
	for i in range(1,numclusters+1):
		revdict[i] = []

for val in finaldict:
	print len(finaldict[val])
print np.unique(Y)
Y = np.ndarray.tolist(Y)
a = np.unique(Y)
for val in a:
	print Y.count(val)


# c = 5
# for x in range(numiterations):
# 	for val1 in datarr:
# 		minn = 1000000000
# 		i = 1
# 		for val2 in clusterdict:
# 			dist = distpoly(clusterdict[val2],val1,c)
# 			# print dist,"sufbcaisdhbcfilsdnfclihfbelaihrdhbclaeirbhd "
# 			if dist < minn:
# 				minn = dist
# 				ind = i
# 			i = i + 1
# 		revdict[ind].append(val2)
# 	# for val in revdict:
# 	# 	print len(revdict[val])
# 	clusterdict = revdict
# 	for i in range(1,numclusters+1):
# 		revdict[i] = []








