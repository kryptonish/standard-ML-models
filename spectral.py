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




iris = datasets.load_iris()
X = iris.data[:, :4]  # we only take the first two features.
numclusters = 3
datarr = []
datarr = X
Y = iris.target
datarr = np.array(datarr)
maxx = 0
dists = []
for val1 in datarr:
	for val2 in datarr:
		norm = np.linalg.norm(val1 - val2)
		dists.append(norm)
		if norm > maxx:
			maxx = norm
print maxx
graphadj = np.zeros((150, 150))
dist = 1.1
for i in range(len(datarr)):
	for j in range(len(datarr)):
		idist = np.linalg.norm(datarr[i] - datarr[j])
		if idist < dist:
			graphadj[i][j] = idist

# for val in graphadj:|
# 	print np.count_nonzero(val)
	# print len(val)

degree = np.zeros((150, 150))

for i in range(len(graphadj)):
	a = np.count_nonzero(graphadj[i])
	degree[i][i] = a

laplacian = degree - graphadj
for val in laplacian:
	print val

eig_val, eig_vec = np.linalg.eig(laplacian)
indexes = eig_val.argsort()[::-1]
eig_vec = np.dot(np.transpose(datarr),eig_vec) #convert eigen vectors to the dimensions of the data
topkk = 2
indices = []
for val in indexes:
	if eig_val[val] != 0:
		indices.append(val)

topk = eig_vec[:, indices[-topkk:]].real
print np.shape(topk),"sakbashcsdifgdiugs"
print np.shape(datarr)

finalmat =  np.dot(datarr,topk)
print np.shape(finalmat)


dims = 2
clusterdict = {}
for i in range(1,numclusters+1):
	clusterdict[i] = []

initkmeans = np.random.random((2,dims))

datarr = finalmat
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
gamma = 0.000001
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










