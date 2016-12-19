import numpy as np 
import sklearn as skl 
import csv
from numpy import array
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn import svm

DIMENSIONS = 2


def sqdistance(data):
    distance_matrix = []
    for i in data:
        distance_from_others = []
        for j in data:
            distance_from_others.append(np.linalg.norm(i - j) * np.linalg.norm(i - j))
        distance_matrix.append(np.array(distance_from_others))
    return np.array(distance_matrix)


def rbfkernelmat(datamat, gamma):
    return np.exp(-gamma * datamat)

def linkernelmat(data):
    dotp_matrix = []
    for i in data:
        dotp_with_others = []
        for j in data:
            dotp_with_others.append(np.dot(i,j) * np.dot(i,j))
        dotp_matrix.append(np.array(dotp_with_others))
    return np.array(dotp_matrix)

def projectdata(point,data,gamma,eig_vec,eig_val):
    pair_dist = []
    for row in data:
        dist = np.sum((np.array(point)-np.array(row))**2)
        pair_dist.append(dist)
    pair_dist = np.array(pair_dist)
    gdist = np.exp(-gamma*pair_dist)
    return gdist.dot(eig_vec/eig_val)

def projectdatal(point,data,gamma,eig_vec,eig_val):
    pair_dist = []
    for row in data:
        dist = np.dot(point,row)
        pair_dist.append(dist)
    pair_dist = np.array(pair_dist)
    gdist = np.exp(-gamma*pair_dist)
    return gdist.dot(eig_vec/eig_val)

datarr = []
datarr2 = []

with open('arcene_train.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    i = 0
    for row in spamreader:
        row = row[:-1]
        datarr.append(np.array(row).astype(float))

datarr3 = []
with open('arcene_valid.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    i = 0
    for row in spamreader:
        row = row[:-1]
        datarr3.append(np.array(row).astype(float))

datarr4 = []
with open('arcene_validlabels.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    i = 0
    for row in spamreader:
        row = np.array(row).astype(float)
        datarr4.append(row)


#rbf plotting and others
# datarr = np.array(datarr)
# distance_matrix = sqdistance(datarr)
# var = 10
# rbfkm = rbfkernelmat(distance_matrix, var)
# print np.shape(rbfkm)
# mean = np.mean(rbfkm, axis=0)
# print len(mean)
# var = np.var(rbfkm, axis=0)
# rbfckm = rbfkm - mean
# eig_val, eig_vec = np.linalg.eig(rbfckm)
# indexes = eig_val.argsort()[::-1]
# eig_vec = eig_vec / np.sqrt(eig_val)
# topk = eig_vec[:, indexes[0:DIMENSIONS]].real
# reduce_mat1 = np.dot(rbfkm, topk)


datarr = np.array(datarr)
linkm = linkernelmat(datarr)
print np.shape(linkm)
mean = np.mean(linkm, axis=0)
var = np.var(linkm, axis=0)
linckm = linkm - mean
eig_val, eig_vec = np.linalg.eig(linckm)
indexes = eig_val.argsort()[::-1]
eig_vec = eig_vec / np.sqrt(eig_val)
topk = eig_vec[:, indexes[0:DIMENSIONS]].real
reduce_mat1 = np.dot(linkm, topk)



with open('arcene_trainlabels.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    i = 0
    for row in spamreader:
        row = np.array(row).astype(float)
        datarr2.append(row)

datarr2 = [item for sublist in datarr2 for item in sublist]
datarr4 = [item for sublist in datarr4 for item in sublist]

clf1 = svm.SVC()
clf1.fit(reduce_mat1, datarr2)

gamma = 0.0000001
datarr3 = np.array(datarr3)

final = []
for val in datarr3:
    points = []
    for i in range(DIMENSIONS):
        op = projectdatal(val,datarr,gamma,topk[:,i],eig_val[indexes[i]])
        points.append(op)
    final.append(points)

final = np.array(final)
final = np.real(final)
maxes = np.amax(final,axis = 0)
final[:,0] = final[:,0]/maxes[0]
final[:,1] = final[:,1]/maxes[1]

predictions1 = []
for i in range(len(final)):
    predictions1.append(clf1.predict([final[i]]))
predictions1 = [item for sublist in predictions1 for item in sublist]

datarr4 = np.array(datarr4)
count = 0
preds = []
for i in range(len(datarr4)):
    lim = datarr4[i] - predictions1[i]
    preds.append(lim)
    if lim == 0:
        count = count + 1

print count

