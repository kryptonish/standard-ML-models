import numpy as np 
import sklearn as skl 
import csv
from numpy import array
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from sklearn import svm



def rbfkernelmat(datamat, gamma):
	return np.exp(-gamma * datamat)

def kerdistance(data,total):
	distance_matrix = []
	for i in total:
		distance_from_others = []
		for j in data:
			distance_from_others.append(np.linalg.norm(i - j) * np.linalg.norm(i - j))
		distance_matrix.append(np.array(distance_from_others))
	return np.array(distance_matrix)

def oker1(a,b,gamma):
	val = np.linalg.norm(a - b)
	val = val * val
	return np.exp(-gamma * val)


def sqdistance(p1,p2,gamma):
	dist = np.sum((np.array(p1)-np.array(p2))**2)
	a = np.exp(-gamma*dist)
	return a

datarr = []
datarr2 = []
with open('arcene_train.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    i = 0
    for row in spamreader:
        row = row[:-1]
        datarr.append(np.array(row).astype(float))

with open('arcene_trainlabels.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    i = 0
    for row in spamreader:
        row = np.array(row).astype(float)
        datarr2.append(row)

datarr2 = [item for sublist in datarr2 for item in sublist]
datarr = np.array(datarr)
class1 = []
class2 = []
i=0

for val in datarr:
	if datarr2[i] == 1:
		class1.append(val)
	else:
		class2.append(val)
	i = i + 1
gamma = 0.00000000000000000010
M1 = []
M2 = []

tot = class2 + class1
K1 = kerdistance(class1,tot)
K2 = kerdistance(class2,tot)

x = len(class1)
y = len(class2)


K1 = rbfkernelmat(K1,gamma)
K2 = rbfkernelmat(K2,gamma)




for row in K1:
	M1.append(sum(row)/x)

for row in K2:
	M2.append(sum(row)/y)

I1 = np.identity(44)
I2 = np.identity(56)

A2 = np.full((56,56),1/56)
A1 = np.full((44,44),1/44)

M1 = np.array(M1)
M2 = np.array(M2)

a = M2 - M1

t1 = np.transpose(K1)
t2 = np.transpose(K2)
N1 = np.dot(K1,I1-A1)
N1 = np.dot(N1,t1)
N2 = np.dot(K2,I2-A2)
N2 = np.dot(N2,t2)
N = N1+N2
NINV = np.linalg.inv(N)
alphas = np.dot(NINV,a)

datarr3 = []
with open('arcene_valid.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    i = 0
    for row in spamreader:
        row = row[:-1]
        datarr3.append(np.array(row).astype(float))
datarr3 = np.array(datarr3)


datarr4 = []
with open('arcene_validlabels.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ')
    i = 0
    for row in spamreader:
        row = np.array(row).astype(float)
        datarr4.append(row)
datarr4 = [item for sublist in datarr4 for item in sublist]

trainarr = []
for val1 in datarr:
	summ = 0
	for val2 in datarr:
		summ = summ + oker1(val1,val2,gamma)
	trainarr.append(summ)

trainarr = np.array(trainarr)
datarr2 = np.array(datarr2)

trainarr = trainarr.reshape(100,1)
datarr2 =  datarr2.reshape(100,1)

clf1 = svm.SVC()
clf1.fit(trainarr, datarr2)
print np.shape(datarr3)
trainvalid = []
for val1 in datarr3:
	summ = 0
	for val2 in datarr3:
		summ = summ + oker1(val1,val2,gamma)
	trainvalid.append(summ)
count = 0 
i = 0
for val in trainvalid:
	b = clf1.predict(val)
	if b == datarr4[i]:
		count = count + 1
	i = i + 1

print count
# print np.shape(datarr)
# yarr = []
# for val1 in datarr:
# 	i = 0
# 	sum = 0
# 	for val2 in datarr:
# 		inter = oker1(val1,val2,gamma)

# 		res = alphas[i]*inter
# 		sum = sum + res
# 		i = i + 1
# 	yarr.append(sum)

# zero1 = np.full((44),0)
# zero2 = np.full((56),0)
# c1 = []
# c2 = []

# i = 0
# for val in datarr2:
# 	if val == 1:
# 		c1.append(yarr[i])
# 	if val == -1:
# 		c2.append(yarr[i])

# print len(c1),len(c2)
# # print datarr2
# plt.scatter(c1, zero1,color='red')
# # # plt.scatter(c2, zero2,color='blue')

# plt.show()













































