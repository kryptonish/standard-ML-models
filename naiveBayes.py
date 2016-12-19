import pandas as pd 
import numpy as np 
import random
a= []
b = []
for i in range(0,42):
	a.append(i)
# df = pd.read_csv('census-income.csv',header = a)
df = pd.read_csv('census-income.csv',header = None,sep=',\s', na_values=["?"],engine = "python")
df.fillna(df.mode().iloc[0],inplace=True)

df1 = pd.read_csv('census-incometest.csv',header = None,sep=',\s', na_values=["?"],engine = "python")
df1.fillna(df1.mode().iloc[0],inplace=True)

print len(df1[3])
# a = random.sample(range(1, 19000), 1000)
# df = []
# for val in a:
# 	df.append(df1[val])
# a = df[1:20]
# print a[1] 
# df = a
# df.columns = a

c = df[41].value_counts()
valsum = c[0] + c[1] 
p1 = c[1]/valsum
p2 = c[0]/valsum
col = 41
count1 = 0
count2 = 0

ind1 = [k for k, x in enumerate(df[41]) if x == '50000+.']
ind2 = [k for k, x in enumerate(df[41]) if x == "- 50000."]
# print len(ind1)
# print len(ind2)
testlist = []
for i in range(0,41):
	if df[i].dtype == 'object':
		testlist.append(i)

mydict = {}
for i in range(0,42):
	if df[i].dtype == 'object':
		unique = []
		unique = df[i].unique()
		for items in unique:
			ind3 = [k for k, x in enumerate(df[i]) if x == items]
			ind4 = set(ind1).intersection(ind3)
			ind5 = set(ind2).intersection(ind3)
			P1 = float(len(ind4))/float(len(ind1))
			P2 = float(len(ind5))/float(len(ind2))
			mydict[items] = []
			mydict[items].append(P1)
			mydict[items].append(P2)

counter = 0
c = 1
for i in range(27000,28000):
	prod1 = p1
	prod2 = p2
	for val in testlist:
		prod1 = prod1*mydict[df1[val][i]][0]
		prod2 = prod1*mydict[df1[val][i]][1]
	if prod1 > prod2:
		c = 1
	elif prod1 < prod2:
		c = 2
	if df1[41][i] ==  '50000+.' and c == 2:
		counter = counter + 1
	elif df1[41][i] == "- 50000." and c == 1:
		counter = counter + 1

print counter

final_list1 = np.ar[4692,4661,4713,4668,4658]
final_list2 = [944,934,933,938,941]

mean1 = (sum(final_list1)/5)*2;
mean2 = (sum(final_list2)/5);

mean3 = (mean1 + mean2)/2;






