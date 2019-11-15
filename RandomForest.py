# !/usr/bin/python3
# coding: utf-8

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os
import csv

path = os.getcwd() + '\\'
file = path + 'GreenView.csv'

csv_file = open(file, encoding='utf-8')
csv_reader = csv.reader(csv_file)
header = next(csv_reader)
GV = pd.read_csv("GreenView.csv", dtype={'GreenViewRate': int})

x_data = []
x_header = []
y_data = []
y_header = []
time_y = []

for x in csv_reader:
	if x[0] >= '0':
		x = [float(x) for x in x[1:]]
		x_data.append(x)

for x in range(0, 71):
	y_header.append(x)

x_data = [[int(x) for x in row] for row in x_data]
x_data = np.array(x_data)
x_header = np.array(header[1:])

data = GV.values.tolist()
y_data = [x[0] for x in data]
y_data = np.array(y_data)
# y_data = y_data.reshape(-1, 1)
# y_header = np.array(header[0])

print(x_data)
print(x_header)
print(y_data)
print(y_header)
# print(xdata[:5], len(xdata), type(xdata), xheader[:5])


df = pd.DataFrame(x_data, columns=x_header)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
df['species'] = pd.Categorical.from_codes(y_data, y_header)

train, test = df[df['is_train'] == True], df[df['is_train'] == False]
features = df.columns[:7]
clf = RandomForestClassifier(n_jobs=2)

y, _ = pd.factorize(train['species'])
clf.fit(train[features], y)
preds = np.array(y_header)[clf.predict(test[features])]
print(preds)

pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds'])

mmm = clf.feature_importances_
nnn = list(mmm)

nnn = [x * 100 for x in nnn]
nnn = [float('%.5f' % x) for x in nnn]
print(nnn)
