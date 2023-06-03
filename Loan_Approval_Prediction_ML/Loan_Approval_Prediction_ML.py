import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('LoanApprovalPrediction.csv')


# EDA
obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)

#plt.figure(figsize=(18,36))
#index = 1
#
#for col in object_cols:
#    y = data[col].value_counts()
#    plt.subplot(3,5,index)
#    plt.xticks(rotation=90)
#    sns.barplot(x=list(y.index), y=y)
#    index +=1
#plt.show()


# Data Preprocessing
#data = data.drop(['Loan_ID'], axis=1, inplace= True)
from sklearn import preprocessing
label_enc = preprocessing.LabelEncoder()
for col in list(obj[obj].index):
    data[col] = label_enc.fit_transform(data[col])
#print(data.dtypes)
#print(data)


# EDA
#plt.figure(figsize=(12,6))
#sns.heatmap(data.corr(), cmap='BrBG', linewidths=2, annot=True)
#plt.show()

#sns.catplot(x='Gender', y='Married',
#            hue='Loan_Status',
#            kind='bar',
#            data=data)
#plt.show()


# Replacing N/A values
for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())
#print(data.isna().sum())


# Splitting Data
from sklearn.model_selection import train_test_split

X = data.drop(['Loan_Status', 'Loan_ID'], axis=1)
Y = data['Loan_Status']

#print(X.shape, Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=22)

#print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


# Model Training
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(n_estimators=7, criterion='entropy',random_state=22)
svc = SVC()
lr = LogisticRegression()

for clf in (rfc, knn, svc, lr):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_train)
    print("Accuracy score of train in ",
          clf.__class__.__name__,
          "=",100*metrics.accuracy_score(Y_train, Y_pred))
    
for clf in (rfc, knn, svc, lr):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    print("Accuracy score of test in ",
          clf.__class__.__name__,
          "=",100*metrics.accuracy_score(Y_test, Y_pred))



