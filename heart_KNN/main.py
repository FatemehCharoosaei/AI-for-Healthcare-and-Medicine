# -*- coding: utf-8 -*-


## dataset = https://www.kaggle.com/ronitf/heart-disease-uci/version/1

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split;
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


subjects_df = pd.read_csv('heart.csv')
#features = subjects_df[['age', 'cp', 'trestbps', 'chol']]
plt.scatter(subjects_df['chol'], subjects_df['target'])

subjects = subjects_df.values
features = subjects[:, [2, 6, 9, 10, 11, 12]]
label = subjects[:, 13]


X_train, X_test, Y_train, Y_test = train_test_split(features, label, test_size=.1) #cross validation 


## KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)
knn_predictions = knn_model.predict(X_test)
knn_acc = accuracy_score(knn_predictions, Y_test) * 100


## SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, Y_train)
svm_predictions = svm_model.predict(X_test)
svm_acc = accuracy_score(svm_predictions, Y_test) * 100


pr = precision_recall_fscore_support(Y_test, svm_predictions)
cm = confusion_matrix(Y_test, svm_predictions)
plt.matshow(cm)






