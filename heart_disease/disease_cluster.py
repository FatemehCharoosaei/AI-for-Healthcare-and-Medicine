# -*- coding: utf-8 -*-


## dataset = https://www.kaggle.com/ronitf/heart-disease-uci/version/1

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
from sklearn.metrics import accuracy_score


subjects_df = pd.read_csv('heart.csv')
#features = subjects_df[['age', 'cp', 'trestbps', 'chol']]

subjects = subjects_df.values
features = subjects[:, [2, 6, 9, 10, 11, 12]]
plt.scatter(subjects_df['chol'], subjects_df['target'])



model = KMeans(n_clusters=2)
model = model.fit(features)

plt.scatter(subjects_df['chol'], subjects_df['target'], c=model.labels_.astype(np.float))


results = model.labels_
targets = subjects_df['target'].values

#accuracy = (1 - sum(results ^ targets) / 303) * 100
#acc = accuracy_score(results, targets) * 100
