# -*- coding: utf-8 -*-
"""Socialmedia Ads classification.ipynb
"""

import numpy
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("social media ads classifier.csv")

data["Purchased"] = data["Purchased"].map({0:"Not Purchase",1:"Purchase"})

"""### Data Preparation"""
x = np.array(data[["Age", "EstimatedSalary"]])
y = np.array(data[["Purchased"]])

"""### Splitting"""
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)

"""### Modelling"""
dt = DecisionTreeClassifier()
dt.fit(xtrain, ytrain)

# Saving model to disk
pickle.dump(dt, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[40,50000]]))

