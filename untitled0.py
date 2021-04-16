#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 21:39:28 2021

@author: connorstevens

Lost my code for the first two graphs I uploaded. Note to self: save frequently.
"""
#Import packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#Import data.
df = pd.read_csv('/Users/connorstevens/Downloads/df.csv')


#Survival histograms based on fares.
plt.hist(df['Fare'][df['Survived'] == 1], label = 'Survived', )
plt.hist(df['Fare'][df['Survived'] == 0], label = 'Died', alpha = 0.5)
plt.legend()

#Survival histograms based on fares above 50.
temp = df[df['Fare'] > 40]
plt.hist(temp['Fare'][temp['Survived'] == 1], label = 'Survived', )
plt.hist(temp['Fare'][temp['Survived'] == 0], label = 'Died', alpha = 0.5)
plt.legend()

#Survival histograms based on age.
plt.hist(df['Age_Group'][df['Survived'] == 1], bins = [10, 20, 30, 40, 50, 60, 70, 80], label = 'Survived')
plt.hist(df['Age_Group'][df['Survived'] == 0], bins = [10, 20, 30, 40, 50, 60, 70, 80], label = 'Died', alpha = 0.5)
plt.legend()

#Survival histograms based on Embark location.
plt.hist(df['Embarked'][df['Survived'] == 1], label = 'Survived')
plt.hist(df['Embarked'][df['Survived'] == 0], label = 'Died', alpha = 0.5)
plt.legend()

#Survival histograms based on class.
plt.hist(df['Pclass'][df['Survived'] == 1], label = 'Survived' )
plt.hist(df['Pclass'][df['Survived'] == 0], label = 'Died', alpha = 0.5)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.legend()

#Survival histograms based on number of parents/children aboard.
plt.hist(df['Parch'][df['Survived'] == 1], label = 'Survived', bins = [0, 1, 2, 3, 4, 5, 6])
plt.hist(df['Parch'][df['Survived'] == 0], label = 'Died', alpha = 0.5 , bins = [0, 1, 2, 3, 4, 5, 6])
plt.xlabel('Number of parents/children aboard')
plt.ylabel('Frequency')
plt.legend()

#Survival histograms based on number of siblings/spouses aboard.
plt.hist(df['SibSp'][df['Survived'] == 1], label = 'Survived', bins = [0, 1, 2, 3, 4, 5, 6])
plt.hist(df['SibSp'][df['Survived'] == 0], label = 'Died', alpha = 0.5 , bins = [0, 1, 2, 3, 4, 5, 6])
plt.xlabel('Number of siblings/spouse aboard')
plt.ylabel('Frequency')
plt.legend()

#Survival histograms based on number of sex.
plt.hist(df['Sex'][df['Survived'] == 1], label = 'Survived')
plt.hist(df['Sex'][df['Survived'] == 0], label = 'Died', alpha = 0.5)
plt.xlabel('Men or Women (Woman = 0, Men = 1)')
plt.ylabel('Frequency')
plt.legend()



plt.hist(df['Parch'][df['Survived'] == 0], label = 'Died', alpha = 0.5)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.legend()

sns.pairplot(df)

fig1, axs = plt.subplots(2, 2)
axs[0, 0].hist(df['Age_Group'][df['Survived'] == 1], bins = [10, 20, 30, 40, 50, 60, 70, 80], label = 'Survived')
axs[0, 0].hist(df['Age_Group'][df['Survived'] == 0], bins = [10, 20, 30, 40, 50, 60, 70, 80], label = 'Died', alpha = 0.5)
#axs[0,0].set_title('Age')
axs[0,0].set(xlabel = 'Age')
fig1.legend()
axs[0, 1].hist(df['Pclass'][df['Survived'] == 1], label = 'Survived' )
axs[0, 1].hist(df['Pclass'][df['Survived'] == 0], label = 'Died', alpha = 0.5 )
axs[0,1].set(xlabel = 'Class')
axs[1, 0].hist(df['Sex'][df['Survived'] == 1], label = 'Survived')
axs[1, 0].hist(df['Sex'][df['Survived'] == 0], label = 'Died', alpha = 0.5)
axs[1,0].set(xlabel = 'Sex (Woman = 0, Man = 1)')
axs[1, 1].hist(df['Fare'][df['Survived'] == 1], label = 'Survived')
axs[1, 1].hist(df['Fare'][df['Survived'] == 0], label = 'Died', alpha = 0.5)
axs[1,1].set(xlabel = 'Fare')
fig1.tight_layout()

fig2, axs = plt.subplots(2)
axs[0].hist(df['Parch'][df['Survived'] == 1], label = 'Survived', bins = [0, 1, 2, 3, 5, 6])
axs[0].hist(df['Parch'][df['Survived'] == 0], label = 'Died', alpha = 0.5, bins = [0, 1, 2, 3, 5, 6])
axs[0].set(xlabel = 'Parent/Children Aboard')
fig2.legend()
axs[1].hist(df['SibSp'][df['Survived'] == 1], label = 'Survived', bins = [0, 1, 2, 3, 5])
axs[1].hist(df['SibSp'][df['Survived'] == 0], label = 'Died', alpha = 0.5, bins = [0, 1, 2, 3, 5])
axs[1].set(xlabel = 'Siblings/Spouse Aboard')
fig2.tight_layout()



sns.distplot(df['Age'])

sns.distplot(df['Fare'])

plt.hist(df['Fare'])

mDescStats = df.describe()

np.quantile(df['Fare'], 0.8)

plt.scatter(df['Pclass'], df['Fare'])

sum(df['Pclass'] == 1)
sum(df['Pclass'] == 2)
sum(df['Pclass'] == 3)

"""
DECISION TREE
"""
from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split

#Setting features.
features = ['Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Age_Group']

#Defining features and prediction target
X = df[features]
y = df.Survived

#Defining survival model as decision tree.
survival_model = tree.DecisionTreeClassifier()

#Fit model.
#survival_model.fit(X, y)

#Predict using fitted model.
#prediction = survival_model.predict(X)

#Measure mean absolute error for prediction.
#mean_absolute_error(y, prediction)

#Split dataset into train and test sets.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

#Fit training set.
survival_model.fit(train_X, train_y)

#Predict on test set.
train_prediction = survival_model.predict(val_X)

#Measure mean absolute error for test set predictions.
mean_absolute_error(val_y, train_prediction)

f1_score(val_y, train_prediction)

#Loop through differing number of leaves to compare mean absolute errors.
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = tree.DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    #AUC = auc(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [4, 40, 100, 120, 200, 400, 4000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(my_mae)

#MAE stops decreaseing above 200 leaves

"""
RANDOM FORESTS
"""
from sklearn.ensemble import RandomForestClassifier

#Select random forest for classifier model.
survival_model_forest = RandomForestClassifier(random_state = 0)

#fit model.
survival_model_forest.fit(train_X, train_y)

#Predict on test set.
forest_prediction = survival_model_forest.predict(val_X)

#Measure mean absolute error for test set predictions.
mean_absolute_error(val_y, forest_prediction)

f1_score(val_y, forest_prediction)
  