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
from pprint import pprint
from sklearn.impute import SimpleImputer


#Import data.
df = pd.read_csv('/Users/connorstevens/OneDrive - Vrije Universiteit Amsterdam/DMT/Assignment 1/train.csv')

for i in range(df.shape[0]):
    if(df['Sex'][i] == 'male'):
        df ['Sex'][i] = 1
    else:
        df['Sex'][i] = 0
        
    if(df['Embarked'][i] == 'C'):
        df['Embarked'][i] = 3
    elif(df['Embarked'][i] == 'Q'):
        df['Embarked'][i] = 2
    elif(df['Embarked'][i] == 'S'):
        df['Embarked'][i] = 1
        
df['Age_Group'] = pd.cut(df['Age'], 8, precision = 0, labels = [10, 20, 30, 40, 50, 60, 70, 80])

df['Age'].fillna((df['Age'].median()), inplace=True)
df['Fare'].fillna((df['Fare'].median()), inplace=True)
df['Embarked'].fillna((df['Embarked'].median()), inplace=True)
df['Age_Group'] = pd.cut(df['Age'], 8, precision = 0, labels = [10, 20, 30, 40, 50, 60, 70, 80])


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
survival_model = tree.DecisionTreeClassifier(random_state = 1, max_leaf_nodes= 200)

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
    model = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    F1 = f1_score(val_y, preds_val)
    AUC = auc(val_y, preds_val)
    return(mae, F1)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [4, 40, 100, 120, 200, 400, 4000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(my_mae)

#MAE stops decreaseing above 200 leaves

"""
RANDOM FORESTS
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

#Setting features.
features = ['Sex', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Age_Group']

#Defining features and prediction target
X = df[features]
y = df.Survived

#Select random forest for classifier model.
survival_model_forest = RandomForestClassifier(random_state = 1)


#fit model.
survival_model_forest.fit(X, y)


####RANDOM GRID SEARCH
#Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(survival_model_forest.get_params())

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
survival_model_forest = RandomForestClassifier(random_state = 1)

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
survival_random = RandomizedSearchCV(estimator = survival_model_forest, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
survival_random.fit(train_X, train_y)

#View best parameters.
survival_random.best_params_

#Compare hyperparameters obtained above to standard random forest.
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f}'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

base_model = RandomForestClassifier(n_estimators = 10, random_state = 42)
base_model.fit(train_X, train_y)
base_accuracy = evaluate(base_model, val_X, val_y)

best_random = survival_random.best_estimator_
random_accuracy = evaluate(best_random, val_X, val_y)

predic = best_random.predict(val_X)

mean_absolute_error(val_y, predic)

f1_score(val_y, predic)

####RANDOM GRID SEARCH WITH CROSS-VALIDATION
from sklearn.model_selection import GridSearchCV

# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [4, 8],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
cv_survival_forest = RandomForestClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = cv_survival_forest, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X, y)

gridpred = grid_search.predict(val_X)

f1_score(val_y, gridpred)

#Import test set.
df_test = pd.read_csv('/Users/connorstevens/OneDrive - Vrije Universiteit Amsterdam/DMT/Assignment 1/test.csv')

for i in range(df_test.shape[0]):
    if(df_test['Sex'][i] == 'male'):
        df_test['Sex'][i] = 1
    else:
        df_test['Sex'][i] = 0
        
    if(df_test['Embarked'][i] == 'C'):
        df_test['Embarked'][i] = 3
    elif(df_test['Embarked'][i] == 'Q'):
        df_test['Embarked'][i] = 2
    elif(df_test['Embarked'][i] == 'S'):
        df_test['Embarked'][i] = 1
        
df_test['Age_Group'] = pd.cut(df_test['Age'], 8, precision = 0, labels = [10, 20, 30, 40, 50, 60, 70, 80])

df_test['Age'].fillna((df['Age'].median()), inplace=True)
df_test['Fare'].fillna((df['Fare'].median()), inplace=True)
df_test['Embarked'].fillna((df['Embarked'].median()), inplace=True)

df_test['Age_Group'] = pd.cut(df_test['Age'], 8, precision = 0, labels = [10, 20, 30, 40, 50, 60, 70, 80])



df_test['Survived'] = best_random.predict(df_test[features])

submission = df_test[['PassengerId', 'Survived']]

submission.to_csv('/Users/connorstevens/OneDrive - Vrije Universiteit Amsterdam/DMT/Assignment 1/Submission.csv')
