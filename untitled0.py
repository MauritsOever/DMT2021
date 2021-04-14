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
axs[0,0].set_title('Age')
fig1.legend()
axs[0, 1].hist(df['Pclass'][df['Survived'] == 1], label = 'Survived' )
axs[0, 1].hist(df['Pclass'][df['Survived'] == 0], label = 'Died', alpha = 0.5 )
axs[0,1].set_title('Class')
axs[1, 0].hist(df['Parch'][df['Survived'] == 1], label = 'Survived', bins = [0, 1, 2, 3, 4, 5, 6])
axs[1, 0].hist(df['Parch'][df['Survived'] == 0], label = 'Died', bins = [0, 1, 2, 3, 4, 5, 6], alpha = 0.5)
axs[1,0].set_title('Parents/Children Aboard')
axs[1, 1].hist(df['Parch'][df['Survived'] == 1], label = 'Survived', bins = [0, 1, 2, 3, 4, 5, 6])
axs[1, 1].hist(df['Parch'][df['Survived'] == 0], label = 'Died', bins = [0, 1, 2, 3, 4, 5, 6], alpha = 0.5)
axs[1, 1].set_title('Siblings/Spouse Aboard')
fig1.tight_layout()

fig2, axs = plt.subplots(2)
axs[0].hist(df['Fare'][df['Survived'] == 1], label = 'Survived')
axs[0].hist(df['Fare'][df['Survived'] == 0], label = 'Died', alpha = 0.5)
fig2.legend()
axs[0].set_title('Fare')
axs[1].hist(temp['Fare'][temp['Survived'] == 1], label = 'Survived', )
axs[1].hist(temp['Fare'][temp['Survived'] == 0], label = 'Died', alpha = 0.5)
axs[1].set_title('Fare > 40')
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
