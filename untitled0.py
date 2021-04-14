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
df = pd.read_csv('/Users/connorstevens/OneDrive - Vrije Universiteit Amsterdam/DMT/Assignment 1/train.csv')

df['Age_Group'] = pd.cut(df['Age'], 8, precision = 0, labels = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80'])

sns.pairplot(df)
sns.plt.show()

sns.distplot(df['Age'])

sns.distplot(df['Fare'])

plt.hist(df['Fare'])

mDescStats = df.describe()

np.quantile(df['Fare'], 0.8)

plt.scatter(df['Pclass'], df['Fare'])

sum(df['Pclass'] == 1)
sum(df['Pclass'] == 2)
sum(df['Pclass'] == 3)
