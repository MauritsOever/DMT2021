# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:42:26 2021

@author: Maurits van den Oever
"""
print()
print()
# packages 
# set directory...
import os
os.chdir(r"C:\Users\gebruiker\Documents\GitHub\DMT2021")
import pandas as pd
import numpy as np
import re
import seaborn
import matplotlib
import collections

# load in
df = pd.read_csv(r"C:\Users\gebruiker\Documents\GitHub\DMT2021\ODI-2021.csv")
print("the amount of records are", len(df)) # 313
print("the amount of attributes are", len(df.columns)) # 17

atts = df.columns # get attributes

# kind of attributes 
# range of values -- distribution??
    # programme: category, no range
    # taken a course on ml? --> yes, no, or unknown
    # taken a course on inf rets? --> 1, 0 or unkown
    # taken a course on stats? --> mu, sigma, unkown
    # taken a course on databases? --> ja, nee, unknown
    # gender? --> male, female, unknown
    # chocolate makes u fat? --> slim, fat, no idea, neither
    # birthday? --> dates, but format is all fucked up
    # neighbours sitting nex to u? --> ints, but some are jokers and need to be removed
    # did u stand up? --> yes or no
    # stress level? --> (1-100) some are jokes and NTBR
    # money for competition? --> number from 0-100 max, some need to be removed
    # randn? --> range from 0-10, the rest needs to be removed
    # time you went to bed yesterday? --> times, but need a way to format into observables
    # requisite for good day 1?
    # requisite for good day 2? --> both strings, might be able to sort into categories...

# so lets start the formatting process
# change all the 'course' questions to 1 and 0, treat the unknowns as NaN
df.ML_course[df.ML_course == 'yes'] = 1
df.ML_course[df.ML_course == 'no'] = 0
df.ML_course[df.ML_course == 'unknown'] = np.nan
df['ML_course'] = pd.to_numeric(df['ML_course'])

df.InfRet_course[df.InfRet_course == 'unknown'] = np.nan
df['InfRet_course'] = pd.to_numeric(df['InfRet_course'])

df.stats_course[df.stats_course == 'mu'] = 1
df.stats_course[df.stats_course == 'sigma'] = 0
df.stats_course[df.stats_course == 'unknown'] = np.nan
df['stats_course'] = pd.to_numeric(df['stats_course'])

df.database_course[df.database_course == 'ja'] = 1
df.database_course[df.database_course == 'nee'] = 0
df.database_course[df.database_course == 'unknown'] = np.nan
df['database_course'] = pd.to_numeric(df['database_course'])

# okay we can now do linprob, probit, logit and staff for the course vars

# lets do some more work on the categorical variables:
df['programme'] = df['programme'].str.lower()

for i in range(len(df['programme'])):
    # AI:
    if 'ai' in df.programme[i] or 'artificial' in df.programme[i]:
        df.programme[i] = 'AI'
    
    # QRM
    elif 'qrm' in df.programme[i] or 'risk' in df.programme[i]:
        df.programme[i] = 'QRM'
    
    # Computational Science
    elif 'comp' in df.programme[i] or 'cs' in df.programme[i]:
        df.programme[i] = 'CS'
    
    elif 'language' in df.programme[i]:
        df.programme[i] = 'Human Language Technology'
        
    elif 'ba' in df.programme[i] or 'business' in df.programme[i]:
        df.programme[i] = 'BA'
        
    elif 'informati' in df.programme[i]:
        df.programme[i] = 'Information studies or Data Science'
    
    elif 'finance' in df.programme[i] or 'f&t' in df.programme[i]:
        df.programme[i] = 'FinTech'

    else:
        df.programme[i] = 'Other'


# lets do stress level next:
for i in range(len(df)):
    try:
        if df.stress[i] < 0:
            df.stress[i] = 0
        elif df.stress[i] > 100:
            df.stress[i] = 100
        break
    except TypeError:
        df.stress[i] = float(re.sub('\D','',df.stress[i]))
        if df.stress[i] < 0:
            df.stress[i] = 0
        elif df.stress[i] > 100:
            df.stress[i] = 100

for i in range(len(df)):
    try:
        if df.randn[i] < 0:
            df.randn[i] = 0
        elif df.randn[i] > 10:

df.stress = pd.to_numeric(df.stress)

## do gender next:
for i in range(len(df)):
    if df.gender[i] == 'male':
        df.gender[i] = 1
    elif df.gender[i] == 'female':
        df.gender[i] = 0
    else:
        df.gender[i] = np.nan


# what are the most cited words for a good day?
word_count = {}

for i in range(len(df['goodday1'])):
# for i in range(1):
    str1 = re.sub(r'[^\w\s] ', '', df.iloc[i, 15].lower())
    str2 = re.sub(r'[^\w\s] ', '',df.iloc[i, 16].lower())
    
    words1 = str(str1).split(' ')
    words2 = str(str2).split(' ')
    
    for word in words1:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
            
    for word in words2:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

# now remove the non-sensical keys:
bad_keys = ['good','a',' ', 'no','with','the','nice','and','my','being','day','of','having','in','i','to','when','doing','not','out','stress','is','something','']
for key in bad_keys:
    word_count.pop(key, None)

# Print most common word
n_print = 5

word_counter = collections.Counter(word_count)
for word, count in word_counter.most_common(n_print):
    print(word, ": ", count)


# Create a data frame of the most common words 
# Draw a bar chart
lst = word_counter.most_common(n_print)
commons = pd.DataFrame(lst, columns = ['Word', 'Count']) # can't plot for the life of me ffs

print('Fraction of people that have taken a', df.columns[2], 'is', np.sum(df.iloc[:,2])/len(df))
print('Fraction of people that have taken a',df.columns[3], 'is', np.sum(df.iloc[:,3])/len(df))
print('Fraction of people that have taken a', df.columns[4], 'is', np.sum(df.iloc[:,4])/len(df))
print('Fraction of people that have taken a', df.columns[5], 'is', np.sum(df.iloc[:,5])/len(df))
print('Percentage of males taking DMT is', np.sum(df.iloc[:,6])/len(df))




