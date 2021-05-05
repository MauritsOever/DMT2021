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
        
# okay we can now do linprob, probit, logit and staff for the course vars

