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

# kind of attributes 
# range of values -- distribution??
    # programme: category, no range
    # taken a course on ml? --> yes, no, or unknown
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





