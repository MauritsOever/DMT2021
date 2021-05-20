# -*- coding: utf-8 -*-
"""
Created on Tue May 18 15:35:42 2021

@author: MauritsOever
"""

import pandas as pd

# load in data, see what happens
# fingers crossed...

Train_original = pd.read_csv(r'C:\Users\gebruiker\Documents\GitHub\DMT2021\ASS2\Data\training_set_VU_DM.csv').iloc[1:10001,]
Test_original = pd.read_csv(r'C:\Users\gebruiker\Documents\GitHub\DMT2021\ASS2\Data\test_set_VU_DM.csv').iloc[1:10001,]

Train = Train_original
Test = Test_original


#clean a bit, take the columns we're interested in to predict booking
columns = ['srch_id','site_id','visitor_location_country_id','prop_country_id', 'prop_id',
           'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2'
           ,'prop_log_historical_price','price_usd','promotion_flag', 'srch_destination_id','srch_length_of_stay',
           'srch_booking_window','srch_adults_count','srch_children_count','srch_room_count','srch_saturday_night_bool',
           'year', 'month', 'day', 'booking_bool']

Train = Train[columns]
Test = Test[columns[0:len(columns)-1]]

corr = Train.corr()

# get rid of NA's
Train = Train.fillna(Train.mean())    
Test = Test.fillna(Test.mean())

Train = Train.dropna()
Test = Test.dropna()

# fit and forecast Random forest
from sklearn.ensemble import RandomForestClassifier
ytrain = Train.iloc[:,-1]
xtrain = Train.iloc[:,:-1]

# fit and get probabilities
rf = RandomForestClassifier(n_estimators=100).fit(xtrain, ytrain)
Test['book_prob'] = rf.predict_proba(Test)[:,1]

# get output in format needed...
Test = Test.sort_values(['srch_id','book_prob'], ascending = (True,False))
output = Test[['srch_id','prop_id']]
