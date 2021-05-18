# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:53:14 2021

@author: MauritsOever

"""
import pandas as pd

# load in data, see what happens
# fingers crossed...

Train_full = pd.read_csv(r'C:\Users\gebruiker\Documents\GitHub\DMT2021\ASS2\Data\training_set_VU_DM.csv').iloc[1:10000,]
Test_full = pd.read_csv(r'C:\Users\gebruiker\Documents\GitHub\DMT2021\ASS2\Data\test_set_VU_DM.csv').iloc[1:10000,]

Train = Train_full
Test = Test_full

#clean a bit
columns = ['srch_id','site_id','visitor_location_country_id','prop_country_id', 'prop_id',
           'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2'
           ,'prop_log_historical_price','price_usd','promotion_flag', 'srch_destination_id','srch_length_of_stay',
           'srch_booking_window','srch_adults_count','srch_children_count','srch_room_count','srch_saturday_night_bool','booking_bool']

Train = Train[columns]
Test = Test[columns[0:20]]

#get rid of nans:
    #- try dropping and try replacing w/ means...
    #- factor analysis missed data (FAMD)
Train = Train.fillna(Train.mean())    
Test = Test.fillna(Test.mean())

Train = Train.dropna()
Test = Test.dropna()

# from here, try to implement a sklearn gradient boosting classifier
from sklearn.ensemble import GradientBoostingClassifier
xtrain =  Train.iloc[:,:-1]
ytrain = Train['booking_bool']

# not specify and fit:
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(xtrain, ytrain)
preds = clf.predict_proba(Test)

# predict proba's
Test['book_prob'] = clf.predict_proba(Test)[:,1]

# get output in format needed...
Test = Test.sort_values(['srch_id','book_prob'], ascending = (True,False))
output = Test[['srch_id','prop_id']]



