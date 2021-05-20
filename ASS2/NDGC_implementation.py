# -*- coding: utf-8 -*-
"""
Created on Thu May 20 14:08:32 2021

@author: MauritsOever
"""

import pandas as pd
import numpy as np

# load in data, see what happens
# fingers crossed...

def get_target_col(df):
    df['true_y'] = np.full((len(df),1),0)
    
    df.iloc[df['click_bool']==1,-1] = 1
    df.iloc[df['booking_bool']==1,-1] = 5
    
    
    # for i in range(1,len(df)):
    #     if df['click_bool'][i] == 1:
    #         df['true_y'][i]  = 1
    #     if df['booking_bool'][i] ==1:
    #         df['true_y'][i] = 5

Train_original = pd.read_csv(r'C:\Users\gebruiker\Documents\GitHub\DMT2021\ASS2\Data\training_set_VU_DM.csv').iloc[1:10001,]
# Test_original = pd.read_csv(r'C:\Users\gebruiker\Documents\GitHub\DMT2021\ASS2\Data\test_set_VU_DM.csv').iloc[1:10001,]

get_target_col(Train_original)

# rerun this if you want to refresh data used:
# =============================================================================
Train = Train_original
#Test = Test_original


#clean a bit, take the columns we're interested in to predict booking
columns = ['srch_id','site_id','visitor_location_country_id','prop_country_id', 'prop_id',
           'prop_starrating', 'prop_review_score', 'prop_brand_bool', 'prop_location_score1', 'prop_location_score2'
           ,'prop_log_historical_price','price_usd','promotion_flag', 'srch_destination_id','srch_length_of_stay',
           'srch_booking_window','srch_adults_count','srch_children_count','srch_room_count','srch_saturday_night_bool', 'true_y']

Train = Train[columns]
#Test = Test[columns[0:len(columns)-1]]
# =============================================================================



# fit model here i guessss....

