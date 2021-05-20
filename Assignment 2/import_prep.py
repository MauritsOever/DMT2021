#Import packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import datetime

#Import dataset
raw_data = pd.read_csv("/Users/connorstevens/OneDrive - Vrije Universiteit Amsterdam/DMT/Assignment 2/Data/training_set_VU_DM.csv")


'''
FEATURE ENGINEERING
'''
############################################################

"""
Create target variable which will be used for training models.
5 - The user purchased a room at this hotel
1 - The user clicked through to see more information on this hotel
0 - The user neither clicked on this hotel nor purchased a room at this hotel
"""

#Set 'target' equal to zero by default
raw_data['target'] = 0

#Set 'target equal to 1 if click_bool = 1.
raw_data.loc[raw_data.loc[:,'click_bool'] == 1,'target'] = 1

#Set 'target equal to 5 if booking_bool = 1.
raw_data.loc[raw_data.loc[:,'booking_bool'] == 1,'target'] = 5

'''
Create weekend proximity variable.
'''
#Set date_time to datetime variable.
raw_data['date_time'] = pd.to_datetime(raw_data['date_time'])

#Create day_of_week variable.
raw_data['day_of_week'] = raw_data['date_time'].dt.dayofweek 

#Create hour_of_day variable.
raw_data['hour_of_day'] = raw_data['date_time'].dt.hour

#Plot clicks and books by day_of_week.
plt.hist(raw_data['day_of_week'][raw_data['target'] == 1], bins = 7, label = 'Clicked', alpha = 0.5)
plt.hist(raw_data['day_of_week'][raw_data['target'] == 5], bins = 7, label = 'Booked', alpha = 0.3)
plt.xlabel('Days of the Week')
plt.ylabel('Frequency')
plt.xticks([0, 1, 2, 3, 4, 5, 6],labels = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.legend()
plt.show()

#Plot clicks and books by hour_of_day.
plt.hist(raw_data['hour_of_day'][raw_data['target'] == 1], bins = 24, label = 'Clicked', alpha = 0.5)
plt.hist(raw_data['hour_of_day'][raw_data['target'] == 5], bins = 24, label = 'Booked', alpha = 0.3)
plt.xlabel('Days of the Week')
plt.ylabel('Frequency')
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
plt.legend()
plt.show()

"""
NaN exploration.
"""
#Count number of nan values by column.
nan_info= pd.DataFrame({'nan_count': raw_data.isnull().sum()})
nan_info['nan_ratio'] = raw_data.isnull().sum()/np.shape(raw_data)[0]

#Plot missing values.
msno.matrix(raw_data)
plt.show()

msno.heatmap(raw_data)
plt.show()

#msno.bar(raw_data)

#Plot ratios of missing data.
bars = nan_info.index.values
height = nan_info.nan_ratio
y_pos = np.arange(len(bars))
plt.figure(figsize=(15, 10))
plt.bar(y_pos, height)
plt.xticks(fontsize = 20)
plt.yticks(fontsize=20)
plt.ylabel('Ratio of Missing Values', fontsize = 30)
plt.xlabel('Variables', fontsize = 30)
plt.show()

#Count number of variables within given missing value ratio ranges.
print(sum(nan_info.nan_ratio > 0.5), 'variables have more than 50% of the data missing.')
print(sum(nan_info.nan_ratio > 0.6), 'variables have more than 60% of the data missing.')
print(sum(nan_info.nan_ratio > 0.8), 'variables have more than 80% of the data missing.')
print(sum(nan_info.nan_ratio > 0.9), 'variables have more than 90% of the data missing.')


#Count number of unique properties.
unique_props = len(np.unique(raw_data.price_usd))

#compare average expedia price to average competitor prices (given large missing data)
#to see if mean or median would be a good substitute for missing data.

#Calculate expedia mean.
expedia_mean = np.mean(raw_data.price_usd.dropna())

##Calculate competitor means.

#First calculate competitor prices by using Expedia price and percentage difference.

#Create list of columns names.
comp_df = pd.DataFrame({'competitors' :['comp1_rate', 'comp2_rate', 'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate']})
comp_df['means'] = np.zeros(len(comp_df.competitors))
for count, col in comp_df.competitors:
    comp_df.iloc[count] = np.mean(raw_data[col].dropna())
    

