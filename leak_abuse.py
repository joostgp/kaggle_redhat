# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 13:07:58 2016

@author: joostbloom
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

people = pd.read_csv('./data_ori/people.csv', 
                     parse_dates=['date'], 
                     dtype={'people_id': np.str, 'activity_id': np.str, 'char_38': np.int32})
train = pd.read_csv('./data_ori/act_train.csv', 
                    parse_dates=['date'], 
                    dtype={'people_id': np.str, 'activity_id': np.str, 'outcome': np.int8})
test = pd.read_csv('./data_ori/act_test.csv', 
                   parse_dates=['date'], 
                   dtype={'people_id': np.str, 'activity_id': np.str})

def preprocess(df):
    for col in df.columns:
        if col not in ['people_id', 'activity_id', 'date', 'char_38', 'outcome']:
            if df[col].dtype == 'object':
                df[col].fillna('type 0', inplace=True)
                df[col] = df[col].apply(lambda x: x.split(' ')[1]).astype(np.int32)
            elif df[col].dtype == 'bool':
                df[col] = df[col].astype(np.int8)
    
    return df
    

for df in [people, train, test]:
    df = preprocess(df)

train = pd.merge(train, people, how='left', on='people_id')
test = pd.merge(test, people, how='left', on='people_id')

print('Number of unique groups in train:', train.group_1.nunique())
print('Number of unique groups in test:', test.group_1.nunique())
print('Number of unique groups in test and not in train:', len(set(test.group_1)-set(train.group_1)))
print('Number of unique groups in train and not in test:', len(set(train.group_1)-set(test.group_1)))

grps_not_in_train = set(test.group_1)-set(train.group_1)

print('Number of people_id without leak',test[test.group_1.isin(grps_not_in_train)].shape)


def guess_outcome_v1(x):
    
    fill_if_group_not_in_train = 0.123456
    border_fill = 0.1
    
    """
    10% of groups makes one switch
    2% of groups makes two switches
    1 makes three switches
    Cases:
    a) All in train have same values (around 2/3 of groups)
        - All dates within range same value as mean
            - What if there is a large date gap? - probably not bad if t
        - What to do with dates outside range?
    b) Switch in train (around 1/3 of groups)
    """ 
    #print x.group_1[0]
    
#    if x.group_1[0]==51462:
#        print('Error ding found')
#        #x['interpolated'] = 0
#        return x
#    
#    if x.group_1[0]>51375:
#        print x.sample(1), x.shape, sum(x.outcome.isnull()), x.outcome.mean() 
#    
    
    # If 
    #if x.outcome.mean()==0:
    #    x['interpolated'] = 0
    #    return x
    
    #if x.outcome.mean()==1:
    #    x['interpolated'] = 1
    #    return x
    
    if x.shape[0]-sum(x.outcome.isnull())>0: # and x.shape[0]>2:
        #x.index = x.date_x
        if 0:
            x['interpolated'] = x.outcome.interpolate(method='linear')
            x.interpolated.fillna(method='ffill', inplace=True)
            x.interpolated.fillna(method='bfill', inplace=True)
        else:
            x['ffill'] = x.outcome.fillna(method='ffill')
            x['bfill'] = x.outcome.fillna(method='bfill')
            ffill_null = x['ffill'].isnull()
            bfill_null = x['bfill'].isnull()
            x.loc[ffill_null,'ffill'] = np.abs(x.loc[ffill_null,'bfill'] - border_fill)
            x.loc[bfill_null,'bfill'] = np.abs(x.loc[bfill_null,'ffill'] - border_fill)
            x['interpolated'] = x[['ffill','bfill']].mean(axis=1)
            
    else:
        x['interpolated'] = fill_if_group_not_in_train
    
    
    
    if np.random.random()>0.99:
    #    pass
         print x.sample(1), x.shape, sum(x.outcome.isnull()), x.outcome.mean(), x.interpolated.mean()  
        
    return x.reset_index(drop=True)
    
def kaggle_interp(x):
    x = x.reset_index(drop=True)
    g = x['outcome'].copy() ## g should be a list or a pandas Series.
    if g.shape[0] < 3: ## If we have at most two rows.
        x['filled'] = g ## Will be replaced by a mean.
        return x
    missing_index = g.isnull()
    
    if sum(missing_index)==g.shape[0]: # if no train data is available
        x['filled'] = 0.505669
        return x
    borders = np.append([g.index[0]], g[~missing_index].index, axis=0)
    borders = np.append(borders, [g.index[-1]+1], axis=0)
    forward_border = borders[1:]
    backward_border = borders[:-1]
    forward_border_g = g[forward_border]
    backward_border_g = g[backward_border]
    ## Interpolate borders.
    ## TODO: Why does the script author use the value 0.1?
    border_fill = 0.1
    forward_border_g[forward_border_g.index[-1]] = abs(forward_border_g[forward_border_g.index[-2]]-border_fill)
    backward_border_g[backward_border_g.index[0]] = abs(forward_border_g[forward_border_g.index[0]]-border_fill)
    times = forward_border-backward_border
    forward_x_fill = np.repeat([forward_border_g], times)#.reset_index(drop=True)
    backward_x_fill = np.repeat([backward_border_g], times)#.reset_index(drop=True)
    vec = (forward_x_fill+backward_x_fill)/2
    g[missing_index] = vec[g[missing_index].index] ## Impute missing values only.
    x['filled'] = g
    return x

# Fill test outcome with known dates and groups from train dataset  
meanoutcome = train.groupby(['group_1','date_x']).outcome.agg('mean').to_frame().reset_index()
test = test.merge(meanoutcome, on=['group_1','date_x'], how='left')
#test['outcome']=np.nan

test['istest']=True

all_samples = pd.concat([train, test] )
all_samples.sort_values(['group_1','date_x'], inplace=True)
all_samples = all_samples[['group_1','outcome','date_x','activity_id', 'istest']]

x = all_samples.groupby('group_1').apply(guess_outcome_v1)
#x = x.groupby('group_1').apply(kaggle_interp)
xx = x.reset_index(drop=True)
xx = xx[xx.istest==True][['activity_id','interpolated']]
xx.columns = ['activity_id','outcome']
xx.to_csv('leak_abuse_V1.csv',index=False)
#x = all_samples[all_samples.group_1.isin([8,10])].groupby('group_1').apply(guess_outcome)

benchmark = pd.read_csv('Submission.csv')
benchmark.columns = ['activity_id','outcome_ori']

comp2 = benchmark.merge(xx, on='activity_id', how='right')
comp2['diffe'] = comp2['outcome_ori'] - comp2['outcome']
comp2.diffe.value_counts()
comp2.sort_values('diffe').head()