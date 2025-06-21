# -*- coding: utf-8 -*
    
'''
@author: kuaidouai
'''

import os
import sys
import timeit
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '../../data'))
fresh_comp_offline_dir = os.path.join(data_dir, 'fresh_comp_offline')

mobile_raw_dir = os.path.join(data_dir, 'mobile', 'raw')
os.makedirs(mobile_raw_dir, exist_ok=True)

##### file path
# input 
path_df_D = os.path.join(mobile_raw_dir, 'df_part_1.csv')

# output
path_df_part_1 = os.path.join(mobile_raw_dir, 'df_part_1.csv')
path_df_part_2 = os.path.join(mobile_raw_dir, 'df_part_2.csv')
path_df_part_3 = os.path.join(mobile_raw_dir, 'df_part_3.csv')

path_df_part_1_tar = os.path.join(mobile_raw_dir, 'df_part_1_tar.csv')
path_df_part_2_tar = os.path.join(mobile_raw_dir, 'df_part_2_tar.csv')

path_df_part_1_uic_label = os.path.join(mobile_raw_dir, 'df_part_1_uic_label.csv')
path_df_part_2_uic_label = os.path.join(mobile_raw_dir, 'df_part_2_uic_label.csv')
path_df_part_3_uic       = os.path.join(mobile_raw_dir, 'df_part_3_uic.csv')

########################################################################
'''Step 1: divide the data set to 3 part

    part 1 - train: 11.22~11.27 > 11.28;
    part 2 - train: 11.29~12.04 > 12.05;
    part 3 - test: 12.13~12.18 (> 12.19);
    
    here we omit the geo info
'''

batch = 0
for df in pd.read_csv(
        open(os.path.join(fresh_comp_offline_dir, 'tianchi_fresh_comp_train_user.csv'), 'r'), 
                      parse_dates=['time'], 
                      date_format='%Y-%m-%d %H',
                      index_col=['time'],
                      chunksize = 100000):  # operation on chunk as the data file is too large
    try:
        # 对时间索引进行排序
        df = df.sort_index()
        
        df_part_1     = df['2014-11-22':'2014-11-27']
        df_part_1_tar = df['2014-11-28':'2014-11-28']
        df_part_2     = df['2014-11-29':'2014-12-04']
        df_part_2_tar = df['2014-12-05':'2014-12-05']
        df_part_3     = df['2014-12-13':'2014-12-18']

        df_part_1.to_csv(path_df_part_1,  
                         columns=['user_id','item_id','behavior_type','item_category'],
                         header=False, mode='a')        
        df_part_1_tar.to_csv(path_df_part_1_tar,
                         columns=['user_id','item_id','behavior_type','item_category'],
                         header=False, mode='a')
        df_part_2.to_csv(path_df_part_2,  
                         columns=['user_id','item_id','behavior_type','item_category'],
                         header=False, mode='a')  
        df_part_2_tar.to_csv(path_df_part_2_tar,
                         columns=['user_id','item_id','behavior_type','item_category'],
                         header=False, mode='a')   
        df_part_3.to_csv(path_df_part_3,  
                         columns=['user_id','item_id','behavior_type','item_category'],
                         header=False, mode='a')
        
        batch += 1
        print('chunk %d done.' %batch) 
        
    except StopIteration:
        print("divide the data set finish.")
        break 

########################################################################
'''Step 2 construct U-I-C_label of df_part 1 & 2
                    U-I-C of df_part 3      
'''

##### part_1 #####
# uic
data_file = open(path_df_part_1, 'r')
try:
    df_part_1 = pd.read_csv(data_file, index_col = False)
    df_part_1.columns = ['time','user_id','item_id','behavior_type','item_category']
finally:
    data_file.close()
df_part_1_uic = df_part_1.drop_duplicates(['user_id', 'item_id', 'item_category'])[['user_id', 'item_id', 'item_category']]

data_file = open(path_df_part_1_tar, 'r')
try:
    df_part_1_tar = pd.read_csv(data_file, index_col = False, parse_dates = [0])
    df_part_1_tar.columns = ['time','user_id','item_id','behavior_type','item_category']
finally:
    data_file.close()

# uic + label 
df_part_1_uic_label_1 = df_part_1_tar[df_part_1_tar['behavior_type'] == 4][['user_id','item_id','item_category']]
df_part_1_uic_label_1.drop_duplicates(['user_id','item_id'], keep='last', inplace=True)
df_part_1_uic_label_1['label'] = 1
df_part_1_uic_label = pd.merge(df_part_1_uic, 
                               df_part_1_uic_label_1,
                               on=['user_id','item_id','item_category'], 
                               how='left').fillna(0).astype('int')
df_part_1_uic_label.to_csv(path_df_part_1_uic_label, index=False)

##### part_2 #####
# uic
data_file = open(path_df_part_2, 'r')
try:
    df_part_2 = pd.read_csv(data_file, index_col = False)
    df_part_2.columns = ['time','user_id','item_id','behavior_type','item_category']
finally:
    data_file.close()
df_part_2_uic = df_part_2.drop_duplicates(['user_id', 'item_id', 'item_category'])[['user_id', 'item_id', 'item_category']]

data_file = open(path_df_part_2_tar, 'r')
try:
    df_part_2_tar = pd.read_csv(data_file, index_col = False, parse_dates = [0])
    df_part_2_tar.columns = ['time','user_id','item_id','behavior_type','item_category']
finally:
    data_file.close()

# uic + label 
df_part_2_uic_label_1 = df_part_2_tar[df_part_2_tar['behavior_type'] == 4][['user_id','item_id','item_category']]
df_part_2_uic_label_1.drop_duplicates(['user_id','item_id'], keep='last', inplace=True)
df_part_2_uic_label_1['label'] = 1
df_part_2_uic_label = pd.merge(df_part_2_uic, 
                               df_part_2_uic_label_1,
                               on=['user_id','item_id','item_category'], 
                               how='left').fillna(0).astype('int')
df_part_2_uic_label.to_csv(path_df_part_2_uic_label, index=False)

##### part_3 #####
# uic 
data_file = open(path_df_part_3, 'r')
try:
    df_part_3 = pd.read_csv(data_file, index_col = False)
    df_part_3.columns = ['time','user_id','item_id','behavior_type','item_category']
finally:
    data_file.close()
df_part_3_uic = df_part_3.drop_duplicates(['user_id', 'item_id', 'item_category'])[['user_id', 'item_id', 'item_category']]
df_part_3_uic.to_csv(path_df_part_3_uic, index=False)


print(' - kuaidouai - ')