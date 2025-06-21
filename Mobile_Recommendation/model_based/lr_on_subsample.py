# -*- coding: utf-8 -*
    
'''
@author: kuaidouai

@thoughts:  as the samples are extremely imbalance (N/P ratio ~ 1.2k),
            here we use sub-sample on negative samples.
            1-st: using k_means to make clustering on negative samples (clusters_number ~ 1k)
            2-nd: subsample on each clusters based on the same ratio,
                  the ratio was selected to be the best by testing in random sub_sample + LR
            3-rd: using LR model for training and predicting on sub_sample set.
            
            here is 2-nd & 3-rd step
'''

########## file path ##########
##### input file
# training set keys uic-label with k_means clusters' label
path_df_part_1_uic_label_cluster = "data/mobile/k_means_subsample/df_part_1_uic_label_cluster.csv"
path_df_part_2_uic_label_cluster = "data/mobile/k_means_subsample/df_part_2_uic_label_cluster.csv"
path_df_part_3_uic       = "data/mobile/raw/df_part_3_uic.csv"

# data_set features
path_df_part_1_U   = "data/mobile/feature/df_part_1_U.csv"  
path_df_part_1_I   = "data/mobile/feature/df_part_1_I.csv"
path_df_part_1_C   = "data/mobile/feature/df_part_1_C.csv"
path_df_part_1_IC  = "data/mobile/feature/df_part_1_IC.csv"
path_df_part_1_UI  = "data/mobile/feature/df_part_1_UI.csv"
path_df_part_1_UC  = "data/mobile/feature/df_part_1_UC.csv"

path_df_part_2_U   = "data/mobile/feature/df_part_2_U.csv"  
path_df_part_2_I   = "data/mobile/feature/df_part_2_I.csv"
path_df_part_2_C   = "data/mobile/feature/df_part_2_C.csv"
path_df_part_2_IC  = "data/mobile/feature/df_part_2_IC.csv"
path_df_part_2_UI  = "data/mobile/feature/df_part_2_UI.csv"
path_df_part_2_UC  = "data/mobile/feature/df_part_2_UC.csv"

path_df_part_3_U   = "data/mobile/feature/df_part_3_U.csv"  
path_df_part_3_I   = "data/mobile/feature/df_part_3_I.csv"
path_df_part_3_C   = "data/mobile/feature/df_part_3_C.csv"
path_df_part_3_IC  = "data/mobile/feature/df_part_3_IC.csv"
path_df_part_3_UI  = "data/mobile/feature/df_part_3_UI.csv"
path_df_part_3_UC  = "data/mobile/feature/df_part_3_UC.csv"


# normalize scaler
path_df_part_1_scaler = "data/mobile/k_means_subsample/df_part_1_scaler"
path_df_part_2_scaler = "data/mobile/k_means_subsample/df_part_2_scaler"

# item_sub_set P
path_df_P = "data/fresh_comp_offline/tianchi_fresh_comp_train_item.csv"

import os
##### output file
path_df_result     = "data/mobile/lr/result_sample.csv"
path_df_result_tmp = "data/mobile/lr/df_result_tmp.csv"
os.makedirs(os.path.dirname(path_df_result), exist_ok=True)

# depending package
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import matplotlib.pyplot as plt

import time

def clean_data(data):
    '''数据清理函数，处理NaN值、无穷大值和异常值'''
    # 确保数据是浮点类型
    data = data.astype(np.float64)
    
    # 替换无穷大值为NaN
    data = np.where(np.isinf(data), np.nan, data)
    
    # 处理NaN值，用列的中位数填充（如果全是NaN则用0）
    for i in range(data.shape[1]):
        col = data[:, i]
        if np.isnan(col).all():
            data[:, i] = 0.0
        else:
            median_val = np.nanmedian(col)
            if np.isnan(median_val):
                median_val = 0.0
            data[:, i] = np.where(np.isnan(col), median_val, col)
    
    # 处理极端异常值 - 使用更严格的界限
    for i in range(data.shape[1]):
        col_data = data[:, i]
        if np.std(col_data) > 0:  # 只处理有变化的列
            # 使用更严格的分位数界限
            q1 = np.percentile(col_data, 25)
            q3 = np.percentile(col_data, 75)
            iqr = q3 - q1
            
            # 使用IQR方法定义异常值界限
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # 确保界限是有限的
            if not np.isfinite(lower_bound):
                lower_bound = np.percentile(col_data, 1)
            if not np.isfinite(upper_bound):
                upper_bound = np.percentile(col_data, 99)
                
            # 将异常值替换为边界值
            data[:, i] = np.clip(col_data, lower_bound, upper_bound)
    
    # 最终检查：确保没有无穷大或NaN值
    data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # 检查数据的数值范围，如果太大则进行缩放
    max_val = np.max(np.abs(data))
    if max_val > 1e6:
        data = data / (max_val / 1e3)  # 缩放到合理范围
    
    return data

# some functions
def df_read(path, mode = 'r'):
    '''the definition of dataframe loading function 
    '''
    data_file = open(path, mode)
    try:     df = pd.read_csv(data_file, index_col = False)
    finally: data_file.close()
    return   df

def subsample(df, sub_size):
    '''the definition of sub-sampling function
    @param df: dataframe
    @param sub_size: sub_sample set size
    
    @return sub-dataframe with the same formation of df
    '''
    if sub_size >= len(df) : return df
    else : return df.sample(n = sub_size)

##### loading data of part 1 & 2
df_part_1_uic_label_cluster = df_read(path_df_part_1_uic_label_cluster)
df_part_2_uic_label_cluster = df_read(path_df_part_2_uic_label_cluster)

df_part_1_U  = df_read(path_df_part_1_U )   
df_part_1_I  = df_read(path_df_part_1_I )
df_part_1_C  = df_read(path_df_part_1_C )
df_part_1_IC = df_read(path_df_part_1_IC)
df_part_1_UI = df_read(path_df_part_1_UI)
df_part_1_UC = df_read(path_df_part_1_UC)

df_part_2_U  = df_read(path_df_part_2_U )   
df_part_2_I  = df_read(path_df_part_2_I )
df_part_2_C  = df_read(path_df_part_2_C )
df_part_2_IC = df_read(path_df_part_2_IC)
df_part_2_UI = df_read(path_df_part_2_UI)
df_part_2_UC = df_read(path_df_part_2_UC)

# generation of target testing set
part_1_scaler = pickle.load(open(path_df_part_1_scaler,'rb'))
part_2_scaler = pickle.load(open(path_df_part_2_scaler,'rb'))

##### generation of training set & valid set
def train_set_construct(np_ratio = 1, sub_ratio = 1.0):
    '''
    # generation of train set
    @param np_ratio: int, the sub-sample rate of training set for N/P balanced.
    @param sub_ratio: float ~ (0~1], the further sub-sample rate of training set after N/P balanced.
    '''
    train_part_1_uic_label = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)
    train_part_2_uic_label = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)
    
    frac_ratio = sub_ratio * np_ratio/1200
    for i in range(1,1001,1):
        train_part_1_uic_label_0_i = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == i]
        train_part_1_uic_label_0_i = train_part_1_uic_label_0_i.sample(frac = frac_ratio)
        train_part_1_uic_label = pd.concat([train_part_1_uic_label, train_part_1_uic_label_0_i])
    
        train_part_2_uic_label_0_i = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == i]
        train_part_2_uic_label_0_i = train_part_2_uic_label_0_i.sample(frac = frac_ratio)
        train_part_2_uic_label = pd.concat([train_part_2_uic_label, train_part_2_uic_label_0_i])
    print("training subset uic_label keys is selected.")
    
    # constructing training set
    train_part_1_df = pd.merge(train_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_I,  how='left', on=['item_id'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_C,  how='left', on=['item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_IC, how='left', on=['item_id','item_category'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
    train_part_1_df = pd.merge(train_part_1_df, df_part_1_UC, how='left', on=['user_id','item_category'])
    
    train_part_2_df = pd.merge(train_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_I,  how='left', on=['item_id'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_C,  how='left', on=['item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_IC, how='left', on=['item_id','item_category'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UI, how='left', on=['user_id','item_id','item_category','label'])
    train_part_2_df = pd.merge(train_part_2_df, df_part_2_UC, how='left', on=['user_id','item_category'])
       
    # using all the features without missing value for valid lr model
    train_X_1 = train_part_1_df[['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                           'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                           'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                           'u_b4_rate',
                                           'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                           'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                           'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                           'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                           'i_b4_rate',
                                           'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                           'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                           'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                           'c_b4_rate',
                                           'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                           'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                           'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                           'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                           'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                           'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                           'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                           'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                           'uc_b_count_rank_in_u']].values
    train_y_1 = train_part_1_df['label'].values
    # 数据清理
    train_X_1 = clean_data(train_X_1)
    # feature standardization
    standard_train_X_1 = part_1_scaler.transform(train_X_1)

    # using all the features without missing value for valid lr model
    train_X_2 = train_part_2_df[['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                           'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                           'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                           'u_b4_rate',
                                           'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                           'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                           'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                           'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                           'i_b4_rate',
                                           'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                           'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                           'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                           'c_b4_rate',
                                           'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                           'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                           'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                           'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                           'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                           'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                           'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                           'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                           'uc_b_count_rank_in_u']].values
    train_y_2 = train_part_2_df['label'].values
    # 数据清理
    train_X_2 = clean_data(train_X_2)
    # feature standardization
    standard_train_X_2 = part_2_scaler.transform(train_X_2)
    
    train_X = np.concatenate((standard_train_X_1, standard_train_X_2))
    train_y = np.concatenate((train_y_1, train_y_2)) # type: ignore
    print("train subset is generated.")
    
    return train_X, train_y

    
def valid_set_construct(sub_ratio = 0.1):
    '''
    # generation of valid set
    @param sub_ratio: float ~ (0~1], the sub-sample rate of original valid set
    '''
    valid_part_1_uic_label = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)
    valid_part_2_uic_label = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == 0].sample(frac = sub_ratio)

    for i in range(1,1001,1):
        valid_part_1_uic_label_0_i = df_part_1_uic_label_cluster[df_part_1_uic_label_cluster['class'] == i]
        valid_part_1_uic_label_0_i = valid_part_1_uic_label_0_i.sample(frac = sub_ratio)
        valid_part_1_uic_label = pd.concat([valid_part_1_uic_label, valid_part_1_uic_label_0_i])
    
        valid_part_2_uic_label_0_i = df_part_2_uic_label_cluster[df_part_2_uic_label_cluster['class'] == i]
        valid_part_2_uic_label_0_i = valid_part_2_uic_label_0_i.sample(frac = sub_ratio)
        valid_part_2_uic_label = pd.concat([valid_part_2_uic_label, valid_part_2_uic_label_0_i])
    
    # constructing valid set
    valid_part_1_df = pd.merge(valid_part_1_uic_label, df_part_1_U, how='left', on=['user_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_I,  how='left', on=['item_id'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_C,  how='left', on=['item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_IC, how='left', on=['item_id','item_category'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UI, how='left', on=['user_id','item_id','item_category','label'])
    valid_part_1_df = pd.merge(valid_part_1_df, df_part_1_UC, how='left', on=['user_id','item_category'])
    
    valid_part_2_df = pd.merge(valid_part_2_uic_label, df_part_2_U, how='left', on=['user_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_I,  how='left', on=['item_id'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_C,  how='left', on=['item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_IC, how='left', on=['item_id','item_category'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UI, how='left', on=['user_id','item_id','item_category','label'])
    valid_part_2_df = pd.merge(valid_part_2_df, df_part_2_UC, how='left', on=['user_id','item_category'])
    
    # using all the features without missing value for valid lr model
    valid_X_1 = valid_part_1_df[['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                           'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                           'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                           'u_b4_rate',
                                           'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                           'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                           'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                           'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                           'i_b4_rate',
                                           'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                           'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                           'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                           'c_b4_rate',
                                           'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                           'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                           'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                           'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                           'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                           'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                           'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                           'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                           'uc_b_count_rank_in_u']].values
    valid_y_1 = valid_part_1_df['label'].values
    # feature standardization
    standard_valid_X_1 = part_1_scaler.transform(valid_X_1)

    # using all the features without missing value for valid lr model
    valid_X_2 = valid_part_2_df[['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                           'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                           'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                           'u_b4_rate',
                                           'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                           'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                           'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                           'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                           'i_b4_rate',
                                           'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                           'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                           'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                           'c_b4_rate',
                                           'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                           'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                           'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                           'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                           'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                           'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                           'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                           'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                           'uc_b_count_rank_in_u']].values
    valid_y_2 = valid_part_2_df['label'].values
    # feature standardization
    standard_valid_X_2 = part_2_scaler.transform(valid_X_2)
    
    valid_X = np.concatenate((standard_valid_X_1, standard_valid_X_2))
    valid_y = np.concatenate((valid_y_1, valid_y_2)) # type: ignore
    print("train subset is generated.")
    
    return valid_X, valid_y

#######################################################################
'''Step 1: training for analysis of the best LR model
        (1). selection for best N/P ratio of subsamole
        (2). selection for best C of subsamole regularization
        (3). selection for best cutoff of prediction
'''


########## (1) selection for best N/P ratio in range(1, 100) of subsample
f1_scores = []
np_ratios = []
valid_X, valid_y = valid_set_construct(sub_ratio=0.18)
for np_ratio in range(1, 100, 2):
    t1 = time.time()
    train_X, train_y = train_set_construct(np_ratio=np_ratio, sub_ratio=0.5)
    
    # generation of lr model and fit
    LR_clf = LogisticRegression(penalty='l1', solver='liblinear', verbose=True, max_iter=10000)  # L1 regularization
    LR_clf.fit(train_X, train_y)
    
    # validation and evaluation
    valid_y_pred = LR_clf.predict(valid_X)
    f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
    np_ratios.append(np_ratio)

    print('LR_clf [NP ratio = %d] is fitted' % np_ratio)
    t2 = time.time()
    print('time used %d s' % (t2-t1))
    
# plot the result
f1 = plt.figure(1)
plt.plot(np_ratios, f1_scores, label="penalty='l1'")
plt.xlabel('NP ratio')
plt.ylabel('f1_score')
plt.title('f1_score as function of NP ratio - LR')
plt.legend(loc=4)
plt.grid(True, linewidth=0.3)
plt.show()



########## (2) selection for best regularization strength of subsample
f1_scores = []
cs = []
valid_X, valid_y = valid_set_construct(sub_ratio=0.18)
train_X, train_y = train_set_construct(np_ratio=35, sub_ratio=0.5)
for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    t1 = time.time()
    
    # generation of lr model and fit
    LR_clf = LogisticRegression(C=c, penalty='l1', solver='liblinear', verbose=True)
    LR_clf.fit(train_X, train_y)
    
    # validation and evaluation
    valid_y_pred = LR_clf.predict(valid_X)
    f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
    cs.append(c)

    print('LR_clf [C = %.3f] is fitted' % c)
    t2 = time.time()
    print('time used %d s' % (t2-t1))
    
# plot the result
f1 = plt.figure(1)
plt.plot(cs, f1_scores, label="penalty='l1', np_ratio=35")
plt.xlabel('C')
plt.ylabel('f1_score')
plt.title('f1_score as function of C - LR')
plt.legend(loc=4)
plt.grid(True, linewidth=0.3)
plt.show()


########## (3) selection for best cutoff of prediction
f1_scores = []
cut_offs = []
valid_X, valid_y = valid_set_construct(sub_ratio=0.18)
train_X, train_y = train_set_construct(np_ratio=55, sub_ratio=0.5)
for co in np.arange(0.1,1,0.1):
    t1 = time.time()
    
    # generation of lr model and fit
    LR_clf = LogisticRegression(penalty='l1', solver='liblinear', verbose=True)
    LR_clf.fit(train_X, train_y)
    
    # validation and evaluation
    valid_y_pred = (LR_clf.predict_proba(valid_X)[:,1] > co).astype(int)
    f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
    cut_offs.append(co)

    print('LR_clf [cut_off = %.1f] is fitted' % co)
    t2 = time.time()
    print('time used %d s' % (t2-t1))
    
# plot the result
f1 = plt.figure(1)
plt.plot(cut_offs, f1_scores, label="penalty='l1',np_ratio=55")
plt.xlabel('C')
plt.ylabel('f1_score')
plt.title('f1_score as function of cut_off - LR')
plt.legend(loc=4)
plt.grid(True, linewidth=0.3)
plt.show()


#######################################################################
'''Step 2: training the optimal RF model and predicting on part_3 
'''
##### predicting
# loading feature data
df_part_3_U  = df_read(path_df_part_3_U )   
df_part_3_I  = df_read(path_df_part_3_I )
df_part_3_C  = df_read(path_df_part_3_C )
df_part_3_IC = df_read(path_df_part_3_IC)
df_part_3_UI = df_read(path_df_part_3_UI)
df_part_3_UC = df_read(path_df_part_3_UC)
# for get scale transform mechanism to large scale of data
scaler_3 = preprocessing.StandardScaler()
# process by chunk as ui-pairs size is too big
batch = 0
for pred_uic in pd.read_csv(open(path_df_part_3_uic, 'r'), chunksize = 100000): 
    try:     
        # construct of prediction sample set
        pred_df = pd.merge(pred_uic, df_part_3_U,  how='left', on=['user_id'])
        pred_df = pd.merge(pred_df,  df_part_3_I,  how='left', on=['item_id'])
        pred_df = pd.merge(pred_df,  df_part_3_C,  how='left', on=['item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_IC, how='left', on=['item_id','item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_UI, how='left', on=['user_id','item_id','item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_UC, how='left', on=['user_id','item_category'])
        
        pred_X = pred_df[['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                    'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                    'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                    'u_b4_rate',
                                    'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                    'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                    'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                    'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                    'i_b4_rate',
                                    'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                    'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                    'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                    'c_b4_rate',
                                    'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                    'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                    'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                    'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                    'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                    'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                    'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                    'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                    'uc_b_count_rank_in_u']].values
        # feature standardization
        scaler_3.partial_fit(pred_X)   
        
        batch += 1
        print("prediction chunk %d done." % batch) 
        
    except StopIteration:
        print("prediction finished.")
        break         


train_X, train_y = train_set_construct(np_ratio=35, sub_ratio=1)

# build model and fitting
LR_clf = LogisticRegression(verbose=True, max_iter=1000)
LR_clf.fit(train_X, train_y)

# process by chunk as ui-pairs size is too big
batch = 0
for pred_uic in pd.read_csv(open(path_df_part_3_uic, 'r'), chunksize = 100000): 
    try:     
        # construct of prediction sample set
        pred_df = pd.merge(pred_uic, df_part_3_U,  how='left', on=['user_id'])
        pred_df = pd.merge(pred_df,  df_part_3_I,  how='left', on=['item_id'])
        pred_df = pd.merge(pred_df,  df_part_3_C,  how='left', on=['item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_IC, how='left', on=['item_id','item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_UI, how='left', on=['user_id','item_id','item_category'])
        pred_df = pd.merge(pred_df,  df_part_3_UC, how='left', on=['user_id','item_category'])

        # fill the missing value as -1 (missing value are time features)
        pred_df.fillna(-1, inplace=True)
        
        # using all the features for training RF model
        pred_X = pred_df[['u_b1_count_in_6','u_b2_count_in_6','u_b3_count_in_6','u_b4_count_in_6','u_b_count_in_6', 
                                    'u_b1_count_in_3','u_b2_count_in_3','u_b3_count_in_3','u_b4_count_in_3','u_b_count_in_3', 
                                    'u_b1_count_in_1','u_b2_count_in_1','u_b3_count_in_1','u_b4_count_in_1','u_b_count_in_1', 
                                    'u_b4_rate',
                                    'i_u_count_in_6','i_u_count_in_3','i_u_count_in_1',
                                    'i_b1_count_in_6','i_b2_count_in_6','i_b3_count_in_6','i_b4_count_in_6','i_b_count_in_6', 
                                    'i_b1_count_in_3','i_b2_count_in_3','i_b3_count_in_3','i_b4_count_in_3','i_b_count_in_3',
                                    'i_b1_count_in_1','i_b2_count_in_1','i_b3_count_in_1','i_b4_count_in_1','i_b_count_in_1', 
                                    'i_b4_rate',
                                    'c_b1_count_in_6','c_b2_count_in_6','c_b3_count_in_6','c_b4_count_in_6','c_b_count_in_6',
                                    'c_b1_count_in_3','c_b2_count_in_3','c_b3_count_in_3','c_b4_count_in_3','c_b_count_in_3',
                                    'c_b1_count_in_1','c_b2_count_in_1','c_b3_count_in_1','c_b4_count_in_1','c_b_count_in_1',
                                    'c_b4_rate',
                                    'ic_u_rank_in_c','ic_b_rank_in_c','ic_b4_rank_in_c', 
                                    'ui_b1_count_in_6','ui_b2_count_in_6','ui_b3_count_in_6','ui_b4_count_in_6','ui_b_count_in_6',
                                    'ui_b1_count_in_3','ui_b2_count_in_3','ui_b3_count_in_3','ui_b4_count_in_3','ui_b_count_in_3',
                                    'ui_b1_count_in_1','ui_b2_count_in_1','ui_b3_count_in_1','ui_b4_count_in_1','ui_b_count_in_1', 
                                    'ui_b_count_rank_in_u','ui_b_count_rank_in_uc',
                                    'uc_b1_count_in_6','uc_b2_count_in_6','uc_b3_count_in_6','uc_b4_count_in_6','uc_b_count_in_6', 
                                    'uc_b1_count_in_3','uc_b2_count_in_3','uc_b3_count_in_3','uc_b4_count_in_3','uc_b_count_in_3', 
                                    'uc_b1_count_in_1','uc_b2_count_in_1','uc_b3_count_in_1','uc_b4_count_in_1','uc_b_count_in_1',
                                    'uc_b_count_rank_in_u']].values

        # predicting
        # feature standardization
        standardized_pred_X = scaler_3.transform(pred_X)
        pred_y = (LR_clf.predict_proba(standardized_pred_X)[:,1] > 0.5).astype(int)

        # generation of U-I pairs those predicted to buy
        pred_df['pred_label'] = pred_y
        # add to result csv
        pred_df[pred_df['pred_label'] == 1].to_csv(path_df_result_tmp, 
                                                   columns=['user_id','item_id'],
                                                   index=False, header=False, mode='a')
        
        batch += 1
        print("prediction chunk %d done." % batch) 
        
    except StopIteration:
        print("prediction finished.")
        break         
   
    
#######################################################################
'''Step 3: generation result on items' sub set P 
'''
# loading data
df_P = df_read(path_df_P)
df_P_item = df_P.drop_duplicates(['item_id'])[['item_id']]
df_pred = pd.read_csv(open(path_df_result_tmp,'r'), index_col=False, header=None)
df_pred.columns = ['user_id', 'item_id']

# output result
df_pred_P = pd.merge(df_pred, df_P_item, on=['item_id'], how='inner')[['user_id', 'item_id']]
df_pred_P.to_csv(path_df_result, index=False)



print(' - kuaidouai - ')