import pymssql
import pandas as pd
import numpy as np
import re
from collections import Counter
import datetime
import os

from pylab import rcParams, savefig
import matplotlib.pyplot as plt

import sys
import scoringfunctions as sf
import woe

#%matplotlib inline

def get_coonection(login, password):
    """create connect to sql base"""
    conn = pymssql.connect(
        host="130.193.44.114",
        user=login,
        password=password,
        database='dwh'
    )
    
    return conn

def read_sql(qu, conn):
    """read sql query to pandas dataframe"""
    df = pd.read_sql(qu, conn)
    df.columns = [col.lower() for col in df.columns] 
    
    return df

def read_sql_from_file(path, conn):
    """open sql file and read result to pandas dataframe"""
    with open(path, 'r', encoding='utf-8') as f:
        qu = f.read()
    
    df = read_sql(qu, conn)
    
    return df

def transform_data(src):
    columns_to_date = ['legal_registration_date', 'originator_registration_date'] #all date columns

    columns_to_numeric = ['amount_executed_for_24_month',  'bki_flg_30',
           'bki_flg_90', 'bki_flg_90plus', 'bki_volume', 
           'number_current_contracts', 'number_current_with_similar_sum',
           'number_current_with_the_similar_work', 'number_investors',
           'number_of_executed_for_24_month', 'number_of_the_following_users',
           'number_with_similar_sum_for_24_months',
           'number_with_customer_for_24_months',
           'number_with_similar_work_for_24_months', 'total_current_contracts',]  # all numeric columns

    drop_columns = ['fact_address', 'client_key', 'okved', 'inn', 'originator_name', 'product_name',
                    'days_in_overdue', 'days_after_billing', 'application_status_id', 'application_dttm',
                   'organizational_form_key', 'legal_address', 'ogrn', 'originator_registration_date_dd'] # all columns wich will be drop

    for col in columns_to_date: # convert column to date format and find difference in days between date of application
        src[col] = pd.to_datetime(src[col])
        src[col + '_dd'] = (src['application_dttm'] - src[col]).dt.days 

    for col in columns_to_numeric:    # convert all numeric columns to float format
        src[col] = src[col].astype(float)

    src['executive_production'] = \
    src['executive_production'].fillna('').str.replace(r',[^0-9]|,$|.$|\.[^0-9]| |[^0-9,.]', '').str.replace(',','.')
    src['executive_production'] = pd.to_numeric(src['executive_production'], errors='coerce') #drop all symbols except digits and convert to float

    src['arbitration'] = src['arbitration'].fillna('').apply(lambda x: x.split('₽')[0]).\
        str.replace(r',[^0-9]|,$|.$|\.[^0-9]| |[^0-9,.]', '').str.replace(',','.')  #drop all symbols except digits and convert to float

    src['arbitration'] = pd.to_numeric(src['arbitration'], errors='coerce')

    src['legal_address'] = src['legal_address'].str.lower().str.replace(r'[,\.]','')
    city_words = "".join(src['legal_address']).split(' ')
    city_words = [x for x in city_words if len(x) > 2]  #find most popular cityes

    addr_pop_words = ['москва', 'санкт-петербург'] # leave only moscow and s-petersburg

    for word in addr_pop_words:
        src[word + '_inside_addr'] = src.legal_address.apply(lambda x: 1 if word in x else 0) #generate new dummy columns for cities

    src.days_in_overdue = src.days_in_overdue.fillna(0)

    default_calc_series = src[['days_in_overdue', 'days_after_billing']].apply(tuple, axis=1) #generate series for calculating deafaul flags

    default_days = [30,60,90]  # number of days for overdue flags

    for default_day in default_days:  # check if loan is matured yhen calculate default flag for every defaul flags
        src[f'default_{default_day}_flag'] = default_calc_series.apply(
            lambda x: np.nan if x[1] < default_day else 1 if x[0] > default_day else 0)

    src = src.drop(drop_columns + columns_to_date, 1) # drop all unused columns

    fillna_to_max_columns = ['oldest_loan_daydiff', 'prev_paym_daydiff'] 
    fillna_to_min_columns = ['earlyest_loan_daydiff']
    fillna_to_median_columns = []

    for col in fillna_to_max_columns: # fill null values to maximum value from column multiplied by 23
        src[col] = src[col].fillna(src[col].max() * 2)

    for col in fillna_to_min_columns: # fill null values to big negative value
        src[col] = src[col].fillna(-src[col].min() * 2)

    for col in fillna_to_median_columns: # fill null value to column median
        src[col] = src[col].fillna(src[col].median())

    targets = ['default_30_flag', 'default_60_flag','default_90_flag']

    src[targets] = src[targets].fillna(-1) # for target all null values (not matured loans) fill to -1

    src = src.fillna(0) # fill all null values to 0
    
    return src
    
def create_folder(folder): 
    """create new folder if it does not exist"""
    if not os.path.exists(folder):
        os.makedirs(folder)

def full_modeling(target, pre_clust_df, model_path, id_column):
    """for current target and dataframe (pre_clust_df) transform all features to woe buckets and learn model"""
    targets = [x for x in pre_clust_df.columns if x[:8] == 'default_']
    # folders for result saving
    folder_auc = model_path + '/pictures/roc_auc'
    folder_column_pics = model_path + '/pictures'
    folder_model_output = model_path + '/model_output'
    create_folder(folder_auc)
    create_folder(folder_model_output)
    
    #take only matured loans
    pre_clust_df = pre_clust_df[pre_clust_df[target]>-.5] 
    pre_clust_df = pre_clust_df.set_index(id_column)

    #drop all target columns except current tarhet column
    drop_targets = [col for col in targets if col != target] 
    drop_targets = list(set(drop_targets) & set(pre_clust_df))
    pre_clust_df = pre_clust_df.drop(drop_targets, 1)

    #transform continous variables to bucket columns
    dfPreWoe, clustVarsInfo = sf.continuousVariables(pre_clust_df, columnLimit=10)  
    #trassform to woe columns
    dfPostWoe, woeVarsInfo = sf.woeVariables(dfPreWoe,target)

    #look at information value of variables
    gg = sf.giniGrowth(dfPostWoe,woeVarsInfo,target)
    #chose best columns
    goodColumns, badColumns = sf.chooseColumnsFromIT(gg, badFlag=target, min_limit=0.01)

    #create log regression model
    model = sf.logReg(preLR=dfPostWoe[goodColumns], badFlag=target)
    #save roc_auc picture 
    model.print_roc_curve(to_file=True, folder=folder_auc)

    #generate doc information about model and variables
    intercept, woeOut = sf.modelOutput(folder_model_output, woeVarsInfo, goodColumns, model, gg, rewrite=True)

    #generate and save pictures of feature distribution
    bad_columns = woe.save_pictures(woeVarsInfo, folder = folder_column_pics, badRateLimit=100)