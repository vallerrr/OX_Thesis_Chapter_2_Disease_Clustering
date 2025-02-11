"""
# Created by valler at 11/12/2024
Feature: 

"""
import numpy as np
import pandas as pd
from src import params
from src.data_preprocess import utils


# 1. read files
df_recorder = pd.read_csv(params.data_path.parent / 'downloaded_data/recorder/recorder_participant.csv')
df_phe_db, df_phemap = utils.read_phe_database()

df_recorder_death = df_recorder[(df_recorder['field_name'].str.contains('40000'))|(df_recorder['field_name'].str.contains('40001'))]
# 40000: death date
# 40001: death cause


# 2. read data based on the recorder
df_death_data = pd.DataFrame()
for ind, group in df_recorder_death.groupby('ind'):

    temp = pd.read_csv(f'/Volumes/Valler/Python/OX_Thesis/Chapter_2_Disease_Clustering/Data/downloaded_data/participant/{round(ind)}.csv', low_memory=False)
    cols = group['field_name'].tolist()
    temp = temp[['eid']+cols]
    if len(df_death_data)==0:
        df_cancer_data = temp
    else:
        df_death_data = df_death_data.merge(temp, how='outer', on='eid')


# 3. preprocess the death data
df_death_data['death_date'] = [x if str(x) not in params.nan_str else y for x,y in zip(df_death_data['p40000_i0'],df_death_data['p40000_i1'])]
df_death_data['death_date'] = pd.to_datetime(df_death_data['death_date'], errors='coerce')


df_death_data.to_pickle(params.data_path.parent / 'final_data/death_data.pkl')
