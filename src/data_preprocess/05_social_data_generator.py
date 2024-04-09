"""
# Created by valler at 04/04/2024
Feature: generate the social data as outcome vars for the model
we only use the wave 0 for the social data
"""

import pandas as pd
from src import params
import ast
import os
import numpy as np
from src.data_preprocess import utils
import warnings

# 1. Load the codebook

df_codebook = pd.read_csv(params.codebook_path/'UKB_var_select.csv')

file_count = 0
instance = 0
cate_names = params.cate_names
cate_name = cate_names[0]
iterators = df_codebook.loc[df_codebook['cate_name'] == cate_name, ].iterrows()
replace_dict_basics = params.replace_dict_basics

df = pd.DataFrame()

while len(df.columns) < 5:
    try:
        ind, row = next(iterators)
    except:
        break

    if row['preprocessed_flag'] == 1:

        print(row['field_id'])
        print(f'Processing {row["field_name"]}')
        df, df_codebook = utils.recoding_process_main_for_final_data_generator(ind, row,df_codebook, df, instance, cate_name, file_count, replace_dict_basics)


#df.to_csv(params.preprocessed_path/)
"""
# check zone 

ind = df_codebook.loc[df_codebook['field_id']==20018,].index
row = pd.Series(
temp = utils.data_reader(row, instance)
row.file_names
row = df_codebook.loc[df_codebook['field_id']==20018,]
"""
