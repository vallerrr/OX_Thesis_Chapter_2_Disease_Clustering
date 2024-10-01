"""
# Created by valler at 31/05/2024
Feature: script to generate data descrption for the data

"""
from src import params
from src.data_preprocess import utils
from collections import Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import ast

df_codebook_final = pd.read_csv(params.codebook_path / 'UKB_preprocess_codebook_wave_0_final.csv')
df = pd.read_pickle(params.final_data_path / 'UKB_wave_0_final_standardised.pkl')

record_column = params.disease_record_column
access_date_column = '53'
HES_ICD_ids = params.HES_ICD_ids
level = 'chronic_cate'
weight_control = False

# control zone
chapter_ranges = [x for x in range(1, 16)]+[19] if 'chapter' in level else [x for x in range(1, 18)]
# disease_column = params.disease_columns_dict[level]
disease_column = ['diseases_within_window_phecode_selected_chronic_first_occ','diseases_within_window_phecode_selected_category_chronic_first_occ']
dates_col, df_single_record = utils.import_df_single_record(record_column)

"""
for disease_column in ['diseases_within_window_phecode_selected_chronic_first_occ','diseases_within_window_phecode_selected_category_chronic_first_occ']:
    df_single_record[disease_column] = [list_ if str(list_) != '[None]' else None for list_ in df_single_record[disease_column]]
    df_single_record[disease_column] = df_single_record[disease_column].apply(lambda x: [i for i in x if pd.notnull(i)] if isinstance(x, list) else x)
df_single_record.to_csv(params.intermediate_path / f'{record_column}_complete.csv', index=False)

"""
# ======================================================================================================================
# 0. combine chronic disease information from the specific time window
# ======================================================================================================================

# merge the new columns back to the original df
for columns in disease_column:
    df_single_record[columns] = [ast.literal_eval(x) if str(x) not in params.nan_str else None for x in df_single_record[columns]]


df = pd.merge(df, df_single_record[['eid','weight'] if weight_control else ['eid'] + disease_column], how='left', left_on='eid', right_on='eid')
# dichotomize the disease columns
unique_diseases = list(set(df['diseases_within_window_phecode_selected_chronic_first_occ'].dropna().explode().explode().dropna()))

df_columns = df.columns.tolist()
for disease in unique_diseases:
    if str(disease) not in df_columns:
        df[str(disease)] = [x.count(disease) if str(x) not in params.nan_str else 0 for x in df['diseases_within_window_phecode_selected_chronic_first_occ']]

# add the disease chapter information
chapter_ranges = df_single_record['diseases_within_window_phecode_selected_category_chronic_first_occ'].explode().explode().dropna().unique()
for chapter in chapter_ranges:
    df[f"c_{chapter}"] = [x.count(chapter) if str(x) not in params.nan_str else 0 for x in df['diseases_within_window_phecode_selected_category_chronic_first_occ']]

df.drop(columns=disease_column, inplace=True)
df.to_pickle(params.final_data_path / 'UKB_wave_0_final_standardised_with_disease.pkl')  # this is the file to be used in final analysis


# ======================================================================================================================
# 1. generate the data description
# ======================================================================================================================
df = pd.read_pickle(params.final_data_path / 'UKB_wave_0_final_non_standardised.pkl')

df_descriptive_table = pd.DataFrame(columns=['variable_name', 'variable_description', 'variable_type', 'variable_category', 'variable_values', 'variable_missing_values', 'variable_missing_values_percentage', 'variable_unique_values', 'variable_unique_values_percentage', 'variable_mean', 'variable_std', 'variable_min', 'variable_25%', 'variable_50%', 'variable_75%', 'variable_max'])
for column in df.columns:
    if column in ['eid', '53','55', 'diseases_within_window_all_icd_first_3', 'diseases_within_window_icd_parent_coding', 'diseases_within_window_icd_chapter_coding']:
        continue
    print(f'Processing {column}')
    var_name = df_codebook_final.loc[df_codebook_final['field_id'] == int(column), 'field_name'].values[0]
    df_descriptive_table.loc[len(df_descriptive_table),] = [column,var_name, df[column].dtype, df_codebook_final.loc[df_codebook_final['field_id'] == int(column), 'cate_name'].values[0], df[column].unique(), df[column].isnull().sum(), df[column].isnull().sum()/len(df), len(df[column].unique()), len(df[column].unique())/len(df), df[column].mean(), df[column].std(), df[column].min(), df[column].quantile(0.25), df[column].quantile(0.5), df[column].quantile(0.75), df[column].max()]

# using df_notebook to fill the missing count
def find_missing_count(row):
    field_id = int(row['variable_name'])

    if field_id in df_codebook_final['field_id'].values:
        missing_count = df_codebook_final.loc[df_codebook_final['field_id']==field_id,'missing_count'].values[0]

    else:
        missing_count = None

    return missing_count


df_descriptive_table['variable_missing_values'] = df_descriptive_table.apply(find_missing_count, axis=1)
df_descriptive_table['variable_missing_values_percentage'] = [x/len(df) for x in df_descriptive_table['variable_missing_values']]
df_descriptive_table.to_csv(params.final_data_path/'descriptive_table'/'descriptive_table_wave_0.csv',index=False)

# concat disease information to the non standardised dataframe
df_non_standardised = pd.read_pickle(params.final_data_path / 'UKB_wave_0_final_non_standardised.pkl')
df_non_standardised = pd.merge(df_non_standardised, df[['eid', 'diseases_within_window_all_icd_first_3', 'diseases_within_window_icd_parent_coding', 'diseases_within_window_icd_chapter_coding']], how='left', left_on='eid', right_on='eid')
df_non_standardised.to_pickle(params.final_data_path / 'UKB_wave_0_final_non_standardised.pkl')

