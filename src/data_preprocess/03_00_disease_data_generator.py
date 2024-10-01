"""
# Created by valler at 07/03/2024
Feature: generate the disease data as outcome vars for the model

"""
import pandas as pd
from src import params
from src.data_preprocess import utils
import ast
from matplotlib import pyplot as plt
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
# 1. Load the data
record_column = params.disease_record_column
wave_codebook_path = params.codebook_path / f'UKB_var_select.csv'
df_codebook = pd.read_csv(wave_codebook_path)
df_codebook.preprocessed_flag.value_counts()

HES_ICD_ids = {"all_icd": {"id": '41270', "time": '41280'},
               "main_icd": {"id": '41202', "time": '41262'},
               "second_icd": {"id": '41203', "time": None}}

# ------------------------------------------------------------------------------------------------------------------
# 2. Hospital Inpatient (HES)
# ------------------------------------------------------------------------------------------------------------------

# read the data for each cat and mark the unique count
df_disease = pd.DataFrame()
for key, value in HES_ICD_ids.items():
    print()
    field_id = HES_ICD_ids[key]['id']

    ind = df_codebook.loc[df_codebook['field_id'] == int(field_id)].index[0]
    row = df_codebook.loc[ind, ]

    df_read = utils.data_reader(row)
    if len(df_disease) == 0:
        df_disease = df_read
    else:
        df_disease = pd.merge(df_disease, df_read, on='eid')


    df_disease[f'{key}_uniq_count'] = [len(ast.literal_eval(x)) if pd.notnull(x) else None for x in df_disease[f'p{field_id}']]
    print(df_disease[f'{key}_uniq_count'].describe())

# replace all ICD codes with their characters

for key in HES_ICD_ids.keys():
    df_disease[f'{key}_icd_codes'] = df_disease[f'p{HES_ICD_ids[key]["id"]}'].apply(lambda x: [str(y).split(' ')[0] for y in ast.literal_eval(x)] if pd.notnull(x) else None)


# ------------------------------------------------------------------------------------------------------------------
# 3. ICD10 code file (coding19.tsv)
#    one can skip this step if the file is already created
#    it will not affect result
# ------------------------------------------------------------------------------------------------------------------

df_disease[f'{record_column}_uniq_count'].hist(bins=100)
plt.show()

ICD_file_path ="/Users/valler/Python/OX_Thesis/Chapter_2_Disease_Clustering/Data/downloaded_data/ICD_10/coding19.tsv"
df_ICD = pd.read_csv(ICD_file_path, sep='\t')

def tag_row_type(row):
    if "Chapter" in row['coding']:
        return 'Chapter'
    elif "Block" in row['coding']:
        return 'Block'
    elif len(row['coding'])==3:
        return '3-char'
    elif len(row['coding'])>=4:
        if len(row['coding'])==4:
            return '4-char'
        elif len(row['coding'])==5:
            return '5-char'
        else:
            return 'more-char'
    else:
        return 'error'

df_ICD['type'] = df_ICD.apply(lambda x: tag_row_type(x), axis=1)
df_ICD.to_csv(params.intermediate_path / 'ICD_10.csv', index=False)
df_ICD['type'].value_counts()

# ------------------------------------------------------------------------------------------------------------------
# 4. dates for a single df
# ------------------------------------------------------------------------------------------------------------------
df_single_record = df_disease[['eid',f'p{HES_ICD_ids[record_column]["id"]}',f'{record_column}_icd_codes', f'{record_column}_uniq_count']]
dates_col = [f'p{HES_ICD_ids[record_column]["time"]}_a{x:03d}' for x in range(0,int(df_single_record[f'{record_column}_uniq_count'].max()))]
columns = ['eid',f'{record_column}_icd_codes', f'{record_column}_uniq_count']+dates_col

field_id = HES_ICD_ids[record_column]['time']
ind = df_codebook.loc[df_codebook['field_id'] == int(field_id)].index[0]
row = df_codebook.loc[ind, ]

df_single_record = pd.merge(df_single_record, utils.data_reader(row), on='eid')
df_single_record.rename(columns={f'p{HES_ICD_ids[record_column]["time"]}_a{x}' : f'p{HES_ICD_ids[record_column]["time"]}_a{x:03d}' for x in range(0,int(df_single_record[f'{record_column}_uniq_count'].max()))}, inplace=True)
df_single_record = df_single_record[columns]
df_single_record.to_csv(params.intermediate_path / f'{record_column}_complete.csv', index=False)

# ------------------------------------------------------------------------------------------------------------------
# 5. index_of_first_disease_out_window
# ------------------------------------------------------------------------------------------------------------------
# 5.0 read and concat the access date
access_date_column = '53'
ind = df_codebook.loc[df_codebook['field_id'] == int(access_date_column)].index[0]
row = df_codebook.loc[ind,]
temp = utils.data_reader(row)

df_single_record = df_single_record.merge(left_on='eid',right=temp[['eid',f'p{access_date_column}_i0']].rename(columns={f'p{access_date_column}_i0':f'{access_date_column}'}),right_on='eid',how='inner')
df_single_record[access_date_column] = pd.to_datetime(df_single_record[access_date_column], errors='coerce', format='%Y-%m-%d')

# 5.1 retrieve gender and age on the df_single_record
import numpy as np
columns_to_remain = ['eid', '21022', '31']
df_read = pd.read_csv(params.preprocessed_path / 'UKB_wave_0_Socio-demographics_0.csv')
df_read = df_read[columns_to_remain]
df_single_record = pd.merge(df_read, df_single_record, on='eid', how='left')

temp = df_single_record.copy()
# sort the date columns
temp = temp[columns_to_remain + [ f'{record_column}_uniq_count', f'{record_column}_icd_codes',access_date_column] + dates_col]

# 5.2 sort the diseases by dates, change the order of dates columns
# 5.2.1 reorder the diseases based on their corresponding date

dates_df = temp[dates_col].copy()
sorted_indices = dates_df.apply(np.argsort, axis=1)

def sort_by_indices(list_, index):
    if str(list_) in params.nan_str:
        return None
    else:
        indices = sorted_indices.loc[index].values[:len(list_)]

        sorted_list = [list_[i] for i in indices]

        return sorted_list


# Apply the function to the 'diseases' column
temp[f'{record_column}_icd_codes'] = temp.apply(lambda row: sort_by_indices(row[f'{record_column}_icd_codes'], row.name) if str(row) != 'None' else None, axis=1)

# 5.2.2 Apply the function fo the dates columns
dates_sorted = temp.apply(lambda row: sort_by_indices(row[dates_col], row.name), axis=1)
dataframes = [pd.DataFrame(dates_sorted[i]).T for i in range(len(dates_sorted))]
m = pd.concat(dataframes)
m.columns = dates_col
m.reset_index(drop=True, inplace=True)
temp[dates_col] = m[dates_col]
df_single_record = temp.copy()


# 5.3 find the index of the first disease out of the window
def find_index(row):
    entry_date = row[access_date_column]
    unique_dis = row[f'{record_column}_uniq_count']
    max_index = -1
    if pd.notnull(unique_dis):
        for index in range(0,int(unique_dis)):
            date_col = f'p{HES_ICD_ids[record_column]["time"]}_a{index:03d}'
            if pd.notnull(row[date_col]):
                if pd.to_datetime(row[date_col], errors='coerce', format='%Y-%m-%d')<=entry_date:
                    max_index = index
                else:
                    break
            else:
                break

    return max_index


df_single_record[f'index_of_first_{record_column}_out_window'] = df_single_record.apply(lambda row: find_index(row), axis=1)
# note that the index is actually, the index of the last disease within the window, so we need to add 1 to it in using it in the model (when it's not -1)
df_single_record[f'index_of_first_{record_column}_out_window'] = [x+1 if x!=-1 else x for x in df_single_record[f'index_of_first_{record_column}_out_window']]


df_single_record.to_csv(params.intermediate_path / f'{record_column}_complete.csv', index=False)

