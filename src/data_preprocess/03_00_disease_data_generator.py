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
wave_codebook_path = params.codebook_path / f'UKB_var_select.csv'
df_codebook = pd.read_csv(wave_codebook_path)
df_codebook.preprocessed_flag.value_counts()
intermediate_path = Path('/Users/valler/Python/OX_Thesis/Chapter_2_Disease_Clustering/Data/intermediate_files')

HES_ICD_ids = {"all_icd": {"id": '41270', "time": '41280'},
               "main_icd": {"id": '41202', "time": '41262'},
               "second_icd": {"id": '41203', "time": None}}

# ------------------------------------------------------------------------------------------------------------------
# 2. Hospital Inpatient (HES)
# ------------------------------------------------------------------------------------------------------------------


# read the data for each cat
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

# replace all ICD codes with the first 4 characters

for key in HES_ICD_ids.keys():
    df_disease[f'{key}_first_3'] = df_disease[f'p{HES_ICD_ids[key]["id"]}'].apply(lambda x: [str(y)[:3] for y in ast.literal_eval(x)] if pd.notnull(x) else None)


# ------------------------------------------------------------------------------------------------------------------
# * create a dictionary for the disease 4 Codes:disease names (this is the most detailed level of the ICD codes)
# ------------------------------------------------------------------------------------------------------------------
ICD_disease_dict = {}
for key in HES_ICD_ids.keys():
    diseases = df_disease[f'p{HES_ICD_ids[key]["id"]}'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else None).explode().unique()
    for disease in diseases:
        if pd.notnull(disease):
            disease_digits = str(disease)[:5]
            if disease_digits not in ICD_disease_dict.keys():
                    ICD_disease_dict[disease_digits] = [disease.replace(disease_digits, '')]
            else:
                if ICD_disease_dict[disease_digits][0] != disease.replace(disease_digits, ''):
                    ICD_disease_dict[disease_digits].append(disease.replace(disease_digits, ''))


# ------------------------------------------------------------------------------------------------------------------
# 3. ICD10 code file (coding19.tsv)
# ------------------------------------------------------------------------------------------------------------------
key = list(HES_ICD_ids.keys())[0]
df_disease[f'{key}_uniq_count'].hist(bins=100)
plt.show()
temp = df_disease.icd10_uniq_count.value_counts().reset_index().sort_values(by='count', ascending=True)

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
df_ICD.to_csv(intermediate_path / 'ICD_10.csv', index=False)
df_ICD['type'].value_counts()

# ------------------------------------------------------------------------------------------------------------------
# 4 dates for a single df
# ------------------------------------------------------------------------------------------------------------------
# 1. codes to generate the file (main_icd_complete.csv and all_icd_complete.csv)
record_column = 'all_icd'
df_single_record = df_disease[['eid',f'{record_column}_first_3', f'{record_column}_uniq_count']]
dates_col = [f'p{HES_ICD_ids[record_column]["time"]}_a{x}' for x in range(0,int(df_single_record[ f'{record_column}_uniq_count'].max()))]
columns = ['eid',f'{record_column}_first_3', f'{record_column}_uniq_count']+dates_col

field_id = HES_ICD_ids[record_column]['time']
ind = df_codebook.loc[df_codebook['field_id'] == int(field_id)].index[0]
row = df_codebook.loc[ind, ]

df_single_record = pd.merge(df_single_record, utils.data_reader(row), on='eid')
df_single_record = df_single_record[columns]
df_single_record.to_csv(intermediate_path / f'{record_column}_complete.csv', index=False)

# 2. reads the file
df_single_record = pd.read_csv(intermediate_path / f'{record_column}_complete.csv')
for col in dates_col:
    df_single_record[col] = pd.to_datetime(df_single_record[col], errors='coerce', format='%Y-%m-%d')


# ------------------------------------------------------------------------------------------------------------------
#  Create an empty dictionary to store the dataframes
value_counts_dict = {}
# Iterate over each column in df_read
for column in df_read.columns:
    # Get value counts for each unique value in the column
    value_counts = df_read[column].value_counts()
    # Convert the series to a dataframe
    value_counts_df = value_counts.reset_index()
    # Rename the columns
    value_counts_df.columns = [column, 'count']
    # Store the dataframe in the dictionary
    value_counts_dict[column] = value_counts_df


df_all_dates = pd.DataFrame()

# Iterate over each dataframe in the dictionary
for column, df in value_counts_dict.items():
    # Append the dataframe to the list
    df.rename(columns={column:'date'}, inplace=True)
    df_all_dates= pd.concat([df_all_dates, df], axis=0)

# df_all_dates['date'] = pd.to_datetime(df_all_dates['date'], errors='coerce')
df_date_counts = df_all_dates.groupby('date')['count'].sum().reset_index()

# Rename the columns
df_date_counts.columns = ['date', 'total_count']
df_date_counts.plot(x='date', y='total_count', kind='line', linewidth=0.1)

df_date_counts['date'] = pd.to_datetime(df_date_counts['date'], errors='coerce')

# smooth the total_count by week
df_date_counts['date'].describe()
# find the date that is missing from the date range
date_range = pd.date_range(start='1995-03-28', end='2022-10-31')
missing_dates = date_range[~date_range.isin(df_date_counts['date'])]

df_date_counts['total_count_smooth'] = df_date_counts['total_count'].rolling(window=7).mean()
df_date_counts['total_count_smooth'].plot(kind='line',linewidth=0.1)


# ------------------------------------------------------------------------------------------------------------------
# disease prevalence using #all_icd
# ------------------------------------------------------------------------------------------------------------------
def count_disease(x, disease):
    if str(x) == 'None':
        return 0
    else:
        return x.count(disease)

df_disease_prev = pd.DataFrame()
filed_name = 'all_icd'
field_id = HES_ICD_ids['all_icd']['id']
uniq_diseases = df_disease[f'{filed_name}_first_4'].explode().unique()
disease_dict = {}

for disease in uniq_diseases:
    count = df_disease[f'{filed_name}_first_4'].apply(lambda x: count_disease(x, disease)).sum()
    disease_dict[disease] = count
    df_disease_prev[disease] = [count]

df_disease_prev.to_csv(intermediate_path / 'disease_prevalence_4_digits.csv', index=False)

df_disease_prev = df_disease_prev.T.reset_index()

df_disease_prev.rename(columns={'index':'disease_4', 0:'count'}, inplace=True)
df_disease_prev.drop(df_disease_prev.loc[df_disease_prev['disease_4']=='Unnamed: 22'].index, inplace=True)

df_disease_prev['diseases_1'] = df_disease_prev['disease_4'].apply(lambda x: str(x)[0])
df_disease_prev['diseases_2'] = df_disease_prev['disease_4'].apply(lambda x: int(str(x)[1:3]))
df_disease_prev['diseases_3'] = df_disease_prev['disease_4'].apply(lambda x: str(x)[:3])
