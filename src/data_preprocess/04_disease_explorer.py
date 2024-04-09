"""
# Created by valler at 02/04/2024
Feature: 

"""
import pandas as pd
from src import params
from pathlib import Path
import matplotlib.pyplot as plt
import datetime
record_column = 'main_icd'
import numpy as np
import ast
import warnings
warnings.filterwarnings('ignore')

HES_ICD_ids = {"all_icd": {"id": '41270', "time": '41280'}, "main_icd": {"id": '41202', "time": '41262'}, "second_icd": {"id": '41203', "time": None}}
intermediate_path = Path('/Users/valler/Python/OX_Thesis/Chapter_2_Disease_Clustering/Data/intermediate_files')
df_single_record = pd.read_csv(intermediate_path / f'{record_column}_complete.csv')
dates_col = [f'p{HES_ICD_ids[record_column]["time"]}_a{x}' for x in range(0,int(df_single_record[ f'{record_column}_uniq_count'].max()))]

for col in dates_col:
    df_single_record[col] = pd.to_datetime(df_single_record[col], errors='coerce', format='%Y-%m-%d')
df_single_record[record_column+'_first_3'] = [ast.literal_eval(x) if pd.notnull(x) else None for x in df_single_record[record_column+'_first_3']]

# ------------------------------------------------------------------------------------------------------------------
# define comorbidity
# ------------------------------------------------------------------------------------------------------------------
comorbidity_interval=datetime.timedelta(days=365)

def calculate_comorbidity(row):
    if pd.notnull(row[f'{record_column}_uniq_count']):
        disease_count = int(row[f'{record_column}_uniq_count'])
        comorbidity = [f'{i};{j}' for i in range(0, disease_count - 1) for j in range(i + 1, disease_count) if abs(row[dates_col[i]] - row[dates_col[j]]) <= comorbidity_interval]
        return str(comorbidity)
    return None


df_single_record['comorbidity'] = df_single_record.apply(calculate_comorbidity, axis=1)
# count the comorbidity
df_single_record['comorbidity_count'] = df_single_record['comorbidity'].apply(lambda x: len(ast.literal_eval(x)) if pd.notnull(x) else 0)
df_single_record.to_csv(intermediate_path / f'{record_column}_complete.csv', index=False)


# ------------------------------------------------------------------------------------------------------------------
# disease prevalence using #all_icd
# ------------------------------------------------------------------------------------------------------------------
def count_disease(x, disease):
    if str(x) == 'None':
        return 0
    else:
        return x.count(disease)

df_disease_prev = pd.DataFrame()
filed_name = 'main_icd'
field_id = HES_ICD_ids['all_icd']['id']

# 1. get the total count
uniq_diseases = df_single_record[f'{filed_name}_first_3'].explode().unique()

disease_dict = {}
for disease in uniq_diseases:
    count = df_single_record[f'{filed_name}_first_3'].apply(lambda x: count_disease(x, disease)).sum()
    disease_dict[disease] = count
    df_disease_prev[disease] = [count]
df_disease_prev = df_disease_prev.T.reset_index()

df_disease_prev.rename(columns={'index':'disease_3',0:'count'}, inplace=True)
df_disease_prev.drop(df_disease_prev.loc[df_disease_prev['disease_3']=='Unnamed: 22'].index, inplace=True)
df_disease_prev.to_csv(intermediate_path / f'{record_column}_disease_prevalence_4_digits.csv', index=False)



df_disease_prev= pd.read_csv(intermediate_path / f'{record_column}_disease_prevalence_4_digits.csv')

# 2. year information
def count_year(row, disease):
    if pd.isnull(row[f'{filed_name}_uniq_count']):
        return None
    else:
        diseases_lst = list(row[f'{filed_name}_first_3'])
        if disease in diseases_lst:
            index = [i for i, x in enumerate(diseases_lst) if x == disease]
            years = [row[dates_col[i]].year for i in index]
            return years
        return None

for disease in df_disease_prev['disease_3'].unique():
    print(count)
    df_single_record['years'] = df_single_record.apply(lambda x: count_year(x, disease), axis=1)

    # Flatten the list of years and count the occurrences
    year_counts = pd.Series([y for x in df_single_record['years'].dropna() for y in x]).value_counts().to_dict()
    df_disease_prev.loc[df_disease_prev['disease_3'] == disease, 'years'] = str(year_counts)

df_disease_prev.to_csv(intermediate_path / f'{record_column}_disease_prevalence_4_digits.csv', index=False)

# 3. map the dict in year column to each single year in the dataframe,i.e. every key in the dict has a column in the dataframe
years = range(1992, 2024)
for year in years:
    df_disease_prev[year] = df_disease_prev['years'].apply(lambda x: ast.literal_eval(x).get(year) if pd.notnull(x) and (str(year) in x) else None)

# 4. mark the diseases as chapters
df_ICD = pd.read_csv(intermediate_path / 'ICD10.csv')

def find_parent_id(x):
    if pd.notnull(x):
        parent_id = df_ICD.loc[df_ICD['coding'] == x, 'parent_id'].values[0]
        parent_cat = df_ICD.loc[df_ICD['node_id'] == parent_id, 'coding'].values[0]
        return parent_cat
    return None

df_disease_prev['parent_coding'] = df_disease_prev['disease_3'].apply(find_parent_id)
df_disease_prev['chapter'] = df_disease_prev['parent_coding'].apply(find_parent_id)
df_disease_prev.to_csv(intermediate_path / f'{record_column}_disease_prevalence_4_digits.csv', index=False)

# 5. plot the disease prevalence by year and chapter
df_disease_prev = pd.read_csv(intermediate_path / f'{record_column}_disease_prevalence_4_digits.csv')

df_disease_to_plot = df_disease_prev.groupby('chapter').sum()
df_disease_to_plot.drop(axis=1, columns=['disease_3', 'years', 'parent_coding'], inplace=True)
df_disease_to_plot = df_disease_to_plot.T

fig, ax = plt.subplots(figsize=(10, 7))
df_disease_to_plot.drop(axis=0,index=['count',1992,1993,1994,2023]).reindex(sorted(df_disease_to_plot.columns), axis=1).plot(kind='bar', stacked=True, ax=ax, width=0.8)
handles, labels = ax.get_legend_handles_labels() # reverse the order of legend
ax.legend(reversed(handles), reversed(labels), title='Chapter', title_fontsize='large', fontsize='small', loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
plt.savefig(params.current_path / f'plot/{record_column}_disease_prevalence_by_year_and_chapter.pdf')



