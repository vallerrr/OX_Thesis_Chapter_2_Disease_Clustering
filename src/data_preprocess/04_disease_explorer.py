import os.path
import pandas as pd
from src import params
from pathlib import Path
from src.data_preprocess import utils
import matplotlib.pyplot as plt
import datetime
import numpy as np
import ast
import warnings
import polars as pl
from prefixspan import PrefixSpan
warnings.filterwarnings('ignore')

chronic_control = ['_chronic',''][0]
record_column = params.disease_record_column
HES_ICD_ids = {"all_icd": {"id": '41270', "time": '41280'}, "main_icd": {"id": '41202', "time": '41262'}, "second_icd": {"id": '41203', "time": None}}
intermediate_path = Path('/Users/valler/Python/OX_Thesis/Chapter_2_Disease_Clustering/Data/intermediate_files')
df_codebook = pd.read_csv(params.codebook_path / 'UKB_preprocess_codebook_wave_0.csv')

# read the df_single_record
if os.path.isfile(intermediate_path / f'{record_column}_complete.csv'):
    df_single_record = pd.read_csv(intermediate_path / f'{record_column}_complete.csv')
else:
    df_single_record = utils.create_single_record_df(record_column,HES_ICD_ids)

dates_col = [f'p{HES_ICD_ids[record_column]["time"]}_a{x:03d}' for x in range(0, int(df_single_record[f'{record_column}_uniq_count'].max()))]

# convert the dates to datetime (following chunk of code should always be run)
for col in dates_col:
    df_single_record[col] = pd.to_datetime(df_single_record[col], errors='coerce', format='%Y-%m-%d')

# ------------------------------------------------------------------------------------------------------------------
# 1 define comorbidity
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
# 2 disease prevalence
# ------------------------------------------------------------------------------------------------------------------
def count_disease(x, disease):
    if str(x) == 'None':
        return 0
    else:
        return x.count(disease)

df_disease_prev = pd.DataFrame()
filed_name = record_column

# 2.1. generate the df_disease_prev

uniq_diseases = df_single_record[f'{filed_name}_first_3'].explode().unique()

disease = uniq_diseases[0]

disease_dict = {}
for disease in uniq_diseases:
    if (disease not in disease_dict.keys()) &(not pd.isnull(disease)):
        count = df_single_record[f'{filed_name}_first_3'].apply(lambda x: count_disease(x, disease)).sum()
        disease_dict[disease] = count
        df_disease_prev[disease] = [count]
df_disease_prev = df_disease_prev.T.reset_index()


df_disease_prev.rename(columns={'index': 'disease_3', 0:'count'}, inplace=True)
df_disease_prev.drop(df_disease_prev.loc[df_disease_prev['disease_3']=='Unnamed: 22'].index, inplace=True)
df_disease_prev.drop(index = df_disease_prev.loc[df_disease_prev['disease_3'].isna(),].index,axis=1,inplace=True)
df_disease_prev.to_csv(intermediate_path / f'{record_column}_disease_prevalence_4_digits.csv', index=False)

df_disease_prev = pd.read_csv(intermediate_path / f'{record_column}_disease_prevalence_4_digits.csv')

# 2.2. year information of each disease in the df_disease_prev dataframe
def count_year(row, disease):
    """
    count the year information of specific diseases across all samples
    :param row:
    :param disease:
    :return: {}
    """
    if row[f'{record_column}_uniq_count']==0:
        return None
    else:
        diseases_lst = row[f'{record_column}_first_3']

        if disease in diseases_lst:
            index = [i for i, x in enumerate(diseases_lst) if x == disease]
            years = [row[dates_col[i]].year for i in index]
            return years


uniq_diseases = df_disease_prev['disease_3'].unique().tolist()

df_disease_prev['years'] = [None for x in range(len(df_disease_prev))]
for disease in uniq_diseases:
    print(uniq_diseases.index(disease))
    if str(df_disease_prev.loc[df_disease_prev['disease_3'] == disease, 'years'].values[0])=='None':

        df_single_record['years'] = df_single_record.apply(lambda x: count_year(x, disease), axis=1)

        # Flatten the list of years and count the occurrences
        year_counts = pd.Series([y for x in df_single_record['years'].dropna() for y in x]).value_counts().to_dict()
        df_disease_prev.loc[df_disease_prev['disease_3'] == disease, 'years'] = str(year_counts)

df_disease_prev.to_csv(intermediate_path / f'{record_column}_disease_prevalence_4_digits.csv', index=False)

# 3. map the dict in year column to each single year in the dataframe,i.e. every key in the dict has a column in the dataframe
years = range(1992, 2024)
for year in years:
    df_disease_prev[year] = df_disease_prev['years'].apply(lambda x: ast.literal_eval(x).get(year) if pd.notnull(x) and (str(year) in x) else None)

# 4. mark the diseases as chapters and parent code
df_ICD = pd.read_csv(intermediate_path / 'ICD_10.csv')

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
df_ICD = pd.read_csv(intermediate_path / 'ICD_10.csv')

df_disease_to_plot = df_disease_prev.groupby('chapter').sum()
df_disease_to_plot.drop(axis=1, columns=['disease_3', 'years', 'parent_coding'], inplace=True)

fig, ax = plt.subplots(figsize=(10, 7))
df_disease_to_plot.drop(axis=1, columns=['count','1992', '1993', '1994', '2023']).reindex(sorted(df_disease_to_plot.columns), axis=1).T.plot(kind='bar', stacked=True, ax=ax, width=0.8)
handles, labels = ax.get_legend_handles_labels() # reverse the order of legend
ax.legend(reversed(handles), reversed(labels), title='Chapter', title_fontsize='large', fontsize='small', loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()

plt.savefig(params.current_path / f'plot/{record_column}_disease_prevalence_by_year_and_chapter.pdf')

# --------------------------------------------------------------
# 3 disease prevalence using #all_icd
# --------------------------------------------------------------

df_single_record = pd.read_csv(intermediate_path / f'{record_column}_complete.csv')
# sequence analysis

diseases_db = [ast.literal_eval(x) for x in df_single_record[f'{record_column}_first_3'] if pd.notnull(x)]
ps = PrefixSpan(diseases_db)
df_disease_prev = pd.read_csv(intermediate_path / f'{record_column}_disease_prevalence_4_digits.csv')


trial = ps.topk(1066)  # 1066 is the number of diseases over 1000 counts
# convert trial to df
df_comorbidity = pd.DataFrame(columns=['frequency', 'disease'])
df_comorbidity['frequency'] = [x[0] for x in trial]
df_comorbidity['disease'] = [x[1] for x in trial]
df_comorbidity['disease_count'] = [len(x) for x in df_comorbidity['disease']]
df_comorbidity.drop(df_comorbidity.loc[df_comorbidity['disease_count']==1].index, inplace=True)

# --------------------------------------------------------------
# analysis at the chapter & individual level
# --------------------------------------------------------------
df_disease_prev = pd.read_csv(intermediate_path / f'{record_column}_disease_prevalence_4_digits.csv')
df_single_record = pd.read_csv(intermediate_path / f'{record_column}_complete.csv')
df_single_record[f'{record_column}_first_3'] = [ast.literal_eval(x) if pd.notnull(x) else None for x in df_single_record[f'{record_column}_first_3']]

# df_single_record['icd_chapter_coding'] = [ast.literal_eval(x) if pd.notnull(x) else None for x in df_single_record['icd_chapter_coding']]

df_ICD = pd.read_csv(intermediate_path / 'ICD_10.csv')

# log the chapter information for the 'main_icd_first_3' and replace with numbers
df_single_record['icd_parent_coding'] = [[find_parent_id(i) for i in x] if str(x)!='None' else None for x in df_single_record[f'{record_column}_first_3']]
df_single_record['icd_chapter_coding'] = [[find_parent_id(i) for i in x] if str(x)!='None' else None for x in df_single_record['icd_parent_coding']]


chapter_replace_dict = {f"{x}": y for y,x in zip(range(1, 23),['Chapter I', 'Chapter II', 'Chapter III', 'Chapter IV', 'Chapter V', 'Chapter VI', 'Chapter VII', 'Chapter VIII', 'Chapter IX', 'Chapter X', 'Chapter XI', 'Chapter XII', 'Chapter XIII', 'Chapter XIV', 'Chapter XV', 'Chapter XVI', 'Chapter XVII', 'Chapter XVIII', 'Chapter XIX', 'Chapter XX', 'Chapter XXI', 'Chapter XXII'])}# at the chapter level
df_single_record['icd_chapter_coding'] = [[chapter_replace_dict[i] for i in x] if str(x)!='None' else None for x in df_single_record['icd_chapter_coding']]


df_single_record.to_csv(intermediate_path / f'{record_column}_complete.csv', index=False)

diseases_db = df_single_record['icd_chapter_coding'].dropna().to_list()
ps = PrefixSpan(diseases_db)
trial = ps.topk(10000)
# will the ps.topk() return repeatited count?
df_disease_pattern_by_chapter = pd.DataFrame(columns=['frequency', 'disease'])
df_disease_pattern_by_chapter['frequency'] = [x[0] for x in trial]
df_disease_pattern_by_chapter['disease'] = [x[1] for x in trial]
df_disease_pattern_by_chapter['disease_count'] = [len(x) for x in df_disease_pattern_by_chapter['disease']]

# check the disease by single chapter
for chap in range(1, 23):
    df_disease_pattern_by_chapter[f'chap_{chap}_flag'] = [1 if chap in x else 0 for x in df_disease_pattern_by_chapter['disease']]

df_disease_pattern_by_chapter.to_csv(intermediate_path / f'{record_column}_disease_pattern_by_chapter.csv', index=False)

df_disease_pattern_by_chapter = pl.read_csv(intermediate_path / f'{record_column}_disease_pattern_by_chapter.csv')


# --------------------------------------------------------------
# 4 analysis for diseases occur during the research window
# --------------------------------------------------------------
year_window = [2006, 2010]
