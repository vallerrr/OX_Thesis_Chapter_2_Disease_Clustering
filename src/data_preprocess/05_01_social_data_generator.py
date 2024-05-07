"""
# Created by valler at 04/04/2024
Feature: generate the social data as outcome vars for the model
we only use the wave 0 for the social data
"""
import matplotlib.pyplot as plt
import pandas as pd
from src import params
import ast
import os
import numpy as np
from src.data_preprocess import utils
import warnings
from matplotlib import pyplot as plt
# =========================，，，=============================================================================================
# 1. Load the codebook
# ======================================================================================================================

df_codebook = pd.read_csv(params.codebook_path/'UKB_var_select.csv')
fail_dict = {}
replace_dict_basics = params.replace_dict_basics
cate_names = params.cate_names
# ======================================================================================================================
# 2. control zone
# ======================================================================================================================

file_count = 0
instance = 0
cate_name = cate_names[3]
iterators = df_codebook.loc[df_codebook['cate_name'] == cate_name, ].iterrows()


# ======================================================================================================================
# 3. process zone
# ======================================================================================================================

count = 0
while count <5:
    df = pd.DataFrame()

    while len(df.columns) < 30:
        try:
            ind, row = next(iterators)
        except:
            break

        if row['preprocessed_flag'] == 1:
            try:
                print(row['field_id'])
                print(f'Processing {row["field_name"]}')
                df, df_codebook = utils.recoding_process_main_for_final_data_generator(ind, row,df_codebook, df, instance, cate_name, file_count, replace_dict_basics)
            except Exception as e:
                print(f'Failed to process {row["field_name"]}, {e}')
                if cate_name in fail_dict.keys():
                    fail_dict[cate_name].append({row['field_name']: e})
                else:
                    fail_dict[cate_name] = [{row['field_name']: e}]
    file_count += 1
    count += 1

import json
with open(params.codebook_path / f'fail_dict_wave_{instance}.json', 'w') as f:
    for key, value in fail_dict.items():
        fail_dict[key] = [str(x) for x in value]

    json.dump(fail_dict, f, indent=4)


#  df.to_csv(params.preprocessed_path/)
"""
# check zone 

ind = df_codebook.loc[df_codebook['field_id']==20018,].index
row = pd.Series(
temp = utils.data_reader(row, instance)
row.file_names
row = df_codebook.loc[df_codebook['field_id']==20018,]
"""

# ======================================================================================================================
# 4. information from the df_codebook
# ======================================================================================================================

df_codebook = pd.read_csv(params.codebook_path/'UKB_preprocess_codebook_wave_0.csv')
df_codebook.preprocessed_flag_wave0.value_counts()

# 4.1 cross-tab
df_codebook[['preprocessed_flag', 'preprocessed_flag_wave0']].value_counts(dropna=False).plot(kind='bar')
plt.tight_layout()
plt.show()

df_codebook[['preprocessed_flag_wave0', 'cate_name']].value_counts().plot(kind='bar')
plt.tight_layout()
plt.show()

# Missing values histogram
df_codebook['missing_count'].dropna().plot(kind='hist', bins=100)
plt.title('Missing values histogram')
plt.show()

df_codebook[['preprocessed_flag_wave0', 'cate_name']].value_counts()

# ======================================================================================================================
# 5. selecting variables based on the missing values
# ======================================================================================================================

threshold = 0.2 * 502242  # 20% of the total number of participants
df_codebook['remain_flag'] = [1 if x < threshold else 0 for x in df_codebook['missing_count']]
df_codebook['remain_flag'].value_counts()
temp_remain = df_codebook.loc[df_codebook['remain_flag'] == 1, ]
temp_delete = df_codebook.loc[df_codebook['remain_flag'] == 0, ]
temp_delete['preprocessed_flag'].value_counts()

# mark the rows need to be rechecked
df_to_reprocess = pd.DataFrame()
for ind, row in temp_delete.iterrows():
    if ind not in df_to_reprocess.index:
        field_name = row['field_name']
        print('\n',f'{field_name}', '\ncate_name=', row['cate_name'],'\nmissing_count=',row['missing_count'],f'threshold={round(threshold)}','\npreprocessed_flag=', row['preprocessed_flag'])
        if input('remain? 1-yes') == '1':
            df_to_reprocess = pd.concat([df_to_reprocess, row], axis=1)

temp = df_to_reprocess.reset_index()
all_id = list(temp.loc[temp['index'] == 'field_id',0].values)

missed_rows = ['Average weekly beer plus cider intake','Never eat eggs, dairy, wheat, sugar (pilot)', 'Never eat eggs, dairy, wheat, sugar','Drive faster than motorway speed limit', 'Medication for pain relief, constipation, heartburn','Vitamin and mineral supplements (pilot)','Illness, injury, bereavement, stress in last 2 years','Had major operations' ,'Had other major operations','Diastolic blood pressure, automated reading']
additional_ids = list(df_codebook.loc[df_codebook['field_name'].isin(missed_rows),'field_id'].values)
all_id=all_id+additional_ids

all_id =[10137, 399, 845, 6138, 10722, 6142, 20119, 796, 777, 132, 816, 806, 826, 767, 757, 6143, 22617, 22601, 22611, 22607, 22609, 22608, 22606, 21000, 3659, 10877, 6139, 10860, 6140, 26433, 26422, 26425, 26434, 26431, 26421, 26419, 26430, 26420, 26432, 26423, 26428, 26418, 26427, 26426, 26424, 6146, 4674, 20022, 1677, 20115, 1647, 1618, 1578, 1608, 5364, 1568, 1598, 1339, 1329, 1299, 10912, 10886, 1140, 1130, 10105, 1110, 10749, 1120, 22035, 22036, 22032, 22038, 22039, 22037, 22033, 22034, 10962, 3647, 1001, 914, 10971, 874, 10953, 981, 884, 904, 864, 1090, 1080, 1070, 6164, 924, 2149, 1220, 1170, 1180, 1190, 1160, 1200, 1210, 1239, 1269, 1279, 1259, 1031, 10740, 6160, 1050, 1060, 42014, 40008, 40007, 20510, 20513, 2794, 3591, 2774, 2784, 2814, 2734, 3393, 135, 10004, 6179, 10854, 6155, 10723, 137, 5663, 20127, 5386, 6149, 10006, 136, 3079, 41214, 41218, 94, 95, 102, 4080, 93, 3082, 20258, 3062, 10694, 1807, 20107, 20110, 20111, 3526, 845, 6138, 10722, 6142, 20119, 796, 777, 132, 816, 806, 757, 6143, 1588, 6144, 10855, 1100, 6154, 10007, 6145, 2415, 2844, 4079]
all_id.remove(5386)  # df_codebook.loc[df_codebook['field_name']=='Number of unenthusiastic/disinterested episodes','field_id']

df_codebook['round_2_flag'] = [1 if x in all_id else 0 for x in df_codebook['field_id']]
df_codebook['round_2_flag'].value_counts()

# 32 duplicates are all from the double input of the same field in different categories: education and employment & socio-demographics
df_codebook.to_csv(params.codebook_path/'UKB_preprocess_codebook_wave_0.csv', index=False)

