"""
# Created by valler at 22/04/2024
Feature: 

"""

import matplotlib.pyplot as plt
import pandas as pd
from src import params
import ast
import os
import numpy as np
from src.data_preprocess import utils
import warnings


def register_missings(df_codebook, field_ids, note, file_name, df, field_id, replace_dict=None, missing_count=None,missing = 'mean'):
    if f'p{field_id}' in df.columns:
        missing_count = df[f'p{field_id}'].isnull().sum()
    for field_id in field_ids:
        df_codebook.loc[df_codebook['field_id'] == field_id, 'preprocessed_flag'] = 1
        df_codebook.loc[df_codebook['field_id'] == field_id, 'note'] = "manually code:" + note
        df_codebook.loc[df_codebook['field_id'] == field_id, 'missing_count'] = missing_count
        df_codebook.loc[df_codebook['field_id'] == field_id, 'file_name'] = file_name
        df_codebook.loc[df_codebook['field_id'] == field_id, 'missing'] = missing
        if replace_dict:
            df_codebook.loc[df_codebook['field_id'] == field_id, 'replace_dict'] = str(replace_dict)
    df.to_csv(params.preprocessed_path / file_name, index=False)
    df_codebook.to_csv(params.codebook_path / 'UKB_preprocess_codebook_wave_0.csv', index=False)
    return df_codebook

def read_multiple_fields(field_ids,df_codebook):
    temp=pd.DataFrame()
    rows = df_codebook.loc[df_codebook['field_id'].isin(field_ids),]
    for ind in range(len(rows)):
        if len(temp)==0:
            temp = utils.data_reader(rows.iloc[ind], instance=0)
        else:
            temp = pd.merge(temp, utils.data_reader(rows.iloc[ind], instance=0), on='eid')
    return rows, temp

df_codebook = pd.read_csv(params.codebook_path / 'UKB_preprocess_codebook_wave_0.csv')
all_id = [10137, 399, 845, 6138, 10722, 6142, 20119, 796, 777, 132, 816, 806, 826, 767, 757, 6143, 22617, 22601, 22611, 22607, 22609, 22608, 22606, 21000, 3659, 10877, 6139, 10860, 6140, 26433, 26422, 26425, 26434, 26431, 26421, 26419, 26430, 26420, 26432, 26423, 26428, 26418, 26427, 26426, 26424,
          6146, 4674, 20022, 1677, 20115, 1647, 1618, 1578, 1608, 5364, 1568, 1598, 1339, 1329, 1299, 10912, 10886, 1140, 1130, 10105, 1110, 10749, 1120, 22035, 22036, 22032, 22038, 22039, 22037, 22033, 22034, 10962, 3647, 1001, 914, 10971, 874, 10953, 981, 884, 904, 864, 1090, 1080, 1070, 6164,
          924, 2149, 1220, 1170, 1180, 1190, 1160, 1200, 1210, 1239, 1269, 1279, 1259, 1031, 10740, 6160, 1050, 1060, 42014, 40008, 40007, 20510, 20513, 2794, 3591, 2774, 2784, 2814, 2734, 3393, 135, 10004, 6179, 10854, 6155, 10723, 137, 5663, 20127, 5386, 6149, 10006, 136, 3079, 41214, 41218, 94,
          95, 102, 4080, 93, 3082, 20258, 3062, 10694, 1807, 20107, 20110, 20111, 3526, 845, 6138, 10722, 6142, 20119, 796, 777, 132, 816, 806, 757, 6143, 1588, 6144, 10855, 1100, 6154, 10007, 6145, 2415, 2844, 4079]
all_id.remove(5386)  # df_codebook.loc[df_codebook['field_name']=='Number of unenthusiastic/disinterested episodes','field_id']

instance = 0
df_round_2 = df_codebook.loc[df_codebook['round_2_flag'] == 1,]
df_round_2 = df_round_2.drop_duplicates(subset='field_id', keep='first')

iterators = df_round_2.iterrows()
file_name = f'UKB_wave_{instance}_Round_2_variables_0.csv'
df = pd.read_csv(params.preprocessed_path / file_name)
# 10137,399 Number of incorrect matches in round
# example of the variable with the pilot data, the field_id always go with the one without (pilot) in the title

# type 1: data with pilot information

# no.1
field_ids = [10137, 399]
field_id_ind = 1
recode_type_ind = 0
field_id = field_ids[field_id_ind]

rows = df_codebook.loc[df_codebook['field_id'].isin(field_ids),]
temp = utils.data_reader(rows.iloc[0], instance=0)
temp = pd.merge(temp, utils.data_reader(rows.iloc[1], instance=0), on='eid')
recode_type = ast.literal_eval(rows.iloc[recode_type_ind].recode_type)

df['eid'] = temp['eid']
# step 1: recode check

if 'a' in recode_type.keys():
    array_cols = temp.columns[1:]
    df[f'p{field_id}'] = utils.average(temp[array_cols])

df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note='combined with the pilot data', file_name=file_name, df=df, field_id=field_id,missing = 'median')

# --------------------------------------------------------------------------------------------------------------------------------
# no.2  Qualifications

field_ids = [6138, 10722]
field_id_ind = 1
recode_type_ind = 0
field_id = field_ids[field_id_ind]

rows = df_codebook.loc[df_codebook['field_id'].isin(field_ids),]
temp = utils.data_reader(rows.iloc[0], instance=0)
temp = pd.merge(temp, utils.data_reader(rows.iloc[1], instance=0), on='eid')

# unique of unique values
unique_values = list(set(item for sublist in temp.drop(columns='eid').apply(pd.Series.unique).explode().dropna().tolist() for item in ast.literal_eval(sublist)))
replace_dict = utils.auto_dict({}, unique_values)

for fid in temp.drop(columns='eid').columns:
    print(fid)
    temp[fid] = [ast.literal_eval(x) if not pd.isnull(x) else None for x in temp[fid]]
    temp[fid] = temp[fid].apply(lambda x: [replace_dict.get(item, item) for item in x] if isinstance(x, list) else x)
    temp[fid] = [max(x) if isinstance(x, list) else None for x in temp[fid]]

temp[f'p{field_id}'] = utils.average(temp[temp.drop(columns='eid').columns])

df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note='combined with the pilot data, vals in vals', file_name=file_name, df=df, field_id=field_id, replace_dict=replace_dict,missing='mean')

# --------------------------------------------------------------------------------------------------------------------------------
# no.3 Current Employment status


field_ids = [6142, 20119]
field_id_ind = 0
recode_type_ind = 0
field_id = field_ids[field_id_ind]

rows = df_codebook.loc[df_codebook['field_id'].isin(field_ids),]
temp = utils.data_reader(rows.iloc[0], instance=0)
temp = pd.merge(temp, utils.data_reader(rows.iloc[1], instance=0), on='eid')

# add the information from the 20019 to 6142
temp[f'p{field_id}_i0'] = [ast.literal_eval(x) if not pd.isnull(x) else None for x in temp[f'p{field_id}_i0']]
temp[f'p{field_id}_i0'] = [x if pd.isnull(y) else list(set(x + [y])) if isinstance(x, list) else [y] for x, y in zip(temp[f'p{field_id}_i0'], temp[f'p20119_i0'])]

temp[f'p{field_id}_retired'] = [1 if isinstance(x, list) and 'Retired' in x else 0 if isinstance(x, list) else None for x in temp[f'p{field_id}_i0']]
temp[f'p{field_id}_employed'] = [1 if isinstance(x, list) and 'In paid employment or self-employed' in x else 0 if isinstance(x, list) else None for x in temp[f'p{field_id}_i0']]
temp[f'p{field_id}_unable_to_work'] = [1 if isinstance(x, list) and 'Unable to work because of sickness or disability' in x else 0 if isinstance(x, list) else None for x in temp[f'p{field_id}_i0']]
temp[f'p{field_id}_doing_unpaid_work'] = [1 if isinstance(x, list) and any(item in x for item in ['Doing unpaid or voluntary work', 'Looking after home and/or family', 'Full or part-time student']) else 0 if isinstance(x, list) else None for x in temp[f'p{field_id}_i0']]

df = pd.merge(df, temp[['eid', f'p{field_id}_retired', f'p{field_id}_employed', f'p{field_id}_unable_to_work', f'p{field_id}_doing_unpaid_work']], on='eid')

# record for new columns in the codebook
new_columns = [f'p{field_id}_retired', f'p{field_id}_employed', f'p{field_id}_unable_to_work', f'p{field_id}_doing_unpaid_work']


for column in new_columns:
    print(column)
    df_codebook.loc[len(df_codebook)+1, :] = [column, 'Socio-demographics', field_id, 'self_defined', 'self_defined', None, None, None, 0, None, None, None, {'recode': 'self-derived'}, None, -1, f'derived from field {field_id} when combined with field 20119', None, 1, df[column].isnull().sum(), file_name, 0, 1,None]

# update the codebook for original cats
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=f'derving new 4 columns {new_columns}', file_name=file_name, df=df, field_id=field_id, replace_dict={}, missing_count=df[column].isnull().sum())

# --------------------------------------------------------------------------------------------------------------------------------
# No.4 Job Code - current job, using field 20277
# 20277 is not in the codebook
# we code this variable at the most coarse level of the job code
field_id = 20277
filed_id_in_file = 'p20277_i0'

# read the column
recorder = pd.read_csv(params.recorder_path / 'recorder_participant.csv')
file_num = recorder.loc[recorder['field_name'] == filed_id_in_file, 'ind'].values[0]
temp = pd.read_csv(params.participant_path / f'{round(file_num)}.csv')
temp = temp[['eid', filed_id_in_file]]

unique_values = list(set(temp[filed_id_in_file].dropna().tolist()))
job_code_dict = params.job_code_dict
temp[f'p{field_id}'] = [job_code_dict.get(x, None) for x in temp[filed_id_in_file]]
replace_dict = {'Managers and Senior Officials': 9,
                'Professional Occupations': 8,
                'Associate Professional and Technical Occupations': 7,
                'Personal Service Occupations': 4,
                'Administrative and Secretarial Occupations': 6,
                'Elementary Occupations': 1,
                'Sales and Customer Service Occupations': 3,
                'Process, Plant and Machine Operatives': 2,
                'Skilled Trades Occupations': 5}
temp[f'p{field_id}'] = [replace_dict.get(x, None) for x in temp[f'p{field_id}']]
df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook.loc[len(df_codebook)+1, :] = ['Job Code', 'Employment', field_id, 100073, None, None, None, None, 0, None, None, None, {'recode': 'self-derived','missing': 'median'}, None, -1, f'not in the original df_codebook, derived from field {field_id}, job codes are stored at the most coarse level based on the soc 2000 code, original dict can be found at the params.py', None, 1, df[f'p{field_id}'].isnull().sum(),file_name, 0, 1,'median']

# --------------------------------------------------------------------------------------------------------------------------------
# No.5 Foreign born
field_ids = [3659]

field_id = 3659
rows = df_codebook.loc[df_codebook['field_id'].isin(field_ids),]
temp = utils.data_reader(rows.iloc[0], instance=0)
temp[f'p{field_id}'] = [1 if pd.notnull(x) else 0 for x in temp[f'p{field_id}_i0']]
df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note='if there is repsonse -> 1, else 0', file_name=file_name, df=df, field_id=field_id)

rows = df_codebook.loc[df_codebook['field_id'].isin(field_ids),]


# --------------------------------------------------------------------------------------------------------------------------------
# No.6 Ethnic background - white vs non-white
field_ids = [21000]
field_id = 21000
rows = df_codebook.loc[df_codebook['field_id'].isin(field_ids),]
temp = utils.data_reader(rows.iloc[0], instance=0)
white_group = ['British', 'Irish', 'Any other white background']
temp[f'p{field_id}'] = [1 if x in white_group else 0 for x in temp[f'p{field_id}_i0']]

df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note="white vs non-white, white group = ['British', 'Irish', 'Any other white background']", file_name=file_name, df=df, field_id=field_id)


# --------------------------------------------------------------------------------------------------------------------------------
# No.7 Average total household income before tax
field_ids = [738, 10877]
field_id_ind = 0
recode_type_ind = 0
field_id = field_ids[field_id_ind]

rows, temp = read_multiple_fields(field_ids,df_codebook)
recode_type = ast.literal_eval(rows.iloc[recode_type_ind].recode_type)
replace_dict = ast.literal_eval(rows.iloc[recode_type_ind].replace_dict.replace(' nan: None, ', '').replace(' nan:None, ', '').replace(', nan: None', '').replace(' nan: None', ''))
replace_dict['31,000 to 52,000']=3
replace_dict['18,000 to 31,000']=2

temp.replace(to_replace=replace_dict.keys(), value=replace_dict.values(), inplace=True)
temp[f'p{field_id}'] = utils.average(temp[temp.columns[1:]])

df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')

# update the missing value method
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note="manually add the pilot information (10877) to the original field ", file_name=file_name, df=df, field_id=field_id, replace_dict=replace_dict,missing='median')
rows = df_codebook.loc[df_codebook['field_id'].isin(field_ids),]
# --------------------------------------------------------------------------------------------------------------------------------
# No.8 Gas or solid-fuel cooking/heating
note = "response score = response count"
field_ids = [6139, 10860]

field_id_ind= 0
field_id = field_ids[field_id_ind]
rows, temp = read_multiple_fields(field_ids, df_codebook)

temp[f'p{field_id}_i0'] = [ast.literal_eval(x) if not pd.isnull(x) else None for x in temp[f'p{field_id}_i0']]
temp[f'p{field_ids[1]}_i0'] = [ast.literal_eval(x) if not pd.isnull(x) else None for x in temp[f'p{field_ids[1]}_i0']]
temp[f'p{field_id}_i0'] = [x if not isinstance(y,list) else list(set(x + y)) if isinstance(x, list) else y for x, y in zip(temp[f'p{field_id}_i0'], temp[f'p{field_ids[1]}_i0'])]

count_response =['A gas hob or gas cooker','A gas fire that you use regularly in winter time','An open solid fuel fire that you use regularly in winter time']
false_response = ['Prefer not to answer','Do not know']
temp[f'p{field_id}']=[sum(item in x for item in count_response) if isinstance(x, list) and all(item not in x for item in false_response) else None for x in temp[f'p{field_id}_i0']]

df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='mean')

df_round_2['dealt_flag'] = [1 if x<=79 else 0 for x in df_round_2.index]
# --------------------------------------------------------------------------------------------------------------------------------
# No.9 Attendance/disability/mobility allowance
note = "response score = count of allowance received"
field_ids = [6146]

field_id=6146
rows, temp = read_multiple_fields(field_ids, df_codebook)

temp[f'p{field_id}_i0'] = [ast.literal_eval(x) if not pd.isnull(x) else None for x in temp[f'p{field_id}_i0']]


count_response =['Attendance allowance', 'Disability living allowance', 'Blue badge']
false_response = ['Prefer not to answer','Do not know']
temp[f'p{field_id}']=[sum(item in x for item in count_response) if isinstance(x, list) and all(item not in x for item in false_response) else None for x in temp[f'p{field_id}_i0']]

df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id)
df_round_2.loc[df_round_2['field_id'].isin(field_ids),'dealt_flag']=1

# --------------------------------------------------------------------------------------------------------------------------------
# NO.10  Frequency of friend/family visits
note = 'combined with pilot information'
field_ids = [1031, 10740]
field_id_ind = 0
recode_type_ind = 1
field_id = field_ids[field_id_ind]

rows, temp = read_multiple_fields(field_ids, df_codebook)

recode_type = ast.literal_eval(rows.iloc[recode_type_ind].recode_type)
replace_dict = ast.literal_eval(rows.iloc[recode_type_ind].replace_dict.replace(' nan: None, ', '').replace(' nan:None, ', '').replace(', nan: None', '').replace(' nan: None', ''))
replace_dict['Do not know'] = None
temp.replace(to_replace=replace_dict.keys(), value=replace_dict.values(), inplace=True)
temp[f'p{field_id}'] = utils.average(temp[temp.columns[1:]])

df= pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id, replace_dict=replace_dict)
df_round_2.loc[df_round_2['field_id'].isin(field_ids),'dealt_flag']=1

# --------------------------------------------------------------------------------------------------------------------------------
# No.11  Variation in diet (pilot)
note = 'combined with pilot information'
field_ids = [1548, 10912]
field_id_ind = 0
recode_type_ind = 0
field_id = field_ids[field_id_ind]

rows, temp = read_multiple_fields(field_ids, df_codebook)
recode_type = ast.literal_eval(rows.iloc[recode_type_ind].recode_type)
replace_dict = ast.literal_eval(rows.iloc[recode_type_ind].replace_dict.replace(' nan: None, ', '').replace(' nan:None, ', '').replace(', nan: None', '').replace(' nan: None', ''))
replace_dict = params.clean_replace_dict(replace_dict)
temp.replace(to_replace=replace_dict.keys(), value=replace_dict.values(), inplace=True)
temp[f'p{field_id}'] = utils.average(temp[temp.columns[1:]])

df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id, replace_dict=replace_dict)

df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.12 never eat eggs, dariy, wheat sugar
field_ids = [6144,10855]
note = 'combined with pilot information and is a positive response answer'
field_id_ind = 0
field_id = field_ids[field_id_ind]

rows, temp = read_multiple_fields(field_ids, df_codebook)
all_values = []
for column in temp.columns:
    if column != 'eid':
        unique_values = temp[column].dropna().unique().tolist()
        all_values.extend(unique_values)
unique_values = list(set(item for sublist in all_values for item in ast.literal_eval(sublist)))
replace_dict = utils.auto_dict({}, unique_values)

for fid in temp.drop(columns='eid').columns:
    print(fid)
    temp[fid] = [ast.literal_eval(x) if not pd.isnull(x) else None for x in temp[fid]]
    temp[fid] = temp[fid].apply(lambda x: [replace_dict.get(item, item) for item in x] if isinstance(x, list) else x)
    temp[fid] = [sum(value for value in x if value is not None ) if isinstance(x, list) else None for x in temp[fid]]

temp[f'p{field_id}'] = utils.average(temp[temp.columns[1:]])
missing  = '0'
df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id, replace_dict=replace_dict, missing=missing)
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.13  Duration of walks
field_ids = [874, 10953]
note = 'combined with pilot information'
field_id_ind = 0
field_id = field_ids[field_id_ind]

rows, temp = read_multiple_fields(field_ids, df_codebook)
unique_values = temp['p10953_i0'].unique()
replace_dict = utils.auto_dict({}, unique_values)
temp.replace(to_replace=replace_dict.keys(), value=replace_dict.values(), inplace=True)
temp['p874_i0']= [float(x) if not pd.isnull(x) else None for x in temp['p874_i0']]

temp[f'p{field_ids[field_id_ind]}'] = utils.average(temp[temp.columns[1:]])


df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id, replace_dict=replace_dict)
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.14 Number of days/week of moderate physical activity 10+ minutes
field_ids = [884, 904, 864]
note = ''

rows, temp = read_multiple_fields(field_ids, df_codebook)
for field in field_ids:
    column = f'p{field}_i0'
    temp[column] = [float(x) if not pd.isnull(x) and x not in ['Do not know','Prefer not to answer','Unable to walk'] else None for x in temp[column]]
    if f'p{field}' in df.columns:
        df.drop(columns=[f'p{field}'], inplace=True)
    df = pd.merge(df, temp[['eid', column]], on='eid')
    df.rename(columns={column: f'p{field}'}, inplace=True)
    df_codebook = register_missings(df_codebook=df_codebook, field_ids=[field], note=note, file_name=file_name, df=df, field_id=field, missing= 'median')


df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.15  Time spent driving，using computer, watching TV
field_ids = [1090, 1080, 1070]
note = ''

rows, temp = read_multiple_fields(field_ids, df_codebook)
unique_values = list(set(temp[temp.columns[1:]].apply(pd.Series.unique).explode().dropna().tolist()))
replace_dict = utils.auto_dict({}, unique_values)

for field in field_ids:
    column = f'p{field}_i0'
    temp[column] = temp[column].replace(replace_dict)
    if f'p{field}' in df.columns:
        df.drop(columns=[f'p{field}'], inplace=True)
    df = pd.merge(df, temp[['eid', column]], on='eid')
    df.rename(columns={column: f'p{field}'}, inplace=True)
    df_codebook = register_missings(df_codebook=df_codebook, field_ids=[field], note=note, file_name=file_name, df=df, field_id=field, missing= 'median',replace_dict=replace_dict)


df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.16  Lifetime number of sexual partners
field_ids = [2149]
field_id = 2149
note = 'divided by 10 +1'
rows, temp = read_multiple_fields(field_ids, df_codebook)
unique_values = list(set(temp['p2149_i0'].explode().dropna().tolist()))
replace_dict = utils.auto_dict({}, unique_values)
temp.replace(replace_dict.keys(), replace_dict.values(), inplace=True)

temp[f'p{field_id}'] = [x//10+1 if isinstance(x,float) else None for x in temp['p2149_i0']]
df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')

df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id, missing= 'median',replace_dict=replace_dict)
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.17 sleep related variables
field_ids = [1220, 1170, 1180, 1190, 1160, 1200, 1210]
note = ''
rows, temp = read_multiple_fields(field_ids, df_codebook)

unique_values = list(set(temp[temp.columns[1:]].apply(pd.Series.unique).explode().dropna().tolist()))
replace_dict = utils.auto_dict({}, unique_values)
temp.replace(replace_dict.keys(), replace_dict.values(), inplace=True)

for field in field_ids:
    column = f'p{field}_i0'
    temp[column] = temp[column].replace(replace_dict)
    if f'p{field}' in df.columns:
        df.drop(columns=[f'p{field}'], inplace=True)
    df = pd.merge(df, temp[['eid', column]], on='eid')
    df.rename(columns={column: f'p{field}'}, inplace=True)
    df_codebook = register_missings(df_codebook=df_codebook, field_ids=[field], note=note, file_name=file_name, df=df, field_id=field, missing= 'median',replace_dict=replace_dict)

df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# new file to be saved
file_name = f'UKB_wave_{instance}_Round_2_variables_1.csv'
df = pd.read_csv(params.preprocessed_path / file_name)
df = pd.DataFrame()
df['eid'] = temp['eid']
# --------------------------------------------------------------------------------------------------------------------------------
# No.18  smoking exposures
field_ids = [1239, 1269, 1279, 1259]
note = ''
rows, temp = read_multiple_fields(field_ids, df_codebook)
unique_values = list(set(temp[temp.columns[1:]].apply(pd.Series.unique).explode().dropna().tolist()))
replace_dict = utils.auto_dict({}, unique_values)

for field in field_ids:
    column = f'p{field}_i0'
    temp[column] = temp[column].replace(replace_dict)
    df = pd.merge(df, temp[['eid', column]], on='eid')
    df.rename(columns={column: f'p{field}'}, inplace=True)
    df_codebook = register_missings(df_codebook=df_codebook, field_ids=[field], note=note, file_name=file_name, df=df, field_id=field, missing= 'median',replace_dict=replace_dict)

df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.19  Physical measurements
field_ids_list = [[4079, 94], [95, 102], [4080, 93]]
for field_ids in field_ids_list:
    note = 'code with manual information'
    field_id_ind=0
    field_id = field_ids[field_id_ind]

    rows, temp = read_multiple_fields(field_ids, df_codebook)
    recode_type = ast.literal_eval(rows.iloc[field_id_ind].recode_type)

    for field in field_ids:
        if 'a' in recode_type.keys():
            array_cols = [x for x in temp.columns if x.startswith(f'p{field}')]
            temp[f'p{field}_i0'] = utils.average(temp[array_cols])

    temp[f'p{field_id}'] = utils.average(temp[[f'p{field}_i0' for field in field_ids]])
    df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')

    df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id)
    df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.20  spirometry measurements
field_ids = [3062, 10694]
note = 'code with pilot information'
field_id_ind = 0
field_id = field_ids[field_id_ind]
rows, temp = read_multiple_fields(field_ids, df_codebook)

recode_type = ast.literal_eval(rows.iloc[field_id_ind].recode_type)
for field in field_ids:
    if 'a' in recode_type.keys():
        array_cols = [x for x in temp.columns if x.startswith(f'p{field}')]
        temp[f'p{field}_i0'] = utils.average(temp[array_cols])
temp[f'p{field_id}'] = utils.average(temp[[f'p{field}_i0' for field in field_ids]])
df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')

df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='median')
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.21  parent's age at death
field_ids = [1807, 3526]  # TODO: use information from all waves
note = ''

rows, temp = read_multiple_fields(field_ids, df_codebook)
unique_values = list(set(temp[temp.columns[1:]].apply(pd.Series.unique).explode().dropna().tolist()))
auto_dict = utils.auto_dict({}, unique_values)
temp.replace(auto_dict.keys(), auto_dict.values(), inplace=True)
for field in field_ids:
    column = f'p{field}_i0'
    df = pd.merge(df, temp[['eid', column]], on='eid')
    df.rename(columns={column: f'p{field}'}, inplace=True)
    df_codebook = register_missings(df_codebook=df_codebook, field_ids=[field], note=note, file_name=file_name, df=df, field_id=field, missing= 'median',replace_dict=auto_dict)

df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1
rows = df_codebook.loc[df_codebook['field_id'].isin(field_ids),]

# --------------------------------------------------------------------------------------------------------------------------------
# No.22  deprivation
# multiple deprivation
field_ids = [26410,26427,26426]
field_id_ind = 0
note = 'replace the value by the quantile number for each area (including England, Wales and Scotland) in the field_ids(26410,26427,26426)'
field_id = field_ids[field_id_ind]
rows, temp = read_multiple_fields(field_ids, df_codebook)

# replace the value by the quantile number for each column in the field_ids
for field in field_ids:
    column = f'p{field}'
    deciles = np.linspace(temp[column].min(), temp[column].max(), 11)  # Creates 10 evenly spaced values from 0 to 1
    decile_labels = np.arange(1, 11)
    temp[field] = pd.cut(temp[column], bins=deciles, labels=decile_labels)
    temp[field] = temp[field].astype(float)


temp[f'p{field_id}'] = utils.average(temp[field_ids])
df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='median')
df_codebook.loc[df_codebook['field_id'].isin(field_ids), 'round_2_flag'] = 1

df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.23  fish take
field_ids = [1339, 1329]
note = 'fish intake = oily/non-oily fish intake'
rows, temp = read_multiple_fields(field_ids, df_codebook)
replace_dict = params.replace_dict_basics
replace_dict['Less than one'] = 0.5
replace_dict=params.clean_replace_dict(replace_dict)
temp.replace(replace_dict.keys(), replace_dict.values(), inplace=True)
temp[f'p{field_ids[0]}'] = utils.average(temp[temp.columns[1:]].astype(float))

df = pd.merge(df, temp[['eid', f'p{field_ids[0]}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_ids[0],replace_dict=replace_dict)
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.24  sun exposure
field_ids = [1050, 1060]
note = 'time spent outdoors'
rows, temp = read_multiple_fields(field_ids, df_codebook)
unique_values = temp[f'p{field_ids[1]}_i0'].unique()
replace_dict = utils.auto_dict({}, unique_values)
temp.replace(replace_dict.keys(), replace_dict.values(), inplace=True)
for field in field_ids:
    column = f'p{field}_i0'
    df = pd.merge(df, temp[['eid', column]], on='eid')
    df.rename(columns={column: f'p{field}'}, inplace=True)
    df_codebook = register_missings(df_codebook=df_codebook, field_ids=[field], note=note, file_name=file_name, df=df, field_id=field, missing= 'median',replace_dict=replace_dict)

df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1
rows = df_codebook.loc[df_codebook['field_id'].isin(field_ids),]

# --------------------------------------------------------------------------------------------------------------------------------
# No.25  Salad intake
field_ids = [1299]
rows, temp = read_multiple_fields(field_ids, df_codebook)
note = ' '
replace_dict = params.replace_dict_basics
replace_dict['Less than one'] = 0.5
temp.replace(replace_dict.keys(), replace_dict.values(), inplace=True)
temp.rename(columns={f'p{field_ids[0]}_i0': f'p{field_ids[0]}'}, inplace=True)

df = pd.merge(df, temp[['eid', f'p{field_ids[0]}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_ids[0],replace_dict=replace_dict)
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.26  Drive faster than motorway speed limit
field_ids = [1100]
note = ''
rows, temp = read_multiple_fields(field_ids, df_codebook)
replace_dict = params.replace_dict_basics
replace_dict['Do not drive on the motorway'] = 0
temp.replace(replace_dict.keys(), replace_dict.values(), inplace=True)
temp[f'p{field_ids[0]}'] = utils.average(temp[temp.columns[1:]].astype(float))
df = pd.merge(df, temp[['eid', f'p{field_ids[0]}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_ids[0],replace_dict=replace_dict)
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.27  Duration of moderate physical activity
field_ids = [894, 10962]
field_id_ind = 0
note = 'code with pilot information'
field_id = field_ids[field_id_ind]

rows, temp = read_multiple_fields(field_ids, df_codebook)
unique_values = list(set(temp[temp.columns[1:]].apply(pd.Series.unique).explode().dropna().tolist()))
replace_dict = utils.auto_dict({}, unique_values)
temp.replace(replace_dict.keys(), replace_dict.values(), inplace=True)

temp[f'p{field_id}'] = utils.average(temp[temp.columns[1:]].astype(float))
temp.fillna(0, inplace=True)
df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='0',replace_dict=replace_dict)
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.28  Duration of vigorous physical activity
field_ids = [914, 10971]
field_id_ind = 0
note = 'code with pilot information'
field_id = field_ids[field_id_ind]
rows, temp = read_multiple_fields(field_ids, df_codebook)
replace_dict={'2 to 4 hours': 180.0, 'Do not know': None, '1 to 2 hours': 90.0, 'Less than 30 mins': 15.0, 'Prefer not to answer': None, 'More than 4 hours': 249.0, '30 mins to 1 hour': 45.0}
temp.replace(replace_dict.keys(), replace_dict.values(), inplace=True)

temp[f'p{field_id}'] = utils.average(temp[temp.columns[1:]].astype(float))
temp.fillna(0, inplace=True)
df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='0',replace_dict=replace_dict)

df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.29 Townsend deprivation index at recruitment
field_ids = [22189]
temp = pd.read_csv(params.participant_path / '441.csv')
temp = temp[['eid', 'p22189']]
df = pd.merge(df, temp, on='eid')
df_codebook.loc[len(df_codebook), :] = ['Townsend deprivation index at recruitment', 'Socio-demographics', 22189,100094, 'Baseline characteristics', 1, 0, 0, {441:['p22189']}, 0, None, '441', {'recode': 'self-derived'}, None, 0, 'discovered when checking the MDI derived from field 22189', None, None, df['p22189'].isnull().sum(),file_name, 0, 1,'median']

# --------------------------------------------------------------------------------------------------------------------------------
# No.30  all other deprivation
# 30.1 crime score
field_ids = [26416,26425,26434]
field_id_ind = 0
note = 'replace the value by the quantile number for each area (including England, Wales and Scotland) in the field_ids(26416,26425,26434)'
field_id = field_ids[field_id_ind]
rows, temp = read_multiple_fields(field_ids, df_codebook)

# replace the value by the quantile number for each column in the field_ids
for field in field_ids:
    column = f'p{field}'
    deciles = np.linspace(temp[column].min(), temp[column].max(), 11)  # Creates 10 evenly spaced values from 0 to 1
    decile_labels = np.arange(1, 11)
    temp[field] = pd.cut(temp[column], bins=deciles, labels=decile_labels)
    temp[field] = temp[field].astype(float)


temp[f'p{field_id}'] = utils.average(temp[field_ids])
temp[f'p{field_id}'].hist()
plt.show()
df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='median')
df_codebook.loc[df_codebook['field_id'].isin(field_ids), 'round_2_flag'] = 1
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# 30.2 education score
field_ids = [26414,26431,26421]
field_id_ind = 0
note = 'replace the value by the quantile number for each area (including England, Wales and Scotland) in the field_ids(26414,26431,26421)'
field_id = field_ids[field_id_ind]
rows, temp = read_multiple_fields(field_ids, df_codebook)

# replace the value by the quantile number for each column in the field_ids
for field in field_ids:
    column = f'p{field}'
    deciles = np.linspace(temp[column].min(), temp[column].max(), 11)  # Creates 10 evenly spaced values from 0 to 1
    decile_labels = np.arange(1, 11)
    temp[field] = pd.cut(temp[column], bins=deciles, labels=decile_labels)
    temp[field] = temp[field].astype(float)


temp[f'p{field_id}'] = utils.average(temp[field_ids])
temp[f'p{field_id}'].hist()
plt.show()
df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='median')
df_codebook.loc[df_codebook['field_id'].isin(field_ids), 'round_2_flag'] = 1
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# 30.3 employment score
field_ids = [26412,26429,26419]
field_id_ind = 0
note = 'replace the value by the quantile number for each area (including England, Wales and Scotland) in the field_ids(26412,26429,26419)'
field_id = field_ids[field_id_ind]
rows, temp = read_multiple_fields(field_ids, df_codebook)
temp_p = pd.read_csv(params.participant_path / '656.csv')
temp = pd.merge(temp, temp_p[['eid', 'p26429']], on='eid')
del temp_p
# replace the value by the quantile number for each column in the field_ids
for field in field_ids:
    column = f'p{field}'
    deciles = np.linspace(temp[column].min(), temp[column].max(), 11)  # Creates 10 evenly spaced values from 0 to 1
    decile_labels = np.arange(1, 11)
    temp[field] = pd.cut(temp[column], bins=deciles, labels=decile_labels)
    temp[field] = temp[field].astype(float)


temp[f'p{field_id}'] = utils.average(temp[field_ids])
temp[f'p{field_id}'].hist()
plt.show()
df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='median')
df_codebook.loc[df_codebook['field_id'].isin(field_ids), 'round_2_flag'] = 1
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# 30.4 health score
field_ids = [26413,26430,26420]
field_id_ind = 0
note = 'replace the value by the quantile number for each area (including England, Wales and Scotland) in the field_ids(26413,26430,26420)'
field_id = field_ids[field_id_ind]
rows, temp = read_multiple_fields(field_ids, df_codebook)

# replace the value by the quantile number for each column in the field_ids
for field in field_ids:
    column = f'p{field}'
    deciles = np.linspace(temp[column].min(), temp[column].max(), 11)  # Creates 10 evenly spaced values from 0 to 1
    decile_labels = np.arange(1, 11)
    temp[field] = pd.cut(temp[column], bins=deciles, labels=decile_labels)
    temp[field] = temp[field].astype(float)
temp[f'p{field_id}'] = utils.average(temp[field_ids])
temp[f'p{field_id}'].hist()
plt.show()

df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='median')
df_codebook.loc[df_codebook['field_id'].isin(field_ids), 'round_2_flag'] = 1
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# 30.5 housing score
field_ids = [26415,26432,26423]
field_id_ind = 0
note = 'replace the value by the quantile number for each area (including England, Wales and Scotland) in the field_ids(26415,26432,26423)'
field_id = field_ids[field_id_ind]
rows, temp = read_multiple_fields(field_ids, df_codebook)

for field in field_ids:
    column = f'p{field}'
    deciles = np.linspace(temp[column].min(), temp[column].max(), 11)  # Creates 10 evenly spaced values from 0 to 1
    decile_labels = np.arange(1, 11)
    temp[field] = pd.cut(temp[column], bins=deciles, labels=decile_labels)
    temp[field] = temp[field].astype(float)
temp[f'p{field_id}'] = utils.average(temp[field_ids])
temp[f'p{field_id}'].hist()
plt.show()

df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='median')
df_codebook.loc[df_codebook['field_id'].isin(field_ids), 'round_2_flag'] = 1
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# 30.6 income score
field_ids = [26411,26428,26418]
field_id_ind = 0
note = 'replace the value by the quantile number for each area (including England, Wales and Scotland) in the field_ids(26411,26428,26418)'
field_id = field_ids[field_id_ind]
rows, temp = read_multiple_fields(field_ids, df_codebook)

for field in field_ids:
    column = f'p{field}'
    deciles = np.linspace(temp[column].min(), temp[column].max(), 11)  # Creates 10 evenly spaced values from 0 to 1
    decile_labels = np.arange(1, 11)
    temp[field] = pd.cut(temp[column], bins=deciles, labels=decile_labels)
    temp[field] = temp[field].astype(float)
temp[f'p{field_id}'] = utils.average(temp[field_ids])
temp[f'p{field_id}'].hist()
plt.show()

df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='median')
df_codebook.loc[df_codebook['field_id'].isin(field_ids), 'round_2_flag'] = 1
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# 30.7 living environment score
field_ids = [26417,26424]
field_id_ind = 0
note = 'replace the value by the quantile number for each area (including England, Wales) in the field_ids(26417,26424), scotland is not included and the score is replaced by mean'
field_id = field_ids[field_id_ind]
rows, temp = read_multiple_fields(field_ids, df_codebook)

for field in field_ids:
    column = f'p{field}'
    deciles = np.linspace(temp[column].min(), temp[column].max(), 11)  # Creates 10 evenly spaced values from 0 to 1
    decile_labels = np.arange(1, 11)
    temp[field] = pd.cut(temp[column], bins=deciles, labels=decile_labels)
    temp[field] = temp[field].astype(float)
temp[f'p{field_id}'] = utils.average(temp[field_ids])
temp[f'p{field_id}'].hist()
plt.show()

df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='mean')
df_codebook.loc[df_codebook['field_id'].isin(field_ids), 'round_2_flag'] = 1
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.31  Hearing difficulty/problems
field_ids = [2247,10793]
field_id_ind = 0
field_id = field_ids[field_id_ind]
rows, temp = read_multiple_fields(field_ids, df_codebook)
note = 'code with pilot information'

replace_dict = ast.literal_eval(rows.iloc[field_id_ind].replace_dict)
temp.replace(replace_dict.keys(), replace_dict.values(), inplace=True)
temp[f'p{field_id}'] = utils.average(temp[temp.columns[1:]].astype(float))
temp.fillna(0, inplace=True)
df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='0',replace_dict=replace_dict)
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.32  Illness, injury, bereavement, stress in last 2 years
field_ids = [6145,10721]
field_id_ind = 0
field_id = field_ids[field_id_ind]
note = 'positive response, count the response number'
rows, temp = read_multiple_fields(field_ids, df_codebook)
# unique of unique values
unique_values = list(set(item for sublist in temp.drop(columns='eid').apply(pd.Series.unique).explode().dropna().tolist() for item in ast.literal_eval(sublist)))
replace_dict = utils.auto_dict({}, unique_values)
replace_dict = {'Do not know': None, 'Death of a spouse or partner': 3.0, 'Marital separation/divorce': 1.0, 'Death of a close relative': 2.0, 'Prefer not to answer': None, 'Financial difficulties': 2.0, 'Serious illness, injury or assault to yourself': 4.0, 'Serious illness, injury or assault of a close relative': 1.0, 'None of the above': 0.0}

for fid in temp.drop(columns='eid').columns:
    print(fid)
    temp[fid] = [ast.literal_eval(x) if not pd.isnull(x) else None for x in temp[fid]]
    temp[fid] = temp[fid].apply(lambda x: [replace_dict.get(item, item) for item in x] if isinstance(x, list) else x)
    temp[fid] = [sum([num for num in x if pd.notnull(num)]) if isinstance(x, list) else None for x in temp[fid]]

temp[f'p{field_id}'] = utils.average(temp[temp.drop(columns='eid').columns])
df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='median', replace_dict=replace_dict)
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1

# --------------------------------------------------------------------------------------------------------------------------------
# No.33  Medication for pain relief, constipation, heartburn

field_ids = [6154,10004]
field_id_ind = 0
field_id = field_ids[field_id_ind]
note = 'positive response, count the response number'
rows, temp = read_multiple_fields(field_ids, df_codebook)
# unique of unique values
unique_values = list(set(item for sublist in temp.drop(columns='eid').apply(pd.Series.unique).explode().dropna().tolist() for item in ast.literal_eval(sublist)))
replace_dict = utils.auto_dict({}, unique_values)
replace_dict = {'Do not know': None, 'Omeprazole (e.g. Zanprol)': 1.0, 'Paracetamol': 1.0, 'Laxatives (e.g. Dulcolax, Senokot)': 1.0, 'Codeine': 1.0, 'Ranitidine (e.g. Zantac)': 1.0, 'Prefer not to answer': None, 'Aspirin': 1.0, 'Ibuprofen (e.g. Nurofen)': 1.0, 'None of the above': 0.0}

for fid in temp.drop(columns='eid').columns:
    print(fid)
    temp[fid] = [ast.literal_eval(x) if not pd.isnull(x) else None for x in temp[fid]]
    temp[fid] = temp[fid].apply(lambda x: [replace_dict.get(item, item) for item in x] if isinstance(x, list) else x)
    temp[fid] = [sum([num for num in x if pd.notnull(num)]) if isinstance(x, list) else None for x in temp[fid]]

temp[f'p{field_id}'] = utils.average(temp[temp.drop(columns='eid').columns])
df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='median', replace_dict=replace_dict)
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1


# --------------------------------------------------------------------------------------------------------------------------------
# No.34 Mouth/teeth dental problems
field_ids = [6149, 10006]
field_id_ind = 0
note = 'positive response, count the response number'
field_id = field_ids[field_id_ind]
rows, temp = read_multiple_fields(field_ids, df_codebook)
# unique of unique values
unique_values = list(set(item for sublist in temp.drop(columns='eid').apply(pd.Series.unique).explode().dropna().tolist() for item in ast.literal_eval(sublist)))
replace_dict = utils.auto_dict({}, unique_values)
replace_dict = {'Mouth ulcers': 1.0, 'Painful gums': 1.0, 'Bleeding gums': 1.0, 'Toothache': 1.0, 'Dentures': 1.0, 'Prefer not to answer': None, 'Painful teeth': 1.0, 'Lost or loose teeth': 1.0, 'Loose teeth': 1.0, 'None of the above': 0.0}

for fid in temp.drop(columns='eid').columns:
    print(fid)
    temp[fid] = [ast.literal_eval(x) if not pd.isnull(x) else None for x in temp[fid]]
    temp[fid] = temp[fid].apply(lambda x: [replace_dict.get(item, item) for item in x] if isinstance(x, list) else x)
    temp[fid] = [sum([num for num in x if pd.notnull(num)]) if isinstance(x, list) else None for x in temp[fid]]

temp[f'p{field_id}'] = utils.average(temp[temp.drop(columns='eid').columns])
temp[f'p{field_id}'].hist()
plt.show()

df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='median', replace_dict=replace_dict)
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1


# --------------------------------------------------------------------------------------------------------------------------------
# No.35  Vitamin and mineral supplements
field_ids = [6155,10007]
field_id_ind = 0
note = 'positive response, count the response number'
field_id = field_ids[field_id_ind]
rows, temp = read_multiple_fields(field_ids, df_codebook)
# unique of unique values
unique_values = list(set(item for sublist in temp.drop(columns='eid').apply(pd.Series.unique).explode().dropna().tolist() for item in ast.literal_eval(sublist)))
replace_dict = utils.auto_dict({}, unique_values)
replace_dict = {'Vitamin A': 1.0, 'Fish oil (including cod liver oil': 1.0, 'Evening primrose oil': 1.0, 'Glucosamine': 1.0, 'Vitamin C': 1.0, 'Garlic': 1.0, 'Ginkgo': 1.0, 'Prefer not to answer': None, 'Folic acid or Folate (Vit B9)': 1.0, 'Vitamin B': 1.0, 'Multivitamins +/- minerals': 1.0, 'Vitamin E': 1.0, 'Other supplements, vitamins or minerals': 1.0, 'Vitamin D': 1.0, 'None of the above': None}

for fid in temp.drop(columns='eid').columns:
    print(fid)
    temp[fid] = [ast.literal_eval(x) if not pd.isnull(x) else None for x in temp[fid]]
    temp[fid] = temp[fid].apply(lambda x: [replace_dict.get(item, item) for item in x] if isinstance(x, list) else x)
    temp[fid] = [sum([num for num in x if pd.notnull(num)]) if isinstance(x, list) else None for x in temp[fid]]


temp[f'p{field_id}'] = utils.average(temp[temp.drop(columns='eid').columns])
temp[f'p{field_id}'].hist()
plt.show()

df = pd.merge(df, temp[['eid', f'p{field_id}']], on='eid')
df_codebook = register_missings(df_codebook=df_codebook, field_ids=field_ids, note=note, file_name=file_name, df=df, field_id=field_id,missing='median', replace_dict=replace_dict)
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1


# ======================================================================================================================
# type 3: fill by average
field_ids = [845, 796, 777,1677,1618,22035,22036,22032]
df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = 1
df_codebook.loc[df_codebook['field_id'].isin(field_ids), 'round_2_flag'] = 1
df_codebook.loc[df_codebook['field_id'].isin(field_ids), 'missing'] = 'average'

# or by median
median_ids = [22038,22039,22037,22033,22034]
df_round_2.loc[df_round_2['field_id'].isin(median_ids), 'dealt_flag'] = 1
df_codebook.loc[df_codebook['field_id'].isin(median_ids), 'round_2_flag'] = 1
df_codebook.loc[df_codebook['field_id'].isin(median_ids), 'missing'] = 'median'

# df_codebook.loc[df_codebook['field_id'].isin(median_ids), 'recode_type'] = [ for x in df_codebook.loc[df_codebook['field_id'].isin(median_ids), 'recode_type']]

# type 4: don't need from second round check
field_ids = [4674,20022,816,826,767,757,6143,22611,22607,22609,22608,22606,1588,1578,1608,5364,1568,1598,10105,10749,981,3082,20258,20107,20110,20111]

df_round_2.loc[df_round_2['field_id'].isin(field_ids), 'dealt_flag'] = -1
df_codebook.loc[df_codebook['field_id'].isin(field_ids), 'round_2_flag'] = 0

# ======================================================================================================================
# check the data
# ======================================================================================================================

# remark the round_2_flag in df_codebook
dealt_ids = df_round_2.loc[df_round_2['dealt_flag'] == 1, 'field_id'].tolist()
dealt_ids.append([20277,22189]) # add the manually added fields
for x in [22617,132,22601,6140]: dealt_ids.remove(x)
df_codebook['round_2_flag'] = [1 if x in dealt_ids else 0 for x in df_codebook['field_id']]


df_codebook['final_keep_flag'] = [1 if not ((x==0) & (y ==0)) else 0 for x,y in zip(df_codebook['remain_flag'],df_codebook['round_2_flag'])]
df_codebook['final_keep_flag'].value_counts()
df_codebook_final = df_codebook.loc[df_codebook['final_keep_flag'] == 1,]

df_codebook_final.to_csv(params.codebook_path / 'UKB_preprocess_codebook_wave_0_final.csv', index=False)


# ======================================================================================================================
# standardise and missings
# ======================================================================================================================

# change the column names in both dfs to be just the field_id
df = pd.read_csv(params.preprocessed_path/ 'UKB_wave_0_Round_2_variables_0.csv')
df.rename(columns={column: int(column[1:]) for column in df.columns if column not in  ['eid','p6142_retired', 'p6142_employed', 'p6142_unable_to_work', 'p6142_doing_unpaid_work']}, inplace=True)
df.to_csv(params.preprocessed_path/ 'UKB_wave_0_Round_2_variables_0.csv', index=False)



# missing values
df_codebook_final['missing'] = ['mean' if not x=='median' else 'median' for x in df_codebook_final['missing']]

fail_ids = []
for ind,row in df_codebook_final.iterrows():
    field_id = str(round(row['field_id']))
    missing = row['missing']
    file_path = params.preprocessed_path / row['file_name']
    temp = pd.read_csv(file_path)
    try:
        if missing == 'mean':
            temp[field_id].fillna(temp[field_id].mean(), inplace=True)
        elif missing == 'median':
            temp[field_id].fillna(temp[field_id].median(), inplace=True)
        temp.to_csv(file_path, index=False)
    except:
        fail_ids.append(field_id)
fail_ids = [int(x) for x in fail_ids]
df_codebook_final.loc[df_codebook_final['field_id'].isin(fail_ids), 'missing_fail'] = 1
# ======================================================================================================================
# save the data
# ======================================================================================================================

