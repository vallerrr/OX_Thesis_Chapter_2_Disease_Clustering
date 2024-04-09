"""
Created by valler at 15/02/2024 execute at the same date
Feature: generate the datafile from the codebook
Steps:
"""
import pandas as pd
from src import params
import ast
import os
import numpy as np
from datetime import datetime
from dateutil.parser import parse
from src.data_preprocess import utils



def average(df):
    '''
    average by row, ignoring NAs
    :param df:
    :return:
    '''
    return df.mean(skipna=True, axis=1)

def array_name_by_ins(row, instance):
    if instance == '':
        return [f'p{row.field_id}_a{i}' for i in row['array'].split(';')]
    else:
        return [f'p{row.field_id}_i{instance}_a{i}' for i in row['array'].split(';')]

def print_basic_info(df_read, row):
    null_dict = df_read.isnull().sum()
    print(f"unique values = {list(pd.unique(df_read.values.ravel('K')))}")
    print(f'null in each columns:\n{null_dict}')
    print(f'for field "{row.field_name}", instance_count = {row.instance_count},arrary_count = {row.array_count}')
    return null_dict.to_dict()
def recode_type_generator(row):
    """
    overall recoding information generator
    :return: recode_type
    """
    recode_type = {}
    recode_dict = {'a': 'average', 't': 'this_wave'}
    recode_type['recode'] =  input('recode this field? y')

    # instance
    if pd.isnull(row.instance) :
        recode_type['i'] =  'single_wave'

    elif len(row.instance.split(";")) == 1:
        # only available in one future wave
        recode_type['i'] = 'single_wave'
    else:
        recode_type_i = input('instance recode type')
        try:
            recode_type_i = recode_dict[recode_type_i]
        except:
            print(f'use {recode_type_i} as instance recode type')
        recode_type['i'] = recode_type_i

    # array
    if row.array_count > 0:
        recode_type_a = input('array recode type')
        try:
            recode_type_a = recode_dict[recode_type_a]
        except:
            print(f'use {recode_type_i} as array recode type')
        recode_type['a'] = recode_type_a
    return recode_type

def replace_recode_main(row, df_read):
    """
    packed replacing recode function
    :param row:
    :param df_read:
    :return: replace_dict,df_read
    """

    def manual_dict(replace_dict, unique_val):
        replace_val = input(f'for value "{unique_val}", replace it with value: (z->None)')
        replace_dict[unique_val] = float(replace_val) if not replace_val == 'z' else None
        return replace_dict

    def auto_dict(replace_dict, unique_vals):
        for unique_val in unique_vals:
            try:
                float_value = float(unique_val)
                if not pd.isnull(unique_val):
                    replace_dict[unique_val] = float_value
            except:
                if unique_val in set(unique_vals) - set(replace_dict.keys()):
                    replace_dict = manual_dict(replace_dict, unique_val)
        return replace_dict

    def replacing_recode(df_read,replace_dict):

        unique_vals = list(pd.unique(df_read.values.ravel('K')))
        print(f'unique values = {unique_vals}')

        if all(unique_val in [np.nan, 'No', 'Yes', 'Do not know','Prefer not to answer','Not sure'] for unique_val in unique_vals):

            auto_replace_contrl = True
        else:
            auto_replace_contrl = True if input('do you want to launch the automatic dict? y') == 'y' else False

        if auto_replace_contrl:
            replace_dict = auto_dict(replace_dict, unique_vals)
        else:

            for unique_val in set(unique_vals) - set(replace_dict.keys()):
                if not pd.isnull(unique_val):
                    replace_dict = manual_dict(replace_dict, unique_val)
        replace_dict = {x:y for x,y in zip(replace_dict.keys(),replace_dict.values()) if x in unique_vals}

        df_read.replace(replace_dict, inplace=True)
        return replace_dict, df_read

    # main function body
    if pd.isnull(row['replace_dict']):
        # check whether the replace_dict already existed
        replace_dict, df_read = replacing_recode(df_read, replace_dict_basics)
        df_codebook.loc[ind, 'replace_dict'] = str(replace_dict)
    else:
        replace_dict = row['replace_dict'].replace('nan: None,','').replace('nan:None,','')
        replace_dict = ast.literal_eval(replace_dict)
        print(f'replace dict exists: {replace_dict}')
        unique_complete_check = set(pd.unique(df_read.values.ravel('K')))-set(replace_dict.keys())
        if len(unique_complete_check)>0:
            auto_dict(replace_dict, unique_complete_check)

        df_read.replace(replace_dict, inplace=True)
    return replace_dict, df_read


def recoding_process_main(ind, row):

    df_read = utils.data_reader(row)
    null_dict = print_basic_info(df_read, row)
    df_codebook.loc[ind, 'missing_info_general'] = str(null_dict)

    drop_col = True if input('drop this column? y/n') == 'y' else False
    if not drop_col:
        # preprocessing code

        # step 0: generate recode type dict
        try:
            recode_type = ast.literal_eval(row.recode_type)
        except:
            recode_type = recode_type_generator(row)

        # step 1: recode check
        if recode_type['recode'] =='y':
            replace_dict, df_read = replace_recode_main(row, df_read)
            df_codebook.loc[ind, 'replace_dict'] = str(replace_dict)

            # update the replace_dict
            update_flag = input('update the replace_dict?')
            if update_flag == 'y':
                replace_dict_basics.update(replace_dict)
            elif update_flag == 'n':
                for key in replace_dict.keys():
                    if key in replace_dict_basics.keys():
                        replace_dict_basics.pop(key)
            # if update_flag == any other character, it will ignore the command
        elif recode_type['recode']=='t': # timestamp type
            df_read.apply(lambda col:utils.try_to_datetime(col))

        # step 2: array check
        if 'a' in recode_type.keys():
            # first array then instance
            if pd.isnull(row.instance):
                row.instance = ''
            for ins in row.instance.split(';'):
                array_cols = array_name_by_ins(row, ins)
                if 'average' in recode_type['a']:
                    df_read[f'p{row.field_id}_i{ins}'] = average(df_read[array_cols])

        # step 3: instance check
        if recode_type['i'] == 'average':
            df[f'{row.field_id}'] = average(df_read[[x for x in df_read.columns if 'a' not in x]])
        elif recode_type['i'] == 'this_wave':
            field_name = f'p{row.field_id}'if pd.isnull(row.instance) else f'p{row.field_id}_i{ max(row.instance.split(";") if isinstance(row.instance,str) else row.instance)}'
            df[f'{row.field_id}'] = df_read[field_name]
        elif recode_type['i'].startswith('w'):  # e.g. w1
            ins = int(recode_type['i'].replace('w',''))
            df[f'{row.field_id}'] = df_read[f'p{row.field_id}_i{ins}']
        elif recode_type['i'] == 'single_wave':
            print('this is a single wave variable')
            field_name = f'p{row.field_id}'if pd.isnull(row.instance) else f'p{row.field_id}_i{row.instance}'
            df[f'{row.field_id}'] = df_read[field_name]
        else:
            print('key in unknown wave coding information')

        df_codebook.loc[ind, 'preprocessed_flag'] = 1
        df_codebook.loc[ind, 'recode_type'] = str(recode_type)

        df_codebook.loc[ind, 'missing_count'] = df[f'{row.field_id}'].isnull().sum()
        print(f"missing  count = {df_codebook.loc[ind, 'missing_count']}")
    else:
        df_codebook.loc[ind, 'preprocessed_flag'] = -1

    df_codebook.loc[ind, 'note'] = input('any note?')

    df.to_csv(params.preprocessed_path / f'UKB_wave_{instance}_{cate_name}_{file_count}.csv', index=False)
    df_codebook.to_csv(params.codebook_path / f'UKB_preprocess_codebook_wave_{instance}.csv', index=False)

    return df, df_codebook

    #  {'average_i':'average across instance','this_wave':'do not consider cross wave difference'}


# cognitive function as an example
instance = 0

wave_codebook_path = params.codebook_path / f'UKB_preprocess_codebook_wave_{instance}.csv'
if os.path.isfile(wave_codebook_path):
    df_codebook = pd.read_csv(wave_codebook_path)
else:
    df_codebook = pd.read_csv(params.codebook_path/'UKB_var_select.csv')
    df_codebook['preprocessed_flag'] = [None]*len(df_codebook)



file_count = 0

cate_names = params.cate_names
cate_name = cate_names[4]
iterators = df_codebook.loc[df_codebook['cate_name'] == cate_name, ].iterrows()
replace_dict_basics = params.replace_dict_basics

df = pd.DataFrame()
df = pd.read_csv(f'/Users/valler/Python/OX_Thesis/Chapter_2_Disease_Clustering/Data/preprocessed_data/UKB_wave_4_{cate_name}_{file_count}.csv')


i = 0
while i < 30:
    print(i)
    ind, row = next(iterators)
    if pd.isnull(row['preprocessed_flag']):
        try:
            df, df_codebook = recoding_process_main(ind, row)
            i += 1
        except Exception as e:
            print(e)
            if 'No such file or directory' in str(e):
                continue
            elif input('skip? y/n') == 'y':
                df_codebook.loc[ind,'preprocessed_flag'] = 0
            else:
                if input('break? y/n') == 'y':
                    break
                if input('relaunch? y/n') == 'y':
                    df, df_codebook = recoding_process_main(ind, row)
        i += 1

print(df_codebook['preprocessed_flag'].value_counts())
print(f'left to code = {df_codebook["preprocessed_flag"].isnull().sum()}')

file_count += 1
del df
df = pd.DataFrame()


df_codebook.loc[df.loc[df['preprocessed_flag'].notnull()].index,'preprocessed_flag'] = 0

temp = df_codebook.loc[df_codebook['cate_name'] == cate_name, ]
temp['preprocessed_flag'].value_counts()

'''
# only keep essential columns for df_codebook
df_codebook[params.codebook_basic_columns].to_csv(params.codebook_path/'UKB_var_select.csv',index=False)
print(replace_dict_basics)
df_codebook[['preprocessed_flag','cate_name']].value_counts()
'''

'''
columns = ['20508', '20534', '20546', '20437', '20533', '20517', '20535', '20536',
       '21027']
columns  = [int(x) for x in columns]
df_codebook.loc[df_codebook['field_id'].isin(columns),'preprocessed_flag']=[None]*len(columns)
df_codebook.loc[df_codebook['field_id'].isin(columns),'replace_dict']=[None]*len(columns)
df_codebook.loc[df_codebook['field_id'].isin(columns),'recode_type']=[None]*len(columns)



'''



