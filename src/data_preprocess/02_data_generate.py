"""
Created by valler at 15/02/2024 execute at the same date
Feature: generate the datafile from the codebook
Steps:
"""
import pandas as pd
from src import params
import ast
import os

def data_reader(row):
    """
    the function to retrieve data
    :param row:
    :return: data of selected cols in df
    """
    file_names = [int(x) for x in row.file_names.split(';')]
    ids = ast.literal_eval(row.ids)
    df = pd.DataFrame()
    for file_name in file_names:
        try:
            cols = [id_ for id_ in ids[file_name] if int(id_.split('_')[1].split('i')[-1]) <= instance]
        except:
            cols = ids[file_name]
        temp = pd.read_csv(params.participant_path / f'{file_name}.csv')
        temp = temp.loc[:, cols]
        df = pd.concat([df, temp], axis=1)
    return df

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

    print(f"unique values = {list(pd.unique(df_read.values.ravel('K')))}")
    print(f'null in each columns:\n{df_read.isnull().sum()}')
    print(f'for field "{row.field_name}", instance_count = {row.instance_count},arrary_count = {row.array_count}')

def recode_type_generator(row):
    """
    overall recoding information generator
    :return: recode_type
    """
    recode_type = {}
    recode_dict = {'a': 'average', 't': 'this_wave'}
    recode_type['recode'] = True if input('recode this field? y') == 'y' else False

    # instance
    if pd.isnull(row.instance) :
        recode_type['i']='single_wave'

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

    def replacing_recode(df_read):

        replace_dict = replace_dict_basics
        unique_vals = list(pd.unique(df_read.values.ravel('K')))
        print(f'unique values = {unique_vals}')

        auto_replace_contrl = True if input('do you want to launch the automatic dict for numbers? y') == 'y' else False
        if auto_replace_contrl:
            replace_dict = auto_dict(replace_dict, unique_vals)
        else:
            for unique_val in set(unique_vals) - set(replace_dict.keys()):
                replace_dict = manual_dict(replace_dict, unique_val)
        replace_dict = {x:y for x,y in zip(replace_dict.keys(),replace_dict.values()) if x in unique_vals}

        df_read.replace(replace_dict, inplace=True)
        return replace_dict, df_read

    # main function body
    if pd.isnull(row['replace_dict']):
        # check whether the replace_dict already existed
        replace_dict, df_read = replacing_recode(df_read)

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

def recoding_process(ind, row):

    df_read = data_reader(row)
    print_basic_info(df_read, row)

    drop_col = True if input('drop this column? y/n') == 'y' else False
    if not drop_col:
        # preprocessing code

        # step 0: generate recode type dict
        try:
            recode_type = ast.literal_eval(row.recode_type)
        except:
            recode_type = recode_type_generator(row)

        # step 1: recode check
        if recode_type['recode']:
            replace_dict, df_read = replace_recode_main(row, df_read)
            df_codebook.loc[ind, 'replace_dict'] = str(replace_dict)

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
            field_name = f'p{row.field_id}'if pd.isnull(row.instance) else f'p{row.field_id}_i{instance if str(instance)==max(row.instance.split(";") if isinstance(row.instance,str) else row.instance) else row.instance}'
            df[f'{row.field_id}'] = df_read[field_name]
        elif 'w' in recode_type['i']:  # e.g. w1
            df[f'{row.field_id}'] = df_read[f'p{row.field_id}_i{ins}']
        elif recode_type['i'] == 'single_wave':
            print('this is a single wave variable')
            field_name = f'p{row.field_id}'if pd.isnull(row.instance) else f'p{row.field_id}_i{row.instance}'
            df[f'{row.field_id}'] = df_read[field_name]
        else:
            print('key in unknown wave coding information')

        df_codebook.loc[ind, 'preprocessed_flag'] = 1
        df_codebook.loc[ind, 'recode_type'] = str(recode_type)

        if input('update the replace_dict?') == 'y':
            replace_dict_basics.update(replace_dict)
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
instance = 4

wave_codebook_path = params.codebook_path / f'UKB_preprocess_codebook_wave_{instance}.csv'
if os.path.isfile(wave_codebook_path):
    df_codebook = pd.read_csv(wave_codebook_path)
else:
    df_codebook = pd.read_csv(params.codebook_path/'UKB_var_select.csv')
    df_codebook['preprocessed_flag'] = [None]*len(df_codebook)


df = pd.DataFrame()
# df = pd.read_csv('/Users/valler/Python/OX_Thesis/Chapter_2_Disease_Clustering/Data/preprocessed_data/UKB_wave_4_Lifestyle_4.csv')
cate_names = params.cate_names
cate_name = cate_names[2]
iterators = df_codebook.loc[df_codebook['cate_name'] == cate_name, ].iterrows()

replace_dict_basics = params.replace_dict_basics

file_count = 1

i = 0
while i < 30:
    print(i)
    ind, row = next(iterators)
    if pd.isnull(row['preprocessed_flag']):
        try:
            df, df_codebook = recoding_process(ind, row)
            i += 1
        except Exception as e:
            print(e)
            if 'No such file or directory' in str(e):
                continue
            elif input('break? y/n')=='y':
                break
            elif input('relaunch? y/n')=='y':
                df, df_codebook = recoding_process(ind, row)
                i += 1

print(df_codebook['preprocessed_flag'].value_counts())
print(f'left to code = {df_codebook["preprocessed_flag"].isnull().sum()}')


file_count += 1
del df
df = pd.DataFrame()
# only keep essential columns for df_codebook
# df_codebook[params.codebook_basic_columns].to_csv(params.codebook_path/'UKB_var_select.csv',index=False)
# print(replace_dict_basics)
