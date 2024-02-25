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
    return [f'p{row.field_id}_i{instance}_a{i}' for i in row['array'].split(';')]



def print_basic_info(df_read, row):

    print(f"unique values = {list(pd.unique(df_read.values.ravel('K')))}")
    print(f'null in each columns:\n{df_read.isnull().sum()}')
    print(f'for field "{row.field_name}", instance_count = {row.instance_count},arrary_count = {row.array_count}')

def recode_type_generator():
    """
    overall recoding information generator
    :return: recode_type
    """
    recode_type = {}
    recode_dict = {'a': 'average', 't': 'this_wave'}
    recode_type['recode'] = True if input('recode this field? y') == 'y' else False
    recode_type_i = input('instance recode type')
    try:
        recode_type_i = recode_dict[recode_type_i]
    except:
        print(f'use {recode_type_i} as instance recode type')
    recode_type['i'] = recode_type_i

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
            recode_type = recode_type_generator()

        # step 1: recode check
        if recode_type['recode']:
            replace_dict, df_read = replace_recode_main(row, df_read)
            df_codebook.loc[ind, 'replace_dict'] = str(replace_dict)

        # step 2: array check
        if 'a' in recode_type.keys():
            # first array then instance
            for ins in row.instance.split(';'):
                array_cols = array_name_by_ins(row, ins)
                if 'average' in recode_type['a']:
                    df_read[f'p{row.field_id}_i{ins}'] = average(df_read[array_cols])

        # step 3: instance check
        if recode_type['i'] == 'average':
            df[f'{row.field_id}'] = average(df_read[[x for x in df_read.columns if 'a' not in x]])
        if recode_type['i'] == 'this_wave':

            field_name = f'p{row.field_id}'if pd.isnull(row.instance) else f'p{row.field_id}_i{instance if str(instance)==max(row.instance.split(";")) else row.instance}'
            df[f'{row.field_id}'] = df_read[field_name]

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

replace_dict_basics = {'Prefer not to answer': None, 'Do not know': None, 'half': 0.5, '6+': 6, '2-4 times a week': 2, 'Once a week': 1, 'Less than once a week': 0.5, 'Never': 0, '5-6 times a week': 3, 'Once or more daily': 4, 'Sometimes': 1.0, 'Never/rarely': 0.0, 'Usually': 2.0, 'Always': 3.0, '1': 1.0, '2': 2.0, '3': 3.0, '4+': 4.0, 'quarter': 0.25, '3+': 3.0, 'No': 0.0, 'Yes': 1.0, '4': 4.0, '5': 5.0, 'low': 3.0, 'moderate': 2.0, 'high': 1.0, '5+': 5.0, 'varied': 0.5, 'Between 2 and 3 hours': 2.5, 'Between 30 minutes and 1 hour': 0.75, 'Between 1.5 and 2 hours': 1.75, 'Less than 15 minutes': 0.25, 'Between 15 and 30 minutes': 0.375, 'Over 3 hours': 3.0, 'Between 1 and 1.5 hours': 1.25, '4-5 times a week': 4.5, 'Once in the last 4 weeks': 0.25, '2-3 times a week': 2.5, '2-3 times in the last 4 weeks': 0.625, 'Every day': 7.0, "['Car/motor vehicle', 'Walk']": 3.5, "['Car/motor vehicle']": 4.0, "['Car/motor vehicle', 'Cycle']": 3.0, "['Walk']": 2.0, "['Car/motor vehicle', 'Walk', 'Public transport']": 3.0, "['Car/motor vehicle', 'Public transport']": 3.75, "['Public transport']": 3.0, "['Walk', 'Public transport']": 2.5, "['Car/motor vehicle', 'Walk', 'Cycle']": 2.75, "['Walk', 'Public transport', 'Cycle']": 2.0, "['Car/motor vehicle', 'Public transport', 'Cycle']": 3.25, "['Car/motor vehicle', 'Walk', 'Public transport', 'Cycle']": 2.0, "['Walk', 'Cycle']": 1.5, "['Cycle']": 1.0, "['None of the above']": None, "['Public transport', 'Cycle']": 2.5, "['Prefer not to answer']": None, '1-3 hours': 12.0, '3-5 hours': 4.0, '5-7 hours': 6.0, 'Under 1 hour': 0.5, '7-9 hours': 8.0, '9-12 hours': 10.5, '12+ hours': 12.0, 1.0: 1.0, 2.0: 2.0, 3.0: 3.0, 4.0: 4.0, 5.0: 5.0, '0': 0.0, '6': 6.0, '10': 10.0, '8': 8.0, '20': 20.0, '50': 50.0, '12': 12.0, '40': 40.0, '15': 15.0, '7': 7.0, '25': 25.0, '9': 9.0, '30': 30.0, '13': 13.0, '14': 14.0, '100': 100.0, '11': 11.0, '16': 16.0, '60': 60.0, '18': 18.0, '500': 500.0, '99': 99.0, '34': 34.0, '45': 45.0, '35': 35.0, '999': 999.0, '55': 55.0, '75': 75.0, '80': 80.0, '200': 200.0, '19': 19.0, '90': 90.0, '897': 897.0, '95': 95.0, '150': 150.0, '22': 22.0, '110': 110.0, '42': 42.0, '24': 24.0, '600': 600.0, '300': 300.0, '21': 21.0, '33': 33.0, '36': 36.0, '27': 27.0, '26': 26.0, '400': 400.0, '28': 28.0, '32': 32.0, '96': 96.0, 'Never tan, only burn': 0.0, 'Get moderately tanned': 2.0, 'Get very tanned': 3.0, 'Get mildly or occasionally tanned': 1.0, 'nan': None, 'About your age': 1.0, 'Younger than you are': 0.0, 'Older than you are': 2.0, 'Less than once a year': 0.5, '52': 52.0, '104': 104.0, '48': 48.0, '70': 70.0, '72': 72.0, '29': 29.0, '175': 175.0, '120': 120.0, '65': 65.0, '38': 38.0, '66': 66.0, '108': 108.0, '23': 23.0, '46': 46.0, '54': 54.0, '240': 240.0, '105': 105.0, '320': 320.0, '85': 85.0, '180': 180.0, '250': 250.0, '170': 170.0, '82': 82.0, '365': 365.0, '53': 53.0, '900': 900.0, '162': 162.0, '350': 350.0, '43': 43.0, '102': 102.0, '47': 47.0, '44': 44.0, '56': 56.0, '360': 360.0, '114': 114.0, '160': 160.0, '156': 156.0, '130': 130.0, '190': 190.0, '84': 84.0, '59': 59.0, 'Very fair': 1.0, 'Brown': 1.5, 'Dark olive': 2.0, 'Fair': 0.5, 'Black': 2.5, 'Light olive': 1.25, 'Most of the time': 2.0, 'Do not go out in sunshine': 0.0}
file_count = 4

i=0
while i <30 :
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


file_count += 1
del df
df = pd.DataFrame()
# only keep essential columns for df_codebook
# df_codebook[params.codebook_basic_columns].to_csv(params.codebook_path/'UKB_var_select.csv',index=False)
# print(replace_dict_basics)
df_codebook['preprocessed_flag'].value_counts()
print(f'left to code = {df_codebook["preprocessed_flag"].isnull().sum()}')

