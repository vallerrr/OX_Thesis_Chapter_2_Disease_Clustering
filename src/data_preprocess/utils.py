"""
# Created by valler at 19/03/2024
Feature:

"""
import pandas as pd
from src import params
import ast
import os
import numpy as np
from datetime import datetime
from dateutil.parser import parse
from src import params
import ast

def data_reader(row, instance=4):
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
        if 'eid' not in cols:
            cols = ['eid'] + cols
        temp = pd.read_csv(f'/Volumes/Valler/Python/OX_Thesis/Chapter_2_Disease_Clustering/Data/downloaded_data/participant/{file_name}.csv', low_memory=False)
        temp = temp.loc[:, cols]
        # concat temp and df based on the eid column
        if len(df) == 0:
            df = temp
        else:
            df = pd.merge(df, temp, on='eid')
    return df

def average(df):
    '''
    average by row, ignoring NAs
    :param df:
    :return:
    '''
    return df.mean(skipna=True, axis=1)

def recoding_process_main_for_final_data_generator(ind, row, df_codebook, df, instance, cate_name, file_count, replace_dict_basics):

    df_read = data_reader(row, instance)
    null_dict = df_read.drop(columns=['eid']).isnull().sum().to_dict()
    df_codebook.loc[ind, 'missing_info_general'] = str(null_dict)
    temp = pd.DataFrame(df_read['eid'])

    # df_read=df_read.drop(columns=['eid'])
    # preprocessing code

    # step 0: generate recode type dict
    try:
        recode_type = ast.literal_eval(row.recode_type)
    except:
        raise ValueError(f"recode_type is not a dict for {row.field_name}")

    # step 1: recode check
    if recode_type['recode'] == 'y':
        replace_dict = ast.literal_eval(row['replace_dict'].replace('nan: None,', '').replace('nan:None,', '').replace(', nan: None', '').replace('nan: None', ''))
        df_read.replace(to_replace=replace_dict.keys(), value=replace_dict.values(), inplace=True)
        for column in df_read.columns:
            if not column == 'eid':
                df_read[column] = pd.to_numeric(df_read[column], errors='ignore')
        # if update_flag == any other character, it will ignore the command
    elif recode_type['recode'] == 't':  # timestamp type
        df_read.apply(lambda col: try_to_datetime(col))
    elif recode_type['recode'] == True:
        replace_dict = ast.literal_eval(row['replace_dict'].replace('nan: None,','').replace('nan:None,','').replace(', nan: None','').replace('nan: None',''))
        df_read.replace(to_replace=replace_dict.keys(), value=replace_dict.values(), inplace=True)
        for column in df_read.columns:
            if not column == 'eid':
                df_read[column] = pd.to_numeric(df_read[column], errors='ignore')
    elif recode_type['recode'] == False:
        pass
    else:
        raise ValueError(f"recode type is not recognized for {row.field_name}")

    #  step 2: array check
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
        columns = [x for x in df_read.columns if ('a' not in x) and ('eid' not in x)]
        temp.loc[:,f'{row.field_id}'] = average(df_read[columns])

    elif recode_type['i'] == 'this_wave':
        field_name = f'p{row.field_id}'if pd.isnull(row.instance) else f'p{row.field_id}_i{ max(row.instance.split(";") if isinstance(row.instance,str) else row.instance)}'
        temp.loc[:, f'{row.field_id}'] = df_read[field_name]
    elif recode_type['i'].startswith('w'):  # e.g. w1
        ins = int(recode_type['i'].replace('w',''))
        temp.loc[:, f'{row.field_id}'] = df_read[f'p{row.field_id}_i{ins}']
    elif recode_type['i'] == 'single_wave':
        print('this is a single wave variable')
        field_name = f'p{row.field_id}'if pd.isnull(row.instance) else f'p{row.field_id}_i{row.instance}'
        temp.loc[:, f'{row.field_id}'] = df_read[field_name]
    else:
        print('key in unknown wave coding information')


    if len(df) == 0:
        df = temp.copy()
    else:
        df = pd.merge(df, temp, on='eid')

    df_codebook.loc[ind, f'preprocessed_flag_wave{instance}'] = 1
    df_codebook.loc[ind, 'recode_type'] = str(recode_type)

    missing_count = df[f'{row.field_id}'].isnull().sum()
    df_codebook.loc[ind, 'missing_count'] = missing_count
    print(f"missing  count = {missing_count}")

    file_name = f'UKB_wave_{instance}_{cate_name}_{file_count}.csv'
    df_codebook.loc[ind, 'file_name'] = file_name
    df.to_csv(params.preprocessed_path /file_name, index=False)
    df_codebook.to_csv(params.codebook_path / f'UKB_preprocess_codebook_wave_{instance}.csv', index=False)
    del temp
    return df, df_codebook

    #  {'average_i':'average across instance','this_wave':'do not consider cross wave difference'}

def try_to_datetime(col):
    try:
        return pd.to_datetime(col)
    except ValueError:
        return col

def array_name_by_ins(row, instance):
    if instance == '':
        return [f'p{row.field_id}_a{i}' for i in row['array'].split(';')]
    else:
        return [f'p{row.field_id}_i{instance}_a{i}' for i in row['array'].split(';')]

def average(df):
    '''
    average by row, ignoring NAs
    :param df:
    :return:
    '''
    return df.mean(skipna=True, axis=1)

def replace_recode_main(row, df_read,df_codebook,ind):
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
        replace_dict, df_read = replacing_recode(df_read, params.replace_dict_basics)
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

def create_single_record_df(record_column,HES_ICD_ids):
    """
    create a single record df for a specific record column
    :param record_column: one of ['all_icd','main_icd','second_icd']
    :param HES_ICD_ids: {"all_icd": {"id": '41270', "time": '41280'}, "main_icd": {"id": '41202', "time": '41262'}, "second_icd": {"id": '41203', "time": None}}

    :return: cleaned diseases record df with the eid, the recoded diseases, the unique count of diseases, and the first 3 characters of the diseases and date
    """
    df_codebook = pd.read_csv(params.codebook_path / 'UKB_preprocess_codebook_wave_0.csv')
    # read id
    df_single_record = data_reader(df_codebook.loc[df_codebook['field_id'] == int(HES_ICD_ids[record_column]['id']),].iloc[0])

    df_single_record[record_column] = df_single_record[f"p{HES_ICD_ids[record_column]['id']}"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else None)
    df_single_record[record_column+'_uniq_count'] = df_single_record[record_column].apply(lambda x: len(set(x)) if str(x)!='None' else 0)
    df_single_record[record_column+'_first_3'] = df_single_record[record_column].apply(lambda x: [ele[:3] for ele in x] if str(x)!='None' else None)
    df_single_record.drop(columns=[f"p{HES_ICD_ids[record_column]['id']}"], inplace=True)
    # read time
    df_single_record = pd.merge(left=df_single_record, right=data_reader(df_codebook.loc[df_codebook['field_id'] == int(HES_ICD_ids[record_column]['time']),].iloc[0]), left_on='eid', right_on='eid')

    # reorder the diseases based on their corresponding date
    # retrieve gender and age on the df_single_record
    df_read = pd.read_csv(params.preprocessed_path / 'UKB_wave_0_Socio-demographics_0.csv')
    columns_to_remain = ['eid', '21022', '31']
    df_read = df_read[columns_to_remain]
    df_single_record = pd.merge(df_read, df_single_record, on='eid', how='left')

    temp = df_single_record.copy()
    dates_col = [f'p{HES_ICD_ids[record_column]["time"]}_a{x:03d}' for x in range(0, int(df_single_record[f'{record_column}_uniq_count'].max()))]
    dates_dict = {f'p{HES_ICD_ids[record_column]["time"]}_a{x}': f'p{HES_ICD_ids[record_column]["time"]}_a{x:03d}' for x in range(0, int(df_single_record[f'{record_column}_uniq_count'].max()))}
    temp.rename(columns=dates_dict, inplace=True)

    # sort the date columns
    temp = temp[columns_to_remain+ [f'{record_column}', f'{record_column}_uniq_count', f'{record_column}_first_3']+dates_col]

    # reorder the diseases based on their corresponding date
    dates_df = temp[dates_col].copy()
    sorted_indices = dates_df.apply(np.argsort, axis=1)

    def sort_by_indices(list_, index):
        if str(list_) == 'None':
            return None
        else:
            indices = sorted_indices.loc[index].values[:len(list_)]

            sorted_list = [list_[i]  for i in indices]

            return sorted_list

    # Apply the function to the 'diseases' column
    temp[f'{record_column}_first_3'] = temp.apply(lambda row: sort_by_indices(row[f'{record_column}_first_3'], row.name) if str(row) != 'None' else None, axis=1)

    # Apply the function fo the dates columns
    dates_sorted = temp.apply(lambda row: sort_by_indices(row[dates_col], row.name), axis=1)
    dataframes = [pd.DataFrame(dates_sorted[i]).T for i in range(len(dates_sorted))]
    m = pd.concat(dataframes)
    m.columns = dates_col
    m.reset_index(drop=True, inplace=True)
    temp[dates_col] = m[dates_col]

    df_single_record = temp.copy()
    # df_single_record.to_csv(intermediate_path / f'{record_column}_complete.csv', index=False)

    return df_single_record

def import_df_single_record(record_column,):
    """
    import the single record df
    :param record_column:
    :return:
    """
    df_single_record = pd.read_csv(params.intermediate_path / f'{record_column}_complete.csv')

    dates_col = [f'p{params.HES_ICD_ids[record_column]["time"]}_a{x:03d}' for x in range(0, int(df_single_record[f'{record_column}_uniq_count'].max()))]

    # convert the dates to datetime (following chunk of code should always be run)
    for col in dates_col:
        df_single_record[col] = pd.to_datetime(df_single_record[col], errors='coerce', format='%Y-%m-%d')
    # convert list type columns to list
    list_columns = [f'{record_column}_first_3',f'{record_column}_icd_codes', 'icd_parent_coding', 'icd_chapter_coding','phecode','diseases_within_window_phecode_selected','diseases_within_window_phecode_selected_category']
    for column in list_columns:
        if column in df_single_record.columns.tolist():
            df_single_record[column] = df_single_record[column].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else None)
    return dates_col,df_single_record


def map_icd_to_phecode(icd_codes,df_phemap):
    """
    map the icd codes to phecode and keep the original *count* of the icd codes

    """
    if str(icd_codes) == 'None':
        return None
    else:
        phecode = []
        for icd_code in icd_codes:
            if icd_code in df_phemap.ICD10.tolist():
                rows = df_phemap.loc[df_phemap['ICD10'] == icd_code, 'PHECODE']
                if len(rows) > 1:
                    phecode.append(rows.values.tolist())
                else:
                    phecode.append(rows.values[0])
            else:
                # if the icd code is not in the phemap, then we record it as its original but put tails to recognise them
                phecode.append(f'{icd_code}.4444')
        if len(icd_codes) == len(phecode):
            return phecode
        else:
            print(icd_codes, phecode)
            return None


def clean_unmatched_ICD(row):
    """
    clean the unmatched ICD codes (ends with 4444 in the column), this function can take care of the list of list situation
    :param row: should be not empty
    :return:
    """
    if str(row).count('[')==1:
        updated_row = [None if str(x).endswith('4444') else float(x) for x in row]
    else:
        updated_row=[]
        for ele in row:
            if '[' not in str(ele):
                updated_row.append(None if str(ele).endswith('4444') else ele)
            else:
                updated_row_ele = [None if str(x).endswith('4444') else float(x) for x in ele]
                updated_row.append(updated_row_ele)
    return updated_row


def return_chronoic_code(val,df_phemap,threshold=0.5):
    """
    only returns the phecode that is chronoic, otherwise None
    :param val:
    :param df_phemap:
    :return:
    """

    if '[' in str(val):
        # if those diseases are from one date, executing the function in the list
        # if there is only one condition within the list is chronic, unfold the list
        val_list = []
        for v in val:
            v = float(v)
            if df_phemap.loc[df_phemap['phecode'] == v,'chronic'].values[0]>=threshold:
                val_list.append(v)
            # we don't need to add the non-chronic condition inside the val_list as they are from the same date
            # we tackle with the situation of only 1/0 disease are chronic as below

        if len(val_list)<=1:
            if len(val_list)==1:
                return val_list[0]
            else:
                return None
        else:
            return val_list

    else:
        if pd.notnull(val):
            val = float(val)
            if df_phemap.loc[df_phemap['phecode'] == val,'chronic'].values[0]>=threshold:
                return val
            else:
                return None


def marking_phecode_chronic_chapter(df,df_phemap,df_phe_db):
    """
    Marking the phecode, chronic and chapter for the diseases in df
    :param df:
    :param df_phemap:
    :param df_phe_db:
    :return:
    """
    df['phecode'] = [df_phemap.loc[df_phemap['ICD10'] == x,'PHECODE'].values[0] if x in df_phemap['ICD10'].tolist() else None for x in df['ICD10']]
    df['chronic'] = [df_phemap.loc[df_phemap['ICD10'] == x,'chronic'].values[0] if x in df_phemap['ICD10'].tolist() else None for x in df['ICD10']]
    df['chapter'] = [df_phe_db.loc[df_phe_db['phecode'] == x,'category'].values[0] if x in df_phe_db['phecode'].tolist() else None for x in df['phecode']]
    return df

def read_phe_database():


    df_phe_db = pd.read_csv(params.ICD_path / 'phecode_definitions1.2.csv')  # 1866 Phecodes
    df_phemap = pd.read_csv(params.ICD_path / 'Phecode_map_v1_2_icd10_beta.csv')  # 9066 ICD codes, 1571 Phecodes

    # clean the df_ccir
    #df_ccir = pd.read_csv(params.ICD_path / 'CCIR_v2024-1/CCIR-v2024-1.csv', header=2)  #
    #df_ccir.rename(columns={x: x.replace("'", '').replace(" ", "_").replace("-", "_") for x in df_ccir.columns}, inplace=True)
    #df_ccir['ICD_10_CM_CODE'] = [str(x).replace("'", '') for x in df_ccir['ICD_10_CM_CODE']]

    #df_phemap_CM = pd.read_csv(params.ICD_path / 'Phecode_map_v1_2_icd10cm_beta.csv')
    #df_phemap_CM['ICD10_harmonised'] = [str(x).replace(".", "") for x in df_phemap_CM['icd10cm']]
    #df_phemap_CM = pd.merge(left=df_ccir[['ICD_10_CM_CODE', 'CHRONIC_INDICATOR']], right=df_phemap_CM, left_on='ICD_10_CM_CODE', right_on='ICD10_harmonised', how='right')  #
    #df_phemap_CM['CHRONIC_INDICATOR'].replace({9: 0}, inplace=True)
    # record back to the df_phe file
    #df_phe_db['chronic'] = [df_phemap_CM.loc[df_phemap_CM['phecode'] == float(p), 'CHRONIC_INDICATOR'].sum() / len(df_phemap_CM.loc[df_phemap_CM['phecode'] == float(p), 'CHRONIC_INDICATOR'].notnull()) if p in df_phemap_CM['phecode'].tolist() else None for p in df_phe_db['phecode']]
    #df_phemap['chronic'] = [df_phemap_CM.loc[df_phemap_CM['phecode'] == float(p), 'CHRONIC_INDICATOR'].sum() / len(df_phemap_CM.loc[df_phemap_CM['phecode'] == float(p), 'CHRONIC_INDICATOR'].notnull()) if p in df_phemap_CM['phecode'].tolist() else None for p in df_phemap['PHECODE']]
    # df_phemap.to_csv(params.ICD_path/ 'Phecode_map_v1_2_icd10_beta.csv',index=False)
    # df_phe_db.to_csv(params.ICD_path/ 'phecode_definitions1.2.csv',index=False)

    df_phemap.rename(columns={'PHECODE': 'phecode', 'ICD10': 'ICD10'}, inplace=True)
    df_phe_db.loc[df_phe_db['phecode'] == 296.20000, 'chronic'] = 1  # depression
    df_phe_db.loc[df_phe_db['phecode'] == 296.20000, 'category'] = 'Mental disorders'
    return df_phe_db,df_phemap



def marking_phecode_chronic_chapter(df,df_phemap,df_phe_db,disease_column):
    """
    Marking the phecode, chronic and chapter for the diseases in df
    :param df:
    :param df_phemap:
    :param df_phe_db:
    :return:
    """
    df[f'{disease_column}_phecode'] = [df_phemap.loc[df_phemap['ICD10'] == x,'phecode'].values[0] if x in df_phemap['ICD10'].tolist() else None for x in df[disease_column]]
    df[f'{disease_column}_chronic'] = [df_phemap.loc[df_phemap['ICD10'] == x,'chronic'].values[0] if x in df_phemap['ICD10'].tolist() else None for x in df[disease_column]]
    df['chapter'] = [df_phe_db.loc[df_phe_db['phecode'] == x,'category'].values[0] if x in df_phe_db['phecode'].tolist() else None for x in df['phecode']]
    return df
