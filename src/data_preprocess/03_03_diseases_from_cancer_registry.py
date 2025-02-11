"""
# Created by valler at 29/10/2024
Feature: 

"""
import numpy as np
import pandas as pd
from src import params
from src.data_preprocess import utils


# 1. read files
df_recorder = pd.read_csv(params.data_path.parent / 'downloaded_data/recorder/recorder_participant.csv')
df_phe_db, df_phemap = utils.read_phe_database()

# 2. get the disease list
# 40006: cancer ICD codes
# 40005: cancer date
df_recorder_cancer = df_recorder[(df_recorder['field_name'].str.contains('40006'))|(df_recorder['field_name'].str.contains('40005'))]

# 3. read data based on the recorder
df_cancer_data = pd.DataFrame()
for ind,group in df_recorder_cancer.groupby('ind'):

    temp = pd.read_csv(f'/Volumes/Valler/Python/OX_Thesis/Chapter_2_Disease_Clustering/Data/downloaded_data/participant/{round(ind)}.csv', low_memory=False)
    cols = group['field_name'].tolist()
    temp = temp[['eid']+cols]
    if len(df_cancer_data)==0:
        df_cancer_data = temp
    else:
        df_cancer_data = df_cancer_data.merge(temp, how='outer', on='eid')

disease_columns = [x for x in df_cancer_data.columns if '40006' in x]
df_cancer_data['ICD10'] = df_cancer_data[disease_columns].apply(lambda x: [x[0:6].split(' ')[0] for x in x if str(x) not in params.nan_str], axis=1)
df_cancer_data['ICD10'] = [x if len(x)>0 else None for x in df_cancer_data['ICD10']]
# mark phecodes for each ICD10, order is kept even if there are diseaeses not matching the phemap file,
# 24 ICD10 codes are not in the phemap but under 100 prevalence
df_cancer_data['phecodes'] = [[df_phemap.loc[df_phemap['ICD10'] == disease, 'phecode'].values[0] if disease in df_phemap['ICD10'].tolist() else None for disease in x ]  if str(x) not in params.nan_str else None for x in df_cancer_data['ICD10'] ]

phe_db = [df_cancer_data[x].unique() for x in disease_columns]
phe_db = [item[0:6].split(' ')[0] if str(item) not in params.nan_str else '' for sublist in phe_db for item in sublist] #1881

while '' in phe_db:
    phe_db.remove('')
phe_db = list(set(phe_db)) # 585

# mark chronic conditions
chronic_threshold = params.chronic_threshold
chronic_phecodes = [df_phemap.loc[df_phemap['ICD10']==x,'phecode'].values[0] for x in set(df_phemap[df_phemap['chronic']>chronic_threshold]['ICD10'].tolist()).intersection(set(phe_db))]
df_cancer_data['phecodes_chronic'] = [[disease if disease in chronic_phecodes else None for disease in x]  if str(x) not in params.nan_str else None for x in df_cancer_data['phecodes'] ]

# mark within window or not
dates_col = [col for col in df_cancer_data.columns if col.startswith('p40005')]
df_cancer_data[dates_col] = df_cancer_data[dates_col].apply(pd.to_datetime, errors='coerce')
df_cancer_data['phecode_count'] = [len(x) if str(x) not in params.nan_str else 0 for x in df_cancer_data['phecodes']]
df_cancer_data['date_count'] = df_cancer_data[dates_col].apply(lambda x: sum(~x.isnull()), axis=1)
df_cancer_data['date_diff'] = df_cancer_data['date_count'] - df_cancer_data['phecode_count']

df_cancer_data['date_diff'].value_counts()
# within window tag
df = pd.read_pickle(f'{str(params.final_data_path ).replace("src/data_preprocess/","")}/UKB_wave_0_final_non_standardised.pkl')
df = df[['eid','53','31','21022']]
df = df.merge(df_cancer_data, how='left', on='eid')
df['53'] = df['53'].apply(pd.to_datetime, errors='coerce')

df_cancer_error_record = pd.DataFrame(columns = ['eid','extra_dates','extra_dates_index'])

def mark_within_window(df, df_cancer_error_record, dates_col,disease_col = 'phecodes_chronic',upper_year_interval=0,lower_year_interval=0):

    codes_within_window_list = []
    error_records = []

    # Iterating over the DataFrame rows using iterrows() for better readability
    for idx, row in df.iterrows():
        codes_within_window = []
        phecodes_chronic = row[disease_col]

        # Skip rows with missing 'phecodes_chronic'
        if str(phecodes_chronic) in params.nan_str:
            codes_within_window_list.append(None)
            continue
        # print(phecodes_chronic)
        # Iterate over the date columns with orders

        for date_index, date_col in enumerate(dates_col):
            if pd.notnull(row[date_col]):

                # If the date index exceeds the length of phecodes_chronic, record the error
                threshold_date_upper = row['53'] + pd.Timedelta(days=upper_year_interval*365)
                threshold_date_lower = row['53'] - pd.Timedelta(days=lower_year_interval*365)

                # print(f" recruit time {row['53']}, current date {row[date_col]}, threshold_lower = {threshold_date_lower}, threshold_upper = {threshold_date_upper}")

                if date_index >= len(phecodes_chronic):
                    if (threshold_date_lower < row[date_col]) and (row[date_col] <= threshold_date_upper):
                        #if (row[date_col] <= row['53']) and :
                        # Dates are extra but before the window date
                        error_records.append({
                            'eid': row['eid'],
                            'extra_date': row[date_col],
                            'extra_dates_index': date_index
                        })
                else:
                    phe_code = phecodes_chronic[date_index]

                    if (threshold_date_lower < row[date_col]) and (row[date_col] <= threshold_date_upper):
                        print(phe_code)
                        codes_within_window.append(phe_code)

        # Append the codes within the window or None if empty
        if codes_within_window:
            codes_within_window_list.append(codes_within_window)
        else:
            codes_within_window_list.append(None)

    # Assign the new list to a column in the original DataFrame
    df[f'diseases_within_window_{lower_year_interval}_{upper_year_interval}'] = codes_within_window_list

    # Convert the error records list to a DataFrame and append to df_cancer_error_record
    if error_records:
        df_cancer_error_record = pd.concat([df_cancer_error_record, pd.DataFrame(error_records)], ignore_index=True)

    return df, df_cancer_error_record


# note that we don't have the first occurrence operation in the cancer data, which means that the same disease may appear in different columns
# we choose to ignore it as we will concatenate the diseases from different dates together later in the fo+cancer data

df_new, df_cancer_error_record = mark_within_window(df, df_cancer_error_record, dates_col, disease_col='phecodes_chronic',upper_year_interval=0,lower_year_interval=100)
df_new, df_cancer_error_record = mark_within_window(df_new, df_cancer_error_record, dates_col, disease_col='phecodes_chronic',upper_year_interval=5,lower_year_interval=0)
df_new, df_cancer_error_record = mark_within_window(df_new, df_cancer_error_record, dates_col, disease_col='phecodes_chronic',upper_year_interval=10,lower_year_interval=-5)
df_new, df_cancer_error_record = mark_within_window(df_new, df_cancer_error_record, dates_col, disease_col='phecodes_chronic',upper_year_interval=20,lower_year_interval=-10)

# df_new.rename(columns={'diseases_within_window_0_0':'diseases_within_window_100_0'}, inplace=True)

window_cols = ['diseases_within_window_100_0','diseases_within_window_0_5','diseases_within_window_-5_10','diseases_within_window_-10_20']
df_new_copy = df_new.copy()
# remove Nones from each list and only keep unique values
for col in window_cols:
    df_new[col] = df_new[col].apply(lambda x: None if x is None or all(item is None for item in x) else list(set(item for item in x if item is not None)))

df_cancer_data = df_cancer_data.merge(df_new[['eid', '31', '21022']+window_cols], how='left', on='eid')

#df_new['diseases_within_window'] = df_new['diseases_within_window'].apply(lambda x: None if x is None or all(item is None for item in x) else list(set(item for item in x if item is not None)))
del df,df_new

for col in window_cols:
    df_cancer_data[f'{col}_count'] = [len(x) if str(x) not in params.nan_str else 0 for x in df_cancer_data[col]]
    df_cancer_data[f'{col.replace("diseases","chapter")}'] = [[df_phe_db.loc[df_phe_db['phecode'] == disease, 'category_number'].values[0] for disease in x] if str(x) not in params.nan_str else None for x in df_cancer_data[col]]



# save zone
df_cancer_data.to_pickle(params.intermediate_path / 'cancer_registry_data_all.pkl')

df_cancer_error_record.to_csv(params.intermediate_path / 'cancer_registry_error_record.csv', index=False)
df_cancer_data = df_cancer_data[['eid','31','21022']+window_cols+[f'{col}_count' for col in window_cols]+[f'{col.replace("diseases","chapter")}' for col in window_cols]]
df_cancer_data.to_pickle(params.intermediate_path / 'cancer_registry_data_refined.pkl')



'''
# checking the ICD10 codes
temp = [df_cancer_data[x].unique() for x in disease_columns]
temp = [item[0:6].split(' ')[0] if str(item) not in params.nan_str else '' for sublist in temp for item in sublist] #1881
import numpy as np
while '' in temp:
    temp.remove('')

temp = list(set(temp)) # 585
check= [x for x in temp if x not in df_phemap['ICD10'].tolist() ] # 24 ICD10 codes are not in the phemap
# ['D46.5', 'C96.8', 'C86.4', 'C42.1', 'C86.6', 'C42.3', 'C42.2', 'C88.4', 'C84.8', 'C80.9', 'D47.4', 'C86.0', 'C86.2', 'C86.5', 'D43.9', 'C92.8', 'C81.4', 'C82.3', 'C84.7', 'C94.6', 'C90.3', 'C42.4', 'C80.0', 'C42.0']
[x for x in check if ('C42' not in x ) & ('C86' not in x)]
print(check)
'C43.5' in df_phemap['ICD10'].tolist()
'''



# 4. get the disease list

