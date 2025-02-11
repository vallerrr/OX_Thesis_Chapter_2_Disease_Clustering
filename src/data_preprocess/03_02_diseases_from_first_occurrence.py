"""
# Created by valler at 23/10/2024
Feature:

"""
import pandas as pd
from src import params
from src.data_preprocess import utils
import warnings
warnings.filterwarnings('ignore')

def marking_phecode_chronic_chapter(df,df_phemap,df_phe_db):
    """
    Marking the phecode, chronic and chapter for the diseases in df
    :param df:
    :param df_phemap:
    :param df_phe_db:
    :return:
    """
    df['phecode'] = [df_phemap.loc[df_phemap['ICD10'] == x,'phecode'].values[0] if x in df_phemap['ICD10'].tolist() else None for x in df['ICD10']]
    df['chronic'] = [df_phemap.loc[df_phemap['ICD10'] == x,'chronic'].values[0] if x in df_phemap['ICD10'].tolist() else None for x in df['ICD10']]
    df['chapter'] = [df_phe_db.loc[df_phe_db['phecode'] == x,'category'].values[0] if x in df_phe_db['phecode'].tolist() else None for x in df['phecode']]
    return df


# 1. read files
df_recorder = pd.read_csv(params.data_path.parent / 'downloaded_data/recorder/recorder_participant.csv')
df_phe_db,df_phemap = utils.read_phe_database()

# 2. mark the chronic diseases in the df_recorder

df_first_occ = df_recorder.loc[df_recorder['filed_title'].str.contains('first reported'),] # 1130 rows
df_first_occ['description'] = df_first_occ['filed_title'].str.split('reported').str[1].replace(' (','').replace(')','')
df_first_occ['description']=[str(x).replace(' (','').replace(')','') for x in df_first_occ['description']]
df_first_occ['ICD10'] = df_first_occ['filed_title'].str.extract(r'Date (\w+) first reported ')
df_first_occ = marking_phecode_chronic_chapter(df_first_occ,df_phemap,df_phe_db)

# manually mark some chronic diseases
chronic_phecodes_manual = [296.2] #depression
df_first_occ.loc[df_first_occ['phecode'].isin(chronic_phecodes_manual),'chronic'] = 1


# remove non-chronic diseases
chronic_threshold=params.chronic_threshold
df_first_occ = df_first_occ.loc[df_first_occ['chronic']>0,]  # 582 ICD codes,376 Phecodes
df_first_occ = df_first_occ.loc[df_first_occ['chronic']>chronic_threshold,]  # 319 ICD codes, 239 Phecodes


# append back source report of the diseases
df_first_occ_all = pd.DataFrame()
for disease in df_first_occ['ICD10'].tolist():
    temp = df_recorder.loc[df_recorder['filed_title'].str.contains(f' {disease} '),]
    df_first_occ_all = pd.concat([df_first_occ_all,temp],axis=0)

df_first_occ_all['ICD10'] = df_first_occ_all['filed_title'].str.extract(r' ([A-Z]\d\d) ')
df_first_occ_all = marking_phecode_chronic_chapter(df_first_occ_all,df_phemap,df_phe_db)


# 3. read data based on the recorders and merge them
df_first_occ_data = pd.DataFrame()
for ind,group in df_first_occ_all.groupby('ind'):

    temp = pd.read_csv(f'/Volumes/Valler/Python/OX_Thesis/Chapter_2_Disease_Clustering/Data/downloaded_data/participant/{round(ind)}.csv', low_memory=False)
    cols = group['field_name'].tolist()
    temp = temp[['eid']+cols]
    if len(df_first_occ_data)==0:
        df_first_occ_data = temp
    else:
        df_first_occ_data = df_first_occ_data.merge(temp, how='outer', on='eid')

# df_first_occ_data.rename(columns = {"p130894":"date_F32","p130895":"F32"},inplace=True)
# replace the column name with ICD code
df_first_occ_all['date'] = ['date_' if 'Date' in x else '' for x in df_first_occ_all['filed_title']]
col_dict = {y:f"{x}{z}" for x,y,z in zip(df_first_occ_all['date'],df_first_occ_all['field_name'],df_first_occ_all['ICD10'])}
df_first_occ_data.rename(columns=col_dict, inplace=True)
df_first_occ_data.to_csv(params.final_data_path.parent/ 'intermediate_files/UKB_wave_0_diseases_first_occurrence.csv',index=False)

# replace values
df_first_occ_data = pd.read_csv(params.final_data_path.parent/ 'intermediate_files/UKB_wave_0_diseases_first_occurrence.csv')
unique_values = set(df_first_occ_data['D86'].unique().tolist()+df_first_occ_data['J45'].unique().tolist())
df_first_occ_data.replace({x : 1 for x in unique_values if pd.notnull(x)}, inplace=True)


# 4. compact format

df_first_occ_data_compact = pd.DataFrame()

disease_columns = [col for col in df_first_occ_data.columns if not col.startswith('date_') and col != 'eid']

# Define a function to combine diseases where value == 1
def collapse_diseases(row):
    # Filter out the diseases where the person has the disease (value == 1)
    diseases = [disease for disease in disease_columns if row[disease] == 1]
    # Join the disease codes with a comma
    return ','.join(diseases) if diseases else None

# Apply the function to each row and create the 'diseases' column
df_first_occ_data['diseases'] = df_first_occ_data.apply(collapse_diseases, axis=1)

# Drop the original disease columns if no longer needed
df_cleaned = df_first_occ_data[['eid', 'diseases']]  # Keeping only 'eid' and the new 'diseases' column
# df_cleaned['icd_count'] = [x.count(',')+1 if str(x) not in params.nan_str else 0 for x in df_cleaned['diseases']]
df_cleaned['phecode'] = [[df_first_occ.loc[df_first_occ['ICD10']==icd,'phecode'].values[0] for icd in x.split(',')] if pd.notnull(x) else None for x in df_cleaned['diseases']]
df_cleaned['phe_count'] = [len(x) if str(x) not in params.nan_str else 0 for x in df_cleaned['phecode']]

df_cleaned['chapter'] = [[df_phe_db.loc[df_phe_db['phecode']==phe,'category_number'].values[0] for phe in x] if str(x) not in params.nan_str else None for x in df_cleaned['phecode']]
df_cleaned.to_csv(params.final_data_path/ 'UKB_wave_0_diseases_first_occurrence.csv',index=False)



# dates
dates_col = [col for col in df_first_occ_data.columns if col.startswith('date_')]
df_first_occ_data[dates_col] = df_first_occ_data[dates_col].apply(pd.to_datetime, errors='coerce')
temp = df_first_occ_data[dates_col].copy()
min([temp[col].min() for col in dates_col]) # 1938-04-01 to 2023-09-01

# within window tag
df = pd.read_pickle(f'{str(params.final_data_path ).replace("src/data_preprocess/","")}/UKB_wave_0_final_non_standardised.pkl')
df = df[['eid','53','31','21022']]
df = df.merge(df_first_occ_data, how='left', on='eid')
df['53'] = df['53'].apply(pd.to_datetime, errors='coerce')

df.to_pickle(params.final_data_path / 'UKB_wave_0_diseases_first_occurrence_with_all_dates.pkl')


# if we want to change the date interval, we only need to read the file and run the following code
# params.final_data_path / 'UKB_wave_0_diseases_first_occurrence_with_all_dates.pkl'
df = pd.read_pickle(params.final_data_path / 'UKB_wave_0_diseases_first_occurrence_with_all_dates.pkl')
dates_col = [col for col in df.columns if col.startswith('date_')]
def mark_within_window(row,upper_year_interval=0,lower_year_interval=0):
    """
    Mark the diseases within the window specified by the interval (i.e. the date of the disease is within the interval),
    threshold = date of occurrence+interval
    :param row: operation is done by row
    :param lower_year_interval: years after the date of occurrence
    :param upper_year_interval: years after the date of occurrence
    :return: return the diseases within the window as a list
    """
    codes_within_window = []
    for col in dates_col:

        icd_code = col.split('_')[1]
        threshold_date_lower = row['53'] - pd.Timedelta(days=lower_year_interval*365)
        threshold_date_upper = row['53'] + pd.Timedelta(days=upper_year_interval*365)

        if pd.notnull(row[col]):
            if (threshold_date_lower < row[col]) and (row[col] <= threshold_date_upper):
                codes_within_window.append(icd_code)

    if len(codes_within_window)>0:
        return codes_within_window
    else:
        return None

df['diseases_within_window_pre_recruit'] = df.apply(mark_within_window, axis=1,upper_year_interval=0,lower_year_interval=100)
df['diseases_within_window_0_5_recruit'] = df.apply(mark_within_window, axis=1,upper_year_interval=5,lower_year_interval=0)
df['diseases_within_window_5_10_recruit'] = df.apply(mark_within_window, axis=1,upper_year_interval=10,lower_year_interval=-5)
df['diseases_within_window_10_after_recruit'] = df.apply(mark_within_window, axis=1,upper_year_interval=20,lower_year_interval=-10)

df.to_pickle(params.final_data_path / 'UKB_wave_0_diseases_first_occurrence_with_all_dates.pkl')

# only save the necessary columns
window_cols = ['diseases_within_window_pre_recruit','diseases_within_window_0_5_recruit','diseases_within_window_5_10_recruit','diseases_within_window_10_after_recruit']

df_cleaned = pd.read_csv(params.final_data_path/ 'UKB_wave_0_diseases_first_occurrence.csv')
df_cleaned = df_cleaned[['eid', 'diseases', 'phecode', 'phe_count', 'chapter']]
df_cleaned = df_cleaned.merge(df[['eid','31','21022']+window_cols], how='left', on='eid')

for col in [x.replace('diseases_within_window','') for x in window_cols]:

    df_cleaned[f'diseases_within_window_phecode{col}'] = [[df_phemap.loc[df_phemap['ICD10'] == icd, 'phecode'].values[0] for icd in x] if str(x) not in params.nan_str else None for x in df_cleaned[f'diseases_within_window{col}']]
    df_cleaned[f'chapter_within_window{col}'] = [[df_phe_db.loc[df_phe_db['phecode'] == phe, 'category_number'].values[0] for phe in x] if str(x) not in params.nan_str else None for x in df_cleaned[f'diseases_within_window_phecode{col}']]
    df_cleaned[f'phe_count_within_window{col}'] = [len(x) if str(x) not in params.nan_str else 0 for x in df_cleaned[f'diseases_within_window{col}']]

df_cleaned.to_csv(params.final_data_path / 'UKB_wave_0_diseases_first_occurrence.csv', index=False)


