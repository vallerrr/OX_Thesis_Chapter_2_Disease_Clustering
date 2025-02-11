"""
# Created by valler at 08/11/2024
Feature: 

"""
import pandas as pd
from src import params
import ast
import warnings
warnings.filterwarnings('ignore')
# =============================================================================
# 1. concating diseases from cancer and first occurence
# =============================================================================
# read fo data
df_fo = pd.read_csv(params.final_data_path/'UKB_wave_0_diseases_first_occurrence.csv')
window_cols = ['diseases_within_window_phecode_pre_recruit','diseases_within_window_phecode_0_5_recruit','diseases_within_window_phecode_5_10_recruit','diseases_within_window_phecode_10_after_recruit']
df_fo.rename(columns={x:x.replace("diseases_within_window_","").replace('5_recruit','5').replace('10_recruit','10') for x in window_cols},inplace=True)
window_cols = [x.replace("diseases_within_window_","").replace('5_recruit','5').replace('10_recruit','10') for x in window_cols]
df_fo = df_fo[['eid']+window_cols]  # we don't need count/chapter for now
for col in window_cols:
    df_fo[col] = [ast.literal_eval(x) if str(x) not in params.nan_str else None for x in df_fo[col]]

# read cancer data

df_cancer = pd.read_pickle(f'{str(params.intermediate_path).replace("src/modelling/","")}/cancer_registry_data_refined.pkl')
window_cols_cancer = ['diseases_within_window_100_0','diseases_within_window_0_5', 'diseases_within_window_-5_10','diseases_within_window_-10_20']
df_cancer = df_cancer[['eid']+window_cols_cancer]
df_cancer.rename(columns={x:x.replace("diseases_within_window_","phecode_").replace('_100_0','_pre_recruit').replace('-','').replace('_20','_after_recruit') for x in window_cols_cancer},inplace=True)
window_cols_cancer = [x.replace("diseases_within_window_","phecode_").replace('_100_0','_pre_recruit').replace('-','').replace('_20','_after_recruit') for x in window_cols_cancer]

# combine the two dataframes
df_disease = df_cancer.merge(df_fo[window_cols+['eid']], how='left', left_on='eid', right_on='eid', suffixes=('_cancer', '_fo'))
for disease_column in window_cols:
    df_disease[disease_column] = [
        list(set((x if str(x) not in params.nan_str else []) + (y if str(y) not in params.nan_str else [])))
        for x, y in zip(df_disease[disease_column + '_cancer'], df_disease[disease_column + '_fo'])]

# check whether the results are the same as last version
""" 
df = pd.read_pickle(params.final_data_path/'UKB_wave_0_diseases_first_occurrence_and_cancer.pkl')
temp = [len(set(x)-set(y))==0 for x, y in zip(df['diseases_within_window_phecode'], df_disease['phecode_pre_recruit'])]
df['temp'] = temp
del temp
temp = df.loc[df['temp']==False,['eid','diseases_within_window_phecode']] # 0 
"""

df_phe_db = pd.read_csv('/Users/valler/Python/OX_Thesis/Chapter_2_Disease_Clustering/Data/downloaded_data/ICD_10/phecode_definitions1.2.csv')
df_disease = df_disease[['eid']+window_cols]
# mark chapter
for col in window_cols:
    df_disease[f"{col.replace('phecode','chapter')}"] = [[df_phe_db.loc[df_phe_db['phecode']==disease,'category_number'].values[0] for disease in x] if str(x) not in params.nan_str else None for x in df_disease[col]]

df_disease = df_disease[['eid']+window_cols]
#df_disease=df_disease[['eid','31', '21022',disease_column,'chapter_within_window_phecode_selected']]
#df_disease.rename(columns={disease_column:'diseases_within_window_phecode',"chapter_within_window_phecode_selected":"chapter_within_window_phecode"},inplace=True)
df_disease.to_pickle(params.final_data_path/ 'UKB_wave_0_diseases_first_occurrence_and_cancer.pkl')


# =============================================================================
# 2. frequency of diseases
# =============================================================================
df_disease = pd.read_pickle(params.final_data_path/ 'UKB_wave_0_diseases_first_occurrence_and_cancer.pkl')

# 5. check the prevalence of the diseases
phe_codes_all = df_disease['diseases_within_window_phecode'].dropna().explode().explode().unique().tolist()
while None in phe_codes_all:
    phe_codes_all.remove(None)

df_phe = pd.DataFrame(columns=['phecode', 'prev'])
df_phe['phecode'] = [float(p) if not str(p).endswith('.4444') else p for p in phe_codes_all]
phe_db = df_disease['diseases_within_window_phecode'].dropna().explode().explode().dropna().tolist()
# 1604591 all_icd, 846452 main_icd (including unmatched diseases)  # 800279 with fo+cancer

df_phe['prev'] = [phe_db.count(p) for p in df_phe['phecode']]  # don't forget the comma
df_phe.dropna(inplace=True)
df_phe['category'] = [df_phe_db[df_phe_db['phecode'] == float(p)]['category'].values[0] if isinstance(p,float) else None for p in df_phe['phecode']]
df_phe['describe'] = [df_phe_db[df_phe_db['phecode'] == float(p)]['phenotype'].values[0] if isinstance(p,float) else None for p in df_phe['phecode']]
df_phe.sort_values(by='prev', ascending=False, inplace=True)
df_phe['chronic']=[df_phe_db[df_phe_db['phecode'] == float(p)]['chronic'].values[0] if isinstance(p,float) else None for p in df_phe['phecode']]
df_phe.reset_index(drop=True, inplace=True)
df_phe.to_csv(params.intermediate_path / 'first_occurrence_and_cancer_phecode_chronic.csv',index=False)

df_phe.loc[df_phe.category=='neoplasms','prev'].sum()

lst_final_phecodes = df_phe.loc[df_phe['prev']>500,'phecode'].tolist()
print(lst_final_phecodes)
# [401.1, 495.0, 476.0, 296.2, 564.1, 340.0, 411.3, 721.0, 562.1, 411.8, 351.0, 455.0, 716.2, 600.0, 250.2, 835.0, 411.2, 622.1, 280.1, 618.1, 610.8, 274.1, 496.21, 475.0, 615.0, 365.0, 433.21, 740.1, 569.0, 242.0, 345.0, 362.0, 379.2, 316.0, 555.2, 472.0, 626.8, 54.0, 642.0, 695.3, 557.0, 366.2, 596.5, 428.2, 250.1, 353.0, 567.0, 626.11, 555.1, 281.12, 296.1, 256.0, 371.1, 590.0, 335.0, 303.4, 722.9, 395.1, 362.7, 585.2, 496.3, 433.2, 300.13, 715.2, 737.3, 395.2, 697.0, 448.0, 612.2, 433.3, 696.3, 454.0, 474.2, 295.1, 275.0, 342.0]


# to get the final diseases ordered by chapter
df_phe_temp = df_phe.loc[df_phe['prev']>500].copy()
df_phe_temp.sort_values(by='category',inplace=True)
print(df_phe_temp['phecode'].tolist())

phe_temp_dict = {}
for cat in df_phe_temp['category'].unique():
    phe_temp_dict[cat] =[f'd_{x}' for x in  df_phe_temp.loc[df_phe_temp['category']==cat,'phecode'].tolist()]
print(phe_temp_dict)
# only keep diseases that are in the lst_final_phecodes

phes = pd.DataFrame(df_cleaned[f'diseases_within_window_phecode'].explode().explode())
phes['index'] = phes.index
phes[f'diseases_within_window_phecode_selected'] =  [x if x in lst_final_phecodes else None for x in phes[f'diseases_within_window_phecode']]


df_cleaned[f'diseases_within_window_phecode_selected'] = phes.groupby('index')[f'diseases_within_window_phecode_selected'].apply(list).tolist()
df_cleaned[f'diseases_within_window_phecode_selected'] = [x if str(x)!='[nan]' else None for x in df_cleaned[f'diseases_within_window_phecode_selected']]
df_cleaned[f'diseases_within_window_phecode_selected'] = [[m for m in x if str(m)!='nan'] if str(x)!='None' else None for x in df_cleaned[f'diseases_within_window_phecode_selected']]
df_cleaned[f'diseases_within_window_phecode_selected'] = [x if str(x)!='[]' else None for x in df_cleaned[f'diseases_within_window_phecode_selected']]

df_cleaned['chapter_within_window_phecode_selected'] = [[df_phe_db.loc[df_phe_db['phecode']==phe,'category_number'].values[0] for phe in x] if str(x) not in params.nan_str else None for x in df_cleaned['diseases_within_window_phecode_selected']]


