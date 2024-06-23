"""
# Created by valler at 16/06/2024
Feature: map the ICD codes to phenotypes

steps and thoughts: @16 Jun 2024 
after searching for different resources, the phemap is the most suitable resource for the following reasons:
1. it is a package that is applicable to the UKBB data
2. it provides an overall map instead of a specific map for a specific/limited range of diseases
and the stpes will be: 
1. mapping all the diseases to the phenotype codes
2. exam the unmatched codes and check the missing rate/prevalence 
3. decide the diseases which we want to exam on (threshold)
"""
import pandas as pd
from src import params
from src.data_preprocess import utils
import ast
import warnings
warnings.filterwarnings("ignore")
record_column = 'all_icd'
access_date_column = '53'
HES_ICD_ids = params.HES_ICD_ids

# =============================================================================
# 0. read files
# =============================================================================
df_phe_db = pd.read_csv(params.ICD_path/ 'phecode_definitions1.2.csv')  # 1866 Phecodes
df_phemap = pd.read_csv(params.ICD_path/ 'Phecode_map_v1_2_icd10_beta.csv')  # 9505 ICD codes, 1571 Phecodes
dates_col, df_single_record = utils.import_df_single_record(record_column=record_column)

# clean the df_phemap (delete the nans in PHECODE)
df_phemap = df_phemap.dropna(subset=['PHECODE'])  # 9366 left
df_phemap.reset_index(drop=True, inplace=True)

df_single_record['diseases_within_window_phecode'] = [ast.literal_eval(x) if pd.notnull(x) and str(x) != "[]" else None for x in df_single_record['diseases_within_window_phecode']]
df_single_record['phecode'] = [ast.literal_eval(x) if pd.notnull(x) and str(x) != "[]" else None for x in df_single_record['phecode']]
# =============================================================================
# 1. mapping
# =============================================================================

# 1.1 map the all_icd column to a standard 3/4 digit ICD code
df_single_record[f'{record_column}_codes'] = [[m.split(' ')[0] for m in ast.literal_eval(x)] if pd.notnull(x) else None for x in df_single_record[record_column]]

# 1.2 map the ICD codes to phecode
df_single_record['phecode'] = df_single_record[f'{record_column}_codes'].apply(utils.map_icd_to_phecode, args=(df_phemap,))
df_single_record['phecode_len'] = df_single_record['phecode'].apply(lambda x: len(x) if str(x)!='None' else 0)
df_single_record['phecode'] = [[[float(n) if not str(n).endswith('4444') else n for n in m ] if isinstance(m,list) else float(m) if not str(m).endswith('4444') else m for m in x] if str(x) not in ["[]",'None'] else None for x in df_single_record['phecode']]

# 1.3 generate *diseases_within_window_phecode* column
df_single_record[f'diseases_within_window_phecode'] = df_single_record.apply(lambda x: x['phecode'][0:x['index_of_first_disease_out_window']] if x['index_of_first_disease_out_window']!=-1 else None, axis=1)

# =============================================================================
# 2. check the missing rate of the diseases
# =============================================================================

# 2.1 manual matches for the unmatched diseases
#          note: I48.9.4444 and A09.9.4444 are not included in the phemap -> manually add them
# 2.1.1 I48.9 Atrial fibrillation and atrial flutter, unspecified ->427.20000,Atrial fibrillation and flutter
# 2.1.2 A09.9 Gastroenteritis and colitis of unspecified origin -> 8,Intestinal infection

df_single_record['phecode'] = [str(x).replace('I48.9.4444','427.20000').replace('A09.9.4444','8.00000') for x in df_single_record['phecode']]
df_single_record['diseases_within_window_phecode'] = [str(x).replace('I48.9.4444','427.20000').replace('A09.9.4444','8.00000') for x in df_single_record['diseases_within_window_phecode']]

# 2.1.3 changes we made: we should code depression into a seperate phecode (because it is a common disease and social factors could play important role in it)
for ICD_code in ['F32','F32.0','F32.1','F32.2','F32.3','F32.9']:
    df_phemap.loc[len(df_phemap)] = [ICD_code,296.20000,'295-306.99','psychological disorders']

df_phemap.to_csv(params.ICD_path/ 'Phecode_map_v1_2_icd10_beta.csv',index=False)
for x in ['F32','F32.0','F32.1','F32.2','F32.3','F32.9']:
    df_single_record['phecode'] = [str(x).replace(f'{x}.4444,','296.20000,') for x in df_single_record['phecode']]

# 2.2 get the prevalence of the diseases
phe_codes_all = df_single_record['diseases_within_window_phecode'].dropna().explode().explode().unique().tolist()
while None in phe_codes_all:
    phe_codes_all.remove(None)

df_phe = pd.DataFrame(columns=['phecode', 'prev'])
df_phe['phecode'] = [float(p) if not str(p).endswith('.4444') else p for p in phe_codes_all]  # 4128 phecodes in total

# df_phe['phecode'] = phe_codes_all
# get the prevalence of the diseases
phe_db = df_single_record['diseases_within_window_phecode'].dropna().explode().explode().tolist()  # 1446739

# how many of the None (unmatched) situations in our case?
# phe_db.count('.4444')  # from 106259 to 88292, as we have re-defined the depression phenotypes

df_phe['prev'] = [phe_db.count(p) for p in df_phe['phecode']]  # don't forget the comma
df_phe['category'] = [df_phe_db[df_phe_db['phecode'] == float(p)]['category'].values[0] if isinstance(p,float) else None for p in df_phe['phecode']]


df_single_record.to_csv(params.intermediate_path / f'{record_column}_complete.csv', index=False)
df_phe.to_csv(params.intermediate_path / f'{record_column}_phecode.csv', index=False)
# =============================================================================
# 2. generate the disease database based on the diseases prevalence
# =============================================================================
# set threshold to 5000 cases 0.3%? (prev 4349)
threshold = 0.003 * sum(df_phe['prev'])
temp = df_phe.loc[df_phe['prev'] > threshold]  # 63 diseases
lst_final_phecodes = df_phe.loc[df_phe['prev'] > threshold,'phecode'].tolist()

# [165.1, 365.0, 558.0, 250.2, 272.11, 366.0, 401.1, 208.0, 218.1, 366.2, 285.0, 411.8, 530.1, 550.2, 563.0, 172.2, 185.0, 272.1, 351.0, 411.1, 411.3, 411.4, 411.2, 716.9, 79.0, 455.0, 495.0, 112.0, 300.1, 519.8, 496.0, 317.0, 318.0, 296.2, 41.4, 244.4, 535.0, 994.2, 38.0, 41.0, 172.11, 278.1, 327.3, 427.2, 280.1, 174.11, 198.1, 722.9, 785.0, 454.1, 562.1, 535.8, 530.14, 530.11, 216.0, 153.2, 214.1, 41.1, 276.5, 427.2, 550.1, 716.2, 578.8]

# for the column diseases_within_window_phecode, we only keep the diseases that are in the lst_final_phecodes
phes = pd.DataFrame(df_single_record['diseases_within_window_phecode'].explode().explode())
phes['index'] = phes.index
phes['diseases_within_window_phecode'] = [x if x in lst_final_phecodes else None for x in phes['diseases_within_window_phecode']]

df_single_record['diseases_within_window_phecode_selected'] = phes.groupby('index')['diseases_within_window_phecode'].apply(list).tolist()
df_single_record['diseases_within_window_phecode_selected'] = [x if str(x)!='[nan]' else None for x in df_single_record['diseases_within_window_phecode_selected']]
df_single_record['diseases_within_window_phecode_selected'] = [[m for m in x if str(m)!='nan' ] if str(x)!='None' else None for x in df_single_record['diseases_within_window_phecode_selected']]
df_single_record['diseases_within_window_phecode_selected'] = [x if str(x)!='[]' else None for x in df_single_record['diseases_within_window_phecode_selected']]


# mark the chapter level of phecodes (category)
phe_cate_dict = params.phe_cate_dict
def return_category(row,df_phe_db):
    if pd.notnull(row):
        return phe_cate_dict[df_phe_db.loc[df_phe_db['phecode'] == float(row),'category'].values[0]]
    return None

df_single_record['diseases_within_window_phecode_selected_category'] = [[return_category(m,df_phe_db) for m in x] if str(x) not in ['None','nan'] else None for x in df_single_record['diseases_within_window_phecode_selected']]
temp = df_single_record['diseases_within_window_phecode_selected_category']

#  phe_cate_dict = {cat:i for cat,i in zip(df_phe_db['category'].dropna().unique().tolist(),range(1,len(df_phe_db['category'].dropna().unique().tolist())+1))}

df_single_record.to_csv(params.intermediate_path / f'{record_column}_complete.csv', index=False)


# ==============================================================================
# appendix: Zone of the unmatched situations
# ==============================================================================

ICD_unmatched = [x.replace('.4444','') for x in phe_codes_all if str(x).endswith('4444')]
df_icd = pd.read_csv(params.intermediate_path/'ICD_10.csv')

df_unmatched_ICD = pd.DataFrame(columns=['ICD','ICD3','block', 'chapter', 'prev'])
df_unmatched_ICD['ICD'] = ICD_unmatched
df_unmatched_ICD['ICD3'] = [x.replace('.','')[0:3] for x in df_unmatched_ICD['ICD']]

def find_parent_id(x,df_ICD):
    if pd.notnull(x):
        parent_id = df_ICD.loc[df_ICD['coding'] == x, 'parent_id'].values[0]
        parent_cat = df_ICD.loc[df_ICD['node_id'] == parent_id, 'coding'].values[0]
        return parent_cat
    return None

df_unmatched_ICD['block'] = [find_parent_id(x,df_icd) for x in df_unmatched_ICD['ICD3']]
df_unmatched_ICD['chapter'] = [find_parent_id(x,df_icd) for x in df_unmatched_ICD['block']]
df_unmatched_ICD['prev'] = [phe_db.count(f'{str(p)}.4444') for p in df_unmatched_ICD['ICD']]


