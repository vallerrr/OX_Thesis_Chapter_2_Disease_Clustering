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
df_phemap = pd.read_csv(params.ICD_path/ 'Phecode_map_v1_2_icd10_beta.csv')  # 9066 ICD codes, 1571 Phecodes
df_pop = pd.read_csv(params.intermediate_path.parent/'downloaded_data/uk_pop/uk_population_constituent_countries_level.csv')

dates_col, df_single_record = utils.import_df_single_record(record_column=record_column)

# clean the df_phemap (delete the nans in PHECODE)
df_phemap = df_phemap.dropna(subset=['PHECODE'])  # 9372 left
df_phemap.reset_index(drop=True, inplace=True)

#df_single_record['diseases_within_window_phecode'] = [ast.literal_eval(x) if pd.notnull(x) and str(x) != "[]" else None for x in df_single_record['diseases_within_window_phecode']]
#df_single_record['phecode'] = [ast.literal_eval(x) if pd.notnull(x) and str(x) != "[]" else None for x in df_single_record['phecode']]
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
# phe_db.count('.4444')  # from 106259 to 88292, as we have re-defined the 3 phenotypes

df_phe['prev'] = [phe_db.count(p) for p in df_phe['phecode']]  # don't forget the comma
df_phe['category'] = [df_phe_db[df_phe_db['phecode'] == float(p)]['category'].values[0] if isinstance(p,float) else None for p in df_phe['phecode']]

# deal with the self-defined diseases
# 'I48.9.4444','427.20000'
for (A,B) in [['I48.9.4444','427.2'],['A09.9.4444','8.0']]:
    df_phe.loc[df_phe['phecode']==B,'prev']+=df_phe.loc[df_phe['phecode']==A,'prev'].values[0]
    df_phe.drop(df_phe.loc[df_phe['phecode']==A].index,inplace=True)


df_single_record.to_csv(params.intermediate_path / f'{record_column}_complete.csv', index=False)
df_phe.to_csv(params.intermediate_path / f'{record_column}_phecode.csv', index=False)
# --------------------------------------------------------------------------------------------------
df_phe = pd.read_csv(params.intermediate_path / f'{record_column}_phecode.csv')

# as we have checked the unmatched situations in the appendix, we can drop the cases ending with .4444 in the diseases window in df_single_record

df_single_record['diseases_within_window_phecode'] = [utils.clean_unmatched_ICD(x) if str(x) not in params.nan_str else None for x in df_single_record['diseases_within_window_phecode']]

# ==============================================================================
# 3. mapping the chronic conditions based on the CCIR v2024-1
# mapping between the phecode and CCIR file as we have the final phecode in our case
# ==============================================================================
# clean the df_ccir
df_ccir = pd.read_csv(params.ICD_path/'CCIR_v2024-1/CCIR-v2024-1.csv', header=2)  #
df_ccir.rename(columns={x:x.replace("'",'').replace(" ","_").replace("-","_") for x in df_ccir.columns}, inplace=True)
df_ccir['ICD_10_CM_CODE']=[str(x).replace("'",'') for x in df_ccir['ICD_10_CM_CODE']]

# match the phemap with the chronic indicators
df_phemap['ICD10'] = [str(x).replace(".","") for x in df_phemap['ICD10']]
df_phemap = pd.merge(left=df_ccir[['ICD_10_CM_CODE', 'CHRONIC_INDICATOR']],right=df_phemap,left_on='ICD_10_CM_CODE', right_on='ICD10',how='right')  #
df_phemap['CHRONIC_INDICATOR'].isnull().sum()  # 4564 phecode does not have corresponding chronic condition definition

# new column: diseases_within_window_phecode
df_single_record['diseases_within_window_phecode_chronic'] = [[utils.return_chronoic_code(m, df_phemap) for m in x] if str(x) not in params.nan_str else None for x in df_single_record['diseases_within_window_phecode']]

# =============================================================================
# 4. generate the disease database based on the diseases prevalence
# goal: generate the disease database from the phecode and match the chronic conditions again
#       standardise the prevalence based on the age groups

# =============================================================================

chronic_control = ['_chronic',''][0]
df_single_record[f'diseases_within_window_phecode{chronic_control}'] = [ast.literal_eval(x) if pd.notnull(x) and str(x) not in params.nan_str else None for x in df_single_record[f'diseases_within_window_phecode{chronic_control}']]

phe_db =[x for x in df_single_record[f'diseases_within_window_phecode{chronic_control}'].dropna().explode().explode().tolist() if pd.notnull(x)]  # 1401045 in total, 275329 not null
phe_db = list(set(phe_db))  # 298 unique chronic conditions
df_phe_chronic_with_age = pd.DataFrame(columns=['age', 'gender','phe','count'])

age_range = range(40, 71)
for age in age_range:
    for gender in [0,1]:

        c1 = df_single_record['31'] == gender
        c2 = df_single_record['21022'] == age
        df_single_record_frac = df_single_record.loc[c1&c2, f'diseases_within_window_phecode{chronic_control}']
        phes = [x for x in df_single_record_frac.explode().explode() if pd.notnull(x)]
        for phe in phe_db:
            phe_count = phes.count(phe)
            df_phe_chronic_with_age.loc[len(df_phe_chronic_with_age),] = [age,gender,phe,phe_count]

# standardise to the UK population
df_single_pop = df_single_record[['21022','31']].groupby(['21022','31']).size().reset_index(name='count')
df_single_pop=df_single_pop.loc[df_single_pop['21022'].isin(list(age_range)),] # only keep the age range from 40 to 70

df_pop = df_pop.loc[df_pop['laname23']=='GREAT BRITAIN',['sex','age','population_2011']]  # as the UK Biobank is only running in the great britain not in the north ireland and other regions
df_pop['population_2011']=[int(x.replace(',','')) for x in df_pop['population_2011']]
df_pop['sex']=[0 if x=='F' else 1 for x in df_pop['sex']]
total_pop = df_pop['population_2011'].sum()
df_pop['proportion'] = df_pop['population_2011'] / total_pop

df_single_pop = df_single_pop.merge(df_pop[['age','sex','proportion']], how='inner', left_on=['21022', '31'], right_on=['age', 'sex'])
df_single_pop.drop(columns =['age','sex'],inplace=True)
total_pop_single = df_single_pop['count'].sum()
df_single_pop['standardized_population'] = df_single_pop['proportion'] * total_pop_single
# rounding
df_single_pop['standardized_population'] = df_single_pop['standardized_population'].apply(lambda x: round(x))
# weights for future use

total_pop = df_single_pop['count'].sum()
df_single_pop['sample_proportion'] = df_single_pop['count']/total_pop
# df_single_pop['weight']=df_single_pop['standardized_population']/df_single_pop
df_single_pop['weight'] = df_single_pop['proportion']/df_single_pop['sample_proportion']
df_single_pop.to_csv(params.intermediate_path / 'pop_weights.csv', index=False)

# match it to the df_single_record
df_single_record['weight'] = df_single_record.apply(lambda x: df_single_pop.loc[(df_single_pop['21022']== x['21022']) &(df_single_pop['31']==x['31']),'weight'].values[0] if len(df_single_pop.loc[(df_single_pop['21022']== x['21022']) &(df_single_pop['31']==x['31']),'weight'].values)>0 else 0,axis=1)



# now generate the df_phe_chronic_with_age
total_pop_phe = df_phe_chronic_with_age['count'].sum()
df_phe_chronic_with_age['standardised_count'] = df_phe_chronic_with_age.apply(lambda x: total_pop_phe*df_single_pop.loc[(df_single_pop['21022']==x['age'])&(df_single_pop['31']==x['gender']),'proportion'].values[0],axis=1)
for age in age_range:
    for gender in [0,1]:
        rows = df_phe_chronic_with_age.loc[(df_phe_chronic_with_age['age']==age )& (df_phe_chronic_with_age['gender']==gender) ,]
        age_gender_specific_phe_total = rows['count'].sum()
        age_gender_specific_total = total_pop_phe*df_single_pop.loc[(df_single_pop['21022']==age)&(df_single_pop['31']==gender),'proportion'].values[0]
        df_phe_chronic_with_age.loc[rows.index, 'standardised_count']=[x/age_gender_specific_phe_total*age_gender_specific_total for x in rows['count']]

# save the phe code database that are chronic
df_phe_chronic_with_age.to_csv(params.intermediate_path / f'{record_column}_phecode{chronic_control}.csv', index=False)


# generate df_phe based on the df_phe_chronic_with_age
if chronic_control == '_chronic': # 298 different diseases
    df_phe = df_phe_chronic_with_age.groupby('phe',as_index=False).sum()[['phe','count']]
    df_phe.rename(columns={'count':'prev','phe':'phecode'},inplace=True)
# set threshold to 1000 cases? (prev 4349, 991 when looking at chronic diseases)
threshold = 0.0036 * sum(df_phe['prev'])
temp = df_phe.loc[df_phe['prev'] > threshold]  # 63 diseases, 42 when look at chronic diseases

lst_final_phecodes = df_phe.loc[df_phe['prev'] > threshold,'phecode'].tolist()

# [165.1, 365.0, 558.0, 250.2, 272.11, 366.0, 401.1, 208.0, 218.1, 366.2, 285.0, 411.8, 530.1, 550.2, 563.0, 172.2, 185.0, 272.1, 351.0, 411.1, 411.3, 411.4, 411.2, 716.9, 79.0, 455.0, 495.0, 112.0, 300.1, 519.8, 496.0, 317.0, 318.0, 296.2, 41.4, 244.4, 535.0, 994.2, 38.0, 41.0, 172.11, 278.1, 327.3, 427.2, 280.1, 174.11, 198.1, 722.9, 785.0, 454.1, 562.1, 535.8, 530.14, 530.11, 216.0, 153.2, 214.1, 41.1, 276.5, 427.2, 550.1, 716.2, 578.8]
# chronic diseases: [153.3, 165.1, 185.0, 198.3, 198.4, 244.1, 244.4, 252.1, 272.1, 272.11, 275.5, 281.11, 332.0, 334.0, 335.0, 357.0, 362.4, 394.3, 395.1, 401.1, 411.1, 411.4, 414.0, 416.0, 426.21, 426.31, 426.32, 433.0, 433.31, 443.9, 530.14, 565.1, 593.0, 599.4, 618.2, 624.9, 626.12, 627.1, 634.0, 697.0, 740.11, 747.13]

# for the column diseases_within_window_phecode, we only keep the diseases that are in the lst_final_phecodes
phes = pd.DataFrame(df_single_record[f'diseases_within_window_phecode{chronic_control}'].explode().explode())
phes['index'] = phes.index
phes['diseases_within_window_phecode'] = [x if x in lst_final_phecodes else None for x in phes[f'diseases_within_window_phecode{chronic_control}']]

df_single_record[f'diseases_within_window_phecode_selected{chronic_control}'] = phes.groupby('index')['diseases_within_window_phecode'].apply(list).tolist()
df_single_record[f'diseases_within_window_phecode_selected{chronic_control}'] = [x if str(x)!='[nan]' else None for x in df_single_record[f'diseases_within_window_phecode_selected{chronic_control}']]
df_single_record[f'diseases_within_window_phecode_selected{chronic_control}'] = [[m for m in x if str(m)!='nan' ] if str(x)!='None' else None for x in df_single_record[f'diseases_within_window_phecode_selected{chronic_control}']]
df_single_record[f'diseases_within_window_phecode_selected{chronic_control}'] = [x if str(x)!='[]' else None for x in df_single_record[f'diseases_within_window_phecode_selected{chronic_control}']]


# mark the chapter level of phecodes (category)
phe_cate_dict = params.phe_cate_dict
def return_category(row,df_phe_db):
    if pd.notnull(row):
        return phe_cate_dict[df_phe_db.loc[df_phe_db['phecode'] == float(row),'category'].values[0]]
    return None

df_single_record[f'diseases_within_window_phecode_selected_category{chronic_control}'] = [[return_category(m,df_phe_db) for m in x] if str(x) not in ['None','nan'] else None for x in df_single_record[f'diseases_within_window_phecode_selected{chronic_control}']]
temp = df_single_record[f'diseases_within_window_phecode_selected_category{chronic_control}']

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


df_unmatched_ICD['block'] = [find_parent_id(x, df_icd) for x in df_unmatched_ICD['ICD3']]
df_unmatched_ICD['chapter'] = [find_parent_id(x, df_icd) for x in df_unmatched_ICD['block']]
df_unmatched_ICD['prev'] = [phe_db.count(f'{str(p)}.4444') for p in df_unmatched_ICD['ICD']]


