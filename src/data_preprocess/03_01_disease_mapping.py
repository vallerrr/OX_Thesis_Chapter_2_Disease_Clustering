"""
# Created by valler at 16/06/2024
Feature: map the ICD codes to phenotypes

steps and thoughts: @16 Jun 2024 
after searching for different resources, the phemap is the most suitable resource for the following reasons:
1. it is a package that is applicable to the UKBB data
2. it provides an overall map instead of a specific map for a specific/limited range of diseases
and the steps will be:
1. mapping all the diseases to the phenotype codes
2. exam the unmatched codes and check the missing rate/prevalence 
3. decide the diseases which we want to exam on (threshold)

steps and thoughts: @08 Sep 2024
we will choose chronic diseases in the phecodes based on the CCIR v2024-1 file and only keep the diseases that are in the final list
1. match the phecodes with the chronic conditions
2. only keep diseases that are !before! the survey date
steps:
1. identify the chronic diseases in the phecodes based on their ICD codes
2. identify the dates of the diseases and only keep the diseases that are before the survey date
"""
import pandas as pd
from src import params
from src.data_preprocess import utils
import ast
import warnings
warnings.filterwarnings("ignore")
record_column = params.disease_record_column
access_date_column = '53'
HES_ICD_ids = params.HES_ICD_ids
chronic_control = ['_chronic',''][0]
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
# 1. mapping ICD to Phecode and select the diseases within the window
# =============================================================================

# 1.0 map the ICD codes to phecode
df_single_record['phecode'] = df_single_record[f'{record_column}_icd_codes'].apply(utils.map_icd_to_phecode, args=(df_phemap,))
df_single_record['phecode_len'] = df_single_record['phecode'].apply(lambda x: len(x) if str(x)!='None' else 0)
df_single_record['phecode'] = [[[float(n) if not str(n).endswith('4444') else n for n in m ] if isinstance(m,list) else float(m) if not str(m).endswith('4444') else m for m in x] if str(x) not in ["[]",'None'] else None for x in df_single_record['phecode']]

# 1.1 generate *diseases_within_window_phecode* column
df_single_record[f'diseases_within_window_phecode'] = df_single_record.apply(lambda x: x['phecode'][0:x[f'index_of_first_{record_column}_out_window']] if x[f'index_of_first_{record_column}_out_window']!=-1 else None, axis=1)

# -----------------------------------------------------------------------------
# 1.2. check the missing rate of the diseases
# -----------------------------------------------------------------------------

# 1.2.1 manual matches for the unmatched diseases

#          note: I48.9.4444 and A09.9.4444 are not included in the phemap -> manually add them
# 1.2.1.1 I48.9 Atrial fibrillation and atrial flutter, unspecified ->427.20000,Atrial fibrillation and flutter
# 1.2.1.2 A09.9 Gastroenteritis and colitis of unspecified origin -> 8,Intestinal infection

# those codes are added in the phemap file, we don't need extra manual matches
# df_single_record['phecode'] = [str(x).replace('I48.9.4444','427.20000').replace('A09.9.4444','8.00000') for x in df_single_record['phecode']]
# df_single_record['diseases_within_window_phecode'] = [str(x).replace('I48.9.4444','427.20000').replace('A09.9.4444','8.00000') for x in df_single_record['diseases_within_window_phecode']]

# 1.2.1.3 changes we made: we should code depression into a seperate phecode (because it is a common disease and social factors could play important role in it)
# for ICD_code in ['F32','F32.0','F32.1','F32.2','F32.3','F32.9']:
#     df_phemap.loc[len(df_phemap)] = [ICD_code,296.20000,'295-306.99','psychological disorders']

# df_phemap.to_csv(params.ICD_path/ 'Phecode_map_v1_2_icd10_beta.csv',index=False)
# for x in ['F32','F32.0','F32.1','F32.2','F32.3','F32.9']:
#     df_single_record['phecode'] = [str(x).replace(f'{x}.4444,','296.20000') for x in df_single_record['phecode']]

# 1.2.2 get the prevalence of the diseases -> df_phe
df_single_record['diseases_within_window_phecode'] = [ast.literal_eval(x) if pd.notnull(x) and str(x) not in params.nan_str else None for x in df_single_record['diseases_within_window_phecode']]
phe_codes_all = df_single_record['diseases_within_window_phecode'].dropna().explode().explode().unique().tolist()
while None in phe_codes_all:
    phe_codes_all.remove(None)

df_phe = pd.DataFrame(columns=['phecode', 'prev'])
df_phe['phecode'] = [float(p) if not str(p).endswith('.4444') else p for p in phe_codes_all]

# 5538 phecodes in total for all_icd
# 3397 phecodes in total for main_icd

# get the prevalence of the diseases
phe_db = df_single_record['diseases_within_window_phecode'].dropna().explode().explode().dropna().tolist()
# 1604591 all_icd, 846452 main_icd (including unmatched diseases)


# check the prevalence of the unmathced diseases

print(df_phe.loc[df_phe['phecode'].str.endswith('.4444').fillna(False),].sort_values(by='prev').sum())
# 281688(4003 types) in all_icd, 98662(1907 types) main icd

df_phe['prev'] = [phe_db.count(p) for p in df_phe['phecode']]  # don't forget the comma
df_phe.dropna(inplace=True)
df_phe['category'] = [df_phe_db[df_phe_db['phecode'] == float(p)]['category'].values[0] if isinstance(p,float) else None for p in df_phe['phecode']]

# max disease prev = 3359 in main_icd, mainly from Z30,z09 etc, can be removed as they are not diseases

df_single_record.to_csv(params.intermediate_path / f'{record_column}_complete.csv', index=False)

# --------------------------------------------------------------------------------------------------
df_phe = pd.read_csv(params.intermediate_path / f'{record_column}_phecode.csv')
# as we have checked the unmatched situations in the appendix, we can drop the cases ending with .4444 in the diseases window in df_single_record

df_single_record['diseases_within_window_phecode'] = [utils.clean_unmatched_ICD(x) if str(x) not in params.nan_str else None for x in df_single_record['diseases_within_window_phecode']]
# they are replaced by None, so the position still represents the time order

# ==============================================================================
# 3. mapping the chronic conditions based on the CCIR v2024-1
# mapping between the phecode and CCIR file as we have the final phecode in our case
# ==============================================================================
# clean the df_ccir
df_ccir = pd.read_csv(params.ICD_path/'CCIR_v2024-1/CCIR-v2024-1.csv', header=2)  #
df_ccir.rename(columns={x:x.replace("'",'').replace(" ","_").replace("-","_") for x in df_ccir.columns}, inplace=True)
df_ccir['ICD_10_CM_CODE']=[str(x).replace("'",'') for x in df_ccir['ICD_10_CM_CODE']]


# match the phemap with the chronic indicators

df_phemap_CM = pd.read_csv(params.ICD_path/ 'Phecode_map_v1_2_icd10cm_beta.csv')
df_phemap_CM['ICD10_harmonised'] = [str(x).replace(".","") for x in df_phemap_CM['icd10cm']]
df_phemap_CM = pd.merge(left=df_ccir[['ICD_10_CM_CODE', 'CHRONIC_INDICATOR']], right=df_phemap_CM, left_on='ICD_10_CM_CODE', right_on='ICD10_harmonised', how='right')  #

# record back to the df_phe file
df_phemap_CM['CHRONIC_INDICATOR'].replace({9:0},inplace=True)
df_phe['chronic'] = [df_phemap_CM.loc[df_phemap_CM['phecode'] == float(p),'CHRONIC_INDICATOR'].sum()/len(df_phemap_CM.loc[df_phemap_CM['phecode'] == float(p),'CHRONIC_INDICATOR'].notnull()) if p in df_phemap_CM['phecode'].tolist() else None for p in df_phe['phecode']]
df_phe['description'] = [df_phe_db.loc[df_phe_db['phecode'] == float(p),'phenotype'].values[0] if p in df_phe_db['phecode'].tolist() else None for p in df_phe['phecode']]
df_phe.sort_values(by='prev',ascending=False,inplace=True)
df_phe.loc[df_phe['phecode']==296.20000,'chronic'] = 1  # depression
phes = df_phe.loc[df_phe['prev']>1000,'phecode'].tolist()
# [628.0, 197.0, 1010.0, 614.1, 591.0, 558.0, 560.4, 1019.0, 411.3, 411.4, 740.1, 218.1, 208.0, 530.1, 550.2, 285.0, 480.11, 626.12, 563.0, 1009.0, 214.0, 702.2, 574.1, 495.0, 418.0, 318.0, 272.11, 401.1, 411.1, 306.0, 411.8, 185.0, 550.4, 599.0, 598.0, 512.8, 289.4, 740.11, 608.0, 858.0, 851.0, 665.0, 455.0, 565.0, 615.0, 726.1, 250.2, 562.1, 427.11, 599.5, 600.0, 361.0, 550.1, 1015.0, 394.2, 395.1, 428.2, 681.5, 681.6, 681.3, 573.7, 790.6, 785.0, 540.11, 788.0, 317.1, 317.0, 965.0, 296.2, 969.0, 535.0, 244.4, 850.0, 622.1, 374.0, 871.0, 593.0, 561.0, 535.6, 454.1, 433.2, 278.1, 627.1, 565.1, 172.2, 1011.0, 480.0, 716.9, 716.2, 530.11, 830.0, 41.0, 280.1, 597.1, 564.9, 555.2, 722.9, 559.0, 371.3, 339.0, 459.9, 686.1, 745.0, 740.9, 327.3, 411.2, 706.2, 704.0, 340.0, 578.1, 689.0, 228.0, 701.2, 596.1, 760.0, 172.11, 211.0, 470.0, 250.1, 366.0, 634.0, 366.2, 514.0, 535.8, 594.3, 564.1, 594.1, 159.0, 516.1, 8.5, 726.0, 642.0, 655.0, 635.2, 645.0, 599.2, 530.14, 530.12, 433.31, 960.0, 471.0, 596.0, 216.0, 79.0, 783.0, 599.4, 626.13, 773.0, 735.3, 374.3, 789.0, 351.0, 624.9, 618.2, 626.8, 960.2, 386.9, 626.1, 345.0, 300.1, 574.12, 634.1, 578.8, 520.2, 626.2, 619.5, 646.0, 669.0, 473.0, 530.9, 427.2, 195.1, 507.0, 214.1, 714.1, 496.21, 519.8, 496.0, 604.1, 618.1, 451.2, 701.0, 365.0, 229.0, 835.0, 568.1, 574.3, 335.0, 458.9, 557.1, 819.0, 189.21, 189.0, 727.4, 1002.0, 915.0, 626.0, 619.3, 625.0, 531.2, 854.0, 512.7, 415.0, 661.0, 550.5, 550.0, 652.0, 636.0, 644.0, 735.2, 619.2, 174.11, 198.1, 293.0, 859.0, 531.3, 427.9, 555.1, 653.0, 521.1, 180.3, 870.3, 623.0, 626.14, 272.1, 532.0, 296.1, 574.2, 610.4, 687.4, 427.3, 525.0, 528.0, 614.51, 650.0, 596.5, 537.0, 418.1, 636.3, 443.9, 599.9, 475.0, 687.1, 41.1, 153.2, 735.21, 479.0, 288.11, 477.0, 250.7, 722.6, 619.4, 80.0, 654.1, 175.0, 939.0, 569.0, 153.3, 622.2, 611.3, 220.0, 202.2, 577.1, 798.0, 174.1, 727.1, 635.3, 345.3, 594.8, 702.1, 578.2, 592.1]

df_phe.to_csv(params.intermediate_path / f'{record_column}_phecode.csv', index=False) # it records the phenotypes and their prevalence of the record column


# new column: diseases_within_window_phecode
df_single_record['diseases_within_window_phecode_chronic'] = [[utils.return_chronoic_code(m, df_phe,0.5) for m in x] if str(x) not in params.nan_str else None for x in df_single_record['diseases_within_window_phecode']]



# =============================================================================
# 4. generate the disease database based on the diseases prevalence
# goal: generate the disease database from the phecode and match the chronic conditions again
#       standardise the prevalence based on the age groups
# when selecting the diseases, we should use the first occurrence
# =============================================================================
chronic_control = ['_chronic',''][0]
df_single_record[f'diseases_within_window_phecode{chronic_control}'] = [ast.literal_eval(x) if pd.notnull(x) and str(x) not in params.nan_str else None for x in df_single_record[f'diseases_within_window_phecode{chronic_control}']]

# from here we lost the time order of the diseases

# 4.0 first occurrence and delete the numbers behind the .

phes = pd.DataFrame(df_single_record[f'diseases_within_window_phecode{chronic_control}'].dropna().explode().explode())

phes.dropna(inplace=True)
phes[f'diseases_within_window_phecode{chronic_control}'] = [str(x) for x in phes[f'diseases_within_window_phecode{chronic_control}'] ]
phes['index'] = phes.index
phes.drop_duplicates(inplace=True)
temp = phes.groupby('index')[f'diseases_within_window_phecode{chronic_control}'].apply(list)
df_single_record[f'diseases_within_window_phecode{chronic_control}_first_occ'] = temp


# 4.1 generate the disease database based on the diseases prevalence

phe_db_all =[x for x in df_single_record[f'diseases_within_window_phecode{chronic_control}_first_occ'].dropna().explode().explode()if pd.notnull(x)]
# 216690 not null in all_icd, xxx in main_icd
phe_db = list(set(phe_db_all))  # 887 in all_icd, xxx in main_icd unique chronic conditions
df_phe_chronic = pd.DataFrame(columns=['phe','count'])
df_phe_chronic['phe'] = phe_db
df_phe_chronic['count'] = [phe_db_all.count(p) for p in phe_db]
df_phe_chronic['phe_name'] = [df_phe_db.loc[df_phe_db['phecode'] == float(p),'phenotype'].values[0] for p in df_phe_chronic['phe']]
df_phe_chronic['cat'] = [df_phe_db.loc[df_phe_db['phecode'] == float(p),'category'].values[0] for p in df_phe_chronic['phe']]

df_phe_chronic.to_csv(params.intermediate_path / f'{record_column}_phecode_chronic.csv', index=False)

df_phe_chronic.rename(columns={'count':'prev', 'phe':'phecode'},inplace=True)
# set threshold to 1000 cases? (prev 4349, 991 when looking at chronic diseases)  216690 total in main_icd, 500 cases 4.6% of the total cases
threshold = 1000
temp = df_phe_chronic.loc[df_phe_chronic['prev'] > threshold]
# xx diseases, 92 when look at chronic diseases; xx diseases with prev > 500 in main_icd


lst_final_phecodes = df_phe_chronic.loc[df_phe_chronic['prev'] > threshold,'phecode'].unique().tolist()
print(lst_final_phecodes)
# [165.1, 365.0, 558.0, 250.2, 272.11, 366.0, 401.1, 208.0, 218.1, 366.2, 285.0, 411.8, 530.1, 550.2, 563.0, 172.2, 185.0, 272.1, 351.0, 411.1, 411.3, 411.4, 411.2, 716.9, 79.0, 455.0, 495.0, 112.0, 300.1, 519.8, 496.0, 317.0, 318.0, 296.2, 41.4, 244.4, 535.0, 994.2, 38.0, 41.0, 172.11, 278.1, 327.3, 427.2, 280.1, 174.11, 198.1, 722.9, 785.0, 454.1, 562.1, 535.8, 530.14, 530.11, 216.0, 153.2, 214.1, 41.1, 276.5, 427.2, 550.1, 716.2, 578.8]
# chronic diseases in all_icd with threshold =1000 and first occurrence (92): ['624.9', '557.1', '614.1', '411.4', '618.2', '626.13', '530.14', '401.1', '300.1', '565.1', '317.0', '153.3', '618.1', '622.1', '172.11', '340.0', '250.2', '642.0', '317.1', '530.11', '627.1', '175.0', '716.2', '198.1', '296.2', '189.21', '626.0', '345.0', '574.12', '395.1', '555.2', '327.3', '280.1', '835.0', '366.2', '443.9', '615.0', '172.2', '427.11', '851.0', '625.0', '600.0', '415.0', '530.12', '351.0', '433.31', '496.0', '159.0', '740.1', '562.1', '153.2', '394.2', '596.5', '244.4', '622.2', '626.1', '740.11', '555.1', '272.1', '433.2', '626.14', '626.2', '623.0', '626.8', '475.0', '174.11', '740.9', '411.2', '722.6', '318.0', '185.0', '335.0', '569.0', '285.0', '272.11', '411.1', '250.1', '427.3', '195.1', '365.0', '174.1', '564.1', '596.1', '626.12', '411.3', '722.9', '604.1', '599.4', '411.8', '455.0', '428.2', '495.0']# chronic diseases in all_icd with threshold = 1000: [530.14, 557.1, 565.1, 593.0, 596.1, 599.4, 614.1, 618.2, 619.3, 619.2, 619.4, 622.2, 623.0, 624.9, 625.0, 626.12, 627.1, 626.13, 626.2, 626.14, 634.0, 153.3, 159.0, 185.0, 701.0, 740.11, 244.4, 272.11, 272.1, 335.0, 395.1, 401.1, 411.1, 411.4, 433.31, 443.9, 1002.0]
# chronic diseases in main_icd with threshold = 500, lst_final_phecodes = [530.14, 530.3, 557.1, 565.1, 593.0, 596.1, 599.4, 612.2, 614.1, 618.2, 619.2, 619.3, 619.4, 622.2, 624.9, 626.13, 626.12, 627.1, 626.14, 625.0, 634.0, 153.3, 185.0, 701.0, 709.7, 728.71, 740.11, 335.0, 395.1, 401.1, 411.1, 411.4, 433.31, 443.9, 1002.0]
# phenos = [df_phe_db.loc[df_phe_db['phecode'] == float(p),'phenotype'].values[0] for p in lst_final_phecodes]


first_occurence = True
# for the column diseases_within_window_phecode, we only keep the diseases that are in the lst_final_phecodes
phes = pd.DataFrame(df_single_record[f'diseases_within_window_phecode{chronic_control}{"_first_occ" if first_occurence else ""}'].explode().explode())
phes['index'] = phes.index
phes[f'diseases_within_window_phecode_selected{chronic_control}{"_first_occ" if first_occurence else ""}'] = [x if x in lst_final_phecodes else None for x in phes[f'diseases_within_window_phecode{chronic_control}{"_first_occ" if first_occurence else ""}']]

df_single_record[f'diseases_within_window_phecode_selected{chronic_control}{"_first_occ" if first_occurence else ""}'] = phes.groupby('index')[f'diseases_within_window_phecode_selected{chronic_control}{"_first_occ" if first_occurence else ""}'].apply(list).tolist()
df_single_record[f'diseases_within_window_phecode_selected{chronic_control}{"_first_occ" if first_occurence else ""}'] = [x if str(x)!='[nan]' else None for x in df_single_record[f'diseases_within_window_phecode_selected{chronic_control}{"_first_occ" if first_occurence else ""}']]
df_single_record[f'diseases_within_window_phecode_selected{chronic_control}{"_first_occ" if first_occurence else ""}'] = [[m for m in x if str(m)!='nan' ] if str(x)!='None' else None for x in df_single_record[f'diseases_within_window_phecode_selected{chronic_control}{"_first_occ" if first_occurence else ""}']]
df_single_record[f'diseases_within_window_phecode_selected{chronic_control}{"_first_occ" if first_occurence else ""}'] = [x if str(x)!='[]' else None for x in df_single_record[f'diseases_within_window_phecode_selected{chronic_control}{"_first_occ" if first_occurence else ""}']]


# mark the chapter level of phecodes (category)
phe_cate_dict = params.phe_cate_dict
def return_category(row,df_phe_db):
    if pd.notnull(row):

        return phe_cate_dict[df_phe_db.loc[df_phe_db['phecode'] == float(row),'category'].values[0]]
    return None

df_single_record[f'diseases_within_window_phecode_selected_category{chronic_control}{"_first_occ" if first_occurence else ""}'] = [[return_category(m,df_phe_db) for m in x] if str(x) not in params.nan_str else None for x in df_single_record[f'diseases_within_window_phecode_selected{chronic_control}{"_first_occ" if first_occurence else ""}']]

temp = df_single_record[f'diseases_within_window_phecode_selected_category{chronic_control}{"_first_occ" if first_occurence else ""}']

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


