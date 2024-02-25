"""
QUESTIONS：
1. whether the variables are overlapped in the fields we have identified
2. whether each variable has multiple instances?
-> extract information from recorder
"""

import pandas as pd
from src import params
import re
from collections import defaultdict
import ast
def save_codebook(df):
    df.to_csv(params.codebook_path/'UKB_var_select.csv',index=False)
def compare_two_cats(a,b):
    temp_a = df_codebook.loc[df_codebook['cate_name']==a]
    temp_b = df_codebook.loc[df_codebook['cate_name']==b]
    intersection = set(temp_a['field_id']).intersection(temp_b['field_id'])
    unique_a = df_codebook.loc[df_codebook['field_id'].isin(set(temp_a['field_id']) - set(temp_b['field_id'])),]
    unique_b = df_codebook.loc[df_codebook['field_id'].isin(set(temp_b['field_id'])-set(temp_a['field_id'])),]
    print(f"unique in {a} = {len(a)},"
          f"unique in {b} = {len(b)}")
    return unique_a,unique_b,intersection


df_codebook = pd.read_csv(params.codebook_path/'UKB_var_select.csv')

# question 1
print(f'len orig = {len(df_codebook["field_id"])}, unique = {len(df_codebook["field_id"].unique())}')


df_duplicated = df_codebook.loc[df_codebook.duplicated(subset=['field_id'],keep=False),]

# Early life is duplicated? yes
temp = df_codebook.loc[df_codebook['cate_name']=='Early life']
index_to_remove = temp.index[0:13]
df_codebook.drop(index_to_remove,inplace=True)
df_codebook.reset_index(drop=True,inplace=True)
save_codebook(df_codebook)

# 1 Lifestyle and Diet and alcohol summary-> overlaps go to Lifestyle

a = 'Lifestyle'
b = 'Diet and alcohol summary'
unique_a, unique_b, intersection = compare_two_cats(a,b)
# as there are only 7 unique values that are not in the Lifestyle, add the Diet and alcohol summary to the Lifestyle category
index_of_diet = unique_b.index
df_codebook.loc[index_of_diet,'cate_name'] = [a]*len(index_of_diet)
df_codebook.drop(df_codebook.loc[df_codebook['cate_name']==b,].index,inplace=True)
save_codebook(df_codebook)
# done

# 2 physical measures and physical measure summary

a = 'Physical measure summary'
b = 'Physical measures'

unique_a, unique_b, intersection = compare_two_cats(a,b)
# as there are only 1 unique values that are not in the Lifestyle, add the Diet and alcohol summary to the Lifestyle category
index_of_diet = unique_a.index
df_codebook.loc[index_of_diet,'cate_name'] = [b]*len(index_of_diet)
df_codebook.drop(df_codebook.loc[df_codebook['cate_name']==a,].index,inplace=True)
save_codebook(df_codebook)
# from 1865 vars to 1801
# done


# 3 nans in the cate_name: 38 rows
# all of which is about acceleration time
df_codebook.dropna(subset=['cate_name'],inplace=True)
save_codebook(df_codebook)
# from 1801 to 1763

df_codebook.cate_name.value_counts()


# 4 df_codebook and df_recorder matching
# find the field_id, instance, array of each item in df_codebook and store the length and the item into the df_codebook

df_codebook = pd.read_csv(params.codebook_path/'UKB_var_select.csv')
df_recorder = pd.read_csv(params.recorder_path/'recorder_participant.csv',index_col =0)
id_lst = list(df_recorder.field_name)
ind_lst = list(df_recorder.ind)
def find_field_info(field_id, id_lst,ind_lst):
    def find_instances(ids):
        pattern = re.compile(r'_i(\d+)')
        return list(set([num for id in ids for num in pattern.findall(id)]))

    def find_array(ids):
        pattern = re.compile(r'_a(\d+)')
        return list(set([num for id in ids for num in pattern.findall(id)]))

    pattern = re.compile(rf'p{field_id}(?:_|$)')
    id_ind = [{var: ind} for var,ind in zip(id_lst,ind_lst) if pattern.match(var)]
    ids = [list(id_dict.keys())[0] for id_dict in id_ind]
    instance = find_instances(ids)
    array = find_array(ids)
    ids.sort()
    instance.sort()
    array.sort()
    return id_ind,instance,array


df_codebook[['ids_count','instance_count','array_count','ids','instance','array','file_names','preprocessed_flag']]=[[None,None,None,None,None,None,None,None]]*len(df_codebook)
for ind, row in df_codebook.iterrows():
    field_id = row['field_id']
    ids,instance,array = find_field_info(field_id,id_lst,ind_lst)
    df_codebook.loc[ind, ['ids_count', 'instance_count','array_count']]=[len(ids),len(instance),len(array)]
    df_codebook.loc[ind,'ids'] = ';'.join([str(id)for id in ids])
    df_codebook.loc[ind, 'instance'] = None if len(instance)==0 else ';'.join(instance)
    df_codebook.loc[ind, 'array'] = None if len(array)==0 else ';'.join(array)
    df_codebook.loc[ind,'file_names'] = ';'.join(list(set([str(list(v.values())[0]) for v in ids])))
# store the df_codebook

# store the ids column in the format {filename:['var_names']}
def sort_ids(row):
    # read the data first
    if len(row)>0:
        ids = [{v: k for k, v in ast.literal_eval(x).items()} for x in row.split(';')]  # inverse the key,value
        ids_sorted = defaultdict(list)
        for d in ids:
            for key, value in d.items():
                ids_sorted[key].append(value)
        ids_sorted = dict(ids_sorted)
    else:
        ids_sorted = None
    return ids_sorted

df_codebook.ids = df_codebook['ids'].apply(sort_ids)

df_codebook.to_csv(params.codebook_path/'UKB_var_select.csv',index=False)


# get the filename:count df
counts_dict = {}
for id in df_codebook.ids:
    if id:
        for key in id.keys():
            if key not in counts_dict.keys():
                counts_dict[key]=0

            counts_dict[key]+= len(id[key])
df_counts = pd.DataFrame.from_dict(counts_dict,columns=['count'],orient='index').reset_index(drop=False)



