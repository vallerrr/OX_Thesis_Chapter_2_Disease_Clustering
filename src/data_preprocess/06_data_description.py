"""
# Created by valler at 31/05/2024
Feature: script to generate data descrption for the data

"""


from src import params
from src.data_preprocess import utils
from collections import Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
df_codebook_final = pd.read_csv(params.codebook_path / 'UKB_preprocess_codebook_wave_0_final.csv')
df = pd.read_pickle(params.final_data_path / 'UKB_wave_0_final_standardised.pkl')

record_column = params.disease_record_column
access_date_column = '53'
HES_ICD_ids = params.HES_ICD_ids
level = 'chronic_cate'
weight_control = False

# control zone
chapter_ranges = [x for x in range(1, 16)]+[19] if 'chapter' in level else [x for x in range(1, 18)]
disease_column = params.disease_columns_dict[level]


# ======================================================================================================================
# 0. combine chronic disease information from the specific time window
# ======================================================================================================================



# read the single record file and merge the date columns to the single record file

dates_col, df_single_record = utils.import_df_single_record(record_column)
df[access_date_column] = pd.to_datetime(df[access_date_column], errors='coerce', format='%Y-%m-%d')
df_single_record = pd.merge(df_single_record, df[[access_date_column, 'eid']], how='left', left_on='eid', right_on='eid')


# only select columns before the interview date (in column access_date_column)
'''def find_first_date(row):
    """
    find the first date that is out of the window
    :param row:
    :return: the index of the date, or -1 if all dates are not in the window/no diease
    """
    if row['all_icd_uniq_count'] > 0:
        # access_date = row[access_date_column]
        conditions = [None if pd.isnull(x) else True if (x > row[access_date_column]) else False for x in row[dates_col]]

        if True in conditions:
            first_true_index = conditions.index(True)
        else:
            first_true_index = -1
    else:
        first_true_index = -1
    return first_true_index

df_single_record['index_of_first_disease_out_window'] = df_single_record.apply(find_first_date, axis=1)

for column in list(params.disease_columns_dict.values()):
    if column in df_single_record.columns.tolist():
        print(column)
        df_single_record[column] = df_single_record.apply(lambda x: x[column][0:int(x[f'index_of_first_{record_column}_out_window'])] if x[f'index_of_first_{record_column}_out_window']!=-1 else None, axis=1)
'''

# merge the new three columns back to the original df
df = pd.merge(df, df_single_record[['eid','weight'] if weight_control else ['eid']+ [x for x in list(params.disease_columns_dict.values()) if x in df_single_record.columns.tolist()]], how='left', left_on='eid', right_on='eid')

df_single_record.to_csv(params.intermediate_path / f'{record_column}_complete.csv')
df.to_pickle(params.final_data_path / 'UKB_wave_0_final_standardised_with_disease.pkl')  # this is the file to be used in final analysis


# ======================================================================================================================
# 1. generate the data description
# ======================================================================================================================
df = pd.read_pickle(params.final_data_path / 'UKB_wave_0_final_non_standardised.pkl')

df_descriptive_table = pd.DataFrame(columns = ['variable_name', 'variable_description', 'variable_type', 'variable_category', 'variable_values', 'variable_missing_values', 'variable_missing_values_percentage', 'variable_unique_values', 'variable_unique_values_percentage', 'variable_mean', 'variable_std', 'variable_min', 'variable_25%', 'variable_50%', 'variable_75%', 'variable_max'])
for column in df.columns:
    if column in ['eid', '53','55', 'diseases_within_window_all_icd_first_3', 'diseases_within_window_icd_parent_coding', 'diseases_within_window_icd_chapter_coding']:
        continue
    print(f'Processing {column}')
    var_name = df_codebook_final.loc[df_codebook_final['field_id'] == int(column), 'field_name'].values[0]
    df_descriptive_table.loc[len(df_descriptive_table),] = [column,var_name, df[column].dtype, df_codebook_final.loc[df_codebook_final['field_id'] == int(column), 'cate_name'].values[0], df[column].unique(), df[column].isnull().sum(), df[column].isnull().sum()/len(df), len(df[column].unique()), len(df[column].unique())/len(df), df[column].mean(), df[column].std(), df[column].min(), df[column].quantile(0.25), df[column].quantile(0.5), df[column].quantile(0.75), df[column].max()]

# using df_notebook to fill the missing count
def find_missing_count(row):
    field_id = int(row['variable_name'])

    if field_id in df_codebook_final['field_id'].values:
        missing_count = df_codebook_final.loc[df_codebook_final['field_id']==field_id,'missing_count'].values[0]

    else:
        missing_count = None

    return missing_count


df_descriptive_table['variable_missing_values'] = df_descriptive_table.apply(find_missing_count, axis=1)
df_descriptive_table['variable_missing_values_percentage'] = [x/len(df) for x in df_descriptive_table['variable_missing_values']]
df_descriptive_table.to_csv(params.final_data_path/'descriptive_table'/'descriptive_table_wave_0.csv',index=False)

# concat disease information to the non standardised dataframe
df_non_standardised = pd.read_pickle(params.final_data_path / 'UKB_wave_0_final_non_standardised.pkl')
df_non_standardised = pd.merge(df_non_standardised, df[['eid', 'diseases_within_window_all_icd_first_3', 'diseases_within_window_icd_parent_coding', 'diseases_within_window_icd_chapter_coding']], how='left', left_on='eid', right_on='eid')
df_non_standardised.to_pickle(params.final_data_path / 'UKB_wave_0_final_non_standardised.pkl')

# ======================================================================================================================
# 2. generate the disease description
# ======================================================================================================================
phe_cate_dict = {y:x for x,y in params.phe_cate_dict.items()}
dates_col, df_single_record = utils.import_df_single_record(record_column)
df = pd.read_pickle(params.final_data_path/ 'UKB_wave_0_final_non_standardised.pkl')
if disease_column not in df.columns:
    df = pd.merge(df, df_single_record[['eid', disease_column]], how='left', left_on='eid', right_on='eid')
# merge weight to the dataframe
if weight_control:
    df = pd.merge(df, df_single_record[['eid', 'weight']], how='left', left_on='eid', right_on='eid')
#    average disease count by age and gender
    df[f'{level}_disease_count'] = [len(x)*weight if str(x) not in ['None','nan'] else 0 for x,weight in zip(df[disease_column],df['weight'])]
else:
    df[f'{level}_disease_count'] = [len(x) if str(x) not in ['None', 'nan'] else 0 for x in df[disease_column]]
# ----------------------------------------------------------
# 2.1 plot: average_disease_count_at_recruitment_all_disease
# ----------------------------------------------------------

df_to_plot = df[[f'{level}_disease_count', '31', '21022']].copy()
# 31: gender
# 21022: age at recruitment
# display the disease count by gender and age
df_to_plot = df_to_plot.groupby(['31', '21022']).agg(cases_count=(f'{level}_disease_count', 'count'), disease_count=(f'{level}_disease_count', 'mean')).reset_index()
# set 31 as categorical variable
df_to_plot['31'] = df_to_plot['31'].astype('category')
df_to_plot['31'] = df_to_plot['31'].cat.rename_categories(['Female', 'Male'])
df_to_plot = df_to_plot.loc[(df_to_plot['21022']>=40) & (df_to_plot['21022']<=70),] # delete 40 and 70 as they are severe underpresented


fig, ax = plt.subplots()
# Colors to use for the two categories (adjust as needed)
colors = ['blue', 'orange']
# Plot each category with its own color and label
for i, category in enumerate(df_to_plot['31'].cat.categories):
    df_subset = df_to_plot[df_to_plot['31'] == category]
    ax.scatter(df_subset['21022'], df_subset['disease_count'], color=colors[i], label=category,alpha=0.8)

# Add a legend
ax.legend(title='Gender')
plt.title(f'Average chronic disease count at recruitment by age {"(weighted)" if weight_control else ""}')
plt.xlabel('Age at recruitment')
plt.ylabel('Average disease count')
plt.show()
plt.savefig(params.current_path/f'plot/{level}/average_disease_count_at_recruitment_{record_column}_{"weighted" if weight_control else ""}.pdf')


# ----------------------------------------------------------
# 2.2 plot: Diseases chapter count by age and gender
# ----------------------------------------------------------
def count_chapter(row):
    """
    count the chapter of the diseases for each row in the dataframe
    """
    if str(row[disease_column]) not in ['None','nan','NaN']:
        chapters = [str(x) for x in row[disease_column]]
        count_dict = Counter(chapters)
        if 'weight' in row.index.tolist():
            count_dict = {k: v*row['weight'] for k, v in count_dict.items()}
    else:
        count_dict = None
    return count_dict
import ast
df[disease_column] = [ast.literal_eval(x) if str(x) not in params.nan_str else None for x in df[disease_column] ]

if weight_control:
    df_to_plot = df[[disease_column, '31', '21022','weight']].copy()
else:
    df_to_plot = df[[disease_column, '31', '21022']].copy()

# separate diseases by the chapter name (each chapter will have a separate column)
df_to_plot['chapter_count'] = df_to_plot.apply(count_chapter, axis=1)

chapter_ranges = df[df[disease_column].notnull()][disease_column].apply(lambda x: [int(y) for y in x]).explode().unique()
chapter_ranges.sort()
for column in chapter_ranges:
    df_to_plot[f'chapter_{column}'] = [x[str(column)] if isinstance(x, dict) and str(column) in x.keys() else None for x in df_to_plot['chapter_count']]


df_plot_2 = df_to_plot.pivot_table(index=['21022', '31'],  values=[f'chapter_{x}' for x in chapter_ranges], aggfunc='sum').reset_index()
# remove sparse data points that have age below 40 or over 70 (same as the last plot)
df_plot_2.drop(df_plot_2.loc[(df_plot_2['21022'] < 40) | (df_plot_2['21022'] > 70)].index, inplace=True)

# set 31 as categorical variable
df_plot_2['31'] = df_plot_2['31'].astype('category')
df_plot_2['31'] = df_plot_2['31'].cat.rename_categories(['Female', 'Male'])

#  2.2.1 absolute counts of the diseases
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.rcParams["figure.figsize"] = [16,9]
# Create a subset of the dataframe with only the columns to be plotted
gender = 'Male'
df_subset = df_plot_2[df_plot_2['31'] == gender][['21022'] + [f'chapter_{i}' for i in chapter_ranges]]
# Set '21022' as the index to be used as x-axis
df_subset.set_index('21022', inplace=True)
# Plot the stacked bar chart
df_subset.plot(kind='bar', stacked=True, ax=ax1)
ax1.set_title(gender)
ax1.set_xlabel('Age at recruitment')
ax1.set_ylabel('Count of Disease Chapters')
# remove the legend
ax1.get_legend().remove()

gender = 'Female'
df_subset = df_plot_2[df_plot_2['31'] == gender][['21022'] + [f'chapter_{i}' for i in chapter_ranges]]
# Set '21022' as the index to be used as x-axis
df_subset.set_index('21022', inplace=True)

# Plot the stacked bar chart
df_subset.plot(kind='bar', stacked=True,  ax=ax2)
ax2.set_title(gender)
ax2.set_xlabel('Age at recruitment')
ax2.set_ylabel('Count of Disease Chapters')


handles, labels = ax2.get_legend_handles_labels()  # reverse the order of legend
labels = [phe_cate_dict[int(str(x).replace('chapter_',''))] for x in labels]

ax2.legend(reversed(handles), reversed(labels), title='Chapter', title_fontsize='large', fontsize='small', loc='center left', bbox_to_anchor=(1.0, 0.5))

fig.suptitle('Diseases chapter count by age and gender')
fig.tight_layout(rect=[0, 0.01, 1, 0.98])
# plt.show()

plt.savefig(params.current_path/f'plot/{level}/diseases_chapter_count_{record_column}_by_age_and_gender{"_weighted" if weight_control else ""}.pdf')

# ----------------------------------------------------------
# 2.2.2 relative counts of the diseases
df_plot_2['total_diseases'] = df_plot_2[[f'chapter_{x}' for x in chapter_ranges]].sum(axis=1)
# Calculate the proportion of each chapter
for column in chapter_ranges:
    df_plot_2[f'chapter_{column}_prop'] = df_plot_2[f'chapter_{column}']/df_plot_2['total_diseases']

fig, (ax1, ax2) = plt.subplots(1, 2)
plt.rcParams["figure.figsize"] = [16,9]
# Create a subset of the dataframe with only the columns to be plotted
gender = 'Male'
df_subset = df_plot_2[df_plot_2['31'] == gender][['21022'] + [f'chapter_{i}_prop' for i in chapter_ranges]]
# Set '21022' as the index to be used as x-axis
df_subset.set_index('21022', inplace=True)

# Plot the stacked bar chart
df_subset.plot(kind='bar', stacked=True, ax=ax1)
ax1.set_title(gender)
ax1.set_xlabel('Age at recruitment')
ax1.set_ylabel('Proportion of Disease Chapters')
# remove the legend
ax1.get_legend().remove()


gender = 'Female'
df_subset = df_plot_2[df_plot_2['31'] == gender][['21022'] + [f'chapter_{i}_prop' for i in chapter_ranges]]
# Set '21022' as the index to be used as x-axis
df_subset.set_index('21022', inplace=True)

# Plot the stacked bar chart
df_subset.plot(kind='bar', stacked=True,  ax=ax2)
ax2.set_title(gender)
ax2.set_xlabel('Age at recruitment')
ax2.set_ylabel('Proportion of Disease Chapters')

handles, labels = ax2.get_legend_handles_labels()  # reverse the order of legend
labels = [phe_cate_dict[int(str(x).replace('chapter_','').replace('_prop',''))] for x in labels]
ax2.legend(reversed(handles), reversed(labels), title='Chapter', title_fontsize='large', fontsize='small', loc='center left', bbox_to_anchor=(1.0, 0.5))

fig.suptitle('Diseases chapter count by age and gender (Proportion)')
fig.tight_layout(rect=[0, 0.01, 1, 0.98])
# plt.show()

plt.savefig(params.current_path/f'plot/{level}/diseases_chapter_{record_column}_proportion_by_age_and_gender{"_weighted" if weight_control else ""}.pdf')


# ----------------------------------------------------------
# 2.3 disease occurrence: within chapter (self-repeating rate)
# ----------------------------------------------------------

df_subset = df[[f'{level}_disease_count',disease_column, '31', '21022']].copy()


# self-repeating occurrence
# diseases related vars ['disease_count', 'chapter_1-22']

df_to_plot = df[['31', '21022']+[f'chapter_{x}' for x in chapter_ranges]]

df_to_plot = df_to_plot.pivot_table(index=['31','21022'],values=[f'chapter_{x}' for x in chapter_ranges],aggfunc=np.mean)
df_to_plot.reset_index(inplace=True)
df_to_plot.drop(df_to_plot.loc[(df_to_plot['21022'] <= 40) | (df_to_plot['21022'] >= 70)].index, inplace=True)

# set 31 as categorical variable
df_to_plot['31'] = df_to_plot['31'].astype('category')
df_to_plot['31'] = df_to_plot['31'].cat.rename_categories(['Female', 'Male'])

# remove diseases chapters that are bigger than 15

# within disease co-occurrence plot
fig, (ax1,ax2) = plt.subplots(2, 1)
plt.rcParams["figure.figsize"] = [15, 20]

gender = 'Male'
df_subset = df_to_plot[df_to_plot['31'] == gender][['21022'] + [f'chapter_{i}' for i in chapter_ranges]]
df_subset.set_index('21022', inplace=True)
df_subset.plot(kind='line', ax=ax1)
ax1.set_title(gender)
ax1.set_xlabel('Age at recruitment')
ax1.set_ylabel('Count of Disease Chapters')
for column in df_subset.columns:
    line = ax1.plot(df_subset.index, df_subset[column], label=column)
    last_x, last_y = df_subset.index[-1], df_subset[column].iloc[-1]
    ax1.text(last_x, last_y, phe_cate_dict[int(f'{column}'.replace('chapter_',''))], fontsize=12, verticalalignment='center')
ax1.legend().set_visible(False)

gender = 'Female'
df_subset = df_to_plot[df_to_plot['31'] == gender][['21022'] + [f'chapter_{i}' for i in chapter_ranges]]
df_subset.set_index('21022', inplace=True)
df_subset.plot(kind='line',  ax=ax2)
ax2.set_title(gender)
ax2.set_xlabel('Age at recruitment')
ax2.set_ylabel('Count of Disease Chapters')
for column in df_subset.columns:
    line = ax2.plot(df_subset.index, df_subset[column], label=column)
    last_x, last_y = df_subset.index[-1], df_subset[column].iloc[-1]
    ax2.text(last_x, last_y, phe_cate_dict[int(f'{column}'.replace('chapter_',''))], fontsize=12, verticalalignment='center')
ax2.legend().set_visible(False)

fig.suptitle('Diseases self-repeating rate by chapter (weighted)')
fig.tight_layout(rect=[0, 0.01, 1, 0.98])
# plt.show()

plt.savefig(params.current_path/f'plot/{level}/disease_co_occurrence_{record_column}_within_diseases_by_chapter_chronic.pdf')


# ----------------------------------------------------------
# 2.4 co-occurrence: between diseases (by each chapter)
# ----------------------------------------------------------
df_subset = df[[f'{level}_disease_count',disease_column, '31', '21022']].copy()

# remove the chapters that we are less interested in
def remove_chapters_from_list(row,chapter_ranges):
    """
    remove the chapters that are not in the chapter_ranges
    """
    if str(row) != 'None':
        chapters = row
        chapters = [x for x in chapters if x in chapter_ranges]
    else:
        chapters = None
    return chapters
df_subset[disease_column] = df_subset[disease_column].apply(lambda x: remove_chapters_from_list(x, chapter_ranges))
# update the diesease count
if weight_control:
    df_subset[f'disease_count'] = [len(x)*weight if str(x) not in params.nan_str  else 0 for x,weight in zip(df[disease_column],df['weight'])]
else:
    df_subset[f'disease_count'] = [len(x) if str(x) not in params.nan_str else 0 for x in df[disease_column]]

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

# Sample data: list of transactions
dataset = [x for x in df_subset[disease_column].tolist() if str(x)!='None']

# Initialize TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)

# Convert to DataFrame for the FP-Growth function
df_fp = pd.DataFrame(te_ary, columns=te.columns_)

# Using fpgrowth to find frequent itemsets with a minimum support of 0.05
frequent_itemsets = fpgrowth(df_fp, min_support=0.01, use_colnames=True)



fig, (ax1,ax2) = plt.subplots(2,1)
plt.rcParams["figure.figsize"] = [8,13]

# bar chart of the support of the frequent itemsets
frequent_itemsets['itemsets_name'] = [', '.join([phe_cate_dict[int(m)] for m in list(x)])for x in frequent_itemsets['itemsets']]
frequent_itemsets_1 = frequent_itemsets.sort_values(by='support')
ax1.bar(frequent_itemsets_1['itemsets_name'], frequent_itemsets_1['support'], color='skyblue')
ax1.set_xlabel('Disease Chapter(s)')
ax1.set_ylabel('Proportion')
ax1.set_title(' ')
ax1.set_xticklabels(frequent_itemsets_1['itemsets_name'], rotation=90, fontsize=12)
# ax1.set_yticklabels(fontsize=14)
ax1.spines[['right', 'top']].set_visible(False)
ax1.text(0.2, 1.03, 'a. Proportion of Frequent Itemsets', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize=14)
# plt.title('Disease co-occurrence between diseases by chapter: bar view')

# network view

frequent_itemsets['itemsets'] = [[phe_cate_dict[int(m)]for m in x ] for x in frequent_itemsets['itemsets']]

G = nx.Graph()
# Adding nodes and edges based on itemsets and their supports
for _, row in frequent_itemsets.iterrows():
    items = row['itemsets']
    if len(items) == 2:
        G.add_edge(items[0], items[1], weight=row['support'], label=f"{row['support']:.2f}")

# Use a layout that might help reduce edge overlapping
pos = nx.spring_layout(G, scale=5, seed=42)

# change the pos manually

'''pos[11][1] = pos[11][1]-2
pos[11][0] = pos[11][0]+0.5
pos[9][0] = pos[9][0]+1.5
pos[10][0] = pos[10][0]-1.3
pos[10][1] = pos[10][1]-1.5
pos[19][1] = pos[11][1]
pos[19][0] = pos[19][0]-1.5'''
# Create the plot with specific dimensions


# Nodes
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800,ax=ax2)
# Edges
edges = G.edges(data=True)
edge_labels = nx.get_edge_attributes(G, 'label')

nx.draw_networkx_edges(G, pos, edgelist=edges, width=[d['weight']*12 for u, v, d in edges],ax=ax2)


nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif',ax=ax2)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=12,ax=ax2)

ax2.spines[['right', 'top','bottom','left']].set_visible(False)
# ax2.set_title('Network view of paired chapters')
ax2.text(0.2, 1, 'b. Network view of paired chapters', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize=14)

fig.suptitle(f'Disease co-occurrence between diseases by chapter {"(weighted)" if weight_control else ""}')
fig.tight_layout(rect=[0, 0.01, 1, 0.98])

#plt.show()
plt.savefig(params.current_path/f'plot/{level}/disease_co_occurrence_between_diseases_by_{level}_{record_column}_{"_weighted" if weight_control else ""}.pdf')
temp= df[['31','21022']].value_counts().reset_index()
