"""
# Created by valler at 31/05/2024
Feature: script to generate data descrption for the data

"""

import pandas as pd
from src import params
from src.data_preprocess import utils
from collections import Counter

from matplotlib import pyplot as plt
df_codebook_final = pd.read_csv(params.codebook_path / 'UKB_preprocess_codebook_wave_0_final.csv')
df = pd.read_pickle(params.final_data_path / 'UKB_wave_0_final_standardised.pkl')

# ======================================================================================================================
# 0. combine disease information from the specific time window
# ======================================================================================================================
record_column = 'all_icd'
access_date_column = '53'
HES_ICD_ids = params.HES_ICD_ids
# read the single record file and merge the date columns to the single record file

dates_col,df_single_record = utils.import_df_single_record(record_column)
df[access_date_column] = pd.to_datetime(df[access_date_column], errors='coerce', format='%Y-%m-%d')
df_single_record = pd.merge(df_single_record, df[[access_date_column, 'eid']], how='left', left_on='eid', right_on='eid')

def find_first_date(row):
    """
    find the first date that is out of the window
    :param row:
    :return: the index of the date, or -1 if all dates are not in the window/no diease
    """
    if row['all_icd_uniq_count'] > 0:
        # access_date = row[access_date_column]
        conditions = [None if pd.isnull(x) else True if (x > row[access_date_column]) else False for x in row[dates_col]]

        #print(row[access_date_column],conditions)
        #print(row[dates_col].values)
        # record the index of the first True
        if True in conditions:
            first_true_index = conditions.index(True)
        else:
            first_true_index = -1
    else:
        first_true_index = -1
    return first_true_index

df_single_record['index_of_first_disease_out_window'] = df_single_record.apply(find_first_date, axis=1)

for column in ['all_icd_first_3', 'icd_parent_coding', 'icd_chapter_coding']:
    df_single_record[f'diseases_within_window_{column}'] = df_single_record.apply(lambda x: x[column][0:x['index_of_first_disease_out_window']] if x['index_of_first_disease_out_window']!=-1 else None, axis=1)

# merge the new three columns back to the original df
df = pd.merge(df, df_single_record[['eid', 'diseases_within_window_all_icd_first_3', 'diseases_within_window_icd_parent_coding', 'diseases_within_window_icd_chapter_coding']], how='left', left_on='eid', right_on='eid')

for column in ['all_icd_first_3', 'icd_parent_coding', 'icd_chapter_coding']:
    df[f'diseases_within_window_{column}'] = [x if str(x)!='[]' else None for x in  df[f'diseases_within_window_{column}']]

df_single_record.to_csv(params.intermediate_path / f'{record_column}_complete.csv')
df.to_pickle(params.final_data_path / 'UKB_wave_0_final_standardised_with_disease.pkl')


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

df_descriptive_table['variable_missing_values']= df_descriptive_table.apply(find_missing_count, axis=1)
df_descriptive_table['variable_missing_values_percentage'] = [x/len(df) for x in df_descriptive_table['variable_missing_values']]
df_descriptive_table.to_csv(params.final_data_path/'descriptive_table'/'descriptive_table_wave_0.csv',index=False)

# concat disease information to the non standardised dataframe
df_non_standardised = pd.read_pickle(params.final_data_path / 'UKB_wave_0_final_non_standardised.pkl')
df_non_standardised = pd.merge(df_non_standardised, df[['eid', 'diseases_within_window_all_icd_first_3', 'diseases_within_window_icd_parent_coding', 'diseases_within_window_icd_chapter_coding']], how='left', left_on='eid', right_on='eid')
df_non_standardised.to_pickle(params.final_data_path / 'UKB_wave_0_final_non_standardised.pkl')

# ======================================================================================================================
# 2. generate the disease description
# ======================================================================================================================

df = pd.read_pickle(params.final_data_path/ 'UKB_wave_0_final_non_standardised.pkl')

# average disease count by age and gender
import matplotlib.pyplot as plt
df['disease_count'] = [len(x) if str(x) != 'None' else 0 for x in df[f'diseases_within_window_all_icd_first_3']]


# ----------------------------------------------------------
# 2.1 plot: average_disease_count_at_recruitment_all_disease
# ----------------------------------------------------------

df_to_plot = df[['disease_count', '31', '21022']].copy()
# 31: gender
# 21022: age at recruitment
# display the disease count by gender and age
df_to_plot = df_to_plot.groupby(['31', '21022']).agg(cases_count=('disease_count', 'count'), disease_count=('disease_count', 'mean')).reset_index()
# set 31 as categorical variable
df_to_plot['31'] = df_to_plot['31'].astype('category')
df_to_plot['31'] = df_to_plot['31'].cat.rename_categories(['Female', 'Male'])

fig, ax = plt.subplots()
# Colors to use for the two categories (adjust as needed)
colors = ['blue', 'orange']
df_to_plot = df_to_plot.loc[df_to_plot['cases_count'] > 100,]
# Plot each category with its own color and label
for i, category in enumerate(df_to_plot['31'].cat.categories):
    df_subset = df_to_plot[df_to_plot['31'] == category]
    ax.scatter(df_subset['21022'], df_subset['disease_count'], color=colors[i], label=category)

# Add a legend
ax.legend(title='Gender')
plt.title('Average disease count at recruitment')
plt.xlabel('Age at recruitment')
plt.ylabel('Average disease count')

plt.savefig(params.current_path/'plot/average_disease_count_at_recruitment_all_disease.pdf')


# ----------------------------------------------------------
# 2.2 plot: Diseases chapter count by age and gender
# ----------------------------------------------------------



def count_chapter(row):
    """
    count the chapter of the diseases for each row in the dataframe
    """
    if str(row) != 'None':
        chapters = row['diseases_within_window_icd_chapter_coding']
        count_dict = Counter(chapters)
    else:
        count_dict = None
    return count_dict


df_to_plot = df[['diseases_within_window_icd_chapter_coding', '31', '21022']].copy()
# separate diseases by the chapter name (each chapter will have a separate column)
df_to_plot['chapter_count'] = df_to_plot.apply(count_chapter, axis=1)

for column in range(1, 23):
    df_to_plot[f'chapter_{column}'] = [x[column] if len(x)>0 else None for x in df_to_plot['chapter_count']]

df_plot_2 = df_to_plot.pivot_table(index=['21022', '31'],  values=[f'chapter_{x}' for x in range(1,23)], aggfunc='sum').reset_index()
# remove sparse data points that have age below 40 or over 70 (same as the last plot)
df_plot_2.drop(df_plot_2.loc[(df_plot_2['21022'] < 40) | (df_plot_2['21022'] > 70)].index, inplace=True)

# set 31 as categorical variable
df_plot_2['31'] = df_plot_2['31'].astype('category')
df_plot_2['31'] = df_plot_2['31'].cat.rename_categories(['Female', 'Male'])

# plot
fig, (ax1, ax2) = plt.subplots(1, 2)

plt.rcParams["figure.figsize"] = [16,9]
# Create a subset of the dataframe with only the columns to be plotted
gender = 'Male'
df_subset = df_plot_2[df_plot_2['31'] == gender][['21022'] + [f'chapter_{i}' for i in range(1, 23)]]
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
df_subset = df_plot_2[df_plot_2['31'] == gender][['21022'] + [f'chapter_{i}' for i in range(1, 23)]]
# Set '21022' as the index to be used as x-axis
df_subset.set_index('21022', inplace=True)

# Plot the stacked bar chart
df_subset.plot(kind='bar', stacked=True,  ax=ax2)
ax2.set_title(gender)
ax2.set_xlabel('Age at recruitment')
ax2.set_ylabel('Count of Disease Chapters')

handles, labels = ax2.get_legend_handles_labels() # reverse the order of legend
ax2.legend(reversed(handles), reversed(labels), title='Chapter', title_fontsize='large', fontsize='small', loc='center left', bbox_to_anchor=(1.0, 0.5))

fig.suptitle('Diseases chapter count by age and gender')
plt.savefig(params.current_path/'plot/diseases_chapter_count_by_age_and_gender.pdf')
#plt.show()

