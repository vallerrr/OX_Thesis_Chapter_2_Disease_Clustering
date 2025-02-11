"""
# Created by valler at 13/09/2024
Feature: 

"""

from src import params
from src.data_preprocess import utils
from collections import Counter
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import ast


# =================================================================================================
# 0. prepare the data
# =================================================================================================

record_column = params.disease_record_column
access_date_column = '53'
HES_ICD_ids = params.HES_ICD_ids
level = 'chronic_cate'
weight_control = False

# control zone
chapter_ranges = [x for x in range(1, 16)]+[19] if 'chapter' in level else [x for x in range(1, 18)]
#disease_column = params.disease_columns_dict[level]
disease_columns = ['diseases_within_window_phecode_selected_chronic_first_occ','diseases_within_window_phecode_selected_category_chronic_first_occ']
dates_col, df_single_record = utils.import_df_single_record(record_column)


phe_cate_dict = {y:x for x,y in params.phe_cate_dict.items()}
df = pd.read_pickle(params.final_data_path/ 'UKB_wave_0_final_non_standardised.pkl')

for disease_column in disease_columns:
    if disease_column not in df.columns.tolist():
            df = pd.merge(df, df_single_record[['eid', disease_column]], how='left', left_on='eid', right_on='eid')
        # merge weight to the dataframe
    if weight_control:
        df = pd.merge(df, df_single_record[['eid', 'weight']], how='left', left_on='eid', right_on='eid')
    #    average disease count by age and gender
        df[f'{level}_disease_count'] = [len(x)*weight if str(x) not in params.nan_str else 0 for x,weight in zip(df[disease_column],df['weight'])]
    else:
        df[disease_column] = [ast.literal_eval(x) if str(x) not in params.nan_str else None for x in df[disease_column]]
        df[f'{level}_disease_count'] = [len(x) if str(x) not in params.nan_str else 0 for x in df[disease_column]]


# =================================================================================================
# 1. disease network by gender
# =================================================================================================

df_subset = df[[f'{level}_disease_count',disease_columns[0], '31', '21022']].copy()

gender_dict = {0:'Female',1:'Male'}

# Step 3: Visualize the network
fig, axs = plt.subplots(2,1,figsize=(12,12))

ax0 = axs[0]
ax1 = axs[1]

for gender in [1,0]:
    dataset = [x for x in df_subset.loc[df_subset['31']==gender,disease_columns[0]].tolist() if str(x)!='None']

    transaction_frequencies = [Counter(transaction) for transaction in dataset]

    # Get all unique items
    all_items = sorted(set(item for transaction in dataset for item in transaction))

    # Create a frequency matrix where rows are transactions and columns are items
    frequency_matrix = pd.DataFrame([{item: count[item] for item in all_items} for count in transaction_frequencies],
                                    columns=all_items).fillna(0)


    co_occurrence_matrix = np.dot(frequency_matrix.T, frequency_matrix)

    # Set diagonal to zero to remove self-loops (optional, if you don't want self-links)
    np.fill_diagonal(co_occurrence_matrix, 0)

    # Convert to DataFrame for easier interpretation
    co_occurrence_df = pd.DataFrame(co_occurrence_matrix, index=frequency_matrix.columns, columns=frequency_matrix.columns)

    threshold = 0.01
    # replace the value below the threshold with 0
    co_occurrence_df = co_occurrence_df.applymap(lambda x: 0 if x<threshold*sum(sum(co_occurrence_df.values))else x)
    co_occurrence_df.columns = [df_phe_db.loc[df_phe_db['phecode']==int(x),'phenotype'].values[0] for x in co_occurrence_df.columns]

    # Step 2: Create a network graph using NetworkX
    G = nx.Graph()

    # Add nodes (items)
    for item in co_occurrence_df.columns:
        if sum(co_occurrence_df[item]) > 0:
            #print(item)
            G.add_node(item)

    edge_set = set()
    # Add edges (co-occurrences) if the co-occurrence value is greater than 0
    for i, item1 in enumerate(co_occurrence_df.columns):
        for j, item2 in enumerate(co_occurrence_df.columns):
            if co_occurrence_df.iloc[i, j] > 0:
                edge_set.add((item1, item2))
                if (item2, item1) not in edge_set:
                    #print(item1, item2, co_occurrence_df.iloc[i, j])
                    G.add_edge(item1, item2, weight=round(co_occurrence_df.iloc[i, j]/sum(sum(co_occurrence_df.values))*100,1))

    # remove edges with weight less than 0.02
    # G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if d['weight'] < 0.01])



    if gender == 0:
        labels = {'Hypothyroidism': 'Hypothyroidism',
                  'Disorders of lipoid metabolism': 'Disorders of lipoid metabolism',
                  'Hypertension': 'Hypertension',
                  'Ischemic Heart Disease': 'Ischemic Heart Disease',
                  'Other symptoms/disorders or the urinary system': 'Other symptoms/disorders \nor the urinary system',
                  'Noninflammatory female genital disorders': 'Noninflammatory female \ngenital disorders',
                  'Symptoms involving female genital tract': 'Symptoms involving \nfemale genital tract',
                  'Disorders of menstruation and other abnormal bleeding from female genital tract': 'Disorders of menstruation and other \nabnormal bleeding from female genital tract',
                  'Menopausal and postmenopausal disorders': 'Menopausal and postmenopausal disorders'}
        pos = nx.shell_layout(G)
        pos["Symptoms involving female genital tract"][1] = pos["Symptoms involving female genital tract"][1]-0.7
        pos["Symptoms involving female genital tract"][0] = pos["Symptoms involving female genital tract"][0]-0.4
        pos['Other symptoms/disorders or the urinary system'][0] = pos['Other symptoms/disorders or the urinary system'][0]-0.3
        pos['Disorders of lipoid metabolism'][0] = pos['Disorders of lipoid metabolism'][0]+1
        pos['Disorders of lipoid metabolism'][1] = pos['Disorders of lipoid metabolism'][1]+0.1
        pos['Noninflammatory female genital disorders'][0] = pos['Noninflammatory female genital disorders'][0]-1.1
        ax = ax0
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=11, ax=ax,labels=labels)

    else:
        ax = ax1
        pos = nx.spring_layout(G, seed=1234)
        pos['Ischemic Heart Disease'][0] = pos['Ischemic Heart Disease'][0] + 0.15
        pos['Ischemic Heart Disease'][1] = pos['Ischemic Heart Disease'][1] + 0.14
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=11, ax=ax)

    edge_labels = nx.get_edge_attributes(G, 'weight')

    edges = G.edges(data=True)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=[d['weight']/5 for u, v, d in edges],ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='grey', font_size=13,ax=ax)
    # remove edges with weight less than 0.02
    ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)

    ax.set_title('{}.Disease Co-occurrence Network View {}'.format('A' if gender ==0 else "B",gender_dict[gender]),fontsize=14,fontweight='bold', loc='left')

    #
fig.tight_layout()
#plt.show()
plt.savefig(params.data_path.parent.parent/f'plot/abstract/diseases_network_view.pdf')

# =================================================================================================
# 2. by education level
# =================================================================================================
df_subset = df[[f'{level}_disease_count',disease_columns[0], '21022','31','10722']].copy()
df_subset[f'10722'].replace({3:1,2:1, list(df_subset['10722'].unique())[1]:1, 4:2,1:2,5:3},inplace=True)
df_subset=df_subset[(df_subset['21022']>40) & (df_subset['21022']<70)]

df_subset.drop(df_subset.loc[df_subset['chronic_cate_disease_count']==0].index, inplace=True)


# Separate the dataframe by the values in column '31'
df_0 = df_subset[df_subset['31'] == 0]
df_1 = df_subset[df_subset['31'] == 1]

# Plotting
import seaborn as sns

# Create a matplotlib figure with two subplots

# Create lmplot for female (df_0)
lm_female = sns.lmplot(x='21022', y='chronic_cate_disease_count', data=df_0, hue='10722', scatter=False)
lm_female.fig.suptitle('Female')
lm_female.set_xlabels('Age')
lm_female.set_ylabels('Chronic Disease Count')
legend = lm_female._legend
legend.set_title('Education')  # Set a custom title for the legend

plt.savefig(params.data_path.parent.parent/f'plot/abstract/diseases_by_education_female.pdf')


# Create lmplot for female (df_0)
lm_female = sns.lmplot(x='21022', y='chronic_cate_disease_count', data=df_1, hue='10722', scatter=False)
lm_female.fig.suptitle('Male')
lm_female.set_xlabels('Age')
lm_female.set_ylabels('Chronic Disease Count')
lm_female._legend.remove()

plt.savefig(params.data_path.parent.parent/f'plot/abstract/diseases_by_education_male.pdf')

