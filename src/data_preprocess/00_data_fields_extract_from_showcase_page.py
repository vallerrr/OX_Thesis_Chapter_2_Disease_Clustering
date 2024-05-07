"""
this script enables extracting information from the UKB showcase webpage
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from src import params

ids = [1007,1009]

var_select_path = params.current_path / 'Data/codebook/UKB_var_select.csv'
df_main = pd.read_csv(var_select_path)

# URL of the page to scrape
for id in ids:
    url = f'https://biobank.ctsu.ox.ac.uk/showcase/label.cgi?id={id}'

    # Fetch the page
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract table headers and rows
    # id = int(re.findall(r'id=(\d*)', url)[0])
    main_div = soup.find('div', class_='main')
    cate_name = main_div.find(string=True, recursive=False).strip()

    table = soup.find('table')
    headers = [header.text for header in table.find_all('th')]
    rows = []
    for row in table.find_all('tr'):
        cols = row.find_all('td')
        if cols:
            rows.append([col.text.strip() for col in cols])

    # Create a DataFrame
    df = pd.DataFrame(rows)
    if len(df.columns)>3:
        df=df.loc[:,0:2]
    df.columns=headers
    df['cate_id'] = [id] * len(df)
    df['cate_name'] = [cate_name] * len(df)
    df.rename(columns={'Field ID': 'field_id', 'Category': 'original_cat', 'Description': 'field_name'}, inplace=True)
    df = df.reindex(['field_name', 'cate_name', 'field_id', 'cate_id', 'original_cat'], axis=1)

    # Save the DataFrame to a CSV file
    df_main = pd.concat([df_main, df], axis=0)

    df_main.to_csv(var_select_path, index=False)

    print(df_main.cate_id.unique())

Health_outcomes = df_main.loc[df_main]


# --------------------------------------------------------------------------------------------------------------------------------
# get the tree structure of some specifc field
# --------------------------------------------------------------------------------------------------------------------------------

url = 'https://biobank.ctsu.ox.ac.uk/showcase/field.cgi?id=20277'
# Fetch the page
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
# find all tables in the soup object
tables = soup.find_all('table')
table = tables[2]

# Extract table headers and rows
def extract_tree_structure(ul_tag, parent_id=''):
    data = []
    for li_tag in ul_tag.find_all('li', recursive=False):
        input_tag = li_tag.find('input')
        label_tag = li_tag.find('label')
        span_tag = li_tag.find('span', class_='tree_subtree_total')

        category = label_tag.text.strip()
        count = span_tag.text.strip() if span_tag else '-'
        item_id = input_tag['id'] if input_tag else ''

        data.append((item_id, parent_id, category, count))

        # Recursively extract child tree structure
        ul_tag = li_tag.find('ul')
        if ul_tag:
            data.extend(extract_tree_structure(ul_tag, parent_id=item_id))

    return data


# Extract tree structure data
tree_structure_data = extract_tree_structure(table.find('ul', class_='tree'))
# Create DataFrame for tree structure
tree_structure_df = pd.DataFrame(tree_structure_data, columns=['Item ID', 'Parent ID', 'Category', 'Count'])

# generate the dictionary for this tree structure
job_code_tree_dict = {}

for index, row in tree_structure_df.iterrows():
    item_id = row['Item ID']
    if len(item_id)>6:
        parent_id = item_id[0:6]
        parent_category = tree_structure_df.loc[tree_structure_df['Item ID'] == parent_id, 'Category'].values[0]
        job_code_tree_dict[row['Category']] = parent_category
