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
