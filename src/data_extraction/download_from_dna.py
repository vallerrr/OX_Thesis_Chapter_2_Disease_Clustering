import os
from pathlib import Path
import requests
import re

records = '/Users/valler/Python/OX_Thesis/Chapter_2_Disease_Clustering/src/data_extraction/DNAnexus_export_urls-20240217-170536.txt'
urls = open(records).read().split('\n')

out_path = Path.cwd()/'Data/participant'

except_list = []
for url in urls[271:500]:
    print(urls.index(url))
    try:
        response = requests.get(url)
        file_name = re.findall('\d*.csv',url)[0]
        if response.status_code == 200:
            with open(f'{out_path}/{file_name}', 'wb') as file:
                file.write(response.content)
        else:
            print("Failed to retrieve data from the URL")
    except:
        except_list+=[url]
