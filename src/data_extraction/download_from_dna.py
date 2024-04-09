import os
from pathlib import Path
import requests
import re

records = '/Users/valler/Python/OX_Thesis/Chapter_2_Disease_Clustering/src/data_extraction/DNAnexus_export_urls-20240402-220628.txt'
urls = open(records).read().split('\n')

out_path = Path.cwd()/'Data/downloaded_data/participant'
x = urls[0]
files = ['832', '830', '828', '836', '829', '833', '834', '837', '831', '835','826', '824', '825', '823','828','815']
pioritiy_list = [x for x in urls if x[-7:-4] in files]

for url in pioritiy_list:
    urls.remove(url)

except_list = []
for url in urls:
    print(urls.index(url))
    try:
        response = requests.get(url)
        file_name = re.findall('\d*.csv',url)[0]
        if response.status_code == 200:
            with open(f'{out_path}/{file_name}', 'wb') as file:
                file.write(response.content)
        else:
            print("Failed to retrieve data from the URL")
    except Exception as e:
        except_list+=[url]


# check the data being downloaded
download_num_lst = range(0,980)
file_lst = [int(x.replace('.csv','')) for x in os.listdir(out_path) if x.endswith('.csv')]
non_covered = list(set(download_num_lst)-set(file_lst))
non_covered.sort()
print(non_covered)
