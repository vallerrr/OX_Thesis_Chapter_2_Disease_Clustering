from pathlib import Path


#  paths
current_path = Path.cwd()
recorder_path = current_path/'Data/downloaded_data/recorder'
codebook_path = current_path/'Data/codebook'
participant_path = current_path/'Data/downloaded_data/participant'
preprocessed_path = current_path/'Data/preprocessed_data'

# recoding section
cate_names = ['Cognitive function summary', 'Lifestyle', 'Socio-demographics', 'Early life', 'Health outcomes', 'Physical measures', 'Family history', 'Education and employment']
codebook_basic_columns = ['field_name', 'cate_name', 'field_id', 'cate_id', 'original_cat','ids_count', 'instance_count', 'array_count', 'ids', 'instance','array', 'file_names', 'recode_type','replace_dict','preprocessed_flag', 'note']
replace_dict_basics = {'Prefer not to answer': None, 'Do not know': None, 'half': 0.5, '6+': 6, '2-4 times a week': 2, 'Once a week': 1, 'Less than once a week': 0.5, 'Never': 0, '5-6 times a week': 3, 'Once or more daily': 4, 'Sometimes': 1.0, 'Never/rarely': 0.0, 'Usually': 2.0, 'Always': 3.0, '52,000 to 100,000': 4.0, '18,000 to 30,999': 2.0, 'Greater than 100,000': 5.0, '31,000 to 51,999': 3.0, 'Less than 18,000': 1.0,  'Less than a year': 0.5, 'Two': 2.0, 'One': 1.0, 'Three': 3.0, 'Four or more': 4.0, 'Rent - from private landlord or letting agency': 2.5, 'Own outright (by you or someone in your household)': 0.5, 'Own with a mortgage': 1.0, 'Rent - from local authority, local council, housing association': 3.0, 'Live in accommodation rent free': 2.0, 'Pay part rent and part mortgage (shared ownership)': 1.5, 'None of the above': None, 'Mobile or temporary structure (i.e. caravan)': 3.0, 'Sheltered accommodation': 4.0, 'A flat, maisonette or apartment': 2.0, 'A house or bungalow': 1.0, 'Care home': 4.0}
