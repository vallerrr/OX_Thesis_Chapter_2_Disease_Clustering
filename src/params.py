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
replace_dict_basics = {'Prefer not to answer': None, 'Do not know': None, 'half':0.5, '6+':6,'2-4 times a week':2, 'Once a week':1, 'Less than once a week':0.5, 'Never':0, '5-6 times a week':3, 'Once or more daily':4}
