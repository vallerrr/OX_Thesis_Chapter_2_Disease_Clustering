"""
# Created by valler at 13/09/2024
Feature: 

"""



"""
# in 03_01 around line 164
# standardise to the UK population
df_phe_chronic_with_age = pd.DataFrame(columns=['age', 'gender','phe','count'])
age_range = range(40, 71)
for age in age_range:
    for gender in [0,1]:

        c1 = df_single_record['31'] == gender
        c2 = df_single_record['21022'] == age
        df_single_record_frac = df_single_record.loc[c1&c2, f'diseases_within_window_phecode{chronic_control}']
        phes = [x for x in df_single_record_frac.explode().explode() if pd.notnull(x)]
        for phe in phe_db:
            phe_count = phes.count(phe)
            df_phe_chronic_with_age.loc[len(df_phe_chronic_with_age),] = [age,gender,phe,phe_count]


df_single_pop = df_single_record[['21022','31']].groupby(['21022','31']).size().reset_index(name='count')
df_single_pop=df_single_pop.loc[df_single_pop['21022'].isin(list(age_range)),] # only keep the age range from 40 to 70

df_pop = df_pop.loc[df_pop['laname23']=='GREAT BRITAIN',['sex','age','population_2011']]  # as the UK Biobank is only running in the great britain not in the north ireland and other regions
df_pop['population_2011']=[int(x.replace(',','')) for x in df_pop['population_2011']]
df_pop['sex']=[0 if x=='F' else 1 for x in df_pop['sex']]
total_pop = df_pop['population_2011'].sum()
df_pop['proportion'] = df_pop['population_2011'] / total_pop

df_single_pop = df_single_pop.merge(df_pop[['age','sex','proportion']], how='inner', left_on=['21022', '31'], right_on=['age', 'sex'])
df_single_pop.drop(columns =['age','sex'],inplace=True)
total_pop_single = df_single_pop['count'].sum()
df_single_pop['standardized_population'] = df_single_pop['proportion'] * total_pop_single
# rounding
df_single_pop['standardized_population'] = df_single_pop['standardized_population'].apply(lambda x: round(x))
# weights for future use

total_pop = df_single_pop['count'].sum()
df_single_pop['sample_proportion'] = df_single_pop['count']/total_pop
# df_single_pop['weight']=df_single_pop['standardized_population']/df_single_pop
df_single_pop['weight'] = df_single_pop['proportion']/df_single_pop['sample_proportion']
df_single_pop.to_csv(params.intermediate_path / 'pop_weights.csv', index=False)

# match it to the df_single_record
df_single_record['weight'] = df_single_record.apply(lambda x: df_single_pop.loc[(df_single_pop['21022']== x['21022']) &(df_single_pop['31']==x['31']),'weight'].values[0] if len(df_single_pop.loc[(df_single_pop['21022']== x['21022']) &(df_single_pop['31']==x['31']),'weight'].values)>0 else 0,axis=1)



# now generate the df_phe_chronic_with_age
total_pop_phe = df_phe_chronic_with_age['count'].sum()
df_phe_chronic_with_age['standardised_count'] = df_phe_chronic_with_age.apply(lambda x: total_pop_phe*df_single_pop.loc[(df_single_pop['21022']==x['age'])&(df_single_pop['31']==x['gender']),'proportion'].values[0],axis=1)
for age in age_range:
    for gender in [0,1]:
        rows = df_phe_chronic_with_age.loc[(df_phe_chronic_with_age['age']==age )& (df_phe_chronic_with_age['gender']==gender) ,]
        age_gender_specific_phe_total = rows['count'].sum()
        age_gender_specific_total = total_pop_phe*df_single_pop.loc[(df_single_pop['21022']==age)&(df_single_pop['31']==gender),'proportion'].values[0]
        df_phe_chronic_with_age.loc[rows.index, 'standardised_count']=[x/age_gender_specific_phe_total*age_gender_specific_total for x in rows['count']]

# save the phe code database that are chronic
df_phe_chronic_with_age.to_csv(params.intermediate_path / f'{record_column}_phecode{chronic_control}.csv', index=False)

# generate df_phe based on the df_phe_chronic_with_age
#if chronic_control == '_chronic': # 298 different diseases
#    df_phe_chronic = df_phe_chronic_with_age.groupby('phe',as_index=False).sum()[['phe','count']]
#    df_phe_chronic.rename(columns={'count':'prev','phe':'phecode'},inplace=True)
"""
