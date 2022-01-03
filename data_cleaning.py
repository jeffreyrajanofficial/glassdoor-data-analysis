"""
Data cleaning and Feature Engineering
"""
import pandas as pd

df = pd.read_csv("glassdoor_raw.csv")
#Parsing the salary info
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)

salary_range = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
clean_salary_range = salary_range.apply(lambda x: x.lower().replace('k', '').replace('$', '').replace('per hour', '').replace('employer provided salary:', ''))

df['min_salary'] = clean_salary_range.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = clean_salary_range.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary+df.max_salary)/2

df['company_name_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis=1)
df['company_age'] = df.Founded.apply(lambda x: x if x<1 else 2022 - x)

df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1])
df['job_state'] = df.job_state.apply(lambda x: x.strip().strip() if x.strip().lower() != 'los angeles' else 'CA')
df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis=1)


df['python'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df['R'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() or 'rstudio' in x.lower() else 0)
df['spark'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
df['aws'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
df['excel'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)

print(df.excel.value_counts())

def title_simplifier(title):
    if 'data scientist' in title.lower():
        return 'data scientist'
    elif 'data engineer' in title.lower():
        return 'data engineer'
    elif 'analyst' in title.lower():
        return 'analyst'
    elif 'machine learning' in title.lower():
        return 'mle'
    elif 'manager' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    else:
            return 'na'
        
def seniority(title):
    if 'sr' in title.lower() or 'senior' in title.lower():
        return 'senior'
    elif 'jr' in title.lower() or 'jr.' in title.lower() or 'junior' in title.lower():
        return'jr'
    else:
        'na'

df['job_simp'] = df['Job Title'].apply(title_simplifier)
df['seniority'] = df['Job Title'].apply(seniority)
#Job Description Length
df['desc_len'] = df['Job Description'].apply(lambda x: len(x))
#Compititor Count
df['num_comp'] = df['Competitors'].apply(lambda x: len(x.split(',')) if x != -1 else 0)
df['min_salary'] = df.apply(lambda x: x.min_salary*2 if x.hourly == 1 else x.min_salary, axis=1)
df['max_salary'] = df.apply(lambda x: x.max_salary*2 if x.hourly == 1 else x.max_salary, axis=1)

print(df.columns)

# df.to_csv('glassdoor_data_cleaned.csv', index=False)


