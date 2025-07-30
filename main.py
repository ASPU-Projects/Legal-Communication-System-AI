# Import Libraries
import pandas as pd
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
import string
import requests,json
import pyterrier as pt
from Preprocessing.Preprocess import split_by_language,compute_similarity,convert2json

if not pt.started():
    pt.init()


### Load The Data From The APIs

# Get Lawyers From Lawyers API
lawyer_response = requests.get("http://osamanaser806-32078.portmap.io:32078/api/v1/ai/lawyers")
print(f"status code : {lawyer_response.status_code}")


# Get Rates From Rates API 
rate_response = requests.get("http://osamanaser806-32078.portmap.io:32078/api/v1/ai/rates")
print(f"status code : {rate_response.status_code}")


# Get Agencies From Agencies API
agency_response = requests.get("http://osamanaser806-32078.portmap.io:32078/api/v1/ai/agencies")
print(f"status code : {agency_response.status_code}")


# Get Issues From Issues API
issue_response = requests.get("http://osamanaser806-32078.portmap.io:32078/api/v1/ai/issues")
print(f"status code : {issue_response.status_code}")



### Lawyers Data Perprocessing
lawyers_response = lawyer_response.json()
lawyers_pere = pd.DataFrame(lawyers_response['lawyers'])
lawyers = lawyers_pere.rename(columns={
    "id":"lawyer_id"
}).drop(["name","email","union_number","affiliation_date","phone","rates","avatar"],axis=1)
# lawyers.head()


### Rates Data Preproceesing
rates_response = rate_response.json()
rates = pd.DataFrame(rates_response["rates"])
rates.dropna(inplace=True)
rates.drop(["id"],axis=1,inplace=True)
# rates['rating'] = rates['rating'].apply(lambda x :f"{x}star")
# rates.head()


### Agencies Data Perprcessing
agencies_response = agency_response.json()
agencies = pd.DataFrame(agencies_response["agencies"]).rename(columns={
    "id":"agency_id"
})
agencies = agencies[["agency_id","lawyer_id"]]
# agencies.head()


### Issues Data Preprocessing
issues_response = issue_response.json()
issues = pd.DataFrame(issues_response["issues"]).drop(["base_number","record_number","id","start_date","end_date","status"],axis=1)
# issues.head()


### Merge Lawyers With Rates
lawyers_with_rates = lawyers.merge(rates,on="lawyer_id")
# lawyers_with_rates.head()


### Merge Lawyers_with_rates with Agencies
lawyers_with_rates= pd.merge(agencies,lawyers_with_rates,on=["lawyer_id"],how="inner")
# lawyers_with_rates.head()


### Merge Them With Issues
lawyers_with_rates = pd.merge(issues,lawyers_with_rates,on="agency_id",how="inner")
# lawyers_with_rates.head()


### Add Data To Text Column
lawyers_with_rates["Text"] = lawyers_with_rates[['court_name','address','union_branch','estimated_cost']].astype(str).agg(" ".join,axis=1)
# lawyers_with_rates.head()


### Text Data Preprocessing
lawyers_with_rates['processed_text'] = lawyers_with_rates['Text'].apply(split_by_language)
# lawyers_with_rates.head()




def json_format(df):
    results = convert2json(df)
    return results


def recommendation(user_text):
    results = compute_similarity(lawyers_with_rates,user_text)
    # print(results.values)

    filtered_df = lawyers_pere[lawyers_pere['id'].isin(results.values)]
    # filtered_df.head()
    return json_format(filtered_df)
