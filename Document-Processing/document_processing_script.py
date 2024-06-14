import requests
from typing import (
    List,
)
import re
import json
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import time


start_time = time.time()

load_dotenv()

COS_INSTANCE_ID=os.getenv("COS_INSTANCE_ID")
COS_ENDPOINT_URL=os.getenv("COS_ENDPOINT_URL")
COS_BUCKET_NAME=os.getenv("COS_BUCKET_NAME")

QUERY_LLM_ENDPOINT_URL=os.getenv("QUERY_LLM_ENDPOINT_URL")
INDEX_NAME=os.getenv("INDEX_NAME")
WXD_REQUEST_BODY_FILE = open('config/wxd_request_body.json')
WXD_REQUEST_BODY = json.load(WXD_REQUEST_BODY_FILE)
WXD_REQUEST_BODY_FILE.close()
DOCUMENT_PROCESSING_CONFIG_FILE = open('config/document_processing_config.json')
DOCUMENT_PROCESSING_CONFIG_FILE_JSON = json.load(DOCUMENT_PROCESSING_CONFIG_FILE)
QUESTIONS_LIST = DOCUMENT_PROCESSING_CONFIG_FILE_JSON['questions']
FILE_NAMES = DOCUMENT_PROCESSING_CONFIG_FILE_JSON['file_names']
MARKER_FILE = DOCUMENT_PROCESSING_CONFIG_FILE_JSON['marker_file']
NEW_FILES_ONLY = DOCUMENT_PROCESSING_CONFIG_FILE_JSON['new_files_only']
DOCUMENT_PROCESSING_CONFIG_FILE.close()

API_KEY = os.getenv("IBM_CLOUD_API_KEY")
WX_URL=os.getenv("WX_URL")
WX_PROJECT_ID=os.getenv("WX_PROJECT_ID")
ANSWER_PROCESSING_MODEL_ID=os.getenv("ANSWER_PROCESSING_MODEL_ID")

# retrieves temporary bearer token from your IBM Cloud apikey
def get_bearer_token(apikey):
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }

    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": apikey
    }

    response = requests.post(url, headers=headers, data=data, )

    return "Bearer " + response.json()['access_token']

# lists all files in your cloud object storage instance
def list_files() -> List[str]:
    headers = {
            "ibm-service-instance-id": COS_INSTANCE_ID,
            "Authorization": get_bearer_token(API_KEY),
        }
    params = {
        "list-type" : 2,
    }
    response = requests.request("GET", COS_ENDPOINT_URL + COS_BUCKET_NAME, params=params, headers=headers)
    data = response.text
    file_names = re.findall(r"<Key>(.*?)</Key>", data)

    isTruncated = re.findall(r"<IsTruncated>(.*?)</IsTruncated>", data)[0]

    while isTruncated == 'true':
        continuation_token = re.findall(r"<NextContinuationToken>(.*?)</NextContinuationToken>", data)
        params = {
            "list-type": 2,
            "continuation-token": continuation_token[0]
        }
        response = requests.request("GET", COS_ENDPOINT_URL + COS_BUCKET_NAME, params=params, headers=headers)
        data = response.text
        file_names += re.findall(r"<Key>(.*?)</Key>", data)
        isTruncated = re.findall(r"<IsTruncated>(.*?)</IsTruncated>", data)[0]

    return file_names

# lists all new files in your cloud object storage instance
def list_new_files() -> List[str]:
    headers = {
            "ibm-service-instance-id": COS_INSTANCE_ID,
            "Authorization": get_bearer_token(API_KEY),
        }
    params = {
        "list-type" : 2,
        "start-after": MARKER_FILE
    }
    response = requests.request("GET", COS_ENDPOINT_URL + COS_BUCKET_NAME, params=params, headers=headers)
    data = response.text
    file_names = re.findall(r"<Key>(.*?)</Key>", data)

    isTruncated = re.findall(r"<IsTruncated>(.*?)</IsTruncated>", data)[0]

    while isTruncated == 'true':
        continuation_token = re.findall(r"<NextContinuationToken>(.*?)</NextContinuationToken>", data)
        params = {
            "list-type": 2,
            "continuation-token": continuation_token[0],
        }
        response = requests.request("GET", COS_ENDPOINT_URL + COS_BUCKET_NAME, params=params, headers=headers)
        data = response.text
        file_names += re.findall(r"<Key>(.*?)</Key>", data)
        isTruncated = re.findall(r"<IsTruncated>(.*?)</IsTruncated>", data)[0]

    return file_names

# check if we specified file names we want to test first
if not FILE_NAMES:
    if NEW_FILES_ONLY:
        FILE_NAMES = list_new_files()
    else:
        FILE_NAMES = list_files()
    if FILE_NAMES:
        MARKER_FILE = FILE_NAMES[-1]

with open('config/document_processing_config.json', 'r') as file:
    data = json.load(file)

data['marker_file'] = MARKER_FILE

with open('config/document_processing_config.json', 'w') as file:
    json.dump(data, file)

# print all your filenames
print("\n\nFile names:\n")
print(FILE_NAMES)
print("\n\n")

answers_list = []

# makes a watsonx Discovery query on a file
def wx_discovery_call(file_name, query):
    WXD_REQUEST_BODY['question'] = query
    WXD_REQUEST_BODY['es_index_name'] = INDEX_NAME
    WXD_REQUEST_BODY['filters']['file_name'] = file_name
    response = requests.request("POST", QUERY_LLM_ENDPOINT_URL, json=WXD_REQUEST_BODY)
    return response.json()['llm_response']


# makes a watsonx Discovery query for all files
print("watsonx Discovery answers:\n")
for file_name in FILE_NAMES:
    response_object = {}
    response_object['fileName'] = file_name
    response_object['answers'] = ""
    for question in QUESTIONS_LIST:
        response_object['answers'] += wx_discovery_call(file_name, question) + "\n"
    print(response_object)
    print("\n\n")
    answers_list.append(response_object)




# instantiate watsonx.ai
wml_credentials = {
    "url": WX_URL,
    "apikey": API_KEY
}
project_id = WX_PROJECT_ID

generate_parameters = {
    "decoding_method": "greedy",
    "min_new_tokens": 1,
    "max_new_tokens": 100,
    "repetition_penalty": 1,
    "stop_sequences": ['}']
}

llm_model_id = ANSWER_PROCESSING_MODEL_ID

llm_model = WatsonxLLM(apikey=wml_credentials['apikey'],
                         url=wml_credentials['url'],
                         project_id=project_id,
                         model_id=llm_model_id,
                         params=generate_parameters)


# query llm on a response object
def llm_entity_extraction(response_object):

    query = response_object['answers']

    template_file = open("config/answer_processing_instructions.txt")
    template = template_file.read()
    template_file.close()

    prompt = PromptTemplate(input_variables=["query"],template=template)
    llm_chain = prompt | llm_model
    prompt_results = llm_chain.invoke(query)
    return prompt_results


extracted_answers = []

# queries the llm to summarize all answers for all files

for answer in answers_list:
    extracted_answers.append(llm_entity_extraction(answer).strip())


print('Extracted json objects:\n')
print(extracted_answers)
print('\n\n')

end_time = time.time()

print(f"Execution time:  {round(end_time - start_time,2)}")