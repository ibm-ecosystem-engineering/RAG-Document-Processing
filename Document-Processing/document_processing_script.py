import requests
from typing import (
    List,
)
import re
import json
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain.prompts import PromptTemplate
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
QUESTIONS_LIST_FILE = open('config/questions_list.json')
QUESTIONS_LIST = json.load(QUESTIONS_LIST_FILE)['questions']
QUESTIONS_LIST_FILE.close()

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

file_names = list_files()

# print all your filenames
print("\n\nFile names:\n")
print(file_names)
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
for file_name in file_names:
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
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.STOP_SEQUENCES: ['}'],
    GenParams.REPETITION_PENALTY: 1,
}
llm_model_id = ANSWER_PROCESSING_MODEL_ID
model = Model(
model_id=llm_model_id,
params=generate_parameters,
credentials=wml_credentials,
project_id=project_id
)
llm_model = WatsonxLLM(model=model)

# query llm on a response object
def llm_entity_extraction(response_object):

    query = response_object['answers']

    template_file = open("config/answer_processing_instructions.txt")
    template = template_file.read()
    template_file.close()

    prompt = PromptTemplate(input_variables=["query"],template=template)
    llm_chain = LLMChain(prompt=prompt, llm=llm_model)
    prompt_results = llm_chain.run(query)
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