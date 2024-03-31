import time
import openai
import google.generativeai as palm
from google.api_core import retry
from google.generativeai.types import safety_types
palm.configure(api_key='your own plm api key')
API_KEY = 'your own openai api key'
RESOURCE_ENDPOINT = ''

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2023-03-15-preview"

import random
random.seed(42)

def request_api_chatgpt(messages, count=10):
    try:
        completion = openai.ChatCompletion.create(
            engine="ChatGPT",
            messages=messages,
            n=count,
            temperature=0.7
        )
        ret = [completion["choices"][i]["message"]["content"].strip() for i in range(len(completion["choices"]))]
        if len(ret) == 1:
            return ret[0]
        return ret
    except Exception as E:
        print(E)
        time.sleep(1)
        return request_api_chatgpt(messages)


@retry.Retry()
def request_api_palm(messages, count=8):
    model = 'models/text-bison-001'
    completion = palm.generate_text(
        model=model,
        prompt=messages,
        temperature=0.7,
        candidate_count=count,
        safety_settings=[
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_DEROGATORY,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_VIOLENCE,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },

            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_TOXICITY,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": safety_types.HarmCategory.HARM_CATEGORY_MEDICAL,
                "threshold": safety_types.HarmBlockThreshold.BLOCK_NONE,
            },
        ]
    )
    ret = [item['output'] for item in completion.candidates]
    if len(ret) == 1:
        return ret[0]
    return [item['output'] for item in completion.candidates]