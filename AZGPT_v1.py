#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 18:09:18 2023

@author: a60156666
"""

import os
import json
import requests
import re
import pandas as pd
import string
from elasticsearch import Elasticsearch

import tiktoken
import openai
from openai.embeddings_utils import distances_from_embeddings
from embedding import get_embedding

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient



service_endpoint = "https://azure-shinhan-cognitive-search-poc.search.windows.net"
key = "ZeALEYwKh3PpOH1PwY456ub7YAsKCIpYeLknAshVvOAzSeAjEaXi"

class AzureGPT:
    def __init__(self, index_name):
        self.az = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
        self.index_name = index_name

        # FIXME: remove .strip()
        self.model_engine = "text-davinci-003"
        self.model_max_tokens = 3000
        
        self.api_key = ("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")
        openai.api_key = self.api_key
        
        self.max_tokens = 2000
        self.split_max_tokens = 1000

        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def search(self, query):
        az_query = {
            "query_string": {"query": query}
        }
        results = self.az.search(search_text=query, semantic_configuration_name="default", top=1)
        return results


    def _loan_results_to_string(self, results):
        answer = ""
        for result in results:
            answer+=(result["name"]+result["details"][:300])
        return answer

    def _split_into_many(self, text):
        sentences = []
        for sentence in re.split(r'[{}]'.format(string.punctuation), text):
            sentence = sentence.strip()
            if sentence and (any(char.isalpha() for char in sentence) or any(char.isdigit() for char in sentence)):
                sentences.append(sentence)

        n_tokens = [len(self.tokenizer.encode(" " + sentence))
                    for sentence in sentences]

        chunks = []
        tokens_so_far = 0
        chunk = []

        for sentence, token in zip(sentences, n_tokens):
            if tokens_so_far + token > self.split_max_tokens and chunk:
                chunks.append(". ".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0

            if token > self.split_max_tokens:
                continue

            chunk.append(sentence)
            tokens_so_far += token + 1

        if chunk:
            chunks.append(". ".join(chunk) + ".")

        return chunks


    def _create_emb_dict_list(self, long_text):
        shortened = self._split_into_many(long_text)

        embeddings_dict_list = []

        for text in shortened:
            n_tokens = len(self.tokenizer.encode(text))
            embeddings = get_embedding(input=text)
            embeddings_dict = {}
            embeddings_dict["text"] = text
            embeddings_dict["n_tokens"] = n_tokens
            embeddings_dict["embeddings"] = embeddings
            embeddings_dict_list.append(embeddings_dict)

        return embeddings_dict_list

    def _create_context(self, question, df):
        """
        Create a context for a question by finding the most similar context from the dataframe
        """

        # Get the embeddings for the question
        q_embeddings = get_embedding(input=question)

        # Get the distances from the embeddings
        df['distances'] = distances_from_embeddings(
            q_embeddings, df['embeddings'].values, distance_metric='cosine')

        returns = []
        cur_len = 0

        # Sort by distance and add the text to the context until the context is too long
        for i, row in df.sort_values('distances', ascending=True).iterrows():
            # Add the length of the text to the current length
            cur_len += row['n_tokens'] + 4

            # If the context is too long, break
            if cur_len > self.max_tokens:
                break

            returns.append(row["text"])

        # Return the context and the length of the context
        return "\n\n###\n\n".join(returns), cur_len

    def _gpt_api_call(self, query, input_token_len, context):
        body = {
            "model": self.model_engine,
            "prompt": "아래 문서를 기반으로 합니다.\"\n\nC문서: " + context + "\n\n---\n\n이 질문에 대한 간결한 답변을 제공하십시오: "+query,
            "max_tokens": self.model_max_tokens - input_token_len,
            "n": 1,
            "temperature": 0.5,
            "stream": True,
        }

        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}

        resp = requests.post("https://api.openai.com/v1/completions",
                             headers=headers,
                             data=json.dumps(body),
                             stream=True)
        return resp

    def gpt_answer(self, query, az_results=None, text_results=None):
        # Generate summaries for each search result
        if text_results:
            input_token_len = len(self.tokenizer.encode(text_results))
            if input_token_len < self.max_tokens:
                context = text_results
            else:
                emb_dict_list = self._create_emb_dict_list(text_results)
                df = pd.DataFrame(columns=["text", "n_tokens", "embeddings"])
                for emb_dict in emb_dict_list:
                    df = df.append(emb_dict, ignore_index=True)

                context, input_token_len = self._create_context(
                    question=query,
                    df=df)
                
        elif az_results:
            result_json_str = self._loan_results_to_string(az_results)
            if not result_json_str:
                result_json_str = "No results found"

            input_token_len = len(self.tokenizer.encode(result_json_str))
            if input_token_len < self.max_tokens:
                context = result_json_str
            else:
                df = pd.DataFrame(columns=["text", "n_tokens", "embeddings"])

                for az_result in az_results:
                    embeddings_dict_list = az_result
                    for embeddings_dict in embeddings_dict_list:
                        df = df.append(embeddings_dict, ignore_index=True)

                context, input_token_len = self._create_context(
                    question=query,
                    df=df)
                
        else:
            assert False, "Must provide either az_results or text_results"

        return self._gpt_api_call(query, input_token_len, context)


# Example usage
if __name__ == "__main__":
    azgpt = AzureGPT("products-1")
    query = "비과세 대상 대출 상품 중에서 금리가 가장 낮은 상품을 알려줘"
    res = azgpt.search(query=query)
    res_str = azgpt._loan_results_to_string(res)

    res = azgpt.gpt_answer(query=query, az_results=res)
    print(res.text)

    res = azgpt.gpt_answer(query=query, text_results=res_str)
    
    json_to_dict = []
    body = res.text.split("\n\n")
    for body_single in body:
        try:
            json_to_dict.append(json.loads(body_single[6:]))
        except:
            print(body_single)
            
    final_string = ""
    for j in json_to_dict:
        try:
            print(j)
            final_string += j['choices'][0]['text']
        except:
            print(j)
            
    print(final_string)