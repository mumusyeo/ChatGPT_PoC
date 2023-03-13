import os
import openai
from sentence_transformers import SentenceTransformer

EMB_USE_OPENAI = os.getenv('EMB_USE_OPENAI', '1')


def _get_openai_embedding(input):
    openai.api_key = ("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")
    return openai.Embedding.create(
        input=input, engine='text-embedding-ada-002')['data'][0]['embedding']


def _get_transformer_embedding(input):
    model = SentenceTransformer("jhgan/ko-sroberta-multitask")
    embedding = model.encode(input)
    return embedding


def get_embedding(input):
    if EMB_USE_OPENAI == '1':
        return _get_openai_embedding(input)
    else:
        return _get_transformer_embedding(input)
