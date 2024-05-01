from functools import cache

import os
import requests

import psycopg
from dotenv import load_dotenv,find_dotenv

from tqdm import tqdm


from torch import Tensor
from numpy import ndarray
import torch
from sentence_transformers import SentenceTransformer

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

model = SentenceTransformer('intfloat/e5-large-v2')

load_dotenv(find_dotenv())

db_string = os.getenv("DB_CONN") or "postgresql://postgres:password@localhost:5433/rec_llm"
connection = psycopg.connect(db_string)


URL = "http://localhost:8000/embedding_store"

@cache
def encode_text(text):
    return model.encode(text, device=device)

def read_post() -> list:
    READ_POSTS_QUERY = "SELECT * FROM posts;"
    with connection.cursor() as cursor:
        cursor.execute(READ_POSTS_QUERY)
        return cursor.fetchall()

def read_prompts() -> list:
    READ_PROMPTS_QUERY = "SELECT * FROM prompts"
    with connection.cursor() as cursor:
        cursor.execute(READ_PROMPTS_QUERY)
        return cursor.fetchall()

def embed(embedding_type: str, id: str, text: str):
    data = {
        "id": id,
        "type": embedding_type,
        "text": text
    }
    # print(data)
    # response = requests.post(URL, json=data)
    # return response.json()
    try:
        encoding = encode_text(text)
        if type(encoding) == Tensor:
            encoding = encoding.numpy().tolist()
        elif type(encoding) == ndarray:
            encoding = encoding.tolist()
        else:
            raise Exception("Encoding failed")

        if embedding_type == 'prompt':
            QUERY = "INSERT INTO prompts_emb (id, embedding) VALUES (%s, %s);"
        if embedding_type == 'post':
            QUERY = "INSERT INTO posts_emb (id, embedding) VALUES (%s, %s);"

        with connection.cursor() as cursor:
            r = cursor.execute(QUERY, (id, encoding))
            connection.commit()

            if r.rowcount == 1:
                return {"success": True}

            else:
                return {"error": "Failed to insert embedding"}

    except Exception as e:
        return {"error": str(e)}

def main():
    prompts = read_prompts()
    posts = read_post()

    failed_prompts = []
    failed_posts = []

    for prompt in tqdm(prompts):
        status = embed("prompt", prompt[0], prompt[1])
        if "error" in status:
            failed_prompts.append((prompt[0], status))

    print("Failed prompts: ", failed_prompts)
    for post in tqdm(posts):
        text = ""
        try:
            text = str(post[1]) + str(post[2])
        except:
            text = str(post[1])
        status = embed("post", post[0], text)
        if "error" in status:
            failed_posts.append((post[0], status))


    print("Failed posts: ", failed_posts)
if __name__ == '__main__':
    main()
