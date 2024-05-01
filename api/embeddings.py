import uvicorn
from sentence_transformers import SentenceTransformer
from functools import cache

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from torch import Tensor
from numpy import ndarray

import psycopg
# load env file
from dotenv import load_dotenv,find_dotenv
import os
# use the find_dotenv function to locate the file
load_dotenv(find_dotenv())

db_string = os.getenv("DB_CONN") or "postgresql://postgres:password@localhost:5432/rec_llm"
# define a psycopg3 connection to postgres
conn = psycopg.connect(db_string)
conn.autocommit = True

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('intfloat/e5-large-v2')

@cache
def encode_text(text):
    return model.encode(text)

class EmbeddingsRequest(BaseModel):
    text: str

@app.post("/embedding")
async def embeddings(request: EmbeddingsRequest):
    try:
        encoding = encode_text(request.text)
        if type(encoding) == Tensor:
            return {"embedding": encoding.numpy().tolist()}
        elif type(encoding) == ndarray:
            return {"embedding": encoding.tolist()}
        else:
            print(type(encoding))
            raise Exception("Encoding failed")
    except Exception as e:
        return {"error": str(e)}


class EmbeddingsStoreRequest(BaseModel):
    id: str
    type: str
    text: str

@app.post("/embedding_store")
async def embeddings_store(request: EmbeddingsStoreRequest):
    try:
        encoding = encode_text(request.text)

        encoding = encoding.numpy().tolist() if type(encoding) == Tensor else encoding.tolist() if type(encoding) == ndarray else None
        if encoding == None:
            raise Exception("Encoding failed")
        r = None

        with conn.cursor() as cur:
            if request.type == 'prompt':
                r = cur.execute("INSERT INTO prompts_emb (id, embedding) VALUES (%s, %s);", (request.id, encoding))
                conn.commit()
            elif request.type == 'post':
                r = cur.execute("INSERT INTO posts_emb (id, embedding) VALUES (%s, %s);", (request.id, encoding))
                conn.commit()
            else:
                raise Exception("Invalid type")
            if r.rowcount == 0:
                raise Exception("Insert failed")
        return {"embedding": encoding}
    except Exception as e:
        return {"error": str(e)}

# listen at port 8765
