from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
# from utils.get_user_feed import get_user_feed, get_user_recommendations
import random
import numpy as np

import torch
# import cosine similarity
from torch.nn.functional import cosine_similarity
from pydantic import BaseModel

from run_inference import infer, process_input, get_embedding

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

CATEGORIES = [
    'autos',
 'entertainment',
 'finance',
 'foodanddrink',
 'health',
 'kids',
 'lifestyle',
 'middleeast',
 'movies',
 'music',
 'news',
 'northamerica',
 'sports',
 'travel',
 'tv',
 'video',
 'weather',
 'random']

class Post(BaseModel):
    category: str
    user_id: str | None = None

def get_user_posts(user_id: str) -> tuple[list[dict] | None, Exception | None]:
    user_preference_query = "SELECT category FROM user_preferences WHERE user_id = %s"
    with conn.cursor() as cur:
        cur.execute(user_preference_query, (user_id,))
        user_preferences = cur.fetchone()
        if not user_preferences:
            return [], None
        user_preferences = user_preferences[0].split(",")
        category_placeholders = ",".join(["%s"] * len(user_preferences))
        QUERY = f"""
        SELECT * FROM posts p
        join posts_emb e on p.id = e.id
        WHERE category IN ({category_placeholders})
        ORDER BY RANDOM() LIMIT 500"""
        if len(user_preferences) == 1:
            QUERY = f"""
            SELECT * FROM posts p
            join posts_emb e on p.id = e.id
            WHERE category = %s
            ORDER BY RANDOM() LIMIT 500"""
        elif len(user_preferences) == 0:
            QUERY = f"""
            SELECT * FROM posts p
            join posts_emb e on p.id = e.id
            ORDER BY RANDOM() LIMIT 500"""
        cur.execute(QUERY, tuple(user_preferences)) # type: ignore
        posts = [{"id": row[0], "title": row[1], "abstract": row[2], "category": row[4], "subcategory": row[5], "embedding": row[7]} for row in cur.fetchall()]
        return posts, None

def get_candidates(category: str) -> tuple[list[dict] | None, Exception | None]:
    # no user id and is random
    # not random, user id doesnt matter
    QUERY = f"""
        select * from posts p
        join posts_emb e on p.id = e.id
        where p.category = '{category}'
        order by random()
        limit 1000;
    """
    if category == 'random':
        QUERY = f"""
            select * from posts p
            join posts_emb e on p.id = e.id
            order by random()
            limit 1000;
        """

    try:
        with conn.cursor() as cur:
            cur.execute(QUERY) # type: ignore
            posts = [
                {
                    "id": row[0],
                    "title": row[1],
                    "abstract": row[2],
                    "category": row[4],
                    "subcategory": row[5],
                    "embedding": row[7]
                } for row in cur.fetchall()
            ]
            return posts, None
    except Exception as e:
        return None, e

def get_current_prompt() -> tuple[dict| None, Exception | None]:
    QUERY = """
    SELECT p.id, p.prompt, p.prompt_type, p.category, p.subcategory, e.embedding
    FROM prompts p
    JOIN rec_flags r ON p.id = r.value::integer
    JOIN prompts_emb e ON p.id = e.id
    WHERE r.flag = 'active_prompt';
    """

    prompt_dict = {}
    try:
        with conn.cursor() as cur:
            cur.execute(QUERY)
            prompt: list = cur.fetchone() # type: ignore
            prompt_dict = {
                "id": prompt[0],
                "prompt": prompt[1],
                "type": prompt[2],
                "category": prompt[3],
                "subcategory": prompt[4],
                "embedding": prompt[5]
            }
            return prompt_dict, None
    except Exception as e:
        return None, e


@app.post("/posts")
async def get_posts(post: Post):
    candidates = []
    # cases
    # has user id and is random
    # has user id and is not random
    # does not have user id and is random
    # does not have user id and is not random

    if post.user_id and post.category == 'random':
        # handles has user id and is random
        candidates, err = get_user_posts(post.user_id)
        if err:
            return {"error": str(err)}


    if not candidates:

        candidates, err = get_candidates(post.category)
        if err:
            return {"error": str(err)}

    prompt, err = get_current_prompt()
    if err or type(prompt) != dict or type(candidates) != list:
        return {"error": str(err)}

    candidates = [
        c | {
                'cosine_score':
                    cosine_similarity(
                        torch.Tensor(eval(c['embedding'])),
                        torch.Tensor(eval(prompt['embedding'])),
                        dim=0
                    )
        } for c in candidates
    ]
    if prompt['category'] == post.category:
        ...
        # use the ml inference
        # if prompt['type'] == 'suppress':
        # invert the cosine score
        if prompt['type'] == 'suppress':
            candidates = [
                c | {
                    'cosine_score': 1 - c['cosine_score']
                } for c in candidates
            ]

        processed_posts = [
            process_input(
                post=c['embedding'],
                prompt=prompt['embedding'],
                prompt_type=prompt['type'],
                post_category=c['category'],
                prompt_category=prompt['category']
            ) for c in candidates]

        inferred = infer(torch.tensor(processed_posts, dtype=torch.float32, device=device))

        if type(inferred) != list:
            inferred = [inferred]
        z = zip(candidates, inferred)
        candidates = [
            {
                "id": c['id'],
                "title": c['title'],
                "abstract": c['abstract'],
                "category": c['category'],
                "subcategory": c['subcategory'],
                "cosine_score": c['cosine_score'],
                "inferred": i
            } for c, i in z]

        candidates = [
            c | {
                'alignment_score': 0.25*c['cosine_score'].item() + 0.75*c['inferred']
            } for c in candidates
        ]

        candidates.sort(key=lambda x: x['alignment_score'], reverse=True if prompt['type'] == 'boost' else False)

        for c in candidates:
            c['cosine_score'] = c['cosine_score'].item()
        return candidates[:15]


    candidates.sort(key=lambda x: x['cosine_score'], reverse=True if prompt['type'] == 'boost' else False)


    for c in candidates:
        c.pop('embedding')
        c['cosine_score'] = c['cosine_score'].item()
    return candidates[:15]

def get_posts_and_prompts() -> tuple[list[dict], list[dict]]:
    posts, prompts = None, None
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM prompts ORDER BY RANDOM() LIMIT 1")
        prompts = [{"id": row[0], "prompt": row[1], "type": row[2], "category": row[3], "subcategory": row[4]} for row in cur.fetchall()]  # Extract the first element from each row
        cur.execute("SELECT * FROM posts WHERE subcategory = %s ORDER BY RANDOM() LIMIT 500", (prompts[0]['subcategory'],))
        posts = [{"id": row[0], "title": row[1], "abstract": row[2], "category": row[4], "subcategory": row[5]} for row in cur.fetchall()]  # Extract the first element from each row
        cur.execute("SELECT * FROM posts WHERE category = %s ORDER BY RANDOM() LIMIT 500", (prompts[0]['category'],))
        # append to posts
        posts += [{"id": row[0], "title": row[1], "abstract": row[2], "category": row[4], "subcategory": row[5]} for row in cur.fetchall()]
        cur.execute("SELECT * FROM posts WHERE category != %s ORDER BY RANDOM() LIMIT 500", (prompts[0]['category'],))
        # append to posts
        posts += [{"id": row[0], "title": row[1], "abstract": row[2], "category": row[4], "subcategory": row[5]} for row in cur.fetchall()]

        return posts, prompts

@app.get("/train")
async def train(request: Request, response_class = HTMLResponse):
    posts, prompts = get_posts_and_prompts()

    # make 10 combinations of post and prompt
    combinations = []
    try:
        for i in range(15):
            post = random.choice(posts)
            prompt = random.choice(prompts)
            combinations.append({"post": post, "prompt": prompt})
            continue
    except Exception as e:
        print(e)
        return {"error": "Add some posts first"}
    # return the array of combinations
    return {"combinations": combinations}

@app.get("/")
async def root(request: Request, response_class = HTMLResponse):
    return templates.TemplateResponse(
     name="index.html",
     context={"request": request}
    )

class Label(BaseModel):
    post: str
    prompt: int
    label: float


"""
The label endpoint is used to store the labels in the database
Test using:
curl -X POST -H "Content-Type: application/json" -d '{"prompt": 3, "post": "N30302", "label": 0.5}' {URL}/label
"""
@app.post("/label")
async def label(label: Label):
    # print(label)
    with conn.cursor() as cur:
        r = cur.execute("INSERT INTO labels (post_id, prompt_id, label) VALUES (%s, %s, %s)", (label.post, label.prompt, label.label))
        conn.commit()
        if r.rowcount == 1:
            return {"label": label}
        else:
            return {"status": "error"}

class Prompt(BaseModel):
    prompt: str
    type: str
    category: str
    subcategory: str
@app.post("/prompt")
async def prompt(prompt: Prompt):
    # check if prompt not empty string
    if not prompt.prompt:
        return {"error": "Prompt cannot be empty"}

    if prompt.type not in ["suppress", "boost"]:
        return {"error": "Invalid prompt type"}

    if prompt.category not in CATEGORIES:
        return {"error": "Invalid category"}

    prompt_embedding = get_embedding(prompt.prompt)
    if type(prompt_embedding) == torch.Tensor:
        prompt_embedding = prompt_embedding.numpy().tolist()
    elif type(prompt_embedding) == np.ndarray:
        prompt_embedding = prompt_embedding.tolist()
    else:
        return {"error": "Failed to encode prompt"}

    with conn.cursor() as cur:
        QUERY = "SELECT id FROM prompts WHERE prompt = %s AND prompt_type = %s AND category = %s AND subcategory = %s"
        r = cur.execute(QUERY, (prompt.prompt, prompt.type, prompt.category, prompt.subcategory))
        if r.rowcount == 1:
            return {"status": "Cannot create duplicate prompt"}

    flag = 0
    # get the prompt from the request body and insert it into the database
    with conn.cursor() as cur:
        r = cur.execute("INSERT INTO prompts (prompt, prompt_type, category, subcategory) VALUES (%s, %s, %s, %s)", (prompt.prompt, prompt.type, prompt.category, prompt.subcategory))
        conn.commit()
        if r.rowcount == 1:
            flag = 1
        else:
            return {"status": "error"}

    if flag == 1:
        with conn.cursor() as cur:
            QUERY = "SELECT id FROM prompts WHERE prompt = %s AND prompt_type = %s AND category = %s AND subcategory = %s"
            r = cur.execute(QUERY, (prompt.prompt, prompt.type, prompt.category, prompt.subcategory))

            if r.rowcount == 1:
                prompt_id = cur.fetchone()[0]
                cur.execute("INSERT INTO prompts_emb (id, embedding) VALUES (%s, %s)", (prompt_id, prompt_embedding))
                conn.commit()
                return {"status": "success"}


@app.get("/prompts")
async def get_prompts():
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM prompts")
        prompts = [
            {
                "id": row[0],
                "prompt": row[1],
                "type": row[2],
                "category": row[3]
            } for row in cur.fetchall()
            ]
        return prompts


class ActivePrompt(BaseModel):
    prompt: str

@app.post("/set_active_prompt")
async def set_active_prompt(prompt: ActivePrompt):
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM prompts WHERE id = %s", (prompt.prompt,))
        print(prompt.prompt)
        prompt_id = cur.fetchone()
        # if no prompt is found, return an error
        if not prompt_id:
            return {"error": "Prompt not found"}

        cur.execute("UPDATE rec_flags SET value = %s WHERE flag = 'active_prompt';", (prompt_id))
        conn.commit()
        return {"status": "success"}


@app.get("/active_prompt")
async def get_active_prompt():
    QUERY = """
    SELECT p.id, p.prompt, p.prompt_type, p.category, p.subcategory
    FROM prompts p
    JOIN rec_flags r ON p.id = r.value::integer WHERE r.flag = 'active_prompt';
    """
    try:
        with conn.cursor() as cur:
            cur.execute(QUERY)
            prompt: list = cur.fetchone() # type: ignore
            return {"prompt": {
                "id": prompt[0],
                "prompt": prompt[1],
                "type": prompt[2],
                "category": prompt[3],
                "subcategory": prompt[4]
            }}
    except Exception as e:
        return {"error": str(e)}


@app.get("/categories")
async def get_categories():
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT category FROM posts")
        categories = [row[0] for row in cur.fetchall()]
        return categories

class Category(BaseModel):
    category: str

@app.post("/subcategories")
async def get_subcategories(category: Category):
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT(subcategory) FROM posts WHERE category = %s;", (category.category,))
        subcategories = [row[0] for row in cur.fetchall()]
        return subcategories


@app.get("/training_data")
async def get_training_data():
    query = """
        SELECT
            l.id,
            post.title, post.abstract, post.category post_category, post.subcategory post_subcategory,
            prompt.prompt, prompt.prompt_type, prompt.category prompt_category, prompt.subcategory prompt_subcategory,
            l.label FROM LABELS l
            JOIN POSTS post ON l.post_id = post.id
            JOIN PROMPTS prompt ON l.prompt_id = prompt.id
        ;"""
    with conn.cursor() as cur:
        cur.execute(query)
        data = [
            {
                "id": row[0],
                "post": {"title": row[1], "abstract": row[2], "category": row[3], "subcategory": row[4]},
                "prompt": {"prompt": row[5], "type": row[6], "category": row[7], "subcategory": row[8]},
                "label": row[9]
            } for row in cur.fetchall()
        ]
        return data


@app.get("/dataset_size")
async def get_dataset_size():
    query = "SELECT COUNT(*) FROM labels;"
    with conn.cursor() as cur:
        cur.execute(query)
        size = cur.fetchone()[0]
        return {"size": size}


class UserPreference(BaseModel):
    user_id: str
    category: list[str]

@app.post("/set_user_preferences")
async def set_user_preferences(user_preference: UserPreference):
    preferred_categories = user_preference.category
    # check if the preferred categories are in CATEGORIES
    for category in preferred_categories:
        if category not in CATEGORIES:
            return {"error": "Invalid category"}

    preferred_categories = ",".join(preferred_categories)

    # check if user_preferences already exist
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM user_preferences WHERE user_id = %s", (user_preference.user_id,))
        user_preferences = cur.fetchone()
        if user_preferences:
            cur.execute("UPDATE user_preferences SET category = %s WHERE user_id = %s", (preferred_categories, user_preference.user_id))
            conn.commit()
            return user_preference

        # insert the user preferences into the database

        with conn.cursor() as cur:
            cur.execute("INSERT INTO user_preferences (user_id, category) VALUES (%s, %s)", (user_preference.user_id, preferred_categories))
            conn.commit()
            return user_preference


class UserTelemetry(BaseModel):
    user_id: str
    post: str
    action: str
    data: str

VALID_ACTIONS = ["click", "like", "dislike", "share", "comment", "hover", "leave", "login"]

@app.post("/telemetry")
async def telemetry(user_telemetry: UserTelemetry):
    if user_telemetry.action not in VALID_ACTIONS:
        return {"error": "Invalid action"}

    # check if post exists
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM posts WHERE id = %s", (user_telemetry.post,))
        post = cur.fetchone()
        if not post:
            return {"error": "Post not found"}

    with conn.cursor() as cur:
        cur.execute("INSERT INTO telemetry (user_id, post, action, data) VALUES (%s, %s, %s, %s)", (user_telemetry.user_id, user_telemetry.post, user_telemetry.action, user_telemetry.data))
        conn.commit()
        return user_telemetry
