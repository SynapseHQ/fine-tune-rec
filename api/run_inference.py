from functools import cache

import warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer

import numpy as np
import requests
import torch

from model import get_model

embedding_model = SentenceTransformer('intfloat/e5-large-v2')

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# def get_categories() -> list[str]:
#     API = "https://train.synapse.com.np/categories"
#     response = requests.get(API)
#     return response.json()

# CATEGORIES: list[str] = get_categories()
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
 'weather']
CATEGORIES.sort()

def load_model(path):
    model =  get_model()
    if device == 'cpu':
        model.load_state_dict(torch.load(path, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(path))
    return model

MODEL = load_model('./model.bin')
MODEL.to(device)
MODEL.eval()

def infer(processed_tensor: torch.Tensor) -> float | list[float]:
    with torch.no_grad():
        logits = MODEL(processed_tensor)
        # recursively flatten logits and return an float or list of floats
        logits = logits.cpu().numpy().flatten().tolist()
        return logits

REQUIRED_FIELDS = [
    # 'title',
    # 'abstract',
    'prompt',
    'prompt_type',
    'post_category',
    'prompt_category'
]

# @cache
# def get_embedding(text: str | list[str]):
#     return embedding_model.encode(text)

def process_input(**kwargs):
    for field in REQUIRED_FIELDS:
        if field not in kwargs:
            raise ValueError(f"Missing required field: {field}")

    # try:
    #     post = kwargs['title'] + kwargs['abstract']
    # except:
    #     post = kwargs['title']
    prompt = kwargs['prompt']
    post = kwargs['post']

    embedded_prompt = None
    embedded_post = None
    # if kwargs['prompt_embedding'] == 'False':
    #     embedded_prompt = get_embedding(prompt)

    # if kwargs['post_embedding'] == 'False':
    #     embedded_post = get_embedding(post)

    prompt_type = kwargs['prompt_type']

    prompt_type_map = {
        'boost': 0,
        'suppress': 1
    }

    try:
        prompt_type = prompt_type_map[prompt_type]
    except KeyError:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

    post_category = kwargs['post_category']
    prompt_category = kwargs['prompt_category']

    # check if post_category and prompt_category are in the list of categories
    if post_category not in CATEGORIES:
        raise ValueError(f"Invalid post category: {post_category}")

    if prompt_category not in CATEGORIES:
        raise ValueError(f"Invalid prompt category: {prompt_category}")

    # make a one hot encoding of the categories
    post_category_index = CATEGORIES.index(post_category)
    prompt_category_index = CATEGORIES.index(prompt_category)

    post_category = [0] * len(CATEGORIES)
    post_category[post_category_index] = 1

    prompt_category = [0] * len(CATEGORIES)
    prompt_category[prompt_category_index] = 1

    prompt = np.array(eval(prompt))
    post = np.array(eval(post))

    features = np.concatenate([
        prompt,
        post,
        [prompt_type],
        post_category,
        prompt_category
    ])


    # print(features.shape)
    # print(features)
    return features

def get_embedding(text: str | list[str]):
    return embedding_model.encode(text, device=device)

def main():
    title = input("Enter title: ")
    abstract = input("Enter abstract: ")
    prompt = input("Enter prompt: ")
    prompt_type = input("Enter prompt type: ")
    post_category = input("Enter post category: ")
    prompt_category = input("Enter prompt category: ")
    process_input(title=title, abstract=abstract, prompt=prompt, prompt_type=prompt_type, post_category=post_category, prompt_category=prompt_category)

if __name__ == '__main__':
    main()
