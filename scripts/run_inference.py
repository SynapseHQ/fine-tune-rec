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

def get_categories() -> list[str]:
    API = "https://train.synapse.com.np/categories"
    response = requests.get(API)
    return response.json()

CATEGORIES: list[str] = get_categories()
CATEGORIES.sort()

def load_model(path):
    model =  get_model()
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
    'title',
    'abstract',
    'prompt',
    'prompt_type',
    'post_category',
    'prompt_category'
]

@cache
def get_embedding(text: str | list[str]):
    return embedding_model.encode(text)

def process_input(**kwargs):
    for field in REQUIRED_FIELDS:
        if field not in kwargs:
            raise ValueError(f"Missing required field: {field}")

    post = kwargs['title'] + kwargs['abstract']
    prompt = kwargs['prompt']

    embedded_post = get_embedding(post)
    embedded_prompt = get_embedding(prompt)

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

    embedding = np.concatenate([
        embedded_post,
        embedded_prompt,
    ])

    print(embedding.shape)
    # shape of embedding in (2048 ,)

    # concat the one hot encoded categories
    features = np.concatenate([
        embedding,
        post_category,
        prompt_category
    ])

    # concat the prompt_type
    features = np.concatenate([
        features,
        [prompt_type]
    ])


    # print(features.shape)
    # print(features)
    return features

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
