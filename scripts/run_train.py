import warnings
warnings.filterwarnings("ignore")

import os
from functools import cache

import requests
from tqdm import tqdm

import pandas as pd
import polars as pl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score

from sentence_transformers import SentenceTransformer

from model import get_model

embedding_model = SentenceTransformer('intfloat/e5-large-v2')

BATCH_SIZE= 10
EPOCHS = 100


device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device}")

@cache
def embed(text: str):
    return embedding_model.encode(text)

def get_from_api():
    url = "https://train.synapse.com.np/training_data"
    response = requests.get(url)
    return response.json()

def get_categories():
    api = "https://train.synapse.com.np/categories"
    response = requests.get(api)
    return response.json()

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def preprocess(raw_data):
    categories = get_categories()
    categories.sort()

    df = pl.DataFrame(raw_data)
    # concat the title and abstract
    df = df.with_columns(
        pl.concat_str([
            pl.col("post_title"),
            pl.col("post_abstract"),
        ]).alias("post")
    ).with_columns( # handle null values for post
        pl.when(pl.col("post").is_null())
        .then(pl.col("post_title"))
        .otherwise(pl.col("post")).alias("post")
    ).with_columns( # create embeddings for the post and prompt
        pl.col("post").map_elements(embed).alias("post_embedding"),
        pl.col("prompt_prompt").map_elements(embed).alias("prompt_embedding"),
        pl.col("label").cast(pl.Float32).alias("label") # cast the label to a float
    )

    df = df.to_pandas()


    df['prompt_type'] = df['prompt_type'].map({'suppress': 0, 'boost': 1})
    df = df[['id', 'post_embedding', 'prompt_embedding', 'prompt_type', 'post_category', 'prompt_category', 'label']]

    for category in categories:
        df[f'prompt_{category}'] = (df['prompt_category'] == category).astype(int)

    for category in categories:
        df[f'post_{category}'] = (df['post_category'] == category).astype(int)

    df = df.drop(columns=['post_category', 'prompt_category'])

    # bin the column label
    return df

def train_test_split(df, test_size=0.2):
    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    test_idx = idx[:int(n*test_size)]
    train_idx = idx[int(n*test_size):]
    train, test = df.iloc[train_idx], df.iloc[test_idx]

    # train = balance_train_data(train)

    # create a validation split
    n = len(train)
    idx = np.arange(n)
    np.random.shuffle(idx)
    val_idx = idx[:int(n*test_size)]
    train_idx = idx[int(n*test_size):]
    train, val = train.iloc[train_idx], train.iloc[val_idx]


    return train, val, test

class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index]

        label = row['label']
        # features are all columns except the label and id
        post_embedding = torch.tensor(row['post_embedding'], dtype=torch.float32)
        prompt_embedding = torch.tensor(row['prompt_embedding'], dtype=torch.float32)

        prompt_type = torch.tensor([row['prompt_type']])
        features = row.drop(['label', 'id', 'post_embedding', 'prompt_embedding', 'prompt_type'])


        post_categories = features.filter(like='post_').values
        prompt_categories = features.filter(like='prompt_').values


        # convert post_categories from numpy ndarray to tensor
        post_categories = torch.tensor(post_categories.astype(np.float32), dtype=torch.float32)
        prompt_categories = torch.tensor(prompt_categories.astype(np.float32), dtype=torch.float32)

        feature = torch.cat([
            prompt_embedding, post_embedding, prompt_type,
            post_categories,
            prompt_categories])

        return feature, label
    def __len__(self):
        return len(self.df)

def train(dataloader, model, loss_fn, optimizer):
    losses = []
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        pred = pred.squeeze(1)
        loss = loss_fn(pred, y)
        losses.append(loss.item())
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # if batch % 100 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return losses

def validate(dataloader, model, loss_fn):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = pred.squeeze(1)
            val_loss = loss_fn(pred, y)
            val_losses.append(val_loss.item())
    return val_losses

def test(dataloader, model, loss_fn, is_test=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = pred.squeeze(1)
            test_loss += loss_fn(pred, y).item()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    test_loss /= num_batches

    # Compute regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    ev_score = explained_variance_score(y_true, y_pred)

    if is_test:
        print(f"Test Loss: {test_loss:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Explained Variance Score: {ev_score:.4f}")

    return test_loss, y_true, y_pred

def main():
    data = get_from_api()
    data = [flatten_dict(d) for d in data]
    df = preprocess(data)
    train_split, val_split, test_split = train_test_split(df)
    train_dataset = CustomDataset(train_split)
    validation_dataset = CustomDataset(val_split)
    test_dataset = CustomDataset(test_split)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5e-3)

    epoch_losses = []
    with tqdm(total=EPOCHS) as pbar:
        val_losses = 0
        for t in range(EPOCHS):
            # print(f"Epoch {t+1}\n-------------------------------")
            batch_losses = train(train_loader, model, loss_fn, optimizer)
            epoch_losses.extend(batch_losses)
            loss, _, _ = test(test_loader, model, loss_fn)
            loss = f"{loss:.3f}"
            # pbar.set_postfix({'loss': loss, 'epoch': t})
            val_losses = validate(validation_loader, model, loss_fn)
            pbar.set_postfix({'loss': loss, 'val_loss': np.mean(val_losses), 'epoch': t})
            pbar.update(1)

    print("Test set:")
    loss = test(test_loader, model, loss_fn, is_test=True)
    print("\nTrain set:")
    loss = test(train_loader, model, loss_fn, is_test=True)
    print("\nValidation set:")
    loss = test(validation_loader, model, loss_fn, is_test=True)
    # print(f"Test Error: {loss:.3f}")

    torch.save(model.state_dict(), './model.bin')

if __name__ == "__main__":
    main()
