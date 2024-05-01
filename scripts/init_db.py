import polars as pl

import psycopg

import os

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_CONN = os.getenv("DB_CONN") or "postgresql://postgres:password@localhost:5432/rec_llm"

QUERIES = {
    "create_posts": """
        CREATE TABLE posts (
            id text PRIMARY KEY,
            title text,
            abstract text,
            url text,
            category text,
            subcategory text
        );
    """,
    "create_prompts": """
        CREATE TABLE prompts (
            id serial PRIMARY KEY,
            prompt text,
            prompt_type text,
            category text,
            subcategory text
        );
    """,
    "create_labels": """
        CREATE TABLE labels (
            id serial PRIMARY KEY,
            post_id text REFERENCES posts(id),
            prompt_id integer REFERENCES prompts(id),
            label double precision
        );
    """,
    "posts_embeddings": """
        CREATE TABLE posts_emb (
            id text PRIMARY KEY,
            embedding vector(1024)
        );
    """,
    "prompts_embeddings": """
        CREATE TABLE prompts_emb (
            id bigserial PRIMARY KEY,
            embedding vector(1024)
        );
    """
}

def create_tables():
    with psycopg.connect(DB_CONN) as conn:
        with conn.cursor() as cur:
            for query in QUERIES.values():
                try:
                    print(query)
                    cur.execute(query) # type: ignore
                    conn.commit()
                except Exception as e:
                    if e == psycopg.errors.DuplicateTable:
                        print("Table already exists")
                        continue
def save_posts():
    post = pl.read_csv(
        source="../data/MINDsmall_train/news.tsv",
        separator="\t",
        has_header=False,
        new_columns=["id", "category", "sub_category", "title", "abstract", "url", "title_entities", "abstract_entities"],
        ignore_errors=True
    )

    post = post.select(
        [
            "id",
            "title",
            "abstract",
            "url",
            "category",
            "sub_category"
        ]
    )

    post = post.rename({"sub_category": "subcategory"})

    post_count = post.height
    db_count = 0
    query = "SELECT COUNT(*) FROM posts"
    with psycopg.connect(DB_CONN) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            count = cur.fetchone()
            if count is not None:
                db_count = count[0]

    if db_count == post_count:
        print(f"{db_count} posts already exist in the database")
        return


    rows_affected = post.write_database(
        "posts",
        DB_CONN,
        if_table_exists="append"
    )

    print(f"Inserted {rows_affected} rows into posts table")


def save_labels():
    labels = pl.read_csv("../data/rec_llm_public_labels.csv")
    rows_affected = labels.write_database(
        "labels",
        DB_CONN,
        if_table_exists="append"
    )

    print(f"Inserted {rows_affected} rows into labels table")

def main():
    # create_tables()
    # save_posts()
    save_labels()


if __name__ == "__main__":
    main()
