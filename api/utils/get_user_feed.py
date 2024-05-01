import polars as pl
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("intfloat/e5-small-v2")


def parse_entity_from_string(s: str) -> list[dict[str, str | float | list[int] | list[str]]]:
    if s == "":
        return []
    else:
        # parse array of json
        return eval(s)

def flatten_entities(entity):
    return ",".join([e['Label'] for e in entity])

behaviors_columns = [
    "impression_id",
    "user_id",
    "time",
    "history",
    "impressions"
]

news_columns = [
    "news_id",
    "category",
    "sub_category",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities"
]

behaviors = pl.read_csv('../data/MINDsmall_dev/behaviors.tsv', separator='\t', has_header=False, new_columns=behaviors_columns)
behaviors = behaviors.select(
    behaviors['user_id'],
    behaviors['history'].map_elements(lambda s: s.split(' ')).alias('history'),
    behaviors['impressions'].map_elements(lambda s: s.split(' ')).alias('impressions')
)

news = pl.read_csv('../data/MINDsmall_dev/news.tsv', separator='\t', has_header=False, new_columns=news_columns, ignore_errors=True)

# remove where abstract is null
news = news.filter(pl.col("abstract").is_not_null())
news = news.select(
    news['news_id'],
    news['title'],
    news['abstract'],
    news['title_entities'].map_elements(
        lambda x: parse_entity_from_string(x),
        return_dtype=pl.Object
    ).map_elements(
        flatten_entities).alias('title_entities'),
    news['abstract_entities'].map_elements(
        lambda x: parse_entity_from_string(x),
        return_dtype=pl.Object
    ).map_elements(flatten_entities).alias('abstract_entities')
)

news = news.select(
    news['news_id'],
    news['title'],
    news['abstract'],
    news['title_entities'],
    news['abstract_entities'],
    ("Title: " + pl.col("title") + ";Abstract: " + pl.col("abstract") + ";Title Entities: " + pl.col("title_entities") + ";Abstract Entities: " + pl.col("abstract_entities")).alias('text')
)

# drop second half
news = news.head(news.shape[0] // 2)

news = news.with_columns(
            pl.Series(model.encode(news['text'].to_list())).alias('embedding')
    )


# news = news.select(
#     news['news_id'],
#     news['category'],
#     news['sub_category'],
#     news['title'],
#     news['abstract'],
#     news['url'],
#     news['title_entities'].map_elements(
#         lambda s: parse_entity_from_string(s),
#         return_dtype=pl.Object
#     ).alias('title_entities'),
#     news['abstract_entities'].map_elements(
#         lambda s: parse_entity_from_string(s),
#         return_dtype=pl.Object
#     ).alias('abstract_entities')
# )

def generate_user_interactions(user_id):
    user_behaviors = behaviors.filter(behaviors['user_id'] == user_id)
    user_history = user_behaviors['history']
    user_history = user_history.explode()
    user_history = list(set(user_history))
    user_impressions = user_behaviors['impressions']
    user_impressions = user_impressions.explode()
    user_impressions = list(set(user_impressions))
    return user_history, user_impressions

def get_news_candidates(user_history, user_impressions):
    # filter user_impressions that ends with 1
    user_positive_impressions = [impression for impression in user_impressions if impression.endswith('1')]
    # remove the trailing -\d from each impression
    user_positive_impressions = [impression[:-2] for impression in user_positive_impressions]

    # concat user_news_history and user_positive_impressions
    user_history = user_history + user_positive_impressions

    user_news_history = news.filter(pl.col("news_id").is_in(user_history))

    # get the counts of categories and sub categories in the user_news_history
    category_counts = user_news_history.group_by('category').agg(pl.count('category').alias('count'))
    sub_category_counts = user_news_history.group_by('sub_category').agg(pl.count('sub_category').alias('count'))

    # get the category and sub_category with the two highest counts
    category = category_counts.sort('count',descending=True).limit(2)
    sub_category = sub_category_counts.sort('count', descending=True).limit(2)

    # get all news with the category and sub_category
    category_news = user_news_history.filter(pl.col("category").is_in(category['category']))
    sub_category_news = user_news_history.filter(pl.col("sub_category").is_in(sub_category['sub_category']))

    # get news that don't belong to category and sub_category
    # other_news = user_news_history.filter(
    #     ~pl.col("news_id").is_in(category_news['news_id']) &
    #     ~pl.col("news_id").is_in(sub_category_news['news_id'])
    # )

    # sample 3 items from other_news
    # other_news = other_news.sample(3)

    # concat category_new, sub_category_news and other_news
    # all_news = pl.concat([category_news, sub_category_news, other_news])
    all_news = pl.concat([category_news, sub_category_news])
    return all_news




def get_user_recommendations(prompt: str, user_id: str) -> list[dict[str, str]]:
    query = model.encode(prompt)
    # h, i = generate_user_interactions(user_id)
    # news_candidates = get_news_candidates(h, i)
    news_candidates = news.clone()
    # news_candidates = news_candidates.unique('news_id')

    news_candidates = news_candidates.with_columns(
        news_candidates['embedding'].map_elements(
            lambda s: util.dot_score(query, s)
        ).alias('similarity')
        #     pl.Series(model.encode(news_candidates['text'].to_list())).map_elements(
        #         lambda s: util.dot_score(query, s)
        # ).alias('similarity')
    )

    # sort by similarity
    # remove duplicate news_candidates by news_id


    news_candidates = news_candidates.sort('similarity', descending=True)
    # print the news id and smilarity
    # print(news_candidates[['news_id', 'similarity']])
    news_candidates = news_candidates.head(10)
    news_candidates = news_candidates.to_dicts()
    return news_candidates


def get_user_feed(user_id:str) -> list[dict[str, str]]:
    user_history, user_impressions = generate_user_interactions(user_id)
    print(len(user_history), len(user_impressions))
    # filter user_impressions that ends with 1
    user_positive_impressions = [impression for impression in user_impressions if impression.endswith('1')]
    # remove the trailing -\d from each impression
    user_positive_impressions = [impression[:-2] for impression in user_positive_impressions]

    # concat user_news_history and user_positive_impressions
    user_history = user_history + user_positive_impressions

    user_news_history = news.filter(pl.col("news_id").is_in(user_history)).sort('news_id', descending=False)
    # remove where user_news_history['abstract'] is null
    user_news_history = user_news_history.filter(pl.col("abstract").is_not_null())

    user_news_history = user_news_history.unique('news_id').sort('news_id', descending=True)
    user_news_history = user_news_history.to_dicts()
    return user_news_history
