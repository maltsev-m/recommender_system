import os
import pickle
from typing import List
from datetime import datetime

import psycopg2
import pandas as pd
import uvicorn
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, Depends
from loguru import logger
from sqlalchemy import create_engine

from database import SQLALCHEMY_DATABASE_URL
from schema import PostGet


app = FastAPI()


def get_db():
    db = psycopg2.connect(
        SQLALCHEMY_DATABASE_URL,
        cursor_factory=RealDictCursor,
    )
    return db

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_features():
    # Загрузка уникальных постов и юзеровов с лайками
    logger.info('loading liked posts')

    user_post_like_query = """
        SELECT distinct post_id, user_id
        FROM public.feed_data
        WHERE action = 'like'
    """
    liked_posts = batch_load_sql(user_post_like_query)

    # Загрузка фичей по юзерам и постам
    logger.info('loading users and posts features')

    users_features = batch_load_sql('maksim_maltsev_users_lesson_22')
    posts_features = batch_load_sql('maksim_maltsev_posts_lesson_22')

    return [liked_posts, posts_features, users_features]

def load_models():
    # Загрузка модели
    logger.info('loading model')

    model_path = 'service/model_lgbm.pkl'

    with open(model_path , 'rb') as file:
        model = pickle.load(file)
        return model

# Положим модель и фичи в соответствующие переменные при поднятии сервиса
model = load_models()
features = load_features()
logger.info('service is up and running')
    
@app.get("/post/recommendations/", response_model=List[PostGet])
def get_post_recom(id: int, time: datetime, limit: int = 10, db = Depends(get_db)) -> List[PostGet]:

    # Фильтрация фичей пользователя по id
    logger.info(f'loading user {id} features')
    user_features = features[2].loc[features[2].user_id == id]
    user_features.drop('user_id', axis=1, inplace=True)

    # Загрузим фичи постов для юзера
    logger.info(f'loading post features for user {id}')
    post_features = features[1].drop('text', axis=1)
    content = features[1][['post_id', 'text', 'topic']]

    # Объединим фичи
    logger.info('concating features')
    add_user_features = dict(zip(user_features.columns, user_features.values[0]))
    user_id_df = post_features.assign(**add_user_features)
    user_id_df = user_id_df.set_index('post_id')

    # Добавим фичи из даты
    logger.info('adding date features')
    user_id_df['hour'] = time.hour
    user_id_df['month'] = time.month

    # Определим для юзера вероятности лайка постов
    logger.info(f'predicting posts for user {id}')
    predicts = model.predict_proba(user_id_df)[:,1]
    user_id_df['predicts'] = predicts
    
    # Уберем лайкнутые ранее посты
    logger.info(f'deliting liked posts by user {id}')
    liked_posts = features[0]
    liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
    user_id_predict = user_id_df[~user_id_df.index.isin(liked_posts)]

    # Рекомендуем топ постов
    post_predict = user_id_predict.sort_values('predicts', ascending=False)[:limit].index

    return [
        PostGet(**{
            "id": i,
            "text": content[content.post_id == i].text.values[0],
            "topic": content[content.post_id == i].topic.values[0]
        }) for i in post_predict
    ]

if __name__ == '__main__':
    uvicorn.run(app)