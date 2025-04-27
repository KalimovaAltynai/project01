from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import psycopg2
from psycopg2.extras import RealDictCursor
from passlib.hash import bcrypt
import os
import bcrypt
from fastapi import Body
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import inspect
import inspect2
inspect.getargspec = inspect2.getargspec
import pymorphy2

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Подключение к БД
conn = psycopg2.connect(
    dbname="sbornik",
    user="postgres",
    password="Kalimova2003",
    host="localhost",
    port="5432"
)
conn.autocommit = True

# CORS чтобы фронт работал
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модель статьи
class Article(BaseModel):
    id: int
    title: str
    abstract: str
    keywords: List[str]
    tables_count: int
    figures_count: int
    file_name: str
    authors: List[str]

# Главная страница
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Получение всех статей
@app.get("/articles")
def get_articles():
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("""
            SELECT id, title, abstract, keywords, tables_count, figures_count, file_name
            FROM articles
        """)
        articles = cursor.fetchall()

    if not articles:
        return {"articles": [], "filters": {}}

    for article in articles:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT a.name
                FROM authors a
                JOIN article_authors aa ON a.id = aa.author_id
                WHERE aa.article_id = %s
            """, (article['id'],))
            authors = [row[0] for row in cur.fetchall()]
            article['authors'] = authors

    # Вычисляем фильтры
    keywords_counts = [len(a["keywords"]) for a in articles]
    figures_counts = [a["figures_count"] for a in articles]
    tables_counts = [a["tables_count"] for a in articles]

    filters = {
        "keywords": {"min": min(keywords_counts), "max": max(keywords_counts)},
        "figures": {"min": min(figures_counts), "max": max(figures_counts)},
        "tables": {"min": min(tables_counts), "max": max(tables_counts)},
    }

    return {"articles": articles, "filters": filters}

# Одна статья по ID
@app.get("/article/{article_id}", response_class=HTMLResponse)
async def view_article(request: Request, article_id: int):
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("""
            SELECT id, title, abstract, keywords, tables_count, figures_count, file_name
            FROM articles
            WHERE id = %s
        """, (article_id,))
        article = cursor.fetchone()

    if article is None:
        raise HTTPException(status_code=404, detail="Статья не найдена")

    return templates.TemplateResponse("article.html", {"request": request, "article": article})

# Список всех авторов
@app.get("/authors")
def get_authors():
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT DISTINCT a.name
            FROM authors a
            JOIN article_authors aa ON a.id = aa.author_id
        """)
        authors = [row[0] for row in cursor.fetchall()]
    return authors

# Скачать PDF файл
@app.get("/download/{file_name}", response_class=FileResponse)
def download_file(file_name: str):
    file_path = os.path.join("articles", file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(path=file_path, filename=file_name, media_type='application/pdf')



@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Проверяем есть ли пользователь
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    if user:
        cursor.close()
        raise HTTPException(status_code=400, detail="Пользователь уже существует")

    # Хешируем пароль
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    # Сохраняем пользователя
    cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_password))
    cursor.close()
    return {"message": "Регистрация успешна"}


@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Ищем пользователя по имени
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    cursor.close()

    if user and bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
        return {"message": "Login successful", "username": user["username"]}
    else:
        raise HTTPException(status_code=401, detail="Неверный логин или пароль")

morph = pymorphy2.MorphAnalyzer()
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def normalize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^а-яa-z\s]', '', text)
    words = text.split()
    lemmas = [morph.parse(word)[0].normal_form for word in words]
    return lemmas

def jaccard_similarity(str1: str, str2: str) -> float:
    set1 = set(normalize(str1))
    set2 = set(normalize(str2))
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def combined_similarity(query: str, article: dict) -> float:
    query_emb = model.encode(query)
    abstract_emb = model.encode(article["abstract"])
    cos_sim = cosine_similarity([query_emb], [abstract_emb])[0][0]
    jac_sim = jaccard_similarity(query, article["title"])
    return 0.7 * cos_sim + 0.3 * jac_sim

@app.post("/semantic_search")
async def semantic_search(data: dict = Body(...)):
    query = data["query"]

    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("""
        SELECT id, title, abstract, keywords, tables_count, figures_count, file_name
        FROM articles
    """)
    articles = cursor.fetchall()
    cursor.close()

    for article in articles:
        cur = conn.cursor()
        cur.execute("""
            SELECT a.name
            FROM authors a
            JOIN article_authors aa ON a.id = aa.author_id
            WHERE aa.article_id = %s
        """, (article['id'],))
        authors = [row[0] for row in cur.fetchall()]
        article['authors'] = authors
        cur.close()

    threshold = 0.5
    results = []

    for article in articles:
        score = combined_similarity(query, article)
        if score >= threshold:
            results.append((score, article))

    results.sort(key=lambda x: x[0], reverse=True)

    return [r[1] for r in results]
