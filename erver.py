from parser_tools import extract_toc_from_pdf, parse_pdf_metadata, process_new_issue
import shutil
from fastapi import UploadFile, File, Form
from fastapi import UploadFile, File
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
from typing import Dict
import re
import fitz  # PyMuPDF
from fastapi import APIRouter
from fastapi import Query
from typing import Optional
from fastapi.staticfiles import StaticFiles
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
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
def get_articles(user_id: Optional[int] = Query(None), issue_ids: Optional[List[int]] = Query(None)):
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:

        final_issue_ids = []

        # Если issue_ids переданы в запросе — используем их
        if issue_ids:
            final_issue_ids = issue_ids
        elif user_id:
            cursor.execute("""
                SELECT issue_id FROM user_issues WHERE user_id = %s
            """, (user_id,))
            issue_rows = cursor.fetchall()
            final_issue_ids = [row['issue_id'] for row in issue_rows]

        if final_issue_ids:
            cursor.execute("""
                SELECT id, title, abstract, keywords, tables_count, figures_count, file_name
                FROM articles
                WHERE issue_id = ANY(%s)
            """, (final_issue_ids,))
        else:
            cursor.execute("""
                SELECT id, title, abstract, keywords, tables_count, figures_count, file_name
                FROM articles
            """)

        articles = cursor.fetchall()

    if not articles:
        return {"articles": [], "filters": {}}

    # Добавляем авторов
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
        "keywords": {"min": min(keywords_counts) if keywords_counts else 0,
                     "max": max(keywords_counts) if keywords_counts else 0},
        "figures": {"min": min(figures_counts) if figures_counts else 0,
                    "max": max(figures_counts) if figures_counts else 0},
        "tables": {"min": min(tables_counts) if tables_counts else 0,
                   "max": max(tables_counts) if tables_counts else 0},
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

    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    if user:
        cursor.close()
        raise HTTPException(status_code=400, detail="Пользователь уже существует")

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    cursor.execute(
        "INSERT INTO users (username, password) VALUES (%s, %s) RETURNING id",
        (username, hashed_password)
    )
    new_user_id = cursor.fetchone()['id']
    cursor.close()

    return {"message": "Регистрация успешна", "user_id": new_user_id}



@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    cursor.close()

    if user and bcrypt.checkpw(password.encode('utf-8'), user["password"].encode('utf-8')):
        return {"message": "Login successful", "username": user["username"], "user_id": user["id"], "role": user["role"]}
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

router = APIRouter()

def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def check_requirements(article: dict, full_text: str, reference_count: int) -> Dict:

    violations = []
    recommendations = []
    score = 100

    # Проверка аннотации
    annotation_words = len(article['abstract'].split())
    if annotation_words > 50:
        violations.append(f"Аннотация превышает 50 слов ({annotation_words})")
        recommendations.append("Сократите аннотацию до 50 слов")
        score -= 10
    elif annotation_words < 30:
        violations.append(f"Аннотация слишком короткая ({annotation_words})")
        recommendations.append("Добавьте больше содержания в аннотацию")
        score -= 10

    # Проверка ключевых слов
    keyword_count = len(article['keywords'])
    if keyword_count < 4:
        violations.append(f"Слишком мало ключевых слов ({keyword_count})")
        recommendations.append("Добавьте еще ключевые слова (минимум 4)")
        score -= 10
    elif keyword_count > 10:
        violations.append(f"Слишком много ключевых слов ({keyword_count})")
        recommendations.append("Уменьшите количество ключевых слов до 10")
        score -= 5

    # Проверка структуры
    required_sections = ["введение", "проблема", "методы", "результаты", "заключение"]
    section_results = {}
    for section in required_sections:
        match = re.search(rf"\b{section}\b", full_text, re.IGNORECASE)
        section_results[section] = bool(match)
        if not match:
            violations.append(f"Отсутствует раздел: {section.capitalize()}")
            recommendations.append(f"Добавьте раздел {section.capitalize()}")
            score -= 8

    # Проверка ссылок на таблицы и рисунки
    if not re.search(r"таблица\s+\d", full_text, re.IGNORECASE):
        violations.append("Отсутствуют ссылки на таблицы")
        recommendations.append("Добавьте ссылки на таблицы в тексте")
        score -= 5

    if not re.search(r"рис(унок|\.)\s*\d", full_text, re.IGNORECASE):
        violations.append("Отсутствуют ссылки на рисунки")
        recommendations.append("Добавьте ссылки на рисунки в тексте")
        score -= 5

    # Проверка списка литературы
    refs = re.findall(r"\[\d+(?:[;,]\s*\d+)*\]", full_text)
    if reference_count > 20:
        violations.append(f"Слишком много источников ({reference_count})")
        recommendations.append("Оставьте не более 20 источников")
        score -= 5
    elif reference_count < 5:
        violations.append(f"Слишком мало источников ({reference_count})")
        recommendations.append("Добавьте минимум 5 научных источников")
        score -= 5

    violations_summary = []
    if re.search(r"таблица\s+\d", full_text, re.IGNORECASE):
        violations_summary.append(" Ссылки на таблицы найдены")
    else:
        violations_summary.append(" Нет ссылок на таблицы")

    if re.search(r"рис(унок|\.)\s*\d", full_text, re.IGNORECASE):
        violations_summary.append(" Ссылки на рисунки найдены")
    else:
        violations_summary.append(" Нет ссылок на рисунки")

    violations_summary.append(f" Источников из БД: {reference_count}")

    return {
        "score": max(score, 0),
        "violations": violations,
        "recommendations": recommendations,
        "structure_check": section_results,
        "annotation_words": annotation_words,
        "keyword_count": keyword_count,
        "references_found": reference_count,
        "checks_summary": violations_summary

    }



@app.get("/check_requirements/{article_id}")
def check_article_requirements(article_id: int):
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
        article = cursor.fetchone()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT reference_text FROM references_list WHERE article_id = %s", (article_id,))
            references = cur.fetchall()
            reference_count = len(references)

        if not article:
            return {"error": "Статья не найдена"}

    file_path = os.path.join("articles", article["file_name"])
    if not os.path.exists(file_path):
        return {"error": f"PDF файл не найден: {file_path}"}

    try:
        full_text = extract_text_from_pdf(file_path)
    except Exception as e:
        return {"error": f"Ошибка чтения PDF: {str(e)}"}

    result = check_requirements(article, full_text, reference_count)

    return result

from parser_tools import extract_toc_from_pdf, process_new_issue, process_folder

@app.post("/upload_issue")
async def upload_issue(title: str = Form(...), year: int = Form(...), file: UploadFile = File(...)):
    uploads_dir = "uploads"
    os.makedirs(uploads_dir, exist_ok=True)
    save_path = os.path.join(uploads_dir, file.filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM issues WHERE title = %s AND year = %s", (title, year))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Выпуск с таким названием и годом уже существует.")

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO issues (title, year, file_name)
            VALUES (%s, %s, %s) RETURNING id
        """, (title, year, file.filename))
        issue_id = cur.fetchone()[0]


    toc_path = save_path.replace(".pdf", "_toc.txt")
    toc_extracted = extract_toc_from_pdf(save_path, toc_path)

    if not toc_extracted:
        return {"error": "Не удалось извлечь оглавление из PDF."}

    try:
        process_new_issue(save_path, toc_path, issue_id)


    except Exception as e:
        return {"error": f"Ошибка при обработке сборника: {str(e)}"}

    return {"message": "Сборник загружен, статьи распарсены и сохранены", "issue_id": issue_id}

@app.get("/issues")
def get_issues():
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute("SELECT id, title, year FROM issues")
        return cursor.fetchall()

@app.post("/select_issues")
async def select_issues(user_id: int = Form(...), issue_ids: List[int] = Form(...)):
    with conn.cursor() as cursor:
        for issue_id in issue_ids:
            cursor.execute("""
                INSERT INTO user_issues (user_id, issue_id)
                VALUES (%s, %s)
                ON CONFLICT DO NOTHING
            """, (user_id, issue_id))
    return {"message": "Выпуски успешно выбраны"}
