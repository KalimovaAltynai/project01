import base64

import numpy as np
import pandas as pd
import plotly.express as px
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from datetime import datetime, date
import shutil
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
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
from typing import Optional, List
from fastapi import Query

@app.get("/articles")
def get_articles(
    user_id: Optional[int] = Query(None),
    issue_ids: Optional[List[int]] = Query(None),
    passed: Optional[bool] = Query(None)  # 👈 новый параметр фильтрации
):
    with conn.cursor(cursor_factory=RealDictCursor) as cursor:

        final_issue_ids = []

        if issue_ids:
            final_issue_ids = issue_ids
        elif user_id:
            cursor.execute("""
                SELECT issue_id FROM user_issues WHERE user_id = %s
            """, (user_id,))
            issue_rows = cursor.fetchall()
            final_issue_ids = [row['issue_id'] for row in issue_rows]

        # === Сборка SQL-запроса ===
        base_query = """
            SELECT id, title, abstract, keywords, tables_count, figures_count, file_name, requirements_passed
            FROM articles
        """
        conditions = []
        params = []

        if final_issue_ids:
            conditions.append("issue_id = ANY(%s)")
            params.append(final_issue_ids)

        if passed is not None:
            conditions.append("requirements_passed = %s")
            params.append(passed)

        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)

        cursor.execute(base_query, tuple(params))
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


@app.get("/article/{article_id}", response_class=HTMLResponse)
def view_article(request: Request, article_id: int, user_id: Optional[int] = Query(None)):
    today = date.today()

    # ✅ Логирование просмотра
    if user_id:
        log_user_action(user_id, f"Просмотр статьи ID {article_id}")

    # ✅ Фиксация в таблице просмотров
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO article_views(article_id, view_date, view_count)
            VALUES (%s, %s, 1)
            ON CONFLICT (article_id, view_date)
            DO UPDATE SET view_count = article_views.view_count + 1
            """,
            (article_id, today)
        )

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM articles WHERE id=%s", (article_id,))
        art = cur.fetchone()

    if not art:
        raise HTTPException(404, "Статья не найдена")

    return templates.TemplateResponse("article.html", {
        "request": request,
        "article": art,
        "user_id": user_id  # 👈 передаём в шаблон, чтобы подставлять в ссылку на скачивание
    })
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
def download_file(file_name: str, user_id: Optional[int] = Query(None)):
    # 1) Найти article_id по имени файла
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM articles WHERE file_name = %s", (file_name,))
        row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Статья не найдена для данного файла")

    article_id = row[0]

    # ✅ Логирование
    if user_id:
        log_user_action(user_id, f"Скачал статью ID {article_id}")

    # ✅ Фиксация скачивания
    today = date.today()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO article_downloads(article_id, download_date, download_count)
            VALUES (%s, %s, 1)
            ON CONFLICT (article_id, download_date)
            DO UPDATE SET download_count = article_downloads.download_count + 1
            """,
            (article_id, today)
        )

    # ✅ Проверка и отправка PDF
    file_path = os.path.join("articles", file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Файл не найден на сервере")

    return FileResponse(path=file_path, filename=file_name, media_type="application/pdf")


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
        log_user_action(user["id"], "Вход в систему")
        return {"message": "Login successful", "username": user["username"], "user_id": user["id"],"role": user["role"]}


@app.get("/admin/users")
def list_users():
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id,username,role,created_at,last_login FROM users ORDER BY id")
        users = cur.fetchall()
    # Сериализуем datetime
    for u in users:
        u["created_at"] = u["created_at"].isoformat()
        u["last_login"] = u["last_login"].isoformat()
    return JSONResponse(jsonable_encoder({"users": users}))

@app.post("/admin/users")
def create_user(username: str = Form(...), password: str = Form(...), role: str = Form("user")):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users(username,password,role,created_at,last_login) "
            "VALUES(%s,%s,%s,NOW(),NOW())",
            (username, hashed, role)
        )
    return JSONResponse({"message": "User created"})

@app.put("/admin/users/{user_id}")
def update_user(
        user_id: int,
        username: Optional[str] = Form(None),
        password: Optional[str] = Form(None),
        role:     Optional[str] = Form(None)
):
    if not any([username, password, role]):
        raise HTTPException(400, "Nothing to update")
    with conn.cursor() as cur:
        if username:
            cur.execute("UPDATE users SET username=%s WHERE id=%s", (username, user_id))
        if password:
            hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            cur.execute("UPDATE users SET password=%s WHERE id=%s", (hashed, user_id))
        if role:
            cur.execute("UPDATE users SET role=%s WHERE id=%s", (role, user_id))
    return JSONResponse({"message": "User updated"})

@app.delete("/admin/users/{user_id}")
def delete_user(user_id: int):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM users WHERE id=%s", (user_id,))
    return JSONResponse({"message": "User deleted"})

@app.get("/admin/reports/views")
def get_views():
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT a.title, v.view_date, v.view_count
                FROM article_views v
                JOIN articles a ON a.id = v.article_id
                ORDER BY v.view_date
            """)
            rows = cur.fetchall()
            reports = [
                {
                    "title": row["title"],
                    "view_date": row["view_date"],
                    "view_count": row["view_count"]
                }
                for row in rows
            ]
            return {"reports": reports}
    except Exception as e:
        print("Ошибка при получении просмотров:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


# === Маршрут для скачиваний ===
@app.get("/admin/reports/downloads")
def get_downloads():
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT a.title, d.download_date, d.download_count
                FROM article_downloads d
                JOIN articles a ON a.id = d.article_id
                ORDER BY d.download_date
            """)
            rows = cur.fetchall()
            reports = [
                {
                    "title": row["title"],
                    "download_date": row["download_date"],
                    "download_count": row["download_count"]
                }
                for row in rows
            ]
            return {"reports": reports}
    except Exception as e:
        print("Ошибка при получении скачиваний:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


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

    log_user_action(user_id, f"Выбрал выпуски: {', '.join(map(str, issue_ids))}")
    return {"message": "Выпуски успешно выбраны"}

@app.get("/admin/logs")
def get_user_logs():
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT ua.id, u.username, ua.action, ua.action_time
            FROM user_actions ua
            JOIN users u ON ua.user_id = u.id
            ORDER BY ua.action_time DESC
            LIMIT 100
        """)
        logs = cur.fetchall()
    return {"logs": logs}
@app.post("/update_requirements_flags")
def update_all_requirements():
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id, file_name FROM articles")
        articles = cur.fetchall()

    updated = 0
    for article in articles:
        # Парсим текст статьи
        file_path = os.path.join("articles", article["file_name"])
        full_text = extract_text_from_pdf(file_path)

        # Количество ссылок
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM references_list WHERE article_id = %s", (article["id"],))
            ref_count = cur.fetchone()[0]

        # Проверка
        result = check_requirements(article, full_text, ref_count)
        passed = result["score"] >= 85  # например, 85+ баллов считается соответствием

        with conn.cursor() as cur:
            cur.execute(
                "UPDATE articles SET requirements_passed = %s WHERE id = %s",
                (passed, article["id"])
            )
            updated += 1

    return {"message": f"Обновлено {updated} статей"}

@app.post("/admin/clusterize")
async def clusterize_abstracts(request: Request):
    try:
        form = await request.form()
        num_clusters = int(form.get("num_clusters", 3))
        issue_ids = [int(i) for i in form.getlist("issue_ids")]

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, title, abstract FROM articles
                WHERE issue_id = ANY(%s) AND abstract IS NOT NULL
            """, (issue_ids,))
            rows = cur.fetchall()

        if not rows:
            return JSONResponse(status_code=400, content={"error": "Нет аннотаций"})

        df = pd.DataFrame(rows)
        embeddings = model.encode(df["abstract"].tolist(), show_progress_bar=False)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(embeddings)

        # ⬇️ Сохраняем кластеры статей
        with conn.cursor() as cur:
            for i, row in df.iterrows():
                cur.execute("""
                    INSERT INTO annotation_clusters (article_id, cluster_id)
                    VALUES (%s, %s)
                    ON CONFLICT (article_id) DO UPDATE SET cluster_id = EXCLUDED.cluster_id
                """, (row['id'], int(row['cluster'])))

        # ⬇️ Сохраняем центры кластеров
        centroids = kmeans.cluster_centers_
        with conn.cursor() as cur:
            for idx, vec in enumerate(centroids):
                cur.execute("""
                    INSERT INTO cluster_centroids (cluster_id, center_vector)
                    VALUES (%s, %s)
                    ON CONFLICT (cluster_id) DO UPDATE SET center_vector = EXCLUDED.center_vector
                """, (idx, vec.tolist()))

        # ⬇️ 3D визуализация
        pca = PCA(n_components=3).fit_transform(embeddings)
        df['x'], df['y'], df['z'] = pca[:, 0], pca[:, 1], pca[:, 2]

        fig = px.scatter_3d(
            df,
            x='x', y='y', z='z',
            color=df['cluster'].astype(str),
            hover_name=df['title'],
            title="Кластеризация аннотаций",
            width=900, height=700
        )

        html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        encoded_html = base64.b64encode(html.encode("utf-8")).decode("utf-8")
        return {"plot_html": encoded_html, "message": "Кластеризация завершена"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})



@app.post("/cluster_search")
async def cluster_search(data: dict = Body(...)):
    try:
        query = data.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Пустой запрос")

        # Вектор запроса
        query_vec = model.encode([query])[0]

        # Загрузка центров кластеров
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT cluster_id, center_vector FROM cluster_centroids")
            rows = cur.fetchall()
            cluster_centers = {r["cluster_id"]: np.array(r["center_vector"]) for r in rows}

        if not cluster_centers:
            raise HTTPException(status_code=400, detail="Центры кластеров не найдены")

        # Определяем ближайший кластер
        best_cluster = max(
            cluster_centers.items(),
            key=lambda kv: cosine_similarity([query_vec], [kv[1]])[0][0]
        )[0]

        # Загружаем статьи из этого кластера
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT a.id, a.title, a.abstract, a.keywords, a.tables_count, a.figures_count,
                       a.file_name, ac.cluster_id
                FROM articles a
                JOIN annotation_clusters ac ON ac.article_id = a.id
                WHERE ac.cluster_id = %s AND a.abstract IS NOT NULL
            """, (best_cluster,))
            rows = cur.fetchall()

        if not rows:
            return []

        df = pd.DataFrame(rows)
        embeddings = model.encode(df['abstract'].tolist(), show_progress_bar=False)
        similarities = cosine_similarity([query_vec], embeddings)[0]
        df['similarity'] = similarities

        df = df.sort_values(by='similarity', ascending=False).head(20)

        result = []
        for i, row in df.iterrows():
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT a.name FROM authors a
                    JOIN article_authors aa ON aa.author_id = a.id
                    WHERE aa.article_id = %s
                """, (row['id'],))
                authors = [r[0] for r in cur.fetchall()]

            result.append({
                "id": row["id"],
                "title": row["title"],
                "abstract": row["abstract"],
                "keywords": row["keywords"],
                "tables_count": row["tables_count"],
                "figures_count": row["figures_count"],
                "file_name": row["file_name"],
                "authors": authors
            })

        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка кластерного поиска: {str(e)}")



def log_user_action(user_id: int, action: str):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO user_actions (user_id, action)
            VALUES (%s, %s)
        """, (user_id, action))