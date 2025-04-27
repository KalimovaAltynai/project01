# ✅ BACKEND: api.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import psycopg2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    return psycopg2.connect(
        dbname="bobi",
        user="postgres",
        password="Kalimova2003",
        host="localhost",
        port="5432"
    )

class Article(BaseModel):
    id: int
    title: str
    authors: List[str]
    university: str
    abstract: str
    keywords: List[str]
    num_keywords: int
    num_references: int
    num_images: int
    num_tables: int

@app.get("/")
def root():
    return {"message": "Добро пожаловать в API публикаций!"}

@app.get("/articles", response_model=List[Article])
def get_articles(
    author: Optional[str] = Query(None),
    min_keywords: int = 0,
    max_keywords: int = 100,
    min_tables: int = 0,
    max_tables: int = 100,
    min_images: int = 0,
    max_images: int = 100
):
    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        SELECT a.articles_id, a.title,
               ARRAY_AGG(DISTINCT au.name),
               u.university_name, a.annotation, a.keywords,
               a.num_keywords, a.num_references, a.num_images, a.num_tables
        FROM articles a
        JOIN article_authors aa ON a.articles_id = aa.article_id
        JOIN authors au ON aa.author_id = au.authors_id
        JOIN universities u ON au.university_id = u.id
        WHERE a.num_keywords BETWEEN %s AND %s
          AND a.num_tables BETWEEN %s AND %s
          AND a.num_images BETWEEN %s AND %s
    """
    params = [min_keywords, max_keywords, min_tables, max_tables, min_images, max_images]

    if author:
        query += " AND LOWER(au.name) LIKE LOWER(%s)"
        params.append(f"%{author}%")

    query += """
        GROUP BY a.articles_id, a.title, u.university_name, a.annotation, a.keywords,
                 a.num_keywords, a.num_references, a.num_images, a.num_tables
        ORDER BY a.title
    """
    cur.execute(query, tuple(params))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [Article(
        id=row[0], title=row[1], authors=row[2], university=row[3], abstract=row[4],
        keywords=row[5], num_keywords=row[6], num_references=row[7], num_images=row[8], num_tables=row[9]
    ) for row in rows]

@app.get("/article/{article_id}", response_model=Article)
def get_article(article_id: int):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT a.articles_id, a.title,
               ARRAY_AGG(DISTINCT au.name),
               u.university_name, a.annotation, a.keywords,
               a.num_keywords, a.num_references, a.num_images, a.num_tables
        FROM articles a
        JOIN article_authors aa ON a.articles_id = aa.article_id
        JOIN authors au ON aa.author_id = au.authors_id
        JOIN universities u ON au.university_id = u.id
        WHERE a.articles_id = %s
        GROUP BY a.articles_id, a.title, u.university_name, a.annotation, a.keywords,
                 a.num_keywords, a.num_references, a.num_images, a.num_tables
    """, (article_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Статья не найдена")

    return Article(
        id=row[0], title=row[1], authors=row[2], university=row[3], abstract=row[4],
        keywords=row[5], num_keywords=row[6], num_references=row[7], num_images=row[8], num_tables=row[9]
    )

@app.get("/articles/ranges")
def get_article_ranges():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT MIN(num_keywords), MAX(num_keywords),
               MIN(num_tables), MAX(num_tables),
               MIN(num_images), MAX(num_images)
        FROM articles
    """)
    r = cur.fetchone()
    cur.close()
    conn.close()
    return {
        "min_keywords": r[0], "max_keywords": r[1],
        "min_tables": r[2], "max_tables": r[3],
        "min_images": r[4], "max_images": r[5]
    }

@app.get("/authors", response_model=List[str])
def get_authors():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT name FROM authors ORDER BY name")
    names = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return names
