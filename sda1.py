import os
import fitz  # PyMuPDF
import re
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parser_tools import extract_toc_from_pdf, process_new_issue, parse_pdf_metadata


from pod import save_to_db

def parse_pdf_metadata(filepath, filename):
    doc = fitz.open(filepath)
    page_texts = [page.get_text() for page in doc]
    full_text = "\n".join(page_texts)
    doc.close()

    text = full_text.replace("-\n", "").replace("\n", " ").strip()
    lines = full_text.splitlines()

    # --- УДК ---
    udc_match = re.search(r"\b[УуU][ДдKk][Кк]?\b[\s:]*([\d.]+)", text)
    udc = udc_match.group(1).strip() if udc_match else "Не найден"

    # --- Название ---
    title = "Не найден"

    if udc_match:
        after_udc = text[udc_match.end():]
        abstract_match = re.search(r"А[нн]нотац[ияи]|annotation", after_udc, re.IGNORECASE)

        title_region = after_udc[:abstract_match.start()] if abstract_match else after_udc[:500]

        # Поиск кандидатов на заголовок
        title_candidates = re.findall(r"[А-ЯЁA-Z][А-ЯЁA-Z\s,\-]{10,}", title_region)

        if title_candidates:
            # Берём самый длинный кандидат
            title = max(title_candidates, key=len).strip()

    # Если заголовок так и не найден — подставляем название из имени файла
    if title == "Не найден" and filename:
        clean_name = re.sub(r"^\d+_", "", filename)  # убираем "01_", "11_" и т.п. в начале
        clean_name = clean_name.replace(".pdf", "").replace(".PDF", "").strip()
        title = clean_name

    # --- Аннотация ---
    abstract = "Не найдена"
    abstract_match = re.search(r"А[нн]нотаци[яи][.:]?\s*(.*?)Ключевые слова", text, re.DOTALL)
    if abstract_match:
        abstract = abstract_match.group(1).strip()

    # --- Ключевые слова ---

    keywords = []

    # Ищем текст от "Ключевые слова" до первой точки ИЛИ до "Введение" / "1." / "I."
    keywords_match = re.search(r"Ключевые слова[.:]?\s*(.*?)(?:\.\s|Введение|1\.|I\.|Проблема|$)", text, re.DOTALL | re.IGNORECASE)

    if keywords_match:
        raw_keywords = keywords_match.group(1).strip()

        # Разбиваем по запятым
        keywords = [kw.strip() for kw in raw_keywords.split(",") if kw.strip()]

    # --- Авторы и организация ---
    author_line = ""
    org_line = ""
    for i, line in enumerate(lines):
        if re.search(r"\b[А-ЯЁ]\.\s?[А-ЯЁ]\.\s?[А-ЯЁа-яё\-]+", line):
            author_line = line.strip()
            if i + 1 < len(lines) and re.search(r"(г\.|университет|институт)", lines[i + 1], re.IGNORECASE):
                org_line = lines[i + 1].strip()
            break

    authors = author_line if author_line else "Не найдены"
    organization = org_line if org_line else "Не найдена"

    # --- Список литературы ---
    def extract_references(lines):
        references = []
        collecting = False
        current_ref = ""

        for line in lines:
            line = line.strip()
            if re.search(r"список литературы", line, re.IGNORECASE):
                collecting = True
                continue
            if not collecting:
                continue
            if re.fullmatch(r"[-–—\s\d]+", line):
                continue
            if re.match(r"^20\d{2}\.?\s*[–—\-]?\s*Т\.", line) and len(current_ref.strip()) > 20:
                current_ref += " " + line
                continue
            is_numbered = re.match(r"^\d+[\.\)]\s+", line)
            is_russian_fio = re.match(r"^[А-ЯЁ][а-яё\-]+(\s+[А-ЯЁ]\.)+", line)
            is_english_fio = re.match(r"^[A-Z][a-z]+(\s+[A-Z]\.)+", line)
            if is_numbered or is_russian_fio or is_english_fio:
                cleaned = current_ref.strip()
                if cleaned and (len(cleaned) >= 30 or re.search(r"[/.]", cleaned)):
                    references.append(cleaned)
                current_ref = line
            else:
                current_ref += " " + line

        if current_ref.strip():
            cleaned = current_ref.strip()
            if len(cleaned) >= 30 or re.search(r"[/.]", cleaned):
                references.append(cleaned)

        return references

    references_list = extract_references(lines)
    table_numbers = set(re.findall(r"\b(?:таблица|табл\.)\s*(\d+)", text, re.IGNORECASE))
    figure_numbers = set(re.findall(r"\bрис(?:унок)?\.?\s*(\d+)", text, re.IGNORECASE))

    return {
        "title": title,
        "authors": authors,
        "university": organization,
        "abstract": abstract,
        "keywords": keywords,
        "references": references_list,
        "tables": len(table_numbers),
        "figures": len(figure_numbers)
    }


def process_folder(folder_path, issue_id):
    article_count = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            data = parse_pdf_metadata(filepath, filename)

            print(f"\nФайл: {filename}")
            for key, value in data.items():
                print(f"{key}: {value}")
            print("-" * 50)

            save_to_db(data, filename, issue_id)
            article_count += 1

    print(f"\n✅ Всего обработано статей: {article_count}")




