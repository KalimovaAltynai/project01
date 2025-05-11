
import fitz
import re
import os
from pod import save_to_db



# --- Было в utils.py ---
def extract_toc_from_pdf(pdf_path: str, toc_output_path: str) -> bool:
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f" Ошибка открытия PDF: {e}")
        return False

    start_toc = None
    end_toc = None

    for page_num in range(min(15, doc.page_count)):
        text = doc.load_page(page_num).get_text("text")
        if re.search(r"\b(ОГЛАВЛЕНИЕ|СОДЕРЖАНИЕ)\b", text, re.IGNORECASE):
            if start_toc is None:
                start_toc = page_num + 1
        if re.search(r"\.{4,}\s\d+", text):
            end_toc = page_num + 1
        elif start_toc and end_toc:
            break

    if not start_toc or not end_toc:
        print("️ Оглавление не найдено.")
        return False

    toc_text = ""
    for page_num in range(start_toc - 1, end_toc):
        toc_text += doc.load_page(page_num).get_text("text") + "\n"

    os.makedirs(os.path.dirname(toc_output_path), exist_ok=True)
    with open(toc_output_path, "w", encoding="utf-8") as f:
        f.write(toc_text)

    print(f" Оглавление сохранено в {toc_output_path}")
    return True

# --- Было в sda1.py ---
def parse_pdf_metadata(filepath, filename):
    doc = fitz.open(filepath)
    page_texts = [page.get_text() for page in doc]
    full_text = "\n".join(page_texts)
    doc.close()

    text = full_text.replace("-\n", "").replace("\n", " ").strip()
    lines = full_text.splitlines()

    udc_match = re.search(r"\b[УуU][ДдKk][Кк]?\b[\s:]*([\d.]+)", text)
    udc = udc_match.group(1).strip() if udc_match else "Не найден"

    title = "Не найден"
    if udc_match:
        after_udc = text[udc_match.end():]
        abstract_match = re.search(r"А[нн]нотац[ияи]|annotation", after_udc, re.IGNORECASE)
        title_region = after_udc[:abstract_match.start()] if abstract_match else after_udc[:500]
        title_candidates = re.findall(r"[А-ЯЁA-Z][А-ЯЁA-Z\s,\-]{10,}", title_region)
        if title_candidates:
            title = max(title_candidates, key=len).strip()

    if title == "Не найден" and filename:
        clean_name = re.sub(r"^\d+_", "", filename)
        clean_name = clean_name.replace(".pdf", "").replace(".PDF", "").strip()
        title = clean_name

    abstract = "Не найдена"
    abstract_match = re.search(r"А[нн]нотаци[яи][.:]?\s*(.*?)Ключевые слова", text, re.DOTALL)
    if abstract_match:
        abstract = abstract_match.group(1).strip()

    keywords = []
    keywords_match = re.search(r"Ключевые слова[.:]?\s*(.*?)(?:\.\s|Введение|1\.|I\.|Проблема|$)", text, re.DOTALL | re.IGNORECASE)
    if keywords_match:
        raw_keywords = keywords_match.group(1).strip()
        keywords = [kw.strip() for kw in raw_keywords.split(",") if kw.strip()]

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

# --- process_new_issue также был в utils.py ---
def process_new_issue(pdf_path: str, toc_path: str, issue_id: int):
    output_folder = "articles"
    os.makedirs(output_folder, exist_ok=True)

    try:
        with open(toc_path, "r", encoding="utf-8") as f:
            toc_lines = f.readlines()
    except Exception as e:
        print(f" Не удалось открыть оглавление '{toc_path}': {e}")
        return

    cleaned_lines = [line for line in toc_lines if not re.match(r'^\s*—\s*\d+\s*—\s*$', line)]
    entries = []
    final_line_indices = [i for i, line in enumerate(cleaned_lines) if re.search(r'\.{5,}\s*\d+\s*$', line)]

    def is_author_line(text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        return re.search(r'[А-ЯЁA-Z]\.(?:\s|$|,)', text) and re.search(r'[а-яёa-z]', text)

    for idx, final_idx in enumerate(final_line_indices):
        prev_final_idx = final_line_indices[idx - 1] if idx > 0 else -1
        start_idx = prev_final_idx + 1
        while start_idx < len(cleaned_lines) and not is_author_line(cleaned_lines[start_idx]):
            start_idx += 1
        while start_idx < len(cleaned_lines) and is_author_line(cleaned_lines[start_idx]):
            start_idx += 1
        title_lines = [cleaned_lines[j].rstrip('\n') for j in range(start_idx, final_idx + 1)]
        if title_lines:
            last_line = re.sub(r'\.{5,}\s*\d+\s*$', '', title_lines[-1]).strip()
            title_lines[-1] = last_line
        article_title = " ".join(line.strip() for line in title_lines).strip()
        page_match = re.search(r'(\d+)\s*$', cleaned_lines[final_idx].strip())
        start_page = int(page_match.group(1)) if page_match else None
        if article_title and start_page:
            entries.append((article_title, start_page))

    try:
        pdf_doc = fitz.open(pdf_path)
    except Exception as e:
        print(f" Ошибка открытия PDF '{pdf_path}': {e}")
        return

    total_pages = pdf_doc.page_count
    for index, (title, start_page) in enumerate(entries):
        start_index = start_page - 1
        end_index = entries[index + 1][1] - 2 if index < len(entries) - 1 else total_pages - 1
        if start_index < 0 or end_index < start_index or end_index >= total_pages:
            print(f" Пропуск: ошибка диапазона страниц для '{title}'")
            continue
        safe_title = re.sub(r'[\\/\*?"<>|]', '_', title).strip().rstrip('.')
        filename = f"{index+1:02d}_{safe_title}.pdf"
        output_path = os.path.join(output_folder, filename)
        try:
            new_doc = fitz.open()
            new_doc.insert_pdf(pdf_doc, from_page=start_index, to_page=end_index)
            new_doc.save(output_path)
            new_doc.close()
            print(f" Сохранено: {output_path}")
            data = parse_pdf_metadata(output_path, filename)
            save_to_db(data, filename, issue_id)
        except Exception as e:
            print(f" Ошибка при обработке '{title}': {e}")
    pdf_doc.close()
    print("Выпуск успешно обработан.")
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

            save_to_db(data, filename, issue_id)  # <--- передаём issue_id
            article_count += 1

    print(f"\n Всего обработано статей: {article_count}")
