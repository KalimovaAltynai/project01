import fitz  # PyMuPDF for PDF handling
import re
import os

# Пути к входным файлам (предполагается, что они находятся в текущей директории)
toc_filename = "table_of_contents.txt"
pdf_filename = "elibrary_67962613_84684731.pdf"
output_folder = "articles"

# Создаем папку для сохранения статей, если не существует
os.makedirs(output_folder, exist_ok=True)

try:
    # Открываем и читаем файл оглавления
    with open(toc_filename, "r", encoding="utf-8") as f:
        toc_lines = f.readlines()
except Exception as e:
    print(f"Ошибка: не удалось открыть файл оглавления '{toc_filename}': {e}")
    exit(1)

# Удаляем из списка строк любые строки, представляющие номера страниц (например, "— 3 —")
cleaned_lines = []
for line in toc_lines:
    if re.match(r'^\s*—\s*\d+\s*—\s*$', line):
        continue
    cleaned_lines.append(line)

def is_author_line(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    if re.search(r'[А-ЯЁA-Z]\.(?:\s|$|,)', text) and re.search(r'[а-яёa-z]', text):
        return True
    return False

entries = []
final_line_indices = [i for i, line in enumerate(cleaned_lines) if re.search(r'\.{5,}\s*\d+\s*$', line)]

for idx, final_idx in enumerate(final_line_indices):
    prev_final_idx = final_line_indices[idx - 1] if idx > 0 else -1
    start_idx = prev_final_idx + 1

    while start_idx < len(cleaned_lines) and not is_author_line(cleaned_lines[start_idx]):
        start_idx += 1

    author_lines = []
    while start_idx < len(cleaned_lines) and is_author_line(cleaned_lines[start_idx]):
        author_lines.append(cleaned_lines[start_idx].strip())
        start_idx += 1

    title_lines = [cleaned_lines[j].rstrip('\n') for j in range(start_idx, final_idx + 1)]
    if title_lines:
        last_line = title_lines[-1]
        match = re.search(r'\.{5,}\s*\d+\s*$', last_line)
        if match:
            last_line = last_line[:match.start()]
        else:
            last_line = re.sub(r'\s*\d+\s*$', '', last_line)
        title_lines[-1] = last_line.strip()

    article_title = " ".join(line.strip() for line in title_lines).strip()

    page_match = re.search(r'(\d+)\s*$', cleaned_lines[final_idx].strip())
    start_page = int(page_match.group(1)) if page_match else None

    if article_title and start_page is not None:
        entries.append((article_title, start_page))
    else:
        print(f"Предупреждение: не удалось распознать запись оглавления (строка {final_idx})")

# Открываем PDF-файл
try:
    pdf_doc = fitz.open(pdf_filename)
except Exception as e:
    print(f"Ошибка: не удалось открыть PDF-файл '{pdf_filename}': {e}")
    exit(1)

total_pages = pdf_doc.page_count

for index, (title, start_page) in enumerate(entries):
    start_index = start_page - 1
    if index < len(entries) - 1:
        next_start_page = entries[index + 1][1]
        end_index = next_start_page - 2
    else:
        end_index = total_pages - 1

    if start_index < 0 or start_index >= total_pages:
        print(f"Предупреждение: начальная страница {start_page} выходит за границы PDF.")
        continue
    if end_index >= total_pages:
        end_index = total_pages - 1
    if end_index < start_index:
        print(f"Предупреждение: некорректный диапазон страниц для статьи '{title}'.")
        continue

    safe_title = title.replace(':', '-')
    safe_title = re.sub(r'[\\/\*?"<>|]', '_', safe_title)
    safe_title = safe_title.strip().rstrip('.')
    filename = f"{index+1:02d}_{safe_title}.pdf"
    output_path = os.path.join(output_folder, filename)

    try:
        new_doc = fitz.open()
        new_doc.insert_pdf(pdf_doc, from_page=start_index, to_page=end_index)
        new_doc.save(output_path)
        new_doc.close()
        print(f"Сохранен файл: {output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении файла '{output_path}': {e}")

pdf_doc.close()
