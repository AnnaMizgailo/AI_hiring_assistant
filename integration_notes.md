# Интеграция в `AI_hiring_assistant`

## 1) Минимальный путь без рефакторинга
Замените текущий `parsing.py` на файл из этого набора.

Почему этого достаточно:
- `app.py` уже импортирует `save_resume_file` и `parse_document`
- таблицы `candidates`, `profiles`, `documents` уже есть в `database.py`
- текущий `/api/parsing/run` уже вызывает `parse_document(db, document_id)`

После замены:
- `documents.raw_text` будет заполняться текстом резюме
- `profiles.profile_json` будет сохраняться в формате для скоринга
- `candidates.full_name` и `candidates.contacts_json` будут обновляться из распарсенного резюме
- `applications.status` будет переключаться в `PROFILE_READY`

## 2) Что поставить
```bash
pip install -r requirements_resume_parser.txt
```

Для OCR дополнительно нужен системный `tesseract`.
По умолчанию OCR выключен.
Чтобы включить fallback:
```bash
export RESUME_PARSER_ENABLE_OCR=1
```

## 3) Что можно поправить в `app.py`
Сейчас `upload_document()` в `app.py` вручную дублирует логику сохранения файла.
Это не ломает пайплайн, но лучше заменить тело сохранения на вызов:

```python
from parsing import save_resume_file

result = await save_resume_file(
    db=db,
    candidate_id=candidate_id,
    file_bytes=file_bytes,
    filename=file.filename or "",
    mime_type=file.content_type or "",
)
await db.commit()
return {"document_id": result["document_id"]}
```

## 4) Как прогонять руками
1. Создать application / candidate
2. Загрузить резюме через `/api/candidates/{candidate_id}/documents`
3. Вызвать `/api/parsing/run?document_id=...`
4. Проверить:
   - `documents.raw_text`
   - `profiles.profile_json`
   - `candidates.contacts_json`

## 5) Как валидировать качество
Для быстрого smoke-теста:
```bash
pytest tests/test_parsing.py
```

Для мини-оценки на своём JSONL:
```bash
python eval_resume_parser.py data/resume_eval.jsonl
```

Формат JSONL:
```json
{
  "id": "resume_001",
  "raw_text": "полный текст резюме",
  "expected": {
    "full_name": "Иван Петров",
    "desired_position": "Backend Developer",
    "location": "Москва",
    "english_level": "B2",
    "summary": "Backend Python developer",
    "skills": ["Python", "FastAPI"],
    "technologies": ["PostgreSQL"],
    "frameworks": ["FastAPI"],
    "tools": ["Docker", "Git"],
    "languages": ["Python", "SQL"],
    "total_experience_years": 4.5,
    "experience": [
      {
        "company": "AI Factory",
        "position": "Backend Developer",
        "date_start": "2024-01",
        "date_end": "2025-12",
        "description": "Developed FastAPI services"
      }
    ],
    "education": [
      {
        "institution": "МГТУ им. Баумана",
        "degree": "Бакалавр"
      }
    ]
  }
}
```
