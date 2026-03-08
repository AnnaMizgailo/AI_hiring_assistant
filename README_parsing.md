# Resume Parser README

Пошаговая инструкция по запуску и использованию модуля парсинга резюме в проекте **AI Hiring Assistant**.

Этот README рассчитан на локальный запуск:
- на **macOS**
- на **Windows**
- с **Ollama**
- с моделью **`qwen3:1.7b`**
- с парсером резюме, который:
  - принимает PDF/DOCX,
  - извлекает текст,
  - отправляет текст в локальную LLM,
  - сохраняет результат в базу данных.

---

## 1. Что делает парсер

Парсер проходит по такому конвейеру:

1. принимает загруженное резюме,
2. сохраняет файл в `storage_resumes/`,
3. извлекает текст из PDF или DOCX,
4. отправляет текст в локальную модель через Ollama,
5. получает структурированный JSON,
6. нормализует и валидирует результат,
7. сохраняет данные в базу.

На выходе формируется `profile_json` примерно такого вида:

```json
{
  "full_name": "Виктория Красоткина",
  "summary": "...",
  "desired_position": null,
  "location": "Москва, Россия",
  "english_level": null,
  "total_experience_years": 2.1,
  "skills": ["Python", "SQL", "LaTeX"],
  "technologies": ["Python", "SQL"],
  "frameworks": [],
  "tools": ["LaTeX", "GitHub"],
  "languages": [],
  "experience": [
    {
      "company": "Fiverr",
      "position": "Фрилансер",
      "date_start": "2022-07",
      "date_end": null,
      "description": "..."
    }
  ],
  "education": [
    {
      "institution": "МФТИ",
      "degree": "Студент"
    }
  ]
}
```

---

## 2. Что сохраняется и куда

После парсинга данные сохраняются **не в отдельные `.json` файлы**, а в **базу данных**.

Обычно сохраняются:

- `documents.raw_text` — извлечённый текст резюме,
- `profiles.profile_json` — итоговый структурированный JSON,
- `profiles.confidence_json` — служебные данные о качестве парсинга,
- `profiles.missing_fields_json` — какие поля не удалось извлечь,
- `candidates.full_name` — имя кандидата,
- `candidates.contacts_json` — контакты кандидата.

Сам файл резюме сохраняется на диск в папку:

```text
storage_resumes/
```

---

## 3. Требования

### Python

Рекомендуемая версия:

- **Python 3.10+**
- желательно **3.11** или **3.12**

### Системные зависимости

Нужны:
- Python
- pip
- Git
- Ollama

### Python-библиотеки

Минимально нужны библиотеки вроде:

```txt
fastapi
uvicorn
sqlalchemy
pydantic
httpx
python-dotenv
pymupdf
pdfplumber
python-docx
```

Если у тебя уже есть `requirements.txt`, ставь зависимости из него.
Если нет — поставь вручную.

---

## 4. Установка на macOS

### 4.1. Установи Python

Проверь:

```bash
python3 --version
```

### 4.2. Создай виртуальное окружение

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4.3. Установи зависимости

```bash
pip install -r requirements.txt
```

Если `requirements.txt` нет или он неполный:

```bash
pip install fastapi uvicorn sqlalchemy pydantic httpx python-dotenv pymupdf pdfplumber python-docx
```

### 4.4. Установи Ollama

Установи Ollama с официального сайта и проверь:

```bash
ollama --version
```

### 4.5. Скачай модель

```bash
ollama pull qwen3:1.7b
```

### 4.6. Проверь, что модель доступна

```bash
ollama list
```

Ты должен увидеть что-то вроде:

```text
qwen3:1.7b
```

---

## 5. Установка на Windows

### 5.1. Установи Python

Скачай Python с официального сайта.
При установке обязательно включи галочку:

```text
Add Python to PATH
```

Проверь в PowerShell или CMD:

```powershell
python --version
```

### 5.2. Создай виртуальное окружение

В PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Если используешь CMD:

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 5.3. Установи зависимости

```powershell
pip install -r requirements.txt
```

Если файла нет или он неполный:

```powershell
pip install fastapi uvicorn sqlalchemy pydantic httpx python-dotenv pymupdf pdfplumber python-docx
```

### 5.4. Установи Ollama для Windows

Установи Ollama и после установки открой новый PowerShell.
Проверь:

```powershell
ollama --version
```

### 5.5. Скачай модель

```powershell
ollama pull qwen3:1.7b
```

### 5.6. Проверь список моделей

```powershell
ollama list
```

---

## 6. Настройка `.env`

В **корне проекта** создай файл:

```text
.env
```

Пример содержимого:

```env
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_TIMEOUT_SECONDS=600
OLLAMA_MAX_INPUT_CHARS=5000
OLLAMA_NUM_PREDICT=900
OLLAMA_MAX_RETRIES=3
RESUME_STORAGE_DIR=./storage_resumes
```

### Что означают параметры

- `OLLAMA_MODEL` — имя модели в Ollama
- `OLLAMA_URL` — endpoint Ollama API
- `OLLAMA_TIMEOUT_SECONDS` — таймаут ожидания ответа модели
- `OLLAMA_MAX_INPUT_CHARS` — сколько символов текста отправлять модели
- `OLLAMA_NUM_PREDICT` — максимальный размер ответа модели
- `OLLAMA_MAX_RETRIES` — количество повторных попыток при ошибке
- `RESUME_STORAGE_DIR` — папка для хранения резюме

---

## 7. Где должен лежать `parsing.py`

Файл парсера должен лежать в корне проекта рядом с `app.py`.

Пример структуры:

```text
AI_hiring_assistant/
├── .env
├── app.py
├── parsing.py
├── database.py
├── requirements.txt
├── storage_resumes/
└── ...
```

Если у тебя есть готовый улучшенный файл парсера, просто замени старый `parsing.py` новым.

---

## 8. Запуск Ollama

Ollama должен быть запущен локально.

На большинстве систем после установки он стартует как сервис сам.
Если нужно проверить, работает ли API:

```bash
curl http://127.0.0.1:11434/api/tags
```

На Windows в PowerShell можно:

```powershell
curl http://127.0.0.1:11434/api/tags
```

Если сервис работает, ты получишь JSON со списком моделей.

---

## 9. Запуск приложения

Обычно проект запускается через Uvicorn.

На macOS/Linux:

```bash
uvicorn app:app --reload
```

На Windows:

```powershell
uvicorn app:app --reload
```

После запуска backend обычно доступен по адресу:

```text
http://127.0.0.1:8000
```

Swagger UI обычно:

```text
http://127.0.0.1:8000/docs
```

---

## 10. Как работает парсинг шаг за шагом

### Шаг 1. Загрузить резюме

Нужно отправить файл кандидата через endpoint загрузки документа.
После загрузки:
- файл сохранится в `storage_resumes/`,
- в таблице `documents` появится запись со статусом `PENDING`.

### Шаг 2. Запустить парсинг

После загрузки вызови endpoint запуска парсинга, например:

```text
POST /api/parsing/run?document_id=<DOCUMENT_ID>
```

### Шаг 3. Что делает backend

Backend:
1. достаёт запись `Document` из БД,
2. читает файл по `file_path`,
3. извлекает текст,
4. отправляет текст в Ollama,
5. получает JSON,
6. сохраняет результат в `profiles.profile_json`.

### Шаг 4. Что ты получаешь

В ответе API будет примерно:

```json
{
  "status": "DONE",
  "document_id": "...",
  "parse_status": "DONE",
  "profile_json": {
    "full_name": "...",
    "skills": ["Python", "SQL"],
    "experience": [],
    "education": []
  }
}
```

---

## 11. Как понять, что всё работает

Признаки успешного запуска:

1. файл резюме появился в папке `storage_resumes/`
2. в логах есть этапы extraction и вызова Ollama
3. endpoint `/api/parsing/run` возвращает:

```json
{
  "status": "DONE"
}
```

4. в БД появился `profile_json`

---

## 12. Какие логи должны быть в норме

Пример нормальных логов:

```text
INFO parsing: Extracting text kind=pdf path=storage_resumes/xxx__resume.pdf
INFO parsing: Extraction done method=pymupdf quality=0.93 warnings=[]
INFO parsing: Resume text chars=1633
INFO parsing: Ollama request attempt=1 model=qwen3:1.7b timeout=600.0s prompt_chars=3237
INFO parsing: Ollama raw response chars=1284
INFO parsing: Parsed JSON keys=['full_name', 'skills', 'experience', 'education']
INFO parsing: Document parse_status=DONE
```

---

## 13. Типовые ошибки и что делать

### Ошибка: `Ollama parsing failed: empty_or_invalid_json`

Что это значит:
- модель ответила пусто,
- или ответила невалидным JSON,
- или JSON нельзя было извлечь из ответа.

Что делать:
1. проверить, что модель точно скачана:

```bash
ollama list
```

2. проверить `.env`
3. уменьшить вход:

```env
OLLAMA_MAX_INPUT_CHARS=3500
```

4. уменьшить размер ответа:

```env
OLLAMA_NUM_PREDICT=700
```

5. перезапустить приложение

---

### Ошибка: `httpx.ReadTimeout`

Что это значит:
- модель слишком долго отвечает,
- железо не успевает,
- prompt слишком тяжёлый.

Что делать:

```env
OLLAMA_TIMEOUT_SECONDS=600
OLLAMA_MAX_INPUT_CHARS=3500
OLLAMA_NUM_PREDICT=700
```

Если всё равно медленно — попробуй ещё уменьшить `OLLAMA_MAX_INPUT_CHARS`.

---

### Ошибка: PDF распарсился криво

Что это значит:
- PDF с плохой версткой,
- 2 колонки,
- текст после extraction перемешался,
- внутри может быть много переносов.

Что делать:
- попробовать другой extractor,
- проверить, не скан ли это,
- попробовать тот же файл в DOCX,
- уменьшить ожидания: LLM не всегда идеально собирает сложные PDF.

---

### Ошибка: `ModuleNotFoundError`

Не хватает библиотеки.

Решение:

```bash
pip install <missing_package>
```

Например:

```bash
pip install python-dotenv pymupdf pdfplumber python-docx httpx
```

---

## 14. Как проверить, что JSON реально сохранился

### Вариант 1. Через API-ответ

Если в ответе `/api/parsing/run` пришёл `profile_json`, значит он уже сформировался и должен был сохраниться.

### Вариант 2. Через БД

Проверь таблицы:
- `documents`
- `profiles`
- `candidates`

Ищи:
- `documents.raw_text`
- `profiles.profile_json`
- `profiles.confidence_json`
- `profiles.missing_fields_json`

Если у тебя SQLite, можно открыть БД через:
- **DB Browser for SQLite**
- **DataGrip**
- **DBeaver**

---

## 15. Рекомендуемые настройки для слабого ноутбука

Для старого MacBook или Windows-ноутбука:

```env
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_TIMEOUT_SECONDS=600
OLLAMA_MAX_INPUT_CHARS=3500
OLLAMA_NUM_PREDICT=700
OLLAMA_MAX_RETRIES=3
```

Если модель отвечает стабильно, можно попробовать:

```env
OLLAMA_MAX_INPUT_CHARS=5000
OLLAMA_NUM_PREDICT=900
```

---

## 16. Что делать после замены `parsing.py`

1. положить новый `parsing.py` в корень проекта,
2. убедиться, что `.env` лежит рядом с `app.py`,
3. активировать виртуальное окружение,
4. проверить, что Ollama запущен,
5. проверить, что модель `qwen3:1.7b` скачана,
6. запустить backend,
7. загрузить резюме,
8. вызвать `/api/parsing/run?document_id=...`.

---

## 17. Быстрый чек-лист

Перед запуском проверь:

- [ ] установлен Python
- [ ] создано и активировано `venv`
- [ ] установлены зависимости
- [ ] установлен Ollama
- [ ] скачана модель `qwen3:1.7b`
- [ ] создан файл `.env`
- [ ] новый `parsing.py` лежит рядом с `app.py`
- [ ] backend запускается без ошибок
- [ ] `/docs` открывается
- [ ] резюме загружается
- [ ] `/api/parsing/run` возвращает `DONE`

---

## 18. Итог

Текущая реализация парсера рассчитана на локальную работу без платных API.
Она подходит для учебного проекта, демо и защиты, если тебе нужно:

- принимать PDF/DOCX,
- извлекать текст,
- отправлять его в локальную модель,
- получать единый JSON,
- сохранять всё в БД,
- передавать профиль дальше в scoring/screening.

Если качество парсинга на конкретных резюме ещё плавает, это нормально. Самый эффективный путь улучшения дальше:

1. донастроить prompt,
2. улучшить post-processing после LLM,
3. аккуратнее нормализовать skills/experience,
4. собрать небольшой eval-набор из 10–20 резюме.

