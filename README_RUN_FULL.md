# AI Hiring Assistant — полный README по запуску проекта

Этот README описывает запуск проекта с нуля на **Windows**, **macOS** и **Linux**, включая:
- подготовку окружения;
- установку зависимостей;
- запуск **FastAPI backend**;
- запуск **Telegram-бота**;
- установку и проверку **Ollama**;
- создание Telegram-бота через **BotFather**;
- получение данных для **Google Calendar API**;
- очистку хвостов от прошлых запусков;
- частые ошибки и способы их исправления.

---

## 1. Что это за проект

Проект состоит из нескольких частей:

1. **Backend** на FastAPI
   - `app.py`
   - `database.py`
   - `parsing.py`
   - `scoring_rag.py`
   - `calendar_llm.py`

2. **Telegram-бот**
   - `tg_bot.py`

3. **Локальная база**
   - `hr_assistant.db` (SQLite)

4. **Хранилище резюме**
   - папка `storage_resumes/`

5. **Web UI**
   - папка `templates/`
   - папка `static/`

6. **LLM через Ollama**
   - используется для разбора резюме, summary, feedback и части логики

---

## 2. Структура проекта

В рабочем состоянии в корне проекта должны быть примерно такие файлы и папки:

```text
app.py
calendar_llm.py
database.py
ollama_scoring.py
parsing.py
rag_text_utils.py
requirements.txt
requirements_resume_parser.txt
scoring_rag.py
tg_bot.py
hr_assistant.db
.env

static/
templates/
storage_resumes/
data/
tests/
```

Минимально для запуска критичны:

```text
app.py
database.py
parsing.py
scoring_rag.py
tg_bot.py
calendar_llm.py
requirements.txt
.env
static/
templates/
storage_resumes/
```

---

## 3. Что нужно установить заранее

### Для всех ОС

Нужно установить:
- **Python 3.11+**
- **pip**
- **Git** (желательно)
- **Ollama**
- **VS Code** (рекомендуется)
- Telegram на телефоне или ПК
- Google-аккаунт для Calendar API

### Проверка версий

```bash
python --version
pip --version
```

Если у вас Linux/macOS и команда `python` не найдена, попробуйте:

```bash
python3 --version
pip3 --version
```

---

## 4. Как открыть проект

### В VS Code
1. Откройте VS Code.
2. Выберите **File → Open Folder**.
3. Выберите папку проекта.
4. Откройте встроенный терминал: **Terminal → New Terminal**.

---

## 5. Подготовка проекта после старых запусков

Если проект раньше уже запускался, сначала очистите старые процессы и временные данные.

### 5.1 Linux / macOS

#### Остановить backend и бота
```bash
pkill -f "uvicorn app:app" || true
pkill -f "python tg_bot.py" || true
```

#### Проверить процессы
```bash
ps aux | grep -E "uvicorn|tg_bot.py|ollama"
```

#### Освободить порт 8000
```bash
fuser -k 8000/tcp || true
```

#### Проверить порт 11434 (Ollama)
```bash
lsof -i :11434
```

Если Ollama уже запущен как системный сервис, **не надо** запускать его второй раз. Просто проверьте, что он жив:

```bash
curl http://127.0.0.1:11434/api/tags
```

Если хотите перезапустить системный сервис Ollama:

```bash
sudo systemctl restart ollama
```

#### Очистить старую базу и резюме
```bash
rm -f hr_assistant.db
rm -rf storage_resumes/*
mkdir -p storage_resumes
```

---

### 5.2 Windows PowerShell

#### Найти и остановить Python-процессы
```powershell
Get-Process python, python3, uvicorn -ErrorAction SilentlyContinue
```

Остановить конкретный процесс:
```powershell
Stop-Process -Id PID -Force
```

#### Проверить, кто держит порт 8000
```powershell
netstat -ano | findstr :8000
```

Остановить процесс по PID:
```powershell
taskkill /PID PID /F
```

#### Проверить Ollama
```powershell
curl http://127.0.0.1:11434/api/tags
```

#### Очистить базу и резюме
```powershell
Remove-Item hr_assistant.db -ErrorAction SilentlyContinue
Remove-Item storage_resumes\* -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force storage_resumes | Out-Null
```

---

## 6. Создание виртуального окружения

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows CMD
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

Проверка:

```bash
python --version
which python
```

На Windows:

```powershell
python --version
where python
```

---

## 7. Установка зависимостей

### Основные зависимости проекта

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Если хотите поставить и доп. зависимости для parser-части

```bash
pip install -r requirements_resume_parser.txt
```

Если файла нет или он не нужен — можно пропустить.

---

## 8. Установка Ollama

Ollama нужен для:
- разбора резюме;
- генерации summary;
- генерации feedback;
- некоторых LLM-функций внутри проекта.

### Linux/macOS
Установите Ollama по инструкции с официального сайта Ollama.

После установки проверьте:

```bash
ollama --version
```

### Windows
Установите Ollama через официальный установщик для Windows, затем откройте новый терминал и проверьте:

```powershell
ollama --version
```

---

## 9. Загрузка модели Ollama

Проект ожидает модель `qwen3:1.7b`.

Скачайте её один раз:

```bash
ollama pull qwen3:1.7b
```

Проверить список моделей:

```bash
ollama list
```

Ожидаемый результат — в списке есть:

```text
qwen3:1.7b
```

---

## 10. Запуск Ollama

### Если Ollama ещё не запущен

```bash
ollama serve
```

### Если видите ошибку:

```text
bind: address already in use
```

это значит, что Ollama **уже работает**. Тогда запускать её второй раз не надо.

Просто проверьте:

```bash
curl http://127.0.0.1:11434/api/tags
```

Если вернулся JSON со списком моделей — всё нормально.

---

## 11. Создание файла `.env`

Создайте в корне проекта файл **`.env`**.

Минимальный рабочий пример:

```env
BOT_TOKEN=PASTE_YOUR_TELEGRAM_BOT_TOKEN_HERE
API_BASE=http://127.0.0.1:8000

OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_TIMEOUT=120

OLLAMA_URL=http://127.0.0.1:11434/api/generate
OLLAMA_TIMEOUT_SECONDS=600
OLLAMA_MAX_INPUT_CHARS=5000
OLLAMA_NUM_PREDICT=900
OLLAMA_MAX_RETRIES=3

RESUME_STORAGE_DIR=./storage_resumes
DATABASE_URL=sqlite+aiosqlite:///./hr_assistant.db

GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=http://127.0.0.1:8000/api/google/callback
GOOGLE_SCOPES=https://www.googleapis.com/auth/calendar https://www.googleapis.com/auth/calendar.events
```

### Пояснение переменных

- `BOT_TOKEN` — токен Telegram-бота от BotFather
- `API_BASE` — адрес FastAPI backend
- `RECRUITER_ID` — дефолтный recruiter id
- `OLLAMA_BASE_URL` — базовый адрес Ollama
- `OLLAMA_URL` — endpoint генерации
- `RESUME_STORAGE_DIR` — папка для хранения файлов резюме
- `DATABASE_URL` — SQLite база
- `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` — учётные данные Google OAuth
- `GOOGLE_REDIRECT_URI` — callback для Google OAuth
- `GOOGLE_SCOPES` — доступ к календарю

### Можно ли оставить Google-переменные пустыми?

Да. Тогда проект сможет работать в **demo-режиме** календаря.

---

## 12. Как создать Telegram-бота и получить токен

Создание бота делается через **@BotFather** в Telegram.

### Пошагово

1. Откройте Telegram.
2. Найдите **@BotFather**.
3. Нажмите **Start**.
4. Отправьте команду:

```text
/newbot
```

5. BotFather попросит:
   - **название бота** — любое, например:
     - `AI Hiring Assistant Demo`
   - **username бота** — должен заканчиваться на `bot`, например:
     - `ai_hiring_demo_askar_bot`

6. После создания BotFather пришлёт **токен**. Он выглядит примерно так:

```text
1234567890:AAExampleExampleExampleExampleExample
```

7. Скопируйте этот токен в `.env`:

```env
BOT_TOKEN=1234567890:AAExampleExampleExampleExampleExample
```

### Полезные команды BotFather

- `/mybots` — список ваших ботов
- `/revoke` — перевыпустить токен
- `/setdescription` — описание бота
- `/setuserpic` — картинка бота
- `/setcommands` — команды бота

### Если токен утёк
Сразу перевыпустите его через BotFather и обновите `.env`.

---

## 13. Как получить данные для Google Calendar API

Проект использует OAuth для подключения календаря рекрутера.

Нужны:
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REDIRECT_URI`

### 13.1 Создайте Google Cloud Project

1. Откройте Google Cloud Console.
2. Создайте новый проект.
3. Дайте ему имя, например:
   - `ai-hiring-assistant`

### 13.2 Включите Google Calendar API

1. В проекте откройте раздел API/Services.
2. Найдите **Google Calendar API**.
3. Нажмите **Enable**.

### 13.3 Настройте OAuth consent screen

1. Откройте раздел Google Auth platform / Branding.
2. Нажмите **Get Started**, если проект ещё не настроен.
3. Заполните:
   - App name
   - User support email
   - Contact email
4. Выберите тип аудитории:
   - для локального теста обычно подходит **External**
   - если у вас Google Workspace внутри организации, можно **Internal**
5. Сохраните настройки.

### 13.4 Создайте OAuth Client ID

Для этого проекта нужен **Web application**, потому что backend принимает callback на адрес:

```text
http://127.0.0.1:8000/api/google/callback
```

Шаги:
1. Откройте Google Auth platform / Clients.
2. Нажмите **Create Client**.
3. Выберите **Application type → Web application**.
4. Задайте имя, например:
   - `AI Hiring Assistant Local`
5. В раздел **Authorized redirect URIs** добавьте:

```text
http://127.0.0.1:8000/api/google/callback
http://127.0.0.1:8000/api/auth/google/callback
```

Если планируете запуск с `localhost`, можно добавить и второй вариант:

```text
http://localhost:8000/api/google/callback
http://localhost:8000/api/auth/google/callback
```

6. Нажмите **Create**.
7. Скопируйте:
   - Client ID
   - Client Secret

### 13.5 Заполните `.env`

```env
GOOGLE_CLIENT_ID=ваш_client_id
GOOGLE_CLIENT_SECRET=ваш_client_secret
GOOGLE_REDIRECT_URI=http://127.0.0.1:8000/api/google/callback
GOOGLE_SCOPES=https://www.googleapis.com/auth/calendar https://www.googleapis.com/auth/calendar.events
```

### 13.6 Нужно ли скачивать JSON-файл?

Для текущего кода проекта — **нет обязательно**. Здесь удобнее просто взять **Client ID** и **Client Secret** и положить их в `.env`.

---

## 14. Создание обязательных папок

Если папок нет, создайте их:

### Linux / macOS
```bash
mkdir -p templates static storage_resumes
```

### Windows PowerShell
```powershell
New-Item -ItemType Directory -Force templates | Out-Null
New-Item -ItemType Directory -Force static | Out-Null
New-Item -ItemType Directory -Force storage_resumes | Out-Null
```

---

## 15. Запуск backend

### Linux / macOS

```bash
cd ~/merged_project
source .venv/bin/activate
fuser -k 8000/tcp || true
uvicorn app:app --reload
```

### Windows PowerShell

```powershell
cd C:\path\to\final_merged_project
.venv\Scripts\Activate.ps1
uvicorn app:app --reload
```

### Что вы должны увидеть

```text
Uvicorn running on http://127.0.0.1:8000
```

### Проверка в браузере

Откройте:

```text
http://127.0.0.1:8000
```

Если страница открылась — backend работает.

---

## 16. Запуск Telegram-бота

### Linux / macOS

Откройте **второй терминал**:

```bash
cd ~/merged_project
source .venv/bin/activate
python tg_bot.py
```

### Windows PowerShell

Откройте **второе окно PowerShell**:

```powershell
cd C:\path\to\final_merged_project
.venv\Scripts\Activate.ps1
python tg_bot.py
```

Если токен правильный — бот запустится без ошибок.

---

## 17. Как подключить Google Calendar в интерфейсе проекта

1. Запустите backend.
2. Откройте в браузере:

```text
http://127.0.0.1:8000/integrations
```

3. Нажмите кнопку подключения Google.
4. Выполните вход в Google.
5. Подтвердите доступ к календарю.
6. После возврата на callback backend сохранит токены.

Если `GOOGLE_CLIENT_ID` и `GOOGLE_CLIENT_SECRET` пустые, проект будет работать в demo-режиме календаря.

---

## 18. Первый полный тест системы

После запуска backend и бота:

1. Откройте web UI:
   - `http://127.0.0.1:8000`
2. Создайте вакансию как recruiter.
3. Откройте Telegram-бота.
4. Нажмите `/start`.
5. Выберите вакансию.
6. Отправьте резюме PDF или DOCX.
7. Дождитесь обработки резюме.
8. Ответьте на screening questions.
9. Получите:
   - либо feedback,
   - либо интервью-слоты.
10. При наличии Google Calendar подключите календарь и забронируйте слот.

---

## 19. Если хотите запускать совсем с нуля

### Linux / macOS — полный reset

```bash
cd ~/final_merged_project
pkill -f "uvicorn app:app" || true
pkill -f "python tg_bot.py" || true
fuser -k 8000/tcp || true
rm -f hr_assistant.db
rm -rf storage_resumes/*
mkdir -p storage_resumes
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Потом отдельно:

**Терминал 1**
```bash
curl http://127.0.0.1:11434/api/tags
```
Если Ollama отвечает — хорошо. Если нет:
```bash
ollama serve
```

**Терминал 2**
```bash
cd ~/final_merged_project
source .venv/bin/activate
uvicorn app:app --reload
```

**Терминал 3**
```bash
cd ~/final_merged_project
source .venv/bin/activate
python tg_bot.py
```

---

## 20. Если порт уже занят

### Порт 8000 занят

#### Linux / macOS
```bash
lsof -i :8000
fuser -k 8000/tcp || true
```

#### Windows
```powershell
netstat -ano | findstr :8000
taskkill /PID PID /F
```

### Порт 11434 занят

Это почти всегда значит, что Ollama уже работает.

Проверьте:

```bash
curl http://127.0.0.1:11434/api/tags
```

Если есть ответ — оставьте всё как есть.

---

## 21. Частые ошибки и решения

### Ошибка: `BOT_TOKEN is empty`
Причина: не заполнен `BOT_TOKEN` в `.env`.

Решение:
```env
BOT_TOKEN=ваш_реальный_токен
```

---

### Ошибка: `Client error '405 Method Not Allowed' for url 'http://127.0.0.1:11434'`
Причина: запрос идёт в корень Ollama вместо `/api/generate`.

Решение:
```env
OLLAMA_URL=http://127.0.0.1:11434/api/generate
```

Или убедитесь, что код использует `.../api/generate`.

---

### Ошибка: `409 Conflict` на `/decision`
Причина: бот вызывает финальное решение раньше, чем закончился парсинг резюме.

Решение:
- дождаться `parse_status == DONE`;
- убедиться, что в `tg_bot.py` есть polling статуса документа перед вызовом `/decision`.

---

### Ошибка: `candidate profile not ready`
Причина: парсинг не завершён или упал.

Проверьте:
- Ollama запущен;
- модель скачана;
- резюме сохранилось в `storage_resumes/`;
- document status не равен `ERROR`.

---

### Ошибка: `bind: address already in use`
Причина: процесс уже слушает порт.

Решение:
- либо остановить старый процесс;
- либо не запускать второй экземпляр.

---

### Ошибка подключения Google OAuth
Проверьте:
- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REDIRECT_URI`
- redirect URI в Google Cloud Console совпадает с `.env`

Если в `.env` указано:

```env
GOOGLE_REDIRECT_URI=http://127.0.0.1:8000/api/google/callback
```

то **точно такой же URI** должен быть добавлен в Google Cloud.

---

## 22. Рекомендуемая схема запуска

Лучше всего работать в трёх терминалах.

### Терминал 1 — Ollama
```bash
curl http://127.0.0.1:11434/api/tags
```
Если не отвечает:
```bash
ollama serve
```

### Терминал 2 — backend
```bash
cd ~/final_merged_project
source .venv/bin/activate
uvicorn app:app --reload
```

### Терминал 3 — bot
```bash
cd ~/final_merged_project
source .venv/bin/activate
python tg_bot.py
```

---

## 23. Рекомендуемая последовательность проверки

1. Проверить Ollama:
```bash
curl http://127.0.0.1:11434/api/tags
```

2. Проверить backend:
```bash
curl http://127.0.0.1:8000/api/public/jobs
```

3. Проверить web UI:
- открыть `http://127.0.0.1:8000`

4. Проверить бота:
- открыть Telegram
- отправить `/start`

---

## 24. Что делать, если проект был случайно повреждён

Если были удалены файлы:
1. восстановите `.py` модули;
2. восстановите `templates/`;
3. восстановите `static/`;
4. создайте `storage_resumes/`;
5. заново создайте `.env`;
6. переустановите зависимости;
7. удалите старую БД и запустите снова.

---

## 25. Минимальный быстрый старт

Если хотите максимально короткий путь:

1. Создайте `.env`.
2. Установите зависимости:
```bash
pip install -r requirements.txt
```
3. Убедитесь, что Ollama работает:
```bash
curl http://127.0.0.1:11434/api/tags
```
4. Запустите backend:
```bash
uvicorn app:app --reload
```
5. Запустите бота:
```bash
python tg_bot.py
```
6. Создайте вакансию через web UI.
7. Пройдите сценарий через Telegram.

---

## 26. Финальная памятка

Для нормальной работы проекта должны быть одновременно готовы:

- Python-окружение
- зависимости из `requirements.txt`
- корректный `.env`
- работающий backend на `127.0.0.1:8000`
- работающий Ollama на `127.0.0.1:11434`
- загруженная модель `qwen3:1.7b`
- рабочий Telegram-бот с валидным токеном
- при необходимости — подключённый Google Calendar

Если что-то из этого отсутствует, проект либо не стартует, либо проходит сценарий не до конца.
