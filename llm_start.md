# Local RAG Scoring with Ollama

## Требования к установке

### Python

```
Python 3.10+
```
### Ollama

Установка Ollama:

https://ollama.com/download

Проверка:

```
ollama --version
```

---

# 2. Установка LLM модели
Пока очень маленькая, слабенькая
```
ollama pull qwen3:1.7b
```
Проверка:

```
ollama list
```

Должны увидеть:

```
qwen3:1.7b
```

---

# 3. Start Ollama

```
ollama serve
```

By default it runs on:

```
http://localhost:11434
```

---

# 5. Зависимости

```
pip install numpy scikit-learn requests sqlalchemy asyncpg greenlet python-dotenv
```

---

# 6. Переменные среды

Опционально установить:

### Mac / Linux

```
export OLLAMA_MODEL=qwen3:1.7b
```

### Windows PowerShell

```
$env:OLLAMA_MODEL="qwen3:1.7b"
```
---