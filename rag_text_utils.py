import re
from typing import Iterable, List


_STOPWORDS = {
    "and",
    "or",
    "the",
    "with",
    "for",
    "from",
    "that",
    "this",
    "will",
    "have",
    "has",
    "our",
    "your",
    "you",
    "are",
    "need",
    "must",
    "nice",
    "required",
    "requirements",
    "responsibilities",
    "опыт",
    "нужно",
    "важно",
    "будет",
    "ищем",
    "команда",
    "работа",
    "релизы",
    "метрики",
    "обратной",
    "связью",
    "пользователей",
    "уметь",
    "писать",
    "понятные",
    "тикеты",
}


_ALIAS_REPLACEMENTS = [
    ("ё", "е"),
    ("next.js", "nextjs"),
    ("next js", "nextjs"),
    ("node.js", "nodejs"),
    ("node js", "nodejs"),
    ("postgresql", "postgres"),
    ("postgre sql", "postgres"),
    ("ci/cd", "cicd"),
    ("ci cd", "cicd"),
    ("a/b testing", "abtesting"),
    ("ab testing", "abtesting"),
    ("a b testing", "abtesting"),
    ("type script", "typescript"),
    ("rest api", "rest"),
    ("job to be done", "jtbd"),
    ("jobs to be done", "jtbd"),
]


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).replace("\u00a0", " ").replace("\ufeff", "")
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize(text: str) -> str:
    t = clean_text(text).lower()
    t = t.replace("–", "-").replace("—", "-")
    for src, dst in _ALIAS_REPLACEMENTS:
        t = t.replace(src, dst)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: List[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for p in paragraphs:
        if len(p) <= chunk_size:
            if not buf:
                buf = p
            elif len(buf) + len(p) + 2 <= chunk_size:
                buf = (buf + "\n\n" + p).strip()
            else:
                flush()
                buf = p
            continue

        flush()
        start = 0
        while start < len(p):
            end = min(len(p), start + chunk_size)
            part = p[start:end].strip()
            if part:
                chunks.append(part)
            if end >= len(p):
                break
            start = max(end - overlap, start + 1)

    flush()

    out: List[str] = []
    seen = set()
    for c in chunks:
        cc = clean_text(c)
        if not cc:
            continue
        key = normalize(cc)
        if key in seen:
            continue
        seen.add(key)
        out.append(cc)
    return out


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Zа-яА-Я0-9+#./-]{2,}", normalize(text))


def skillish_keywords(values: Iterable[str], limit: int = 64) -> List[str]:
    out: List[str] = []
    seen = set()

    for value in values:
        text = clean_text(str(value or ""))
        if not text:
            continue

        norm = normalize(text)
        if norm and norm not in seen and (
            " " in norm
            or "+" in norm
            or "#" in norm
            or "." in norm
            or "/" in norm
            or norm in {
                "python", "django", "fastapi", "flask", "postgres", "redis", "docker", "kubernetes",
                "react", "typescript", "nextjs", "git", "rest", "graphql", "sql", "pandas",
                "tableau", "abtesting", "airflow", "spark", "mlflow", "playwright", "selenium",
                "jira", "cicd", "celery", "product management", "analytics", "roadmap", "jtbd"
            }
        ):
            seen.add(norm)
            out.append(text)

        for tok in _tokenize(text):
            if tok in _STOPWORDS:
                continue
            if len(tok) < 3:
                continue
            if tok in seen:
                continue
            seen.add(tok)
            out.append(tok)
            if len(out) >= limit:
                return out

    return out[:limit]


def keyword_coverage(resume_text: str, required_keywords: List[str]) -> float:
    if not required_keywords:
        return 0.0

    res = normalize(resume_text)
    seen = set()
    hits = 0
    total = 0

    for kw in required_keywords:
        kk = normalize(kw)
        if not kk or kk in seen:
            continue
        seen.add(kk)
        total += 1
        if kk in res:
            hits += 1

    if total == 0:
        return 0.0
    return round((hits / total) * 100.0, 2)