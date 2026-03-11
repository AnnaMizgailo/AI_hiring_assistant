import re
from typing import List


SECTION_HINTS = [
    "skills",
    "technical skills",
    "навыки",
    "технические навыки",

    "experience",
    "professional experience",
    "опыт",
    "опыт работы",

    "projects",
    "проекты",

    "education",
    "образование",

    "certifications",
    "certificates",
    "сертификаты",
]


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize(text: str) -> str:
    t = clean_text(text).lower()
    t = t.replace("ё", "е")
    t = t.replace("–", "-").replace("—", "-")
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf = ""

    def flush_buf() -> None:
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for p in paragraphs:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            flush_buf()
            if len(p) <= chunk_size:
                buf = p
            else:
                start = 0
                while start < len(p):
                    end = min(start + chunk_size, len(p))
                    chunks.append(p[start:end].strip())
                    if end >= len(p):
                        break
                    start = max(end - overlap, start + 1)
                buf = ""

    flush_buf()

    if overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = []
        for i, c in enumerate(chunks):
            if i == 0:
                overlapped.append(c)
            else:
                prev = overlapped[-1]
                take = prev[-overlap:] if len(prev) > overlap else prev
                overlapped.append((take + "\n" + c).strip())
        chunks = overlapped

    return [c for c in chunks if c]


def keyword_coverage(resume_text: str, job_keywords: List[str]) -> float:
    if not job_keywords:
        return 0.0

    res = normalize(resume_text)
    hits = 0
    seen = set()

    for kw in job_keywords:
        nk = normalize(kw)
        if not nk or nk in seen:
            continue
        seen.add(nk)
        if nk in res:
            hits += 1

    if not seen:
        return 0.0

    return round((hits / len(seen)) * 100, 2)


def extract_resume_query(resume_text: str, max_chars: int = 1800) -> str:
    t = clean_text(resume_text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]

    keep: List[str] = []
    for ln in lines:
        low = ln.lower()
        if any(h in low for h in SECTION_HINTS):
            keep.append(ln)
            continue
        if ("," in ln and len(ln) <= 160) or ("-" in ln and len(ln) <= 160):
            keep.append(ln)

    if len(" ".join(keep)) < 400:
        keep = lines[:30]

    query = " ".join(keep)
    query = re.sub(r"\s+", " ", query).strip()
    return query[:max_chars]
