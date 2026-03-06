
from __future__ import annotations

import datetime as dt
import hashlib
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import Application, Candidate, Document, Profile

logger = logging.getLogger(__name__)

STORAGE_DIR = Path(os.getenv("RESUME_STORAGE_DIR", "./storage_resumes"))
OCR_ENABLED = os.getenv("RESUME_PARSER_ENABLE_OCR", "0") == "1"


class ResumeParsingError(RuntimeError):
    """Base parser error."""


class TextExtractionError(ResumeParsingError):
    """Raised when a document cannot be converted to text."""


@dataclass
class ExtractionResult:
    text: str
    method: str
    quality_score: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class ParsedDateRange:
    start: Optional[dt.date]
    end: Optional[dt.date]
    start_iso: Optional[str]
    end_iso: Optional[str]
    is_present: bool
    raw: str
    confidence: float


MONTH_ALIASES: Dict[str, int] = {
    "jan": 1,
    "january": 1,
    "янв": 1,
    "январь": 1,
    "января": 1,
    "feb": 2,
    "february": 2,
    "фев": 2,
    "февраль": 2,
    "февраля": 2,
    "mar": 3,
    "march": 3,
    "мар": 3,
    "март": 3,
    "марта": 3,
    "apr": 4,
    "april": 4,
    "апр": 4,
    "апрель": 4,
    "апреля": 4,
    "may": 5,
    "май": 5,
    "мая": 5,
    "jun": 6,
    "june": 6,
    "июн": 6,
    "июнь": 6,
    "июня": 6,
    "jul": 7,
    "july": 7,
    "июл": 7,
    "июль": 7,
    "июля": 7,
    "aug": 8,
    "august": 8,
    "авг": 8,
    "август": 8,
    "августа": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "сен": 9,
    "сент": 9,
    "сентябрь": 9,
    "сентября": 9,
    "oct": 10,
    "october": 10,
    "окт": 10,
    "октябрь": 10,
    "октября": 10,
    "nov": 11,
    "november": 11,
    "ноя": 11,
    "ноябрь": 11,
    "ноября": 11,
    "dec": 12,
    "december": 12,
    "дек": 12,
    "декабрь": 12,
    "декабря": 12,
}

PRESENT_MARKERS = {
    "present",
    "current",
    "now",
    "till now",
    "to date",
    "по настоящее время",
    "по наст. время",
    "настоящее время",
    "настоящее время",
    "н.в",
    "н/в",
    "по н.в",
    "по н/в",
    "currently",
    "current time",
    "сейчас",
}

ROLE_KEYWORDS = {
    "developer",
    "engineer",
    "scientist",
    "analyst",
    "manager",
    "architect",
    "intern",
    "lead",
    "head",
    "consultant",
    "administrator",
    "specialist",
    "backend",
    "frontend",
    "fullstack",
    "full-stack",
    "qa",
    "sdet",
    "devops",
    "sre",
    "data",
    "ml",
    "ai",
    "python",
    "java",
    "go",
    "product",
    "project",
    "research",
    "designer",
    "marketing",
    "sales",
    "hr",
    "рaзработчик",
    "разработчик",
    "инженер",
    "аналитик",
    "менеджер",
    "архитектор",
    "стажер",
    "стажёр",
    "тимлид",
    "руководитель",
    "администратор",
    "специалист",
    "бэкенд",
    "бекенд",
    "фронтенд",
    "фулстек",
    "qa",
    "тестировщик",
    "девопс",
    "дата",
    "машин",
    "продукт",
    "проект",
    "дизайнер",
    "маркетолог",
    "рекрутер",
}

COMPANY_HINTS = {
    "llc",
    "ltd",
    "inc",
    "corp",
    "gmbh",
    "company",
    "bank",
    "group",
    "solutions",
    "systems",
    "studio",
    "labs",
    "lab",
    "technologies",
    "technology",
    "university",
    "college",
    "school",
    "academy",
    "ооо",
    "ао",
    "ип",
    "банк",
    "университет",
    "институт",
    "колледж",
    "академия",
    "школа",
    "компания",
    "студия",
    "лаборатория",
}

DEGREE_HINTS = {
    "bachelor",
    "master",
    "phd",
    "mba",
    "b.sc",
    "m.sc",
    "specialist",
    "бакалавр",
    "магистр",
    "специалист",
    "аспирант",
    "аспирантура",
    "кандидат наук",
    "доктор наук",
    "bootcamp",
    "курс",
    "курсы",
}

LOCATION_LABELS = [
    "location",
    "city",
    "address",
    "based in",
    "местоположение",
    "локация",
    "город",
    "адрес",
]

POSITION_LABELS = [
    "desired position",
    "position",
    "role",
    "target role",
    "objective",
    "должность",
    "позиция",
    "желаемая должность",
    "цель",
]

NAME_LABELS = [
    "full name",
    "name",
    "фио",
]

SUMMARY_LABELS = [
    "summary",
    "about",
    "profile",
    "objective",
    "about me",
    "professional summary",
    "о себе",
    "обо мне",
    "профиль",
    "кратко",
]

SECTION_ALIASES: Dict[str, Sequence[str]] = {
    "summary": [
        "summary",
        "about",
        "about me",
        "profile",
        "objective",
        "professional summary",
        "career summary",
        "о себе",
        "обо мне",
        "профиль",
        "кратко",
        "резюме",
    ],
    "skills": [
        "skills",
        "key skills",
        "technical skills",
        "core skills",
        "tech stack",
        "technology stack",
        "stack",
        "technologies",
        "tools",
        "инструменты",
        "навыки",
        "ключевые навыки",
        "технические навыки",
        "стек",
        "стек технологий",
        "технологии",
        "ключевые компетенции",
    ],
    "experience": [
        "experience",
        "work experience",
        "employment",
        "professional experience",
        "career",
        "projects",
        "project experience",
        "опыт",
        "опыт работы",
        "профессиональный опыт",
        "карьера",
        "проекты",
    ],
    "education": [
        "education",
        "academic background",
        "education and training",
        "образование",
        "обучение",
        "учеба",
        "учёба",
    ],
    "languages_human": [
        "languages",
        "language",
        "language proficiency",
        "spoken languages",
        "foreign languages",
        "языки",
        "иностранные языки",
        "язык",
    ],
    "contacts": [
        "contacts",
        "contact information",
        "contact",
        "контакты",
    ],
}

TECH_TAXONOMY: Dict[str, Dict[str, Sequence[str]]] = {
    "languages": {
        "Python": ["python", "py", "python 3", "python3"],
        "SQL": ["sql"],
        "JavaScript": ["javascript", "js"],
        "TypeScript": ["typescript", "ts"],
        "Java": ["java"],
        "C++": ["c++"],
        "C#": ["c#", "c sharp"],
        "Go": ["golang", "go"],
        "Rust": ["rust"],
        "Kotlin": ["kotlin"],
        "PHP": ["php"],
        "Ruby": ["ruby"],
        "Swift": ["swift"],
        "Scala": ["scala"],
        "R": ["r language", " r "],
        "Bash": ["bash", "shell", "shell scripting"],
        "MATLAB": ["matlab"],
    },
    "frameworks": {
        "FastAPI": ["fastapi", "fast api"],
        "Django": ["django"],
        "Flask": ["flask"],
        "aiohttp": ["aiohttp"],
        "DRF": ["django rest framework", "drf"],
        "SQLAlchemy": ["sqlalchemy"],
        "Pydantic": ["pydantic"],
        "Celery": ["celery"],
        "Airflow": ["apache airflow", "airflow"],
        "Pandas": ["pandas"],
        "NumPy": ["numpy"],
        "scikit-learn": ["scikit-learn", "sklearn"],
        "PyTorch": ["pytorch"],
        "TensorFlow": ["tensorflow"],
        "Keras": ["keras"],
        "Hugging Face Transformers": ["huggingface transformers", "transformers", "hugging face"],
        "React": ["react", "react.js", "reactjs"],
        "Next.js": ["next.js", "nextjs"],
        "Vue": ["vue", "vue.js", "vuejs"],
        "Angular": ["angular"],
        "Express": ["express", "express.js", "expressjs"],
        "NestJS": ["nestjs", "nest.js"],
        "Spring": ["spring"],
        "Spring Boot": ["spring boot"],
        ".NET": [".net", "dotnet", "asp.net", "asp.net core"],
        "Laravel": ["laravel"],
        "Ruby on Rails": ["rails", "ruby on rails"],
    },
    "technologies": {
        "PostgreSQL": ["postgresql", "postgres", "postgre"],
        "MySQL": ["mysql"],
        "SQLite": ["sqlite"],
        "Redis": ["redis"],
        "MongoDB": ["mongodb", "mongo"],
        "Elasticsearch": ["elasticsearch", "elastic"],
        "OpenSearch": ["opensearch"],
        "ClickHouse": ["clickhouse"],
        "Kafka": ["kafka", "apache kafka"],
        "RabbitMQ": ["rabbitmq", "rabbit mq"],
        "REST": ["rest api", "rest"],
        "GraphQL": ["graphql"],
        "gRPC": ["grpc"],
        "Microservices": ["microservices", "microservice", "микросервисы", "микросервисная архитектура"],
        "Docker": ["docker"],
        "Kubernetes": ["kubernetes", "k8s"],
        "Linux": ["linux"],
        "Nginx": ["nginx"],
        "CI/CD": ["ci/cd", "cicd", "continuous integration", "continuous delivery"],
        "Git": ["git"],
        "GitHub Actions": ["github actions"],
        "GitLab CI": ["gitlab ci", "gitlab-ci"],
        "Jenkins": ["jenkins"],
        "Terraform": ["terraform"],
        "Ansible": ["ansible"],
        "AWS": ["aws", "amazon web services"],
        "GCP": ["gcp", "google cloud", "google cloud platform"],
        "Azure": ["azure", "microsoft azure"],
        "Prometheus": ["prometheus"],
        "Grafana": ["grafana"],
        "Apache Spark": ["spark", "apache spark"],
        "Hadoop": ["hadoop"],
        "MLflow": ["mlflow"],
        "OpenCV": ["opencv"],
        "NLP": ["nlp", "natural language processing", "обработка естественного языка"],
        "LLM": ["llm", "large language model", "large language models", "gpt", "ollama"],
        "RAG": ["rag", "retrieval augmented generation", "retrieval-augmented generation"],
        "Computer Vision": ["computer vision", "cv"],
        "ETL": ["etl"],
        "Data Analysis": ["data analysis", "аналитика данных"],
        "Pipelines": ["pipeline", "pipelines", "data pipeline", "data pipelines"],
        "Vector Databases": ["vector database", "vector databases", "pgvector", "milvus", "weaviate", "pinecone", "qdrant"],
    },
    "tools": {
        "Docker": ["docker"],
        "Kubernetes": ["kubernetes", "k8s"],
        "Git": ["git"],
        "GitHub": ["github"],
        "GitLab": ["gitlab"],
        "Jira": ["jira"],
        "Confluence": ["confluence"],
        "Notion": ["notion"],
        "Postman": ["postman"],
        "Figma": ["figma"],
        "Slack": ["slack"],
        "Trello": ["trello"],
        "VS Code": ["vscode", "vs code", "visual studio code"],
        "PyCharm": ["pycharm"],
        "Linux": ["linux"],
        "Grafana": ["grafana"],
        "Prometheus": ["prometheus"],
        "Terraform": ["terraform"],
        "Ansible": ["ansible"],
        "Jenkins": ["jenkins"],
        "GitHub Actions": ["github actions"],
        "GitLab CI": ["gitlab ci", "gitlab-ci"],
    },
}


def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = text.replace("\t", " ")
    text = re.sub(r"[ \f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_inline(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;:-")


def normalize_lookup(text: str) -> str:
    text = clean_inline(text).lower().replace("ё", "е")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def dedupe_preserve(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        cleaned = clean_inline(item)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def safe_split_lines(text: str) -> List[str]:
    text = clean_text(text)
    return [line.strip() for line in text.split("\n")]


def strip_bullet(line: str) -> str:
    return re.sub(r"^[\-\*\u2022\u2023\u25E6\u2043\u00b7•]+\s*", "", line).strip()


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text, flags=re.IGNORECASE)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_quality_score(text: str) -> float:
    cleaned = clean_text(text)
    if not cleaned:
        return 0.0
    length_score = min(len(cleaned) / 1200.0, 1.0)
    alpha_chars = sum(ch.isalpha() for ch in cleaned)
    printable_chars = sum(ch.isprintable() and not ch.isspace() for ch in cleaned)
    alpha_ratio = alpha_chars / max(len(cleaned), 1)
    printable_ratio = printable_chars / max(len(cleaned.replace(" ", "")), 1)
    bad_ratio = len(re.findall(r"[□■�]", cleaned)) / max(len(cleaned), 1)
    line_count = max(cleaned.count("\n"), 1)
    line_score = min(line_count / 20.0, 1.0)
    score = (
        0.4 * length_score
        + 0.25 * min(alpha_ratio * 2.0, 1.0)
        + 0.2 * printable_ratio
        + 0.15 * line_score
        - min(bad_ratio * 3.0, 0.3)
    )
    return max(0.0, min(score, 1.0))


def _extension_from_path(file_path: str) -> str:
    return Path(file_path).suffix.lower().strip(".")


def _guess_file_kind(file_path: str, mime_type: str) -> str:
    mime = (mime_type or "").lower()
    ext = _extension_from_path(file_path)
    if "pdf" in mime or ext == "pdf":
        return "pdf"
    if "word" in mime or "officedocument.wordprocessingml" in mime or ext == "docx":
        return "docx"
    if ext in {"txt", "text"} or mime.startswith("text/"):
        return "txt"
    if ext == "doc":
        return "doc"
    return ext or "unknown"


def _extract_text_from_pdf_pymupdf(file_path: str) -> ExtractionResult:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise TextExtractionError("PyMuPDF is not installed") from exc

    warnings: List[str] = []
    parts: List[str] = []
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                parts.append(page.get_text("text", sort=True))
    except Exception as exc:
        raise TextExtractionError(f"PyMuPDF failed to read PDF: {exc}") from exc

    text = clean_text("\n\n".join(parts))
    score = text_quality_score(text)
    if score < 0.22:
        warnings.append("PyMuPDF extracted very little or low-quality text")
    return ExtractionResult(text=text, method="pymupdf", quality_score=score, warnings=warnings)


def _extract_text_from_pdf_pdfplumber(file_path: str) -> ExtractionResult:
    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise TextExtractionError("pdfplumber is not installed") from exc

    warnings: List[str] = []
    parts: List[str] = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=3) or ""
                parts.append(page_text)
    except Exception as exc:
        raise TextExtractionError(f"pdfplumber failed to read PDF: {exc}") from exc

    text = clean_text("\n\n".join(parts))
    score = text_quality_score(text)
    if score < 0.22:
        warnings.append("pdfplumber extracted very little or low-quality text")
    return ExtractionResult(text=text, method="pdfplumber", quality_score=score, warnings=warnings)


def _extract_text_from_pdf_ocr(file_path: str) -> ExtractionResult:
    if not OCR_ENABLED:
        raise TextExtractionError("OCR is disabled")
    try:
        from pdf2image import convert_from_path  # type: ignore
        import pytesseract  # type: ignore
    except ImportError as exc:
        raise TextExtractionError("OCR dependencies are not installed") from exc

    warnings: List[str] = []
    pages = convert_from_path(file_path, dpi=250)
    if not pages:
        raise TextExtractionError("OCR received zero pages")
    texts: List[str] = []
    for image in pages:
        page_text = pytesseract.image_to_string(image, lang="rus+eng")
        texts.append(page_text)
    text = clean_text("\n\n".join(texts))
    score = text_quality_score(text)
    warnings.append("OCR was used because native PDF extraction looked weak")
    return ExtractionResult(text=text, method="ocr", quality_score=score, warnings=warnings)


def _extract_text_from_pdf(file_path: str) -> ExtractionResult:
    attempts: List[ExtractionResult] = []
    errors: List[str] = []

    for extractor in (_extract_text_from_pdf_pymupdf, _extract_text_from_pdf_pdfplumber):
        try:
            attempts.append(extractor(file_path))
        except TextExtractionError as exc:
            errors.append(str(exc))

    if attempts:
        best = max(attempts, key=lambda item: item.quality_score)
        if best.quality_score >= 0.22:
            return best
        if OCR_ENABLED:
            try:
                ocr_result = _extract_text_from_pdf_ocr(file_path)
                if ocr_result.quality_score >= best.quality_score:
                    return ocr_result
            except TextExtractionError as exc:
                errors.append(str(exc))
        best.warnings.extend(errors)
        return best

    if OCR_ENABLED:
        try:
            return _extract_text_from_pdf_ocr(file_path)
        except TextExtractionError as exc:
            errors.append(str(exc))

    raise TextExtractionError("; ".join(errors) or "All PDF extractors failed")


def _extract_text_from_docx(file_path: str) -> ExtractionResult:
    try:
        from docx import Document as DocxDocument  # type: ignore
    except ImportError as exc:
        raise TextExtractionError("python-docx is not installed") from exc

    try:
        doc = DocxDocument(file_path)
    except Exception as exc:
        raise TextExtractionError(f"python-docx failed to read DOCX: {exc}") from exc

    parts: List[str] = []
    for para in doc.paragraphs:
        txt = clean_inline(para.text)
        if txt:
            parts.append(txt)

    for table in doc.tables:
        for row in table.rows:
            cells = [clean_inline(cell.text) for cell in row.cells if clean_inline(cell.text)]
            if cells:
                parts.append(" | ".join(cells))

    text = clean_text("\n".join(parts))
    score = text_quality_score(text)
    warnings: List[str] = []
    if score < 0.22:
        warnings.append("DOCX extracted text looks suspiciously short or noisy")
    return ExtractionResult(text=text, method="python-docx", quality_score=score, warnings=warnings)


def _extract_text_from_txt(file_path: str) -> ExtractionResult:
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            text = Path(file_path).read_text(encoding=encoding)
            text = clean_text(text)
            return ExtractionResult(text=text, method=f"plain-text:{encoding}", quality_score=text_quality_score(text))
        except Exception:
            continue
    raise TextExtractionError("Unable to decode plain text file")


def extract_text_details(file_path: str, mime_type: str) -> ExtractionResult:
    file_kind = _guess_file_kind(file_path, mime_type)
    if file_kind == "pdf":
        result = _extract_text_from_pdf(file_path)
    elif file_kind == "docx":
        result = _extract_text_from_docx(file_path)
    elif file_kind in {"txt", "text"}:
        result = _extract_text_from_txt(file_path)
    elif file_kind == "doc":
        raise TextExtractionError("Legacy .doc is not supported without external converter")
    else:
        raise TextExtractionError(f"Unsupported file type: {file_kind}")

    if not result.text or text_quality_score(result.text) < 0.08:
        raise TextExtractionError("Text extraction returned empty or unusable content")

    return result


def extract_text_from_file(file_path: str, mime_type: str) -> str:
    return extract_text_details(file_path, mime_type).text


def _labeled_value(lines: Sequence[str], labels: Sequence[str], max_lines: int = 40) -> Optional[str]:
    normalized_labels = {normalize_lookup(label) for label in labels}
    for raw_line in lines[:max_lines]:
        line = clean_inline(raw_line)
        if not line:
            continue
        lowered = normalize_lookup(line)
        for label in normalized_labels:
            pattern = re.compile(rf"^{re.escape(label)}\s*[:\-—]?\s*(.+)$", flags=re.IGNORECASE)
            match = pattern.match(lowered)
            if match and match.group(1):
                return clean_inline(match.group(1))
    return None


def _is_probable_heading(line: str) -> Optional[Tuple[str, str]]:
    raw = clean_inline(line)
    if not raw:
        return None
    if len(raw) > 90:
        return None

    normalized = normalize_lookup(raw.rstrip(":"))
    for section, aliases in SECTION_ALIASES.items():
        alias_norms = {normalize_lookup(alias) for alias in aliases}
        if normalized in alias_norms:
            return section, ""
        for alias in alias_norms:
            if normalized.startswith(alias + ":") or normalized.startswith(alias + " -") or normalized.startswith(alias + " —"):
                remainder = re.split(r"[:\-—]", raw, maxsplit=1)[1].strip()
                return section, remainder
    return None


def split_sections(lines: Sequence[str]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {
        "header": [],
        "summary": [],
        "skills": [],
        "experience": [],
        "education": [],
        "languages_human": [],
        "contacts": [],
    }
    current = "header"

    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            sections[current].append("")
            continue

        maybe_heading = _is_probable_heading(cleaned)
        if maybe_heading:
            current, remainder = maybe_heading
            if remainder:
                sections[current].append(remainder)
            continue

        sections.setdefault(current, []).append(cleaned)

    return sections


def _contains_contact_marker(line: str) -> bool:
    lowered = line.lower()
    return bool(
        re.search(r"[\w\.-]+@[\w\.-]+\.\w+", line)
        or re.search(r"(?:\+?\d[\d\-\(\) ]{7,}\d)", line)
        or "linkedin" in lowered
        or "github" in lowered
        or "telegram" in lowered
        or "t.me/" in lowered
    )


def _score_role_like(text: str) -> int:
    lowered = normalize_lookup(text)
    score = 0
    for kw in ROLE_KEYWORDS:
        if kw in lowered:
            score += 2
    if re.search(r"\b(senior|middle|junior|lead|principal|старший|ведущий|младший)\b", lowered):
        score += 1
    if re.search(r"\b(remote|onsite|hybrid|full-time|part-time|удаленно|гибрид)\b", lowered):
        score -= 1
    return score


def _score_company_like(text: str) -> int:
    lowered = normalize_lookup(text)
    score = 0
    for hint in COMPANY_HINTS:
        if hint in lowered:
            score += 2
    if re.search(r"\b(llc|inc|ltd|corp|group|solutions|systems|labs)\b", lowered):
        score += 1
    if re.search(r"\b(ooo|ao)\b", lowered):
        score += 1
    if len(text.split()) <= 4 and _score_role_like(text) == 0 and not _contains_contact_marker(text):
        score += 1
    return score


def is_probable_full_name(text: str) -> bool:
    line = clean_inline(text)
    if not line or len(line) > 70:
        return False
    if _contains_contact_marker(line):
        return False
    if re.search(r"\d", line):
        return False
    if "," in line or "|" in line or "/" in line:
        return False
    if _score_role_like(line) > 0:
        return False
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё'’-]*", line)
    if not 2 <= len(tokens) <= 4:
        return False
    alpha_only = " ".join(tokens)
    if len(alpha_only) < 5:
        return False
    return True


def extract_full_name(lines: Sequence[str]) -> Tuple[Optional[str], float]:
    explicit = _labeled_value(lines, NAME_LABELS, max_lines=30)
    if explicit and is_probable_full_name(explicit):
        return explicit, 0.98

    for line in lines[:12]:
        candidate = clean_inline(line)
        if is_probable_full_name(candidate):
            return candidate, 0.85
    return None, 0.0


def _is_probable_location(line: str) -> bool:
    text = clean_inline(remove_urls(line))
    if not text or len(text) > 70:
        return False
    if _contains_contact_marker(text):
        return False
    if re.search(r"\d", text):
        return False
    if _score_role_like(text) > 0:
        return False
    if text.lower() in {"remote", "hybrid", "onsite", "удаленно", "удалённо", "гибрид"}:
        return True
    if "," in text:
        return True
    words = text.split()
    return 1 <= len(words) <= 4


def extract_location(lines: Sequence[str], name: Optional[str], desired_position: Optional[str]) -> Tuple[Optional[str], float]:
    labeled = _labeled_value(lines, LOCATION_LABELS, max_lines=40)
    if labeled:
        return clean_inline(labeled), 0.96

    for line in lines[:12]:
        candidate = clean_inline(line)
        if not candidate:
            continue
        if name and candidate == name:
            continue
        if desired_position and candidate == desired_position:
            continue
        if _is_probable_location(candidate):
            return candidate, 0.6
    return None, 0.0


def _is_probable_position_line(line: str) -> bool:
    text = clean_inline(line)
    if not text or len(text) > 90:
        return False
    if _contains_contact_marker(text):
        return False
    if is_probable_full_name(text):
        return False
    if re.search(r"\d{4}", text):
        return False
    if _score_role_like(text) > 0:
        return True
    return False


def extract_desired_position(lines: Sequence[str], full_name: Optional[str]) -> Tuple[Optional[str], float]:
    labeled = _labeled_value(lines, POSITION_LABELS, max_lines=40)
    if labeled:
        return clean_inline(labeled), 0.97

    started = False
    for line in lines[:15]:
        candidate = clean_inline(line)
        if not candidate:
            continue
        if full_name and candidate == full_name:
            started = True
            continue
        if _contains_contact_marker(candidate):
            continue
        if _is_probable_position_line(candidate):
            return candidate, 0.78
        if started and len(candidate.split()) <= 8 and len(candidate) <= 80 and not _is_probable_location(candidate):
            return candidate, 0.55
    return None, 0.0


def extract_summary(sections: Dict[str, List[str]], header_lines: Sequence[str], skip_values: Sequence[str]) -> Tuple[Optional[str], float]:
    summary_lines = [clean_inline(line) for line in sections.get("summary", []) if clean_inline(line)]
    if summary_lines:
        text = clean_inline(" ".join(summary_lines[:6]))
        if text:
            return text, 0.92

    skip = {clean_inline(value) for value in skip_values if value}
    candidates: List[str] = []
    for line in header_lines[:20]:
        candidate = clean_inline(line)
        if not candidate or candidate in skip:
            continue
        if _contains_contact_marker(candidate):
            continue
        if _is_probable_heading(candidate):
            continue
        if parse_date_range(candidate) or _is_probable_job_header(candidate):
            if candidates:
                break
            continue
        if len(candidate.split()) < 5:
            continue
        candidates.append(candidate)
    if candidates:
        return clean_inline(" ".join(candidates[:3])), 0.52
    return None, 0.0


def extract_contacts(lines: Sequence[str]) -> Tuple[Dict[str, Any], float]:
    joined = "\n".join(lines)
    emails = dedupe_preserve(match.group(0) for match in re.finditer(r"[\w.\-+%]+@[\w.\-]+\.\w+", joined, flags=re.IGNORECASE))

    phones_raw = dedupe_preserve(match.group(0) for match in re.finditer(r"(?:\+?\d[\d\-\(\) ]{7,}\d)", joined))
    phones = []
    for phone in phones_raw:
        digits = re.sub(r"\D", "", phone)
        if len(digits) < 10:
            continue
        normalized = phone.strip()
        if digits.startswith("8") and len(digits) == 11:
            normalized = "+7 " + f"{digits[1:4]} {digits[4:7]}-{digits[7:9]}-{digits[9:11]}"
        elif digits.startswith("7") and len(digits) == 11 and not phone.startswith("+"):
            normalized = "+7 " + f"{digits[1:4]} {digits[4:7]}-{digits[7:9]}-{digits[9:11]}"
        elif phone.startswith("+"):
            normalized = phone
        else:
            normalized = "+" + digits if len(digits) >= 11 else phone.strip()
        phones.append(normalized)
    phones = dedupe_preserve(phones)

    telegrams = []
    for match in re.finditer(r"(?:telegram|tg|телеграм)[:\s]*(@[A-Za-z0-9_]{4,})", joined, flags=re.IGNORECASE):
        telegrams.append(match.group(1))
    for match in re.finditer(r"(?<!\w)@[A-Za-z0-9_]{4,}(?!\w)", joined):
        telegrams.append(match.group(0))
    for match in re.finditer(r"t\.me/([A-Za-z0-9_]{4,})", joined, flags=re.IGNORECASE):
        telegrams.append("@" + match.group(1))
    telegrams = dedupe_preserve(telegrams)

    contact_conf = 0.0
    if emails:
        contact_conf += 0.35
    if phones:
        contact_conf += 0.35
    if telegrams:
        contact_conf += 0.15

    return {
        "email": emails[0] if emails else None,
        "phone": phones[0] if phones else None,
        "telegram": telegrams[0] if telegrams else None,
        "emails": emails,
        "phones": phones,
        "telegrams": telegrams,
    }, min(contact_conf, 1.0)


def _normalize_english_level(raw: str) -> Optional[str]:
    normalized = normalize_lookup(raw)
    direct = re.search(r"\b(a1|a2|b1|b2|c1|c2)\b", normalized)
    if direct:
        return direct.group(1).upper()

    mapping = {
        "native": "C2",
        "fluent": "C1",
        "advanced": "C1",
        "upper intermediate": "B2",
        "upper-intermediate": "B2",
        "intermediate": "B1",
        "pre intermediate": "A2",
        "pre-intermediate": "A2",
        "elementary": "A1",
        "basic": "A1",
        "родной": "C2",
        "свободно": "C1",
        "продвинутый": "C1",
        "upper intermediate b2": "B2",
        "средний": "B1",
        "ниже среднего": "A2",
        "базовый": "A1",
    }
    for key, value in mapping.items():
        if key in normalized:
            return value
    return None


def extract_english_level(sections: Dict[str, List[str]], all_lines: Sequence[str]) -> Tuple[Optional[str], float]:
    candidates = sections.get("languages_human", []) + list(all_lines[:60])
    for line in candidates:
        lowered = normalize_lookup(line)
        if "english" in lowered or "англий" in lowered:
            level = _normalize_english_level(lowered)
            if level:
                return level, 0.92
    return None, 0.0


def _build_alias_index() -> Tuple[Dict[str, Tuple[str, str]], List[Tuple[re.Pattern[str], str, str]]]:
    alias_to_canonical: Dict[str, Tuple[str, str]] = {}
    regex_patterns: List[Tuple[re.Pattern[str], str, str]] = []

    for category, mapping in TECH_TAXONOMY.items():
        for canonical, aliases in mapping.items():
            for alias in dedupe_preserve([canonical, *aliases]):
                normalized = normalize_lookup(alias)
                alias_to_canonical[normalized] = (category, canonical)
                escaped = re.escape(alias.lower())
                pattern = re.compile(rf"(?<![A-Za-zА-Яа-я0-9_]){escaped}(?![A-Za-zА-Яа-я0-9_])", re.IGNORECASE)
                regex_patterns.append((pattern, category, canonical))

    regex_patterns.sort(key=lambda item: len(item[2]), reverse=True)
    return alias_to_canonical, regex_patterns


ALIAS_TO_CANONICAL, TAXONOMY_PATTERNS = _build_alias_index()


def _split_skill_items(text: str) -> List[str]:
    raw = clean_text(text)
    raw = raw.replace("|", ",").replace("•", ",").replace("·", ",")
    raw = re.sub(r"\s*/\s*", ",", raw)
    raw = re.sub(r"\s*;\s*", ",", raw)
    raw = re.sub(r"\n+", ",", raw)
    parts = [clean_inline(part) for part in raw.split(",")]
    items = []
    for part in parts:
        if not part:
            continue
        if len(part) > 60:
            continue
        if _contains_contact_marker(part):
            continue
        items.append(part)
    return dedupe_preserve(items)


def _canonicalize_skill_item(item: str) -> Optional[Tuple[str, str]]:
    normalized = normalize_lookup(item)
    normalized = normalized.strip()
    if not normalized:
        return None
    if normalized in ALIAS_TO_CANONICAL:
        return ALIAS_TO_CANONICAL[normalized]
    return None


def extract_skills(sections: Dict[str, List[str]], raw_text: str) -> Tuple[Dict[str, List[str]], float]:
    categorized: Dict[str, List[str]] = {
        "skills": [],
        "technologies": [],
        "frameworks": [],
        "tools": [],
        "languages": [],
    }

    explicit_skills_text = "\n".join(sections.get("skills", []))
    explicit_items = _split_skill_items(explicit_skills_text)
    for item in explicit_items:
        canonical = _canonicalize_skill_item(item)
        if canonical:
            category, value = canonical
            if category in categorized:
                categorized[category].append(value)
                categorized["skills"].append(value)
            else:
                categorized["skills"].append(value)
        else:
            categorized["skills"].append(item)

    lowered_text = raw_text.lower()
    for pattern, category, canonical in TAXONOMY_PATTERNS:
        if pattern.search(lowered_text):
            if category in categorized:
                categorized[category].append(canonical)
            categorized["skills"].append(canonical)

    for key in categorized:
        categorized[key] = dedupe_preserve(categorized[key])

    confidence = 0.2
    for key in ("languages", "frameworks", "technologies", "tools"):
        if categorized[key]:
            confidence += 0.18
    if explicit_items:
        confidence += 0.08
    confidence = min(confidence, 0.95)

    return categorized, confidence


MONTH_PATTERN = "|".join(sorted((re.escape(k) for k in MONTH_ALIASES.keys()), key=len, reverse=True))
PRESENT_PATTERN = "|".join(sorted((re.escape(k) for k in PRESENT_MARKERS), key=len, reverse=True))
DATE_TOKEN_PATTERN = rf"""
(?:
    (?:0?[1-9]|1[0-2])[\./-](?:19|20)\d{{2}}
    |
    (?:19|20)\d{{2}}[\./-](?:0?[1-9]|1[0-2])
    |
    (?:{MONTH_PATTERN})\.?\s+(?:19|20)\d{{2}}
    |
    (?:19|20)\d{{2}}\s+(?:{MONTH_PATTERN})\.?
    |
    (?:19|20)\d{{2}}
)
"""
DATE_RANGE_RE = re.compile(
    rf"(?P<start>{DATE_TOKEN_PATTERN})\s*(?:-|–|—|to|until|till|по|до)\s*(?P<end>{DATE_TOKEN_PATTERN}|{PRESENT_PATTERN})",
    flags=re.IGNORECASE | re.VERBOSE,
)


def _parse_single_date(token: str, is_end: bool = False) -> Optional[dt.date]:
    raw = normalize_lookup(token).strip(". ")
    if not raw:
        return None
    if raw in PRESENT_MARKERS:
        today = dt.date.today()
        return dt.date(today.year, today.month, 1)

    match = re.fullmatch(r"(0?[1-9]|1[0-2])[\./-]((?:19|20)\d{2})", raw)
    if match:
        month = int(match.group(1))
        year = int(match.group(2))
        return dt.date(year, month, 1)

    match = re.fullmatch(r"((?:19|20)\d{2})[\./-](0?[1-9]|1[0-2])", raw)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        return dt.date(year, month, 1)

    match = re.fullmatch(rf"({MONTH_PATTERN})\.?\s+((?:19|20)\d{{2}})", raw, flags=re.IGNORECASE)
    if match:
        month = MONTH_ALIASES[normalize_lookup(match.group(1))]
        year = int(match.group(2))
        return dt.date(year, month, 1)

    match = re.fullmatch(rf"((?:19|20)\d{{2}})\s+({MONTH_PATTERN})\.?", raw, flags=re.IGNORECASE)
    if match:
        year = int(match.group(1))
        month = MONTH_ALIASES[normalize_lookup(match.group(2))]
        return dt.date(year, month, 1)

    match = re.fullmatch(r"((?:19|20)\d{2})", raw)
    if match:
        year = int(match.group(1))
        month = 12 if is_end else 1
        return dt.date(year, month, 1)

    return None


def _month_iso(value: Optional[dt.date]) -> Optional[str]:
    if not value:
        return None
    return f"{value.year:04d}-{value.month:02d}"


def parse_date_range(text: str) -> Optional[ParsedDateRange]:
    cleaned = clean_inline(text)
    if not cleaned:
        return None
    match = DATE_RANGE_RE.search(cleaned)
    if not match:
        return None

    start_raw = clean_inline(match.group("start"))
    end_raw = clean_inline(match.group("end"))
    start = _parse_single_date(start_raw, is_end=False)
    is_present = normalize_lookup(end_raw) in PRESENT_MARKERS
    end = _parse_single_date(end_raw, is_end=True)
    if not start or not end:
        return None
    if end < start:
        return None

    confidence = 0.92
    if re.fullmatch(r"(?:19|20)\d{2}", start_raw):
        confidence -= 0.12
    if re.fullmatch(r"(?:19|20)\d{2}", end_raw):
        confidence -= 0.12

    return ParsedDateRange(
        start=start,
        end=end,
        start_iso=_month_iso(start),
        end_iso=None if is_present else _month_iso(end),
        is_present=is_present,
        raw=match.group(0),
        confidence=max(0.4, confidence),
    )


def months_between(start: dt.date, end: dt.date) -> int:
    return max(0, (end.year - start.year) * 12 + (end.month - start.month) + 1)


def _is_short_header_line(line: str) -> bool:
    line = clean_inline(line)
    return bool(line) and len(line) <= 120 and not line.startswith("-")


def _is_probable_job_header(line: str) -> bool:
    text = clean_inline(line)
    if not _is_short_header_line(text):
        return False
    if parse_date_range(text):
        return True
    separators = [" @ ", " at ", " - ", " — ", " | ", " / ", " в "]
    if any(sep in text.lower() for sep in separators) and (_score_role_like(text) > 0 or _score_company_like(text) > 0):
        return True
    if _score_role_like(text) > 1:
        return True
    if _score_company_like(text) > 2 and len(text.split()) <= 4:
        return True
    return False


def _split_blocks(lines: Sequence[str]) -> List[List[str]]:
    blocks: List[List[str]] = []
    current: List[str] = []
    for raw_line in lines:
        line = clean_inline(raw_line)
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        if current and parse_date_range(line) and any(parse_date_range(existing) for existing in current):
            blocks.append(current)
            current = [line]
            continue
        if current and _is_probable_job_header(line) and len(current) >= 4:
            blocks.append(current)
            current = [line]
            continue
        current.append(line)
    if current:
        blocks.append(current)

    if len(blocks) <= 1 and len(lines) > 8:
        strong_blocks: List[List[str]] = []
        current = []
        saw_date = False
        for raw_line in lines:
            line = clean_inline(raw_line)
            if not line:
                continue
            if current and (_is_probable_job_header(line) and saw_date):
                strong_blocks.append(current)
                current = [line]
                saw_date = bool(parse_date_range(line))
                continue
            current.append(line)
            saw_date = saw_date or bool(parse_date_range(line))
        if current:
            strong_blocks.append(current)
        if len(strong_blocks) > len(blocks):
            blocks = strong_blocks

    return blocks


def _split_header_parts(text: str) -> List[str]:
    for separator in [" @ ", " at ", " | ", " / ", " — ", " - ", " в "]:
        if separator in text.lower():
            parts = [clean_inline(part) for part in re.split(re.escape(separator), text, flags=re.IGNORECASE)]
            parts = [part for part in parts if part]
            if len(parts) >= 2:
                return parts
    return [clean_inline(text)]


def _guess_company_position(candidate_lines: Sequence[str], date_line_text: Optional[str]) -> Tuple[Optional[str], Optional[str], List[str]]:
    candidates = [clean_inline(line) for line in candidate_lines if clean_inline(line)]
    used_lines: List[str] = []
    if date_line_text:
        stripped = clean_inline(date_line_text.replace(parse_date_range(date_line_text).raw, "")) if parse_date_range(date_line_text) else clean_inline(date_line_text)
        if stripped:
            candidates = [stripped, *candidates]

    for line in candidates:
        parts = _split_header_parts(line)
        if len(parts) >= 2:
            first, second = parts[0], parts[1]
            first_company = _score_company_like(first)
            first_role = _score_role_like(first)
            second_company = _score_company_like(second)
            second_role = _score_role_like(second)

            if first_company >= second_company and second_role >= first_role:
                used_lines.append(line)
                return first, second, used_lines
            if second_company > first_company and first_role >= second_role:
                used_lines.append(line)
                return second, first, used_lines
            if first_company > 0 and second_role > 0:
                used_lines.append(line)
                return first, second, used_lines
            if second_company > 0 and first_role > 0:
                used_lines.append(line)
                return second, first, used_lines

    if len(candidates) >= 2:
        first, second = candidates[0], candidates[1]
        if _score_role_like(first) >= _score_role_like(second) and _score_company_like(second) >= _score_company_like(first):
            return second, first, [first, second]
        if _score_role_like(second) >= _score_role_like(first) and _score_company_like(first) >= _score_company_like(second):
            return first, second, [first, second]

    if candidates:
        only = candidates[0]
        if _score_role_like(only) > 0:
            return None, only, [only]
        if _score_company_like(only) > 0:
            return only, None, [only]
        return None, only, [only]

    return None, None, []


def _extract_achievements(lines: Sequence[str]) -> List[str]:
    items: List[str] = []
    for raw in lines:
        line = clean_inline(strip_bullet(raw))
        if not line:
            continue
        if raw.strip().startswith(("-", "*", "•", "·", "—")):
            items.append(line)
            continue
        if re.search(r"\b(reduced|improved|increased|decreased|launched|built|optimized|automated|увелич|сниз|ускор|оптимиз|внедрил|разработал)\b", normalize_lookup(line)):
            items.append(line)
            continue
        if re.search(r"\d+%|\d+x|\d+\s*(users|requests|ms|сек|секунд|руб|usd|\$)", line, flags=re.IGNORECASE):
            items.append(line)
    return dedupe_preserve(items)[:10]


def parse_experience_section(lines: Sequence[str], all_lines: Sequence[str]) -> Tuple[List[Dict[str, Any]], float]:
    source_lines = [line for line in lines if clean_inline(line)]
    if not source_lines:
        fallback_lines = [line for line in all_lines if clean_inline(line)]
        first_date_idx = next((idx for idx, line in enumerate(fallback_lines) if parse_date_range(line)), None)
        if first_date_idx is not None:
            start_idx = first_date_idx
            if first_date_idx > 0 and _is_probable_job_header(fallback_lines[first_date_idx - 1]):
                start_idx = first_date_idx - 1
            elif first_date_idx > 1 and _is_probable_job_header(fallback_lines[first_date_idx - 2]):
                start_idx = first_date_idx - 2

            stop_idx = len(fallback_lines)
            for idx in range(first_date_idx + 1, len(fallback_lines)):
                heading = _is_probable_heading(fallback_lines[idx])
                if heading and heading[0] in {"education", "languages_human", "skills", "contacts", "summary"}:
                    stop_idx = idx
                    break
            source_lines = fallback_lines[start_idx:stop_idx]
        else:
            source_lines = fallback_lines

    blocks = _split_blocks(source_lines)
    results: List[Dict[str, Any]] = []

    for block in blocks:
        if len(results) >= 12:
            break

        date_idx = None
        date_info = None
        for idx, line in enumerate(block):
            parsed = parse_date_range(line)
            if parsed:
                date_idx = idx
                date_info = parsed
                break

        if not date_info:
            continue

        block_head = normalize_lookup(" ".join(block[:3]))
        if any(hint in block_head for hint in DEGREE_HINTS) and not any(role in block_head for role in ROLE_KEYWORDS):
            continue

        candidate_header_lines = []
        if date_idx is not None:
            candidate_header_lines.extend(block[max(0, date_idx - 2):date_idx])
            candidate_header_lines.extend(block[date_idx + 1:date_idx + 3])
            date_line = block[date_idx]
        else:
            candidate_header_lines.extend(block[:2])
            date_line = None

        company, position, used_headers = _guess_company_position(candidate_header_lines, date_line)

        if position and _extract_degree(position):
            continue

        description_lines = []
        used_set = {clean_inline(item) for item in used_headers}
        if date_line:
            used_set.add(clean_inline(date_line))
        for line in block:
            if clean_inline(line) in used_set:
                continue
            description_lines.append(strip_bullet(line))

        description = clean_inline(" ".join(description_lines))
        achievements = _extract_achievements(block)

        block_is_job_like = (
            (position is not None and _score_role_like(position) > 0)
            or (company is not None and _score_company_like(company) > 0 and bool(description))
            or bool(re.search(r"\b(api|backend|frontend|python|sql|data|ml|qa|docker|kubernetes|analytics|developer|engineer|devops|analyst|разработ|инженер|аналитик|тестировщик)\b", normalize_lookup(description)))
        )
        if not block_is_job_like:
            continue

        item_confidence = date_info.confidence
        if company:
            item_confidence += 0.12
        if position:
            item_confidence += 0.12
        if description:
            item_confidence += 0.06

        results.append(
            {
                "company": company,
                "position": position,
                "date_start": date_info.start_iso,
                "date_end": date_info.end_iso,
                "duration_months": months_between(date_info.start, date_info.end) if date_info.start and date_info.end else None,
                "description": description,
                "achievements": achievements,
                "_confidence": min(item_confidence, 0.98),
                "_start_dt": date_info.start,
                "_end_dt": date_info.end,
            }
        )

    cleaned_results = []
    for item in results:
        cleaned = {k: v for k, v in item.items() if not k.startswith("_")}
        if cleaned["company"] or cleaned["position"] or cleaned["description"]:
            cleaned_results.append(cleaned)

    confidence = 0.0
    if results:
        confidence = sum(item["_confidence"] for item in results) / len(results)

    return cleaned_results, round(confidence, 3)


def _extract_degree(line: str) -> Optional[str]:
    normalized = normalize_lookup(line)
    for hint in DEGREE_HINTS:
        if hint in normalized:
            return clean_inline(line)
    return None


def parse_education_section(lines: Sequence[str], all_lines: Sequence[str]) -> Tuple[List[Dict[str, Any]], float]:
    source_lines = [line for line in lines if clean_inline(line)]
    if not source_lines:
        fallback_lines = [line for line in all_lines if clean_inline(line)]
        start_idx = None
        for idx, line in enumerate(fallback_lines):
            normalized = normalize_lookup(line)
            if any(hint in normalized for hint in DEGREE_HINTS) or re.search(r"\b(university|institute|college|academy|school|университет|институт|колледж|академия|школа)\b", normalized):
                start_idx = max(0, idx - 1)
                break
        if start_idx is None:
            return [], 0.0
        stop_idx = len(fallback_lines)
        for idx in range(start_idx + 1, len(fallback_lines)):
            heading = _is_probable_heading(fallback_lines[idx])
            if heading and heading[0] in {"experience", "languages_human", "skills", "contacts", "summary"}:
                stop_idx = idx
                break
        source_lines = fallback_lines[start_idx:stop_idx]

    blocks = _split_blocks(source_lines)
    results: List[Dict[str, Any]] = []
    confidences: List[float] = []

    for block in blocks:
        institution = None
        degree = None
        specialization = None
        date_info = None

        for line in block[:4]:
            if not date_info:
                date_info = parse_date_range(line)
            if not institution and _score_company_like(line) > 0:
                institution = clean_inline(line)
            if not institution and re.search(r"\b(university|institute|college|academy|school|университет|институт|колледж|академия|школа)\b", normalize_lookup(line)):
                institution = clean_inline(line)
            if not degree:
                degree = _extract_degree(line)

        if not institution and block:
            institution = clean_inline(block[0])
        if not degree and len(block) >= 2:
            degree = _extract_degree(block[1])

        if len(block) >= 3:
            for line in block[1:4]:
                normalized = normalize_lookup(line)
                if any(hint in normalized for hint in DEGREE_HINTS):
                    continue
                if re.search(r"\b(computer science|software|informatics|математик|информатик|программ|data science|машинное обучение)\b", normalized):
                    specialization = clean_inline(line)
                    break

        if institution or degree or specialization:
            confidence = 0.35
            if institution:
                confidence += 0.25
            if degree:
                confidence += 0.2
            if date_info:
                confidence += 0.2
            confidences.append(min(confidence, 0.95))
            results.append(
                {
                    "institution": institution,
                    "degree": degree,
                    "specialization": specialization,
                    "date_start": date_info.start_iso if date_info else None,
                    "date_end": date_info.end_iso if date_info and not date_info.is_present else (_month_iso(date_info.end) if date_info else None),
                }
            )

    overall = sum(confidences) / len(confidences) if confidences else 0.0
    return results, round(overall, 3)


def total_experience_years(experience: Sequence[Dict[str, Any]]) -> float:
    intervals: List[Tuple[int, int]] = []
    for item in experience:
        start_raw = item.get("date_start")
        end_raw = item.get("date_end")
        if not start_raw:
            continue
        try:
            start_year, start_month = map(int, start_raw.split("-"))
        except Exception:
            continue
        if end_raw:
            try:
                end_year, end_month = map(int, end_raw.split("-"))
            except Exception:
                today = dt.date.today()
                end_year, end_month = today.year, today.month
        else:
            today = dt.date.today()
            end_year, end_month = today.year, today.month
        start_idx = start_year * 12 + start_month
        end_idx = end_year * 12 + end_month
        if end_idx < start_idx:
            continue
        intervals.append((start_idx, end_idx))

    if not intervals:
        return 0.0

    intervals.sort()
    merged: List[List[int]] = []
    for start_idx, end_idx in intervals:
        if not merged or start_idx > merged[-1][1] + 1:
            merged.append([start_idx, end_idx])
        else:
            merged[-1][1] = max(merged[-1][1], end_idx)

    months = sum(end - start + 1 for start, end in merged)
    return round(months / 12.0, 1)


def _missing_fields(profile_json: Dict[str, Any], raw_text: str) -> Dict[str, Any]:
    required = {
        "skills": bool(profile_json.get("skills")),
        "technologies": bool(profile_json.get("technologies")),
        "frameworks": bool(profile_json.get("frameworks")),
        "tools": bool(profile_json.get("tools")),
        "experience": bool(profile_json.get("experience")),
        "summary": bool(profile_json.get("summary")),
        "total_experience_years": profile_json.get("total_experience_years") not in (None, ""),
        "documents.raw_text": bool(clean_text(raw_text)),
    }
    empty_profile_fields = [
        key
        for key in (
            "full_name",
            "summary",
            "desired_position",
            "location",
            "english_level",
            "skills",
            "technologies",
            "frameworks",
            "tools",
            "languages",
            "experience",
            "education",
        )
        if not profile_json.get(key)
    ]
    return {
        "required_for_scoring": [key for key, ok in required.items() if not ok],
        "empty_profile_fields": empty_profile_fields,
        "raw_text_length": len(clean_text(raw_text)),
    }


def parse_resume_to_profile(raw_text: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    cleaned_text = clean_text(raw_text)
    if not cleaned_text:
        raise ResumeParsingError("Resume text is empty after normalization")

    lines = safe_split_lines(cleaned_text)
    sections = split_sections(lines)
    header_lines = [line for line in sections.get("header", []) if clean_inline(line)]

    full_name, full_name_conf = extract_full_name(header_lines or lines)
    desired_position, position_conf = extract_desired_position(header_lines or lines, full_name)
    location, location_conf = extract_location(header_lines or lines, full_name, desired_position)

    contacts, contacts_conf = extract_contacts(lines)
    if location:
        contacts["location"] = location

    summary, summary_conf = extract_summary(
        sections,
        header_lines or lines,
        skip_values=[full_name or "", desired_position or "", location or "", contacts.get("email") or "", contacts.get("phone") or ""],
    )
    english_level, english_conf = extract_english_level(sections, lines)
    skills_data, skills_conf = extract_skills(sections, cleaned_text)
    experience, experience_conf = parse_experience_section(sections.get("experience", []), lines)
    education, education_conf = parse_education_section(sections.get("education", []), lines)
    total_years = total_experience_years(experience)

    profile_json: Dict[str, Any] = {
        "full_name": full_name,
        "summary": summary,
        "desired_position": desired_position,
        "location": location,
        "english_level": english_level,
        "total_experience_years": total_years,
        "skills": skills_data["skills"],
        "technologies": skills_data["technologies"],
        "frameworks": skills_data["frameworks"],
        "tools": skills_data["tools"],
        "languages": skills_data["languages"],
        "experience": experience,
        "education": education,
    }

    confidence_json: Dict[str, Any] = {
        "fields": {
            "full_name": round(full_name_conf, 3),
            "summary": round(summary_conf, 3),
            "desired_position": round(position_conf, 3),
            "location": round(location_conf, 3),
            "contacts": round(contacts_conf, 3),
            "english_level": round(english_conf, 3),
            "skills": round(skills_conf, 3),
            "experience": round(experience_conf, 3),
            "education": round(education_conf, 3),
            "total_experience_years": 0.95 if total_years > 0 else 0.0,
        },
        "warnings": [],
        "contacts": contacts,
    }

    if not experience:
        confidence_json["warnings"].append("Experience section was not parsed confidently")
    if not skills_data["skills"]:
        confidence_json["warnings"].append("No skills detected")
    if not full_name:
        confidence_json["warnings"].append("Full name was not found confidently")

    missing_fields_json = _missing_fields(profile_json, cleaned_text)
    return profile_json, confidence_json, missing_fields_json


async def save_resume_file(
    db: AsyncSession,
    candidate_id: str,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
) -> Dict[str, Any]:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    document_id = str(uuid.uuid4())
    safe_name = Path(filename or f"{document_id}.bin").name
    file_path = STORAGE_DIR / f"{document_id}__{safe_name}"

    file_path.write_bytes(file_bytes)
    file_hash = sha256_bytes(file_bytes)

    doc = Document(
        id=document_id,
        candidate_id=candidate_id,
        file_name=safe_name,
        mime_type=mime_type,
        file_path=str(file_path),
        file_hash=file_hash,
        parse_status="PENDING",
    )
    db.add(doc)
    await db.flush()

    return {"document_id": document_id, "file_path": str(file_path)}


async def upsert_candidate_profile(
    db: AsyncSession,
    candidate_id: str,
    profile_json: Dict[str, Any],
    confidence_json: Dict[str, Any],
    missing_fields_json: Dict[str, Any],
) -> None:
    result = await db.execute(select(Profile).where(Profile.candidate_id == candidate_id))
    profile = result.scalar_one_or_none()

    if profile:
        profile.profile_json = profile_json
        profile.confidence_json = confidence_json
        profile.missing_fields_json = missing_fields_json
        profile.updated_at = dt.datetime.utcnow()
    else:
        profile = Profile(
            candidate_id=candidate_id,
            profile_json=profile_json,
            confidence_json=confidence_json,
            missing_fields_json=missing_fields_json,
            updated_at=dt.datetime.utcnow(),
        )
        db.add(profile)


def _merge_contacts(existing: Optional[Dict[str, Any]], parsed: Dict[str, Any]) -> Dict[str, Any]:
    existing = existing or {}
    result = dict(existing)

    for key in ("email", "phone", "telegram", "location"):
        if parsed.get(key):
            result[key] = parsed.get(key)

    for key in ("emails", "phones", "telegrams"):
        merged = dedupe_preserve([*(existing.get(key) or []), *(parsed.get(key) or [])])
        if merged:
            result[key] = merged

    return result


async def _update_candidate_basics(
    db: AsyncSession,
    candidate_id: str,
    profile_json: Dict[str, Any],
    confidence_json: Dict[str, Any],
) -> None:
    result = await db.execute(select(Candidate).where(Candidate.id == candidate_id))
    candidate = result.scalar_one_or_none()
    if not candidate:
        return

    contacts = confidence_json.get("contacts") or {}
    if profile_json.get("full_name"):
        candidate.full_name = profile_json["full_name"]
    candidate.contacts_json = _merge_contacts(candidate.contacts_json, contacts)


async def parse_document(db: AsyncSession, document_id: str) -> Dict[str, Any]:
    doc_result = await db.execute(select(Document).where(Document.id == document_id))
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise ValueError(f"Document {document_id} not found")

    try:
        extraction = extract_text_details(doc.file_path, doc.mime_type)
        raw_text = extraction.text
        doc.raw_text = raw_text

        profile_json, confidence_json, missing_fields_json = parse_resume_to_profile(raw_text)
        confidence_json["extraction"] = {
            "method": extraction.method,
            "quality_score": round(extraction.quality_score, 3),
            "warnings": extraction.warnings,
        }

        await _update_candidate_basics(db, doc.candidate_id, profile_json, confidence_json)
        await upsert_candidate_profile(
            db,
            doc.candidate_id,
            profile_json,
            confidence_json,
            missing_fields_json,
        )

        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        doc.parse_status = "DONE"
        doc.parsed_at = now
        doc.last_error = None

        apps_result = await db.execute(select(Application).where(Application.candidate_id == doc.candidate_id))
        apps = apps_result.scalars().all()
        for app in apps:
            app.status = "PROFILE_READY"
            app.updated_at = now

        await db.commit()
        return {
            "status": "DONE",
            "candidate_id": doc.candidate_id,
            "profile_json": profile_json,
            "confidence_json": confidence_json,
            "missing_fields_json": missing_fields_json,
        }
    except Exception as exc:
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        doc.parse_status = "ERROR"
        doc.last_error = str(exc)
        doc.parsed_at = now
        await db.commit()
        logger.exception("Failed to parse document %s", document_id)
        return {
            "status": "ERROR",
            "candidate_id": doc.candidate_id,
            "error": str(exc),
        }
