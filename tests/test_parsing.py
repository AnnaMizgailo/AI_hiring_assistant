import pytest

from parsing import parse_date_range, parse_resume_to_profile


RU_RESUME = """
Иван Петров
Senior Python Backend Developer
Москва, Россия
ivan.petrov@example.com | +7 999 123-45-67 | @ivan_petrov

О себе
Backend Python developer with 5+ years of experience building APIs and data services.

Ключевые навыки
Python, FastAPI, Django, PostgreSQL, Redis, Docker, Git, REST, Kafka, SQL

Опыт работы
AI Factory — Senior Backend Developer
Январь 2022 - по настоящее время
- Разработал сервисы на FastAPI и Celery
- Снизил latency API на 30%
- Настроил CI/CD и Docker deployment

Data Lab — Python Developer
03.2019 - 12.2021
- Поддерживал ETL пайплайны
- Работал с PostgreSQL, Airflow, Pandas

Образование
МГТУ им. Баумана
Бакалавр, Прикладная информатика
09.2014 - 06.2018

Языки
English — B2
Russian — Native
"""

MIXED_RESUME = """
John Smith
Data Scientist / ML Engineer
Berlin, Germany
john.smith@mail.com | john.s@altmail.io
+49 170 1234567, +1 (415) 555-0123
Telegram: @johnml

5+ years in machine learning, analytics and LLM/RAG systems.

Machine Learning Engineer @ Vision Labs
Mar 2021 - Present
Built PyTorch models and LLM pipelines.
Improved inference latency by 45%.

Data Analyst | Bank Group
2018 - 2021
Worked with SQL, PostgreSQL, Tableau, Python, pandas.

Education
Technical University of Berlin
M.Sc. Data Science
2016 - 2018

Languages
English - C1
German - B2
"""


def test_parse_date_range_ru_present():
    parsed = parse_date_range("Январь 2022 - по настоящее время")
    assert parsed is not None
    assert parsed.start_iso == "2022-01"
    assert parsed.end_iso is None


def test_parse_resume_russian():
    profile, confidence, missing = parse_resume_to_profile(RU_RESUME)

    assert profile["full_name"] == "Иван Петров"
    assert profile["desired_position"] == "Senior Python Backend Developer"
    assert profile["location"] == "Москва, Россия"
    assert profile["english_level"] == "B2"
    assert "Python" in profile["skills"]
    assert "FastAPI" in profile["frameworks"]
    assert "PostgreSQL" in profile["technologies"]
    assert len(profile["experience"]) == 2
    assert missing["required_for_scoring"] == []
    assert confidence["fields"]["experience"] > 0.7


def test_parse_resume_mixed_without_sections():
    profile, confidence, missing = parse_resume_to_profile(MIXED_RESUME)

    assert profile["full_name"] == "John Smith"
    assert profile["desired_position"] == "Data Scientist / ML Engineer"
    assert profile["english_level"] == "C1"
    assert profile["experience"][0]["company"] == "Vision Labs"
    assert profile["experience"][0]["position"] == "Machine Learning Engineer"
    assert confidence["contacts"]["email"] == "john.smith@mail.com"
    assert confidence["contacts"]["telegram"] == "@johnml"
    assert "tools" in missing["required_for_scoring"]
