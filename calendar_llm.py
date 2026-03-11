from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import Application, Candidate, Job, Profile, RecruiterGoogle, Slot


DEFAULT_TIMEZONE = os.getenv("RECRUITER_TIMEZONE", "Europe/Moscow")
WORKDAY_START_HOUR = int(os.getenv("INTERVIEW_START_HOUR", "8"))
WORKDAY_END_HOUR = int(os.getenv("INTERVIEW_END_HOUR", "17"))
INTERVIEW_SLOT_MINUTES = int(os.getenv("INTERVIEW_SLOT_MINUTES", "30"))
INTERVIEW_DAYS_AHEAD = int(os.getenv("INTERVIEW_DAYS_AHEAD", "3"))
MAX_SLOTS = int(os.getenv("INTERVIEW_MAX_SLOTS", "24"))


def _tz() -> ZoneInfo:
    return ZoneInfo(DEFAULT_TIMEZONE)


def _utcnow_naive() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=None)


def _now_local() -> dt.datetime:
    return dt.datetime.now(_tz())


def _parse_dt(value: Any) -> Optional[dt.datetime]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            return value.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return value.replace(tzinfo=None)
    if isinstance(value, str):
        s = value.strip().replace("Z", "+00:00")
        try:
            parsed = dt.datetime.fromisoformat(s)
        except ValueError:
            return None
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return parsed
    return None


def _to_local(utc_naive: dt.datetime) -> dt.datetime:
    return utc_naive.replace(tzinfo=dt.timezone.utc).astimezone(_tz())


def _local_to_utc_naive(local_dt: dt.datetime) -> dt.datetime:
    if local_dt.tzinfo is None:
        local_dt = local_dt.replace(tzinfo=_tz())
    return local_dt.astimezone(dt.timezone.utc).replace(tzinfo=None)


def _iso_z(value: dt.datetime) -> str:
    return value.replace(microsecond=0).isoformat() + "Z"


def _slot_label(start_utc: dt.datetime, end_utc: dt.datetime) -> str:
    start_local = _to_local(start_utc)
    end_local = _to_local(end_utc)
    return f"{start_local.strftime('%d.%m.%Y %H:%M')} - {end_local.strftime('%H:%M')}"


def _next_business_days(count: int = INTERVIEW_DAYS_AHEAD) -> List[dt.date]:
    days: List[dt.date] = []
    cursor = _now_local().date()
    while len(days) < count:
        cursor += dt.timedelta(days=1)
        if cursor.weekday() < 5:
            days.append(cursor)
    return days


def _day_bounds_utc(day: dt.date) -> Tuple[dt.datetime, dt.datetime]:
    start_local = dt.datetime.combine(day, dt.time(hour=WORKDAY_START_HOUR, minute=0), tzinfo=_tz())
    end_local = dt.datetime.combine(day, dt.time(hour=WORKDAY_END_HOUR, minute=0), tzinfo=_tz())
    return _local_to_utc_naive(start_local), _local_to_utc_naive(end_local)


def _ranges_overlap(start_a: dt.datetime, end_a: dt.datetime, start_b: dt.datetime, end_b: dt.datetime) -> bool:
    return start_a < end_b and start_b < end_a


async def _get_recruiter_google(db: AsyncSession, recruiter_id: str) -> Optional[RecruiterGoogle]:
    result = await db.execute(select(RecruiterGoogle).where(RecruiterGoogle.recruiter_id == recruiter_id))
    return result.scalar_one_or_none()


async def _ollama_generate(prompt: str, system: str = "") -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
    timeout = float(os.getenv("OLLAMA_TIMEOUT", "120"))

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    if system:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{base_url}/api/generate", json=payload)
        response.raise_for_status()
        data = response.json()
        return (data.get("response") or "").strip()


async def get_google_oauth_url(db: AsyncSession, recruiter_id: str) -> str:
    client_id = os.getenv("GOOGLE_CLIENT_ID", "demo-google-client-id")
    redirect_uri = os.getenv("GOOGLE_CALENDAR_REDIRECT_URI", "http://127.0.0.1:8000/api/google/callback")
    scope = os.getenv(
        "GOOGLE_SCOPES",
        "https://www.googleapis.com/auth/calendar https://www.googleapis.com/auth/calendar.events",
    )

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": scope,
        "access_type": "offline",
        "prompt": "consent",
        "state": recruiter_id,
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)


async def _refresh_google_token(db: AsyncSession, recruiter_id: str) -> Optional[str]:
    google_row = await _get_recruiter_google(db, recruiter_id)
    if google_row is None:
        return None

    refresh_token = google_row.refresh_token
    if not refresh_token:
        return google_row.access_token

    if os.getenv("GOOGLE_CLIENT_ID") is None or os.getenv("GOOGLE_CLIENT_SECRET") is None:
        google_row.access_token = f"demo_access_refresh_{recruiter_id}"
        google_row.token_expiry = _utcnow_naive() + dt.timedelta(hours=1)
        google_row.updated_at = _utcnow_naive()
        await db.commit()
        return google_row.access_token

    payload = {
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post("https://oauth2.googleapis.com/token", data=payload)
        response.raise_for_status()
        data = response.json()

    expires_in = int(data.get("expires_in", 3600))
    google_row.access_token = data.get("access_token")
    google_row.token_expiry = _utcnow_naive() + dt.timedelta(seconds=expires_in)
    google_row.updated_at = _utcnow_naive()
    await db.commit()
    return google_row.access_token


async def _get_valid_access_token(db: AsyncSession, recruiter_id: str) -> Optional[str]:
    google_row = await _get_recruiter_google(db, recruiter_id)
    if google_row is None:
        return None
    if google_row.access_token and google_row.token_expiry and google_row.token_expiry > (_utcnow_naive() + dt.timedelta(minutes=2)):
        return google_row.access_token
    return await _refresh_google_token(db, recruiter_id)


async def handle_google_callback(db: AsyncSession, recruiter_id: str, code: str) -> Dict[str, Any]:
    now = _utcnow_naive()
    google_row = await _get_recruiter_google(db, recruiter_id)

    if os.getenv("GOOGLE_CLIENT_ID") is None or os.getenv("GOOGLE_CLIENT_SECRET") is None:
        if google_row is None:
            google_row = RecruiterGoogle(recruiter_id=recruiter_id)
            db.add(google_row)
        google_row.access_token = f"demo_access_{code}"
        google_row.refresh_token = f"demo_refresh_{recruiter_id}"
        google_row.token_expiry = now + dt.timedelta(hours=1)
        google_row.updated_at = now
        await db.commit()
        return {"status": "connected", "mode": "demo"}

    payload = {
        "code": code,
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "redirect_uri": os.getenv("GOOGLE_CALENDAR_REDIRECT_URI", "http://127.0.0.1:8000/api/google/callback"),
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post("https://oauth2.googleapis.com/token", data=payload)
        response.raise_for_status()
        data = response.json()

    if google_row is None:
        google_row = RecruiterGoogle(recruiter_id=recruiter_id)
        db.add(google_row)

    expires_in = int(data.get("expires_in", 3600))
    google_row.access_token = data.get("access_token")
    google_row.refresh_token = data.get("refresh_token") or google_row.refresh_token
    google_row.token_expiry = now + dt.timedelta(seconds=expires_in)
    google_row.updated_at = now
    await db.commit()
    return {"status": "connected", "mode": "google"}


async def get_google_status(db: AsyncSession, recruiter_id: str) -> Dict[str, Any]:
    google_row = await _get_recruiter_google(db, recruiter_id)
    if not google_row:
        return {"connected": False, "mode": "none", "expires_at": None, "calendar_timezone": DEFAULT_TIMEZONE}
    mode = "google" if os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET") else "demo"
    return {
        "connected": bool(google_row.access_token),
        "mode": mode,
        "expires_at": _iso_z(google_row.token_expiry) if google_row.token_expiry else None,
        "calendar_timezone": DEFAULT_TIMEZONE,
    }


async def get_freebusy(db: AsyncSession, recruiter_id: str, time_min: str, time_max: str) -> List[Dict[str, Any]]:
    start = _parse_dt(time_min) or _utcnow_naive()
    end = _parse_dt(time_max) or (start + dt.timedelta(days=5))
    token = await _get_valid_access_token(db, recruiter_id)

    if not token or os.getenv("GOOGLE_CLIENT_ID") is None:
        busy: List[Dict[str, Any]] = []
        for day in _next_business_days(INTERVIEW_DAYS_AHEAD):
            lunch_start = _local_to_utc_naive(dt.datetime.combine(day, dt.time(hour=12, minute=0), tzinfo=_tz()))
            lunch_end = _local_to_utc_naive(dt.datetime.combine(day, dt.time(hour=13, minute=0), tzinfo=_tz()))
            if lunch_end > start and lunch_start < end:
                busy.append({"start": _iso_z(max(lunch_start, start)), "end": _iso_z(min(lunch_end, end))})
        return busy

    body = {"timeMin": _iso_z(start), "timeMax": _iso_z(end), "items": [{"id": "primary"}]}
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post("https://www.googleapis.com/calendar/v3/freeBusy", headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

    return data.get("calendars", {}).get("primary", {}).get("busy", [])


async def _build_candidate_slots_for_days(
    busy_ranges: List[Tuple[dt.datetime, dt.datetime]],
    days: List[dt.date],
    booked_ranges: List[Tuple[dt.datetime, dt.datetime]],
) -> List[Dict[str, Any]]:
    slot_delta = dt.timedelta(minutes=INTERVIEW_SLOT_MINUTES)
    now_utc = _utcnow_naive()
    result: List[Dict[str, Any]] = []
    blocked = busy_ranges + booked_ranges

    for day in days:
        day_start_utc, day_end_utc = _day_bounds_utc(day)
        cursor = day_start_utc
        while cursor + slot_delta <= day_end_utc:
            slot_start = cursor
            slot_end = cursor + slot_delta
            cursor = slot_end

            if slot_start <= now_utc:
                continue

            overlaps = any(
                _ranges_overlap(slot_start, slot_end, block_start, block_end)
                for block_start, block_end in blocked
            )
            if overlaps:
                continue

            result.append(
                {
                    "slot_id": str(uuid.uuid4()),
                    "start_dt": _iso_z(slot_start),
                    "end_dt": _iso_z(slot_end),
                    "label": _slot_label(slot_start, slot_end),
                }
            )
            if len(result) >= MAX_SLOTS:
                return result

    return result


async def create_calendar_event(db: AsyncSession, recruiter_id: str, start_dt: str, end_dt: str, candidate_name: str, description: str) -> Dict[str, Any]:
    token = await _get_valid_access_token(db, recruiter_id)
    start = _parse_dt(start_dt)
    end = _parse_dt(end_dt)
    if not start or not end:
        raise ValueError("invalid start_dt or end_dt")

    if not token or os.getenv("GOOGLE_CLIENT_ID") is None:
        return {"event_id": f"demo-event-{uuid.uuid4()}", "start_dt": _iso_z(start), "end_dt": _iso_z(end)}

    headers = {"Authorization": f"Bearer {token}"}
    start_local = _to_local(start)
    end_local = _to_local(end)
    body = {
        "summary": f"Интервью с кандидатом {candidate_name}",
        "description": description,
        "start": {"dateTime": start_local.isoformat(), "timeZone": DEFAULT_TIMEZONE},
        "end": {"dateTime": end_local.isoformat(), "timeZone": DEFAULT_TIMEZONE},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            headers=headers,
            json=body,
        )
        response.raise_for_status()
        data = response.json()

    return {"event_id": data.get("id", f"google-event-{uuid.uuid4()}"), "start_dt": _iso_z(start), "end_dt": _iso_z(end)}


async def list_job_slots(db: AsyncSession, job_id: str, recruiter_id: str) -> List[Dict[str, Any]]:
    job_result = await db.execute(select(Job).where(Job.id == job_id, Job.recruiter_id == recruiter_id))
    job = job_result.scalar_one_or_none()
    if not job:
        raise ValueError("job not found")

    result = await db.execute(
        select(Slot)
        .where(Slot.job_id == job_id, Slot.recruiter_id == recruiter_id)
        .order_by(Slot.start_dt.asc())
    )
    slots = result.scalars().all()
    return [
        {
            "slot_id": slot.id,
            "start_dt": _iso_z(slot.start_dt) if slot.start_dt else None,
            "end_dt": _iso_z(slot.end_dt) if slot.end_dt else None,
            "label": slot.label,
            "status": slot.status,
            "application_id": slot.application_id,
        }
        for slot in slots
    ]


async def get_available_slots(db: AsyncSession, job_id: str, recruiter_id: str) -> List[Dict[str, Any]]:
    job_result = await db.execute(select(Job).where(Job.id == job_id, Job.recruiter_id == recruiter_id))
    job = job_result.scalar_one_or_none()
    if not job:
        raise ValueError("job not found")

    target_days = _next_business_days(INTERVIEW_DAYS_AHEAD)
    if not target_days:
        return []

    range_start = _day_bounds_utc(target_days[0])[0]
    range_end = _day_bounds_utc(target_days[-1])[1]

    busy_payload = await get_freebusy(db, recruiter_id, _iso_z(range_start), _iso_z(range_end))
    busy_ranges: List[Tuple[dt.datetime, dt.datetime]] = []
    for item in busy_payload:
        start = _parse_dt(item.get("start"))
        end = _parse_dt(item.get("end"))
        if start and end and end > start:
            busy_ranges.append((start, end))

    booked_result = await db.execute(
        select(Slot).where(
            Slot.recruiter_id == recruiter_id,
            Slot.status == "BOOKED",
            Slot.start_dt >= range_start,
            Slot.end_dt <= range_end,
        )
    )
    booked_slots = booked_result.scalars().all()
    booked_ranges = [
        (slot.start_dt, slot.end_dt)
        for slot in booked_slots
        if slot.start_dt and slot.end_dt
    ]

    await db.execute(
        delete(Slot).where(
            Slot.job_id == job_id,
            Slot.recruiter_id == recruiter_id,
            Slot.status == "FREE",
            Slot.start_dt >= range_start,
            Slot.end_dt <= range_end,
        )
    )

    slots = await _build_candidate_slots_for_days(busy_ranges, target_days, booked_ranges)
    created: List[Dict[str, Any]] = []
    for item in slots:
        slot = Slot(
            id=item["slot_id"],
            job_id=job_id,
            recruiter_id=recruiter_id,
            application_id=None,
            start_dt=_parse_dt(item["start_dt"]),
            end_dt=_parse_dt(item["end_dt"]),
            label=item["label"],
            status="FREE",
        )
        db.add(slot)
        created.append({**item, "status": "FREE", "application_id": None})

    await db.commit()
    return created


async def generate_screening_summary(db: AsyncSession, application_id: str) -> str:
    result = await db.execute(select(Application).where(Application.id == application_id))
    app = result.scalar_one_or_none()
    if not app:
        raise ValueError("application not found")

    job_result = await db.execute(select(Job).where(Job.id == app.job_id))
    job = job_result.scalar_one_or_none()
    candidate_result = await db.execute(select(Candidate).where(Candidate.id == app.candidate_id))
    candidate = candidate_result.scalar_one_or_none()
    profile_result = await db.execute(select(Profile).where(Profile.candidate_id == app.candidate_id))
    profile = profile_result.scalar_one_or_none()

    candidate_name = (candidate.full_name or candidate.telegram_username) if candidate else "Candidate"
    score = app.score
    rationale = app.score_rationale or ""
    screening_answers = app.screening_answers_json or {}
    profile_json = profile.profile_json if profile else {}

    prompt = f"""
Сделай краткое screening summary на русском языке для рекрутера.

Вакансия: {job.title if job else 'Unknown role'}
Описание вакансии: {job.description if job else ''}
Кандидат: {candidate_name}
Профиль кандидата JSON: {json.dumps(profile_json, ensure_ascii=False)}
Ответы скрининга JSON: {json.dumps(screening_answers, ensure_ascii=False)}
Score: {score}
Rationale: {rationale}

Верни 5-7 коротких предложений, без приветствия и без markdown.
""".strip()

    try:
        summary = await _ollama_generate(prompt, system="Ты HR-ассистент. Пишешь очень краткие, точные summary для рекрутера.")
    except Exception:
        parts = [f"Кандидат: {candidate_name}.", f"Вакансия: {job.title if job else 'Unknown role'}."]
        if score is not None:
            parts.append(f"Текущий score: {score}/100.")
        if rationale:
            parts.append(f"Причина оценки: {rationale}")
        if screening_answers:
            parts.append(f"Есть ответы скрининга: {json.dumps(screening_answers, ensure_ascii=False)}.")
        summary = " ".join(parts)

    app.screening_summary = summary
    app.updated_at = _utcnow_naive()
    await db.commit()
    return summary


def _extract_candidate_strengths(app: Application, profile: Optional[Profile], candidate: Optional[Candidate]) -> List[str]:
    strengths: List[str] = []
    profile_json = profile.profile_json if profile and profile.profile_json else {}
    skills = profile_json.get("skills") or []
    experience = profile_json.get("experience") or profile_json.get("experience_years")
    education = profile_json.get("education")
    rationale = app.score_rationale or ""
    screening = app.screening_answers_json or {}

    if isinstance(skills, list):
        clean_skills = [str(x).strip() for x in skills if str(x).strip()]
        if clean_skills:
            strengths.append("навыки: " + ", ".join(clean_skills[:3]))
    if experience:
        strengths.append(f"опыт: {experience}")
    if education:
        strengths.append(f"образование: {education}")
    if screening.get("english_level"):
        strengths.append(f"английский: {screening['english_level']}")
    if rationale:
        strengths.append(rationale.strip())
    if candidate and candidate.telegram_username:
        strengths.append(f"кандидат быстро вышел на связь через Telegram @{candidate.telegram_username}")

    seen = set()
    normalized = []
    for item in strengths:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            normalized.append(item)
    return normalized[:3]


async def generate_feedback(db: AsyncSession, application_id: str) -> str:
    result = await db.execute(select(Application).where(Application.id == application_id))
    app = result.scalar_one_or_none()
    if not app:
        raise ValueError("application not found")

    job_result = await db.execute(select(Job).where(Job.id == app.job_id))
    job = job_result.scalar_one_or_none()
    candidate_result = await db.execute(select(Candidate).where(Candidate.id == app.candidate_id))
    candidate = candidate_result.scalar_one_or_none()
    profile_result = await db.execute(select(Profile).where(Profile.candidate_id == app.candidate_id))
    profile = profile_result.scalar_one_or_none()

    candidate_name = (candidate.full_name or candidate.telegram_username or "кандидат") if candidate else "кандидат"
    score = app.score
    rationale = (app.score_rationale or "").strip()
    missing = [str(x).strip() for x in (app.missing_requirements_json or []) if str(x).strip()]
    threshold = job.threshold_score if job else None
    strengths = _extract_candidate_strengths(app, profile, candidate)
    strengths_text = "; ".join(strengths) if strengths else "есть релевантный опыт и мотивация"
    missing_text = ", ".join(missing[:3]) if missing else "часть ключевых требований вакансии"

    prompt = f"""
Напиши персонализированный отказ кандидату на русском языке.

Имя кандидата: {candidate_name}
Вакансия: {job.title if job else 'Unknown role'}
Требуемый порог score: {threshold}
Фактический score: {score}
Rationale: {rationale}
Недостающие требования: {json.dumps(missing, ensure_ascii=False)}
Сильные стороны кандидата: {json.dumps(strengths, ensure_ascii=False)}

Жёсткие требования к ответу:
- 3 абзаца без markdown.
- 1 абзац: короткое уважительное приветствие по имени.
- 2 абзац: мягко объясни отказ, не называй кандидата слабым, не используй шаблонные скобки и заглушки.
- 3 абзац: отметь 1-2 реальные сильные стороны кандидата из входных данных.
- Не обещай оффер или повторное рассмотрение.
- Не пиши от первого лица множественного числа слишком формально; текст должен звучать естественно.
- Не более 900 символов.
""".strip()

    try:
        feedback = await _ollama_generate(prompt, system="Ты HR-ассистент. Пишешь деликатные, конкретные и естественные отказы кандидатам без шаблонных заглушек.")
    except Exception:
        role_title = job.title if job else "выбранную вакансию"
        score_text = f"По итогам первичной оценки ваш результат составил {score} при пороге {threshold}. " if score is not None and threshold is not None else "На текущем этапе мы не можем продолжить процесс. "
        rationale_text = f"Сейчас для этой роли особенно важны {missing_text}. " if missing_text else ""
        feedback = (
            f"Здравствуйте, {candidate_name}! Спасибо за интерес к вакансии {role_title}.\n\n"
            f"{score_text}{rationale_text}Поэтому сейчас мы не готовы пригласить вас на следующий этап отбора.\n\n"
            f"При этом мы отметили ваши сильные стороны: {strengths_text}. Спасибо за время и внимание к нашей вакансии."
        )

    feedback = "\n\n".join(part.strip() for part in feedback.split("\n\n") if part.strip())
    app.feedback_text = feedback
    app.status = "REJECTED"
    app.updated_at = _utcnow_naive()
    await db.commit()
    return feedback


async def book_slot(db: AsyncSession, application_id: str, slot_id: str) -> Dict[str, Any]:
    app_result = await db.execute(select(Application).where(Application.id == application_id))
    app = app_result.scalar_one_or_none()
    if not app:
        raise ValueError("application not found")

    slot_result = await db.execute(select(Slot).where(Slot.id == slot_id))
    slot = slot_result.scalar_one_or_none()
    if not slot:
        raise ValueError("slot not found")
    if slot.status == "BOOKED":
        raise ValueError("slot already booked")

    recruiter_id = slot.recruiter_id
    if not recruiter_id:
        job_result = await db.execute(select(Job).where(Job.id == app.job_id))
        job = job_result.scalar_one_or_none()
        recruiter_id = job.recruiter_id if job else None

    candidate_result = await db.execute(select(Candidate).where(Candidate.id == app.candidate_id))
    candidate = candidate_result.scalar_one_or_none()
    candidate_name = (candidate.full_name or candidate.telegram_username) if candidate else "Candidate"

    if not recruiter_id or not slot.start_dt or not slot.end_dt:
        raise ValueError("slot is missing recruiter_id/start_dt/end_dt")

    summary = app.screening_summary or await generate_screening_summary(db, application_id)

    event = await create_calendar_event(
        db=db,
        recruiter_id=recruiter_id,
        start_dt=_iso_z(slot.start_dt),
        end_dt=_iso_z(slot.end_dt),
        candidate_name=candidate_name,
        description=summary,
    )

    slot.status = "BOOKED"
    slot.application_id = application_id
    slot.google_event_id = event["event_id"]
    slot.updated_at = _utcnow_naive()

    app.status = "SCHEDULED"
    app.updated_at = _utcnow_naive()
    app.scheduled_start_dt = _parse_dt(event["start_dt"])
    app.scheduled_end_dt = _parse_dt(event["end_dt"])
    app.calendar_event_id = event["event_id"]

    await db.commit()
    return {"status": "SCHEDULED", "event_id": event["event_id"], "start_dt": event["start_dt"], "end_dt": event["end_dt"]}