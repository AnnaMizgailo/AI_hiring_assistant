from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import Application, Candidate, Job, Profile, RecruiterGoogle, Slot


def _utcnow_naive() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=None)


def _parse_dt(value: Any) -> Optional[dt.datetime]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
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


def _iso_z(value: dt.datetime) -> str:
    return value.replace(microsecond=0).isoformat() + "Z"


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
    redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/api/google/callback")
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
        "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI", "http://127.0.0.1:8000/api/google/callback"),
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
        return {"connected": False, "mode": "none"}
    mode = "google" if os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET") else "demo"
    return {
        "connected": bool(google_row.access_token),
        "mode": mode,
        "expires_at": _iso_z(google_row.token_expiry) if google_row.token_expiry else None,
    }


async def get_freebusy(db: AsyncSession, recruiter_id: str, time_min: str, time_max: str) -> List[Dict[str, Any]]:
    start = _parse_dt(time_min) or _utcnow_naive()
    end = _parse_dt(time_max) or (start + dt.timedelta(days=5))
    token = await _get_valid_access_token(db, recruiter_id)

    if not token or os.getenv("GOOGLE_CLIENT_ID") is None:
        windows: List[Dict[str, Any]] = []
        cursor = start.replace(hour=10, minute=0, second=0, microsecond=0)
        for _ in range(5):
            if cursor >= end:
                break
            windows.append({"start": _iso_z(cursor), "end": _iso_z(cursor + dt.timedelta(hours=2))})
            cursor = (cursor + dt.timedelta(days=1)).replace(hour=14, minute=0, second=0, microsecond=0)
        return windows

    body = {"timeMin": _iso_z(start), "timeMax": _iso_z(end), "items": [{"id": "primary"}]}
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post("https://www.googleapis.com/calendar/v3/freeBusy", headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

    busy = data.get("calendars", {}).get("primary", {}).get("busy", [])
    busy_ranges = [(_parse_dt(x.get("start")), _parse_dt(x.get("end"))) for x in busy]
    busy_ranges = [(a, b) for a, b in busy_ranges if a and b]
    busy_ranges.sort(key=lambda pair: pair[0])

    windows: List[Dict[str, Any]] = []
    cursor = start
    for busy_start, busy_end in busy_ranges:
        if cursor < busy_start:
            windows.append({"start": _iso_z(cursor), "end": _iso_z(busy_start)})
        if busy_end > cursor:
            cursor = busy_end
    if cursor < end:
        windows.append({"start": _iso_z(cursor), "end": _iso_z(end)})
    return windows


async def build_candidate_slots(free_windows: List[Dict[str, Any]], slot_minutes: int = 30, max_slots: int = 8) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    delta = dt.timedelta(minutes=slot_minutes)
    for window in free_windows:
        start = _parse_dt(window.get("start"))
        end = _parse_dt(window.get("end"))
        if not start or not end or end <= start:
            continue
        cursor = start
        while cursor + delta <= end and len(result) < max_slots:
            slot_start = cursor
            slot_end = cursor + delta
            result.append(
                {
                    "slot_id": str(uuid.uuid4()),
                    "start_dt": _iso_z(slot_start),
                    "end_dt": _iso_z(slot_end),
                    "label": slot_start.strftime("%d.%m %H:%M") + " UTC",
                }
            )
            cursor = slot_end
        if len(result) >= max_slots:
            break
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
    body = {
        "summary": f"Interview with {candidate_name}",
        "description": description,
        "start": {"dateTime": _iso_z(start), "timeZone": "UTC"},
        "end": {"dateTime": _iso_z(end), "timeZone": "UTC"},
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

    result = await db.execute(select(Slot).where(Slot.job_id == job_id).order_by(Slot.start_dt.asc()))
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
    existing = await list_job_slots(db, job_id, recruiter_id)
    free_existing = [s for s in existing if s.get("status") == "FREE" and s.get("start_dt")]
    if free_existing:
        return free_existing

    now = _utcnow_naive()
    windows = await get_freebusy(db, recruiter_id, _iso_z(now), _iso_z(now + dt.timedelta(days=7)))
    slots = await build_candidate_slots(windows)

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
        parts = [f"Кандидат: {candidate_name}.", f"Вакансия: {job.title if job else 'Unknown role'}." ]
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


async def generate_feedback(db: AsyncSession, application_id: str) -> str:
    result = await db.execute(select(Application).where(Application.id == application_id))
    app = result.scalar_one_or_none()
    if not app:
        raise ValueError("application not found")

    job_result = await db.execute(select(Job).where(Job.id == app.job_id))
    job = job_result.scalar_one_or_none()
    candidate_result = await db.execute(select(Candidate).where(Candidate.id == app.candidate_id))
    candidate = candidate_result.scalar_one_or_none()

    candidate_name = (candidate.full_name or candidate.telegram_username) if candidate else "кандидат"
    score = app.score
    rationale = app.score_rationale or ""
    missing = app.missing_requirements_json or []
    threshold = job.threshold_score if job else None

    prompt = f"""
Напиши персонализированный отказ кандидату на русском языке.

Имя кандидата: {candidate_name}
Вакансия: {job.title if job else 'Unknown role'}
Требуемый порог score: {threshold}
Фактический score: {score}
Rationale: {rationale}
Недостающие требования: {json.dumps(missing, ensure_ascii=False)}

Требования:
- Вежливый и профессиональный тон.
- 1 короткое приветствие.
- 1 абзац с честной, но мягкой причиной отказа.
- 1 абзац с сильными сторонами кандидата.
- Без обещаний трудоустройства.
- Без markdown.
""".strip()

    try:
        feedback = await _ollama_generate(prompt, system="Ты HR-ассистент. Пишешь деликатные персонализированные отказы кандидатам.")
    except Exception:
        missing_text = ", ".join(map(str, missing)) if missing else "часть требований вакансии"
        feedback = (
            f"Спасибо, {candidate_name}, за интерес к вакансии {job.title if job else 'Unknown role'}. "
            f"На текущем этапе мы не готовы продолжить процесс: текущий score составляет {score if score is not None else 'N/A'}, "
            f"а для роли сейчас особенно важны следующие требования: {missing_text}. "
            f"При этом мы отмечаем сильные стороны вашего профиля и благодарим за уделённое время."
        )

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
