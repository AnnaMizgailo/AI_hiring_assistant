from typing import Any, Dict, Tuple
import uuid
import os
import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import Document, Candidate, Profile, Application

def _now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"

async def save_resume_file(db: AsyncSession, candidate_id: str, file_bytes: bytes, filename: str, mime_type: str) -> Dict[str, Any]:
    os.makedirs("/mnt/data/storage_resumes", exist_ok=True)
    document_id = str(uuid.uuid4())
    safe_name = filename or f"{document_id}.bin"
    file_path = os.path.join("/mnt/data/storage_resumes", f"{document_id}__{safe_name}")
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    doc = Document(
        id=document_id,
        candidate_id=candidate_id,
        file_name=safe_name,
        mime_type=mime_type,
        file_path=file_path,
        parse_status="PENDING"
    )
    db.add(doc)
    return {"document_id": document_id, "file_path": file_path}

def extract_text_from_file(file_path: str, mime_type: str) -> str:
    raise NotImplementedError()

def parse_resume_to_profile(raw_text: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    raise NotImplementedError()

async def upsert_candidate_profile(db: AsyncSession, candidate_id: str, profile_json: Dict[str, Any], confidence_json: Dict[str, Any], missing_fields_json: Dict[str, Any]) -> None:
    result = await db.execute(
        select(Profile).where(Profile.candidate_id == candidate_id)
    )
    profile = result.scalar_one_or_none()
    if profile:
        profile.profile_json = profile_json
        profile.confidence_json = confidence_json
        profile.missing_fields_json = missing_fields_json
        profile.updated_at = datetime.datetime.utcnow()
    else:
        profile = Profile(
            candidate_id=candidate_id,
            profile_json=profile_json,
            confidence_json=confidence_json,
            missing_fields_json=missing_fields_json,
            updated_at=datetime.datetime.utcnow()
        )
        db.add(profile)

async def parse_document(db: AsyncSession, document_id: str) -> Dict[str, Any]:
    doc_result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise ValueError(f"Document {document_id} not found")

    try:
        raw_text = extract_text_from_file(doc.file_path, doc.mime_type)
        doc.raw_text = raw_text

        profile_json, confidence_json, missing_fields_json = parse_resume_to_profile(raw_text)

        await upsert_candidate_profile(
            db,
            doc.candidate_id,
            profile_json,
            confidence_json,
            missing_fields_json
        )

        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        doc.parse_status = "DONE"
        doc.parsed_at = now

        apps_result = await db.execute(
            select(Application).where(Application.candidate_id == doc.candidate_id)
        )
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
    except Exception as e:
        now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        doc.parse_status = "ERROR"
        doc.last_error = str(e)
        doc.parsed_at = now
        await db.commit()
        return {
            "status": "ERROR",
            "candidate_id": doc.candidate_id,
            "error": str(e)
        }