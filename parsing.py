from typing import Any, Dict, Tuple
import uuid
import os

def _now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"

def save_resume_file(store, candidate_id: str, file_bytes: bytes, filename: str, mime_type: str) -> Dict[str, Any]:
    os.makedirs("/mnt/data/storage_resumes", exist_ok=True)
    document_id = str(uuid.uuid4())
    safe_name = filename or f"{document_id}.bin"
    file_path = os.path.join("/mnt/data/storage_resumes", f"{document_id}__{safe_name}")
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    store.documents[document_id] = {
        "id": document_id,
        "candidate_id": candidate_id,
        "file_name": safe_name,
        "mime_type": mime_type,
        "file_path": file_path,
        "file_hash": None,
        "raw_text": None,
        "parse_status": "PENDING",
        "last_error": None,
        "parsed_at": None,
    }
    return {"document_id": document_id, "file_path": file_path}

def extract_text_from_file(file_path: str, mime_type: str) -> str:
    raise NotImplementedError()

def parse_resume_to_profile(raw_text: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    raise NotImplementedError()

def upsert_candidate_profile(store, candidate_id: str, profile_json: Dict[str, Any], confidence_json: Dict[str, Any], missing_fields_json: Dict[str, Any]) -> None:
    store.profiles[candidate_id] = {
        "candidate_id": candidate_id,
        "profile_json": profile_json,
        "confidence_json": confidence_json,
        "missing_fields_json": missing_fields_json,
        "updated_at": _now_iso(),
    }

def parse_document(store, document_id: str) -> Dict[str, Any]:
    doc = store.documents[document_id]
    try:
        raw_text = extract_text_from_file(doc["file_path"], doc["mime_type"])
        doc["raw_text"] = raw_text
        profile_json, confidence_json, missing_fields_json = parse_resume_to_profile(raw_text)
        upsert_candidate_profile(store, doc["candidate_id"], profile_json, confidence_json, missing_fields_json)
        doc["parse_status"] = "DONE"
        doc["parsed_at"] = _now_iso()
        app_ids = [a["id"] for a in store.applications.values() if a["candidate_id"] == doc["candidate_id"]]
        for app_id in app_ids:
            store.applications[app_id]["status"] = "PROFILE_READY"
            store.applications[app_id]["updated_at"] = _now_iso()
        return {
            "status": "DONE",
            "candidate_id": doc["candidate_id"],
            "profile_json": profile_json,
            "confidence_json": confidence_json,
            "missing_fields_json": missing_fields_json,
        }
    except Exception as e:
        doc["parse_status"] = "ERROR"
        doc["last_error"] = str(e)
        doc["parsed_at"] = _now_iso()
        return {"status": "ERROR", "candidate_id": doc["candidate_id"], "error": str(e)}
