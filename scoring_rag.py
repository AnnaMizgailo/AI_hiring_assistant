from typing import Any, Dict, List, Tuple
import uuid

def _now_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"

def index_job(store, job_id: str) -> int:
    raise NotImplementedError()

def index_resume(store, document_id: str) -> int:
    raise NotImplementedError()

def retrieve_evidence(store, job_id: str, candidate_id: str, k: int = 5) -> List[Dict[str, Any]]:
    raise NotImplementedError()

def calculate_score(requirements_json: Dict[str, Any], profile_json: Dict[str, Any], evidence_chunks: List[Dict[str, Any]]) -> Tuple[int, str, List[str], List[Dict[str, Any]]]:
    raise NotImplementedError()

def save_scoring(store, application_id: str, score: int, rationale: str, missing: List[str], evidence: List[Dict[str, Any]]) -> None:
    a = store.applications[application_id]
    a["score"] = int(score)
    a["score_rationale"] = rationale
    a["missing_requirements_json"] = missing
    a["evidence_snippets_json"] = evidence
    a["status"] = "SCORING_DONE"
    a["updated_at"] = _now_iso()

def score_application(store, application_id: str) -> Dict[str, Any]:
    a = store.applications[application_id]
    job = store.jobs[a["job_id"]]
    candidate_id = a["candidate_id"]
    profile = store.profiles.get(candidate_id)
    if not profile:
        raise RuntimeError("candidate profile not ready")
    requirements_json = job.get("requirements_json") or {}
    evidence_chunks = retrieve_evidence(store, job_id=a["job_id"], candidate_id=candidate_id, k=5)
    score, rationale, missing, evidence = calculate_score(requirements_json, profile["profile_json"], evidence_chunks)
    save_scoring(store, application_id, score, rationale, missing, evidence)
    store_rag_id = str(uuid.uuid4())
    store_rag = {
        "id": store_rag_id,
        "application_id": application_id,
        "top_k_chunks_json": evidence_chunks,
        "prompt_version": "v0",
        "model_version": "unset",
        "created_at": _now_iso(),
    }
    if not hasattr(store, "rag_runs"):
        store.rag_runs = {}
    store.rag_runs[store_rag_id] = store_rag
    return {
        "application_id": application_id,
        "score": int(score),
        "rationale": rationale,
        "missing_requirements": missing,
        "evidence_snippets": evidence,
    }
