"""End-to-end API tests through FastAPI's TestClient."""
from __future__ import annotations

import pytest


# ---------------- system ----------------
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert body["database"] == "ok"
    assert body["embeddings"]


def test_root(client):
    assert client.get("/").status_code == 200


# ---------------- cv ----------------
def test_cv_upload_pdf(client, sample_pdf_bytes):
    response = client.post(
        "/api/v1/cv/upload",
        files={"file": ("cv.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["cv_id"]
    assert body["characters"] > 100
    assert "Python" in body["profile"]["skills"]


def test_cv_upload_docx(client, sample_docx_bytes):
    response = client.post(
        "/api/v1/cv/upload",
        files={"file": ("cv.docx", sample_docx_bytes,
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert response.status_code == 200
    assert "FastAPI" in response.json()["profile"]["skills"]


def test_cv_upload_rejects_bad_type(client):
    response = client.post("/api/v1/cv/upload", files={"file": ("x.jpg", b"12345678901234567890", "image/jpeg")})
    assert response.status_code == 422
    assert response.json()["error_type"] == "cv_parse_error"


def test_cv_upload_rejects_empty(client):
    response = client.post("/api/v1/cv/upload", files={"file": ("x.pdf", b"", "application/pdf")})
    assert response.status_code == 400


def test_get_cv_404(client):
    assert client.get("/api/v1/cv/does-not-exist").status_code == 404


def test_list_cvs(client, cv_id):
    response = client.get("/api/v1/cvs")
    assert response.status_code == 200
    assert any(item["cv_id"] == cv_id for item in response.json())


# ---------------- search ----------------
def test_search_requires_existing_cv(client):
    response = client.post("/api/v1/jobs/search", json={"cv_id": "nope", "filters": {}})
    assert response.status_code == 404


def test_search_starts_and_reports_status(client, cv_id):
    response = client.post(
        "/api/v1/jobs/search",
        json={"cv_id": cv_id, "filters": {"keywords": "python", "limit_per_source": 1}},
    )
    assert response.status_code == 200
    task_id = response.json()["task_id"]

    status_response = client.get(f"/api/v1/jobs/status/{task_id}")
    assert status_response.status_code == 200
    body = status_response.json()
    assert body["status"] in ("pending", "running", "completed", "failed")
    # Scrapers may be blocked in a sandbox; the task must still not crash.
    assert body["error"] is None or isinstance(body["error"], str)


def test_status_404_for_unknown_task(client):
    assert client.get("/api/v1/jobs/status/unknown").status_code == 404


# ---------------- skill gap ----------------
def test_skill_gap_job(client, cv_id):
    response = client.post("/api/v1/skill-gap/job", json={
        "cv_id": cv_id,
        "job_title": "Platform Engineer",
        "job_description": "We need Python, Kubernetes, Terraform and AWS experience.",
        "job_url": "https://example.com/job/1",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert "Python" in body["matching_skills"]
    assert "Terraform" in body["missing_skills"]
    assert 0 <= body["gap_score"] <= 100
    assert body["explanation"]


def test_skill_gap_aggregate(client, cv_id):
    response = client.post("/api/v1/skill-gap/aggregate", json={
        "cv_id": cv_id,
        "top_n": 5,
        "jobs": [
            {"id": "1", "title": "Backend Engineer", "description": "Python FastAPI Kubernetes AWS"},
            {"id": "2", "title": "Platform Engineer", "description": "Kubernetes Terraform AWS Docker"},
        ],
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["jobs_analyzed"] == 2
    skills = [item["skill"] for item in body["top_missing_skills"]]
    # The fixture CV already lists Kubernetes/AWS, so Terraform is the real gap.
    assert "Terraform" in skills
    matching = [item["skill"] for item in body["top_matching_skills"]]
    assert "Kubernetes" in matching
    assert body["learning_priorities"]


def test_skill_gap_aggregate_requires_jobs(client, cv_id):
    response = client.post("/api/v1/skill-gap/aggregate",
                           json={"cv_id": cv_id, "jobs": [], "use_cached_results": False})
    assert response.status_code == 400


# ---------------- applications ----------------
def test_application_full_lifecycle(client, cv_id):
    created = client.post("/api/v1/applications", json={
        "cv_id": cv_id, "job_title": "Backend Engineer", "company": "Fawry",
        "job_url": "https://example.com/job/9", "source": "wuzzuf", "match_score": 82.5,
    })
    assert created.status_code == 201, created.text
    app_id = created.json()["id"]
    assert created.json()["status"] == "saved"

    listed = client.get(f"/api/v1/applications?cv_id={cv_id}")
    assert listed.status_code == 200
    assert any(item["id"] == app_id for item in listed.json())

    patched = client.patch(f"/api/v1/applications/{app_id}",
                           json={"status": "applied", "event_note": "Applied via site"})
    assert patched.status_code == 200
    assert patched.json()["status"] == "applied"
    assert patched.json()["applied_date"]

    client.patch(f"/api/v1/applications/{app_id}", json={"status": "interviewing"})

    timeline = client.get(f"/api/v1/applications/{app_id}/timeline")
    assert timeline.status_code == 200
    events = timeline.json()["events"]
    assert len(events) == 3  # created + applied + interviewing
    assert events[-1]["new_status"] == "interviewing"

    stats = client.get(f"/api/v1/applications/stats?cv_id={cv_id}")
    assert stats.json()["by_status"].get("interviewing") == 1

    assert client.delete(f"/api/v1/applications/{app_id}").status_code == 204
    assert client.get(f"/api/v1/applications/{app_id}/timeline").status_code == 404


def test_application_validation_errors(client, cv_id):
    assert client.post("/api/v1/applications", json={"cv_id": "missing", "job_title": "X"}).status_code == 404
    assert client.post("/api/v1/applications", json={"cv_id": cv_id}).status_code == 422
    assert client.patch("/api/v1/applications/nope", json={"status": "applied"}).status_code == 404


def test_application_rejects_invalid_status(client, cv_id):
    created = client.post("/api/v1/applications", json={"cv_id": cv_id, "job_title": "T"})
    app_id = created.json()["id"]
    assert client.patch(f"/api/v1/applications/{app_id}", json={"status": "banana"}).status_code == 422


# ---------------- alerts ----------------
def test_alert_crud(client, cv_id):
    created = client.post("/api/v1/alerts", json={
        "cv_id": cv_id, "name": "Backend Cairo", "destination": "me@example.com",
        "filters": {"keywords": "backend", "location": "Cairo", "limit_per_source": 3},
    })
    assert created.status_code == 201, created.text
    alert_id = created.json()["id"]
    assert created.json()["is_active"] is True
    assert created.json()["filters"]["keywords"] == "backend"

    listed = client.get(f"/api/v1/alerts?cv_id={cv_id}")
    assert any(item["id"] == alert_id for item in listed.json())

    patched = client.patch(f"/api/v1/alerts/{alert_id}", json={"is_active": False})
    assert patched.json()["is_active"] is False

    assert client.delete(f"/api/v1/alerts/{alert_id}").status_code == 204
    assert client.patch(f"/api/v1/alerts/{alert_id}", json={"is_active": True}).status_code == 404


def test_alert_rejects_bad_email(client, cv_id):
    response = client.post("/api/v1/alerts", json={"cv_id": cv_id, "destination": "not-an-email"})
    assert response.status_code == 422


# ---------------- LLM-dependent endpoints degrade cleanly ----------------
def test_cover_letter_without_key_returns_503(client, cv_id):
    response = client.post("/api/v1/cover-letter", json={"cv_id": cv_id, "job_title": "Backend Engineer"})
    assert response.status_code == 503
    assert response.json()["error_type"] == "llm_unavailable"


def test_mock_interview_without_key_returns_503(client, cv_id):
    response = client.post("/api/v1/mock-interview", json={"cv_id": cv_id, "job_title": "Backend Engineer"})
    assert response.status_code == 503


def test_career_agent_replies_without_key(client, cv_id):
    response = client.post("/api/v1/career-agent/chat",
                           json={"cv_id": cv_id, "message": "Which jobs fit me?"})
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert "OPENROUTER_API_KEY" in body["reply"]


def test_career_agent_persists_history(client, cv_id):
    first = client.post("/api/v1/career-agent/chat",
                        json={"cv_id": cv_id, "session_id": "sess-test", "message": "hello there"})
    assert first.status_code == 200
    history = client.get(f"/api/v1/career-agent/history?cv_id={cv_id}&session_id=sess-test")
    assert history.status_code == 200
    roles = [m["role"] for m in history.json()]
    assert "user" in roles and "assistant" in roles


def test_career_agent_404_for_unknown_cv(client):
    assert client.post("/api/v1/career-agent/chat",
                       json={"cv_id": "nope", "message": "hi"}).status_code == 404
