"""Shared fixtures. Every test runs against a throwaway SQLite file."""
from __future__ import annotations

import io
import os
import tempfile

import pytest

# Point the app at a temp DB *before* app modules import settings.
_TMP_DB = os.path.join(tempfile.mkdtemp(prefix="jobagent-test-"), "test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["SCHEDULER_ENABLED"] = "false"
os.environ.setdefault("OPENROUTER_API_KEY", "")

from fastapi.testclient import TestClient  # noqa: E402

from app.db import init_db  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _db():
    init_db()
    yield


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


SAMPLE_CV_TEXT = """Ahmed Hassan
Senior Backend Engineer
Cairo, Egypt | ahmed@example.com

PROFESSIONAL SUMMARY
Backend engineer with 7 years of experience building scalable REST APIs.

EXPERIENCE
2019 - Present  Senior Software Engineer at Vodafone Egypt
2017 - 2019  Backend Developer at Instabug

SKILLS
Python, FastAPI, Django, PostgreSQL, Redis, Docker, Kubernetes, AWS, REST API, Git

EDUCATION
B.Sc. Computer Science, Cairo University, 2016

CERTIFICATIONS
AWS Certified Solutions Architect
"""


@pytest.fixture(scope="session")
def sample_pdf_bytes() -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    y = 750
    for line in SAMPLE_CV_TEXT.splitlines():
        pdf.drawString(60, y, line[:95])
        y -= 16
        if y < 60:
            pdf.showPage()
            y = 750
    pdf.save()
    return buffer.getvalue()


@pytest.fixture(scope="session")
def sample_docx_bytes() -> bytes:
    import docx

    document = docx.Document()
    for line in SAMPLE_CV_TEXT.splitlines():
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture(scope="session")
def cv_id(client, sample_pdf_bytes) -> str:
    response = client.post(
        "/api/v1/cv/upload",
        files={"file": ("ahmed_cv.pdf", sample_pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200, response.text
    return response.json()["cv_id"]
