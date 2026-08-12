"""CV text extraction + intelligent field extraction."""
from __future__ import annotations

import asyncio

import pytest

from app.services.matching import (
    CVParseError,
    extract_cv_profile,
    extract_cv_text,
    heuristic_profile,
)
from tests.conftest import SAMPLE_CV_TEXT


def test_pdf_parsing(sample_pdf_bytes):
    text = extract_cv_text(sample_pdf_bytes, "cv.pdf")
    assert "Ahmed Hassan" in text
    assert "FastAPI" in text


def test_docx_parsing(sample_docx_bytes):
    text = extract_cv_text(sample_docx_bytes, "cv.docx")
    assert "Ahmed Hassan" in text
    assert "PostgreSQL" in text


def test_unsupported_file_type():
    with pytest.raises(CVParseError):
        extract_cv_text(b"binary", "photo.jpg")


def test_empty_pdf_rejected():
    with pytest.raises(CVParseError):
        extract_cv_text(b"tiny", "cv.pdf")


def test_heuristic_extraction():
    profile = heuristic_profile(SAMPLE_CV_TEXT)
    assert "Python" in profile.skills
    assert "FastAPI" in profile.skills
    assert "Docker" in profile.skills
    assert profile.years_experience == 7.0
    assert any("Cairo University" in e for e in profile.education)
    assert any("AWS Certified" in c for c in profile.certifications)


def test_extraction_falls_back_without_llm():
    """No API key configured -> heuristic path, never an exception."""
    profile, method = asyncio.run(extract_cv_profile(SAMPLE_CV_TEXT))
    assert method == "heuristic"
    assert profile.skills
