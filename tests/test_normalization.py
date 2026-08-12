"""Date/salary normalisation and dedup."""
from __future__ import annotations

from datetime import datetime, timezone

from app.schemas import RawJob
from app.services.normalize import (
    canonical_url,
    monthly_equivalent,
    normalize_and_dedupe,
    normalize_job,
    parse_posted_date,
    parse_salary,
    url_hash,
)

NOW = datetime(2026, 8, 12, tzinfo=timezone.utc)


def test_relative_dates_english():
    assert parse_posted_date("3 days ago", NOW).day == 9
    assert parse_posted_date("yesterday", NOW).day == 11
    assert parse_posted_date("just now", NOW).day == 12


def test_relative_dates_arabic():
    assert parse_posted_date("منذ 3 أيام", NOW).day == 9
    assert parse_posted_date("منذ أسبوع", NOW).day == 5


def test_absolute_dates():
    assert parse_posted_date("2026-05-01").month == 5
    assert parse_posted_date("15/03/2026").day == 15


def test_unparseable_date_returns_none():
    assert parse_posted_date("whenever") is None
    assert parse_posted_date(None) is None
    assert parse_posted_date("") is None


def test_salary_range():
    assert parse_salary("15,000 - 20,000 EGP/month") == (15000.0, 20000.0, "EGP", "monthly")


def test_salary_with_multiplier():
    lo, hi, cur, per = parse_salary("EGP 12k monthly")
    assert lo == 12000.0 and cur == "EGP" and per == "monthly"


def test_salary_never_fabricated():
    assert parse_salary("Confidential") == (None, None, None, None)
    assert parse_salary(None) == (None, None, None, None)


def test_monthly_equivalent():
    assert monthly_equivalent(120000, "yearly") == 10000.0
    assert monthly_equivalent(None, "yearly") is None


def test_canonical_url_strips_tracking():
    assert canonical_url("https://x.com/job/1?utm_source=g") == "https://x.com/job/1"
    assert url_hash("https://x.com/job/1/") == url_hash("https://x.com/job/1?ref=a")


def test_dedupe_keeps_richer_record():
    jobs = normalize_and_dedupe([
        RawJob(title="Dev", company="A", url="https://x.com/1?utm=a", description="short", source="wuzzuf"),
        RawJob(title="Dev", company="A", url="https://x.com/1/", description="a longer description", source="bayt"),
        RawJob(title="Other", company="B", url="https://x.com/2", source="bayt"),
    ])
    assert len(jobs) == 2
    assert "longer" in jobs[0].description


def test_normalize_populates_structured_fields():
    job = normalize_job(RawJob(
        title="  Backend   Dev ", company="Fawry", url="https://x.com/9",
        salary_text="10,000 - 15,000 EGP per month", posted_text="2 days ago", source="wuzzuf",
    ))
    assert job.title == "Backend Dev"
    assert job.salary_min == 10000.0
    assert job.salary_currency == "EGP"
    assert job.posted_date is not None
    assert job.id
