"""Job alerts: seen-job diffing, email rendering, scheduler wiring."""
from __future__ import annotations

import asyncio

import pytest

from app import crud
from app.db import session_scope
from app.schemas import CVProfile, Job
from app.services.notifications import EmailNotConfigured, render_jobs_email, _send_sync
from app.services.scheduler import scheduler_status, start_scheduler


@pytest.fixture()
def alert_ctx():
    with session_scope() as db:
        cv = crud.create_cv(db, "python fastapi", CVProfile(skills=["Python"]), "cv.pdf", "heuristic")
        alert = crud.create_alert(db, {
            "cv_id": cv.id, "name": "Test alert", "destination": "me@example.com",
            "filters": {"keywords": "python"},
        })
        yield cv.id, alert.id


def _job(n: int) -> Job:
    return Job(id=f"j{n}", title=f"Job {n}", company="Co", url=f"https://x.com/job/{n}", source="wuzzuf")


def test_diffing_marks_and_skips_seen_jobs(alert_ctx):
    _, alert_id = alert_ctx
    jobs = [_job(1), _job(2)]

    with session_scope() as db:
        fresh = crud.filter_unseen_jobs(db, alert_id, jobs)
        assert len(fresh) == 2
        assert crud.mark_jobs_seen(db, alert_id, fresh) == 2

    with session_scope() as db:
        # Same two jobs + one new -> only the new one comes back.
        fresh = crud.filter_unseen_jobs(db, alert_id, jobs + [_job(3)])
        assert [j.id for j in fresh] == ["j3"]


def test_diffing_is_url_normalised(alert_ctx):
    _, alert_id = alert_ctx
    with session_scope() as db:
        crud.mark_jobs_seen(db, alert_id, [_job(1)])
    tracked = Job(id="other", title="Job 1", company="Co",
                  url="https://x.com/job/1/?utm_source=email", source="bayt")
    with session_scope() as db:
        assert crud.filter_unseen_jobs(db, alert_id, [tracked]) == []


def test_email_rendering_contains_job_details():
    text, html = render_jobs_email("Backend Cairo", [
        Job(id="1", title="Python Dev", company="Fawry", url="https://x.com/1",
            source="wuzzuf", match_score=87.0, location="Cairo",
            matching_skills=["Python"], missing_skills=["Kubernetes"]),
    ])
    assert "Python Dev" in text and "Fawry" in text and "87%" in text
    assert "https://x.com/1" in html
    assert "Kubernetes" in text


def test_email_without_smtp_config_raises_clear_error():
    with pytest.raises(EmailNotConfigured):
        _send_sync("a@b.com", "s", "t", "<p>h</p>")


def test_alert_run_handles_scraper_outage(alert_ctx, monkeypatch):
    """A full scraping outage must not raise — the alert just finds nothing."""
    from app.services import scheduler as scheduler_module

    async def fake_search(profile, filters, cv_text="", cv_id=""):
        from app.schemas import SourceError
        return {"jobs": [], "errors": [SourceError(source="wuzzuf", error="blocked", error_type="blocked")]}

    import app.agents.graph as graph_module
    monkeypatch.setattr(graph_module, "run_job_search", fake_search)

    _, alert_id = alert_ctx
    outcome = asyncio.run(scheduler_module.run_alert(alert_id, send_email=False))
    assert outcome["new_jobs"] == []
    assert outcome["emailed"] is False


def test_alert_run_detects_new_jobs(alert_ctx, monkeypatch):
    from app.services import scheduler as scheduler_module
    import app.agents.graph as graph_module

    async def fake_search(profile, filters, cv_text="", cv_id=""):
        return {"jobs": [_job(11), _job(12)], "errors": []}

    monkeypatch.setattr(graph_module, "run_job_search", fake_search)
    _, alert_id = alert_ctx

    first = asyncio.run(scheduler_module.run_alert(alert_id, send_email=False))
    assert len(first["new_jobs"]) == 2
    # Second run: same jobs, already seen -> nothing new, no email.
    second = asyncio.run(scheduler_module.run_alert(alert_id, send_email=False))
    assert second["new_jobs"] == []


def test_scheduler_status_reports_disabled_in_tests():
    assert scheduler_status() == "disabled"
    assert start_scheduler() is None
