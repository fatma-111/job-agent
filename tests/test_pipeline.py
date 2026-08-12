"""LangGraph pipeline: fan-out/fan-in, filtering, partial-failure resilience."""
from __future__ import annotations

import asyncio

import pytest

from app.agents.graph import build_graph, run_job_search
from app.agents.nodes import build_search_query, filter_jobs, normalize_jobs, rank_jobs_node
from app.schemas import CVProfile, Job, RawJob, SearchFilters, SourceError
from app.scrapers.base import ScraperBlocked, ScraperError
from app.scrapers.registry import ALL_SOURCES

PROFILE = CVProfile(skills=["Python", "FastAPI", "Docker"], job_titles=["Backend Engineer"])


def test_graph_has_all_nodes():
    nodes = set(build_graph().get_graph().nodes.keys())
    for source in ALL_SOURCES:
        assert f"scrape_{source}" in nodes
    assert {"build_search_query", "normalize_jobs", "filter_jobs", "rank_jobs"} <= nodes


def test_build_query_prefers_explicit_keywords():
    out = build_search_query({"profile": PROFILE, "filters": SearchFilters(keywords="devops engineer")})
    assert out["query"] == "devops engineer"


def test_build_query_falls_back_to_cv_title():
    out = build_search_query({"profile": PROFILE, "filters": SearchFilters()})
    assert out["query"] == "Backend Engineer"


def test_normalize_node_dedupes():
    state = {"raw_jobs": [
        RawJob(title="Dev", url="https://a.com/1", source="wuzzuf"),
        RawJob(title="Dev", url="https://a.com/1", source="bayt"),
    ]}
    assert normalize_jobs(state)["total_found"] == 1


def test_filter_min_salary_excludes_only_known_values():
    jobs = [
        Job(id="1", title="Low", salary_min=5000, salary_period="monthly"),
        Job(id="2", title="High", salary_min=30000, salary_period="monthly"),
        Job(id="3", title="Unknown"),  # no salary data -> must be kept
    ]
    kept = filter_jobs({"jobs": jobs, "filters": SearchFilters(min_salary=10000)})["jobs"]
    titles = {j.title for j in kept}
    assert titles == {"High", "Unknown"}


def test_filter_by_age():
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    jobs = [
        Job(id="1", title="Fresh", posted_date=now - timedelta(days=2)),
        Job(id="2", title="Stale", posted_date=now - timedelta(days=60)),
        Job(id="3", title="Undated"),
    ]
    kept = filter_jobs({"jobs": jobs, "filters": SearchFilters(max_age_days=7)})["jobs"]
    assert {j.title for j in kept} == {"Fresh", "Undated"}


def test_rank_node_scores_all_jobs():
    state = {
        "jobs": [Job(id="1", title="Python Engineer", description="Python FastAPI Docker")],
        "profile": PROFILE,
        "cv_text": "python fastapi docker",
    }
    out = rank_jobs_node(state)
    assert out["jobs"][0].match_score is not None


def test_partial_scraper_failure_still_returns_results(monkeypatch):
    """Two sources succeed, three fail -> results returned, errors recorded."""
    import app.agents.nodes as nodes_module

    class FakeScraper:
        def __init__(self, source):
            self.source = source

        async def search(self, query, location, limit=10):
            if self.source in ("indeed", "linkedin"):
                raise ScraperBlocked(f"{self.source}: blocked")
            if self.source == "tanqeeb":
                raise ScraperError(f"{self.source}: selectors changed")
            return [
                RawJob(
                    title=f"Python Developer ({self.source})",
                    company="TestCo",
                    url=f"https://{self.source}.test/job/1",
                    description="Python FastAPI Docker Kubernetes",
                    salary_text="20,000 EGP monthly",
                    posted_text="2 days ago",
                    source=self.source,
                )
            ]

    def fake_get_scraper_class(source):
        return lambda browser: FakeScraper(source)

    monkeypatch.setattr(nodes_module, "get_scraper_class", fake_get_scraper_class)

    async def fake_enter(self):
        return object()

    from app.agents import graph as graph_module

    class FakeSession:
        async def __aenter__(self):
            return self

        async def close(self):
            return None

    monkeypatch.setattr(graph_module, "BrowserSession", FakeSession)

    result = asyncio.run(
        run_job_search(PROFILE, SearchFilters(limit_per_source=5), cv_text="python", cv_id="x")
    )

    assert len(result["jobs"]) == 2, result["jobs"]
    assert {e.source for e in result["errors"]} == {"indeed", "linkedin", "tanqeeb"}
    assert {e.error_type for e in result["errors"]} == {"blocked", "scrape_error"}
    assert all(j.match_score is not None for j in result["jobs"])
    assert result["jobs"][0].salary_min == 20000.0


def test_total_browser_failure_is_graceful(monkeypatch):
    from app.agents import graph as graph_module
    from app.scrapers.base import ScraperUnavailable

    class DeadSession:
        async def __aenter__(self):
            raise ScraperUnavailable("chromium missing")

        async def close(self):
            return None

    monkeypatch.setattr(graph_module, "BrowserSession", DeadSession)
    result = asyncio.run(run_job_search(PROFILE, SearchFilters(), cv_id="x"))
    assert result["jobs"] == []
    assert len(result["errors"]) == len(ALL_SOURCES)
