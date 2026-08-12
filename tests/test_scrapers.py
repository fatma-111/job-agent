"""Scraper architecture: URLs, selector config, parsing, block handling."""
from __future__ import annotations

import asyncio
import json
import os

import pytest

from app.scrapers.base import BaseScraper, ScraperBlocked, ScraperError, ScraperUnavailable
from app.scrapers.registry import ALL_SOURCES, SCRAPERS, get_scraper_class


def test_all_five_sources_registered():
    assert set(ALL_SOURCES) == {"wuzzuf", "bayt", "tanqeeb", "indeed", "linkedin"}


@pytest.mark.parametrize("source", ALL_SOURCES)
def test_build_url_is_valid(source):
    scraper = SCRAPERS[source]()
    url = scraper.build_url("python developer", "Cairo")
    assert url.startswith("https://")
    assert "python" in url.lower()


@pytest.mark.parametrize("source", ALL_SOURCES)
def test_every_scraper_declares_selectors(source):
    scraper = SCRAPERS[source]()
    for key in ("card", "title", "company"):
        assert scraper.selectors.get(key), f"{source} missing '{key}' selectors"
        assert isinstance(scraper.selectors[key], list)


@pytest.mark.parametrize("source", ALL_SOURCES)
def test_selector_fallback_chains_exist(source):
    """Each scraper needs >1 card selector so a layout change has a fallback."""
    assert len(SCRAPERS[source]().selectors["card"]) >= 2


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        get_scraper_class("monster")


def test_absolute_url_resolution():
    scraper = SCRAPERS["wuzzuf"]()
    assert scraper.absolute_url("/jobs/p/123").startswith("https://wuzzuf.net")
    assert scraper.absolute_url("https://other.com/x") == "https://other.com/x"
    assert scraper.absolute_url("") == ""


def test_selector_overrides_are_applied(tmp_path, monkeypatch):
    """selectors.json lets you repair a scraper without editing code."""
    import app.scrapers.base as base_module

    override_file = tmp_path / "selectors.json"
    override_file.write_text(json.dumps({"wuzzuf": {"card": ["div.brand-new-class"]}}))
    monkeypatch.setattr(base_module, "SELECTOR_OVERRIDE_PATH", str(override_file))

    scraper = SCRAPERS["wuzzuf"]()
    assert scraper.selectors["card"] == ["div.brand-new-class"]
    assert scraper.selectors["title"]  # untouched keys keep their defaults


def test_search_without_session_raises_unavailable():
    scraper = SCRAPERS["bayt"]()
    with pytest.raises(ScraperUnavailable):
        asyncio.run(scraper.search("python"))


def test_block_detection_flags_captcha_pages():
    class FakePage:
        def __init__(self, html):
            self._html = html

        async def content(self):
            return self._html

    scraper = SCRAPERS["indeed"]()
    with pytest.raises(ScraperBlocked):
        asyncio.run(scraper._detect_block(FakePage("<html>Please verify you are human</html>")))
    with pytest.raises(ScraperBlocked):
        asyncio.run(scraper._detect_block(FakePage("<html>authwall - sign in</html>")))
    # A normal page must pass through untouched.
    asyncio.run(scraper._detect_block(FakePage("<html><div class='job'>Python Dev</div></html>")))


def test_parse_card_uses_selector_table():
    """parse_card reads whatever the selector table points at, with fallbacks."""

    class FakeNode:
        def __init__(self, text="", attrs=None):
            self._text = text
            self._attrs = attrs or {}

        async def inner_text(self):
            return self._text

        async def get_attribute(self, name):
            return self._attrs.get(name)

    class FakeCard:
        def __init__(self, mapping):
            self.mapping = mapping

        async def query_selector(self, selector):
            return self.mapping.get(selector)

    scraper = SCRAPERS["wuzzuf"]()
    card = FakeCard({
        "h2.css-m604qf a": FakeNode("Senior Python Developer", {"href": "/jobs/p/42"}),
        "a.css-17s97q8": FakeNode("Fawry"),
        "span.css-5wys0k": FakeNode("Cairo, Egypt"),
        "div.css-4c4ojb": FakeNode("3 days ago"),
    })
    job = asyncio.run(scraper.parse_card(card))
    assert job.title == "Senior Python Developer"
    assert job.company == "Fawry"
    assert job.location == "Cairo, Egypt"
    assert job.url == "https://wuzzuf.net/jobs/p/42"
    assert job.posted_text == "3 days ago"
    assert job.source == "wuzzuf"


def test_parse_card_returns_none_without_title():
    class EmptyCard:
        async def query_selector(self, selector):
            return None

    assert asyncio.run(SCRAPERS["bayt"]().parse_card(EmptyCard())) is None


def test_blocked_error_types():
    assert ScraperBlocked("x").error_type == "blocked"
    assert ScraperError("x").error_type == "scrape_error"
    assert ScraperUnavailable("x").error_type == "unavailable"
