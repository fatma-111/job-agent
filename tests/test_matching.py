"""Embedding similarity, ranking and skill comparison."""
from __future__ import annotations

import numpy as np

from app.schemas import CVProfile, Job
from app.services.matching import (
    compare_skills,
    cosine_similarity,
    embedding_service,
    rank_jobs,
    similarity_to_percentage,
)

PROFILE = CVProfile(
    skills=["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
    job_titles=["Backend Engineer"],
    summary="Backend engineer building APIs",
)


def test_embeddings_produce_normalised_vectors():
    vectors = embedding_service.encode(["python backend", "graphic design"])
    assert vectors.shape[0] == 2
    for vector in vectors:
        assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-4


def test_similar_text_scores_higher_than_unrelated():
    vectors = embedding_service.encode([
        "python fastapi backend engineer api postgresql",
        "python fastapi backend developer rest api docker",
        "florist arranging wedding flowers bouquet",
    ])
    related = cosine_similarity(vectors[0], vectors[1])
    unrelated = cosine_similarity(vectors[0], vectors[2])
    assert related > unrelated


def test_score_percentage_bounds():
    backend = embedding_service.backend
    assert similarity_to_percentage(-1, backend) == 0.0
    assert similarity_to_percentage(2, backend) == 100.0
    assert 0 <= similarity_to_percentage(0.3, backend) <= 100


def test_compare_skills_splits_matching_and_missing():
    matching, missing = compare_skills(PROFILE.skills, "We need Python, FastAPI and Kubernetes")
    assert "Python" in matching and "FastAPI" in matching
    assert "Kubernetes" in missing


def test_ranking_orders_by_relevance():
    jobs = [
        Job(id="a", title="Graphic Designer", description="Photoshop Illustrator branding"),
        Job(id="b", title="Senior Python Backend Engineer",
            description="Python FastAPI PostgreSQL Docker AWS REST API"),
    ]
    ranked = rank_jobs(jobs, PROFILE, "python fastapi postgresql docker aws")
    assert ranked[0].title.startswith("Senior Python")
    assert ranked[0].match_score > ranked[1].match_score
    assert ranked[0].match_explanation
    assert 0 <= ranked[0].match_score <= 100


def test_ranking_empty_list():
    assert rank_jobs([], PROFILE) == []
