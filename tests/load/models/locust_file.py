# tests/load/models/locustfile.py
"""
Locust load test definitions for the book recommendation and similarity API.

Each HttpUser subclass models a distinct type of user behaviour. Weights
control the proportion of each user type spawned during a test run, tuned to
reflect the production traffic distribution: warm behavioral is the dominant
path, cold users account for a meaningful share, and similarity requests are
common but secondary.

All user classes set wait_time = constant(0) for maximum-throughput stress
testing. To simulate realistic think time, replace with between(1, 3).

Usage — interactive web UI (recommended for profiling and exploration):
    locust -f tests/load/models/locustfile.py --host http://localhost:8000

Usage — headless CI or scripted benchmarks:
    locust -f tests/load/models/locustfile.py \
           --host http://localhost:8000       \
           --headless                         \
           --users 50                         \
           --spawn-rate 5                     \
           --run-time 60s                     \
           --csv results/locust_run

The --csv flag writes results/locust_run_stats.csv and
results/locust_run_failures.csv, suitable for archiving alongside
performance_baselines/ JSON files.

To attach py-spy while a Locust run is in progress (from a separate terminal):
    py-spy record --pid <gunicorn_worker_pid> \
                  --output profile.json        \
                  --format speedscope          \
                  --duration 30

Find worker PIDs with: ps aux | grep gunicorn
"""

import random

from locust import HttpUser, between, constant, task

from tests.load.models._constants import (
    COLD_WITHOUT_SUBJECTS_USER_IDS,
    COLD_WITH_SUBJECTS_USER_IDS,
    TEST_BOOK_IDS,
    WARM_USER_IDS,
)


class WarmRecommendationUser(HttpUser):
    """
    Simulates users who have enough ratings to receive ALS-based recommendations.

    Weighted heavily because warm behavioral is the dominant production path.
    Includes a minority of subject-mode requests to represent warm users who
    navigate to subject-discovery contexts.
    """

    weight = 6
    wait_time = constant(0)

    @task(3)
    def recommend_behavioral(self) -> None:
        user_id = random.choice(WARM_USER_IDS)
        self.client.get(
            "/profile/recommend",
            params={"user": str(user_id), "_id": True, "top_n": 200, "mode": "behavioral"},
            name="/profile/recommend [warm/behavioral]",
        )

    @task(1)
    def recommend_subject(self) -> None:
        user_id = random.choice(WARM_USER_IDS)
        self.client.get(
            "/profile/recommend",
            params={"user": str(user_id), "_id": True, "top_n": 200, "mode": "subject"},
            name="/profile/recommend [warm/subject]",
        )


class ColdRecommendationUser(HttpUser):
    """
    Simulates new or infrequent users routed through the cold pipeline.

    Users with favourite subjects set invoke the subject embedding + FAISS
    path; users without subjects fall back to Bayesian popularity. The 3:1
    task weighting reflects that most cold users do have at least some subjects.
    """

    weight = 4
    wait_time = constant(0)

    @task(3)
    def recommend_cold_with_subjects(self) -> None:
        if not COLD_WITH_SUBJECTS_USER_IDS:
            return
        user_id = random.choice(COLD_WITH_SUBJECTS_USER_IDS)
        self.client.get(
            "/profile/recommend",
            params={"user": str(user_id), "_id": True, "top_n": 200, "mode": "subject"},
            name="/profile/recommend [cold/with-subjects]",
        )

    @task(1)
    def recommend_cold_without_subjects(self) -> None:
        if not COLD_WITHOUT_SUBJECTS_USER_IDS:
            return
        user_id = random.choice(COLD_WITHOUT_SUBJECTS_USER_IDS)
        self.client.get(
            "/profile/recommend",
            params={"user": str(user_id), "_id": True, "top_n": 200, "mode": "subject"},
            name="/profile/recommend [cold/no-subjects]",
        )


class SimilarityUser(HttpUser):
    """
    Simulates users browsing book detail pages and triggering similarity requests.

    ALS and hybrid modes are only available for books that have ALS factors;
    422 responses for books without ALS data are marked as successes via
    catch_response so they do not inflate the Locust failure rate — they
    represent a valid application-level outcome, not a server error.
    """

    weight = 4
    wait_time = constant(0)

    @task(3)
    def similar_subject(self) -> None:
        book_id = random.choice(TEST_BOOK_IDS)
        self.client.get(
            f"/book/{book_id}/similar",
            params={"mode": "subject", "top_k": 200},
            name="/book/[id]/similar [subject]",
        )

    @task(2)
    def similar_als(self) -> None:
        book_id = random.choice(TEST_BOOK_IDS)
        with self.client.get(
            f"/book/{book_id}/similar",
            params={"mode": "als", "top_k": 200},
            name="/book/[id]/similar [als]",
            catch_response=True,
        ) as response:
            if response.status_code == 422:
                response.success()

    @task(2)
    def similar_hybrid(self) -> None:
        book_id = random.choice(TEST_BOOK_IDS)
        with self.client.get(
            f"/book/{book_id}/similar",
            params={"mode": "hybrid", "top_k": 200},
            name="/book/[id]/similar [hybrid]",
            catch_response=True,
        ) as response:
            if response.status_code == 422:
                response.success()
