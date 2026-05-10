"""Tests for the eval test case structure and coverage."""
from __future__ import annotations

from app.eval.test_cases import TEST_CASES


class TestTestCases:
    def test_15_cases_present(self):
        assert len(TEST_CASES) == 15

    def test_unique_ids(self):
        ids = [c["id"] for c in TEST_CASES]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"

    def test_category_distribution(self):
        cats = [c["category"] for c in TEST_CASES]
        assert cats.count("baseline") == 5
        assert cats.count("ambiguous") == 5
        assert cats.count("adversarial") == 5

    def test_all_have_required_fields(self):
        required = {"id", "category", "query", "expected_answer_keywords", "expected_no_keywords"}
        for case in TEST_CASES:
            missing = required - set(case.keys())
            assert not missing, f"Case {case['id']} missing: {missing}"

    def test_adversarial_have_no_keywords(self):
        """Every adversarial case must have at least one forbidden keyword."""
        adversarial = [c for c in TEST_CASES if c["category"] == "adversarial"]
        for case in adversarial:
            assert len(case["expected_no_keywords"]) > 0, (
                f"Adversarial case {case['id']} has no expected_no_keywords"
            )

    def test_queries_are_nonempty(self):
        for case in TEST_CASES:
            assert len(case["query"].strip()) > 10, f"Case {case['id']} has too-short query"

    def test_keywords_are_lists(self):
        for case in TEST_CASES:
            assert isinstance(case["expected_answer_keywords"], list)
            assert isinstance(case["expected_no_keywords"], list)

    def test_categories_are_valid(self):
        valid = {"baseline", "ambiguous", "adversarial"}
        for case in TEST_CASES:
            assert case["category"] in valid, f"Invalid category: {case['category']}"
