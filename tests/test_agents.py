"""Test script for all LLM agents.

Run individual agent tests:
    uv run python -m tests.test_agents issue
    uv run python -m tests.test_agents enrichment
    uv run python -m tests.test_agents tagger
    uv run python -m tests.test_agents cluster_namer
    uv run python -m tests.test_agents executive_report
    uv run python -m tests.test_agents faq_synthesizer
    uv run python -m tests.test_agents decomposer
    uv run python -m tests.test_agents decomposer_presup
    uv run python -m tests.test_agents all

Each test invokes the agent with test data and prints the structured response.
"""

import argparse

from dotenv import load_dotenv

from src.agents.cluster_naming_agent import cluster_namer
from src.agents.enrichment_agent import enrichment_extractor
from src.agents.executive_report_agent import executive_report_generator
from src.agents.faq_synthesizer_agent import faq_synthesizer
from src.agents.issue_agent import issue_extractor
from src.agents.question_decomposer_agent import get_question_decomposer
from src.agents.tagger_agent import tagger_extractor
from tests.test_assets import (
    TEST_COMPOUND_QUESTION,
    TEST_EXECUTIVE_DATA_JSON,
    TEST_FAQ_REPRESENTATIVE,
    TEST_FAQ_VARIANTS,
    TEST_ISSUES_TEXT,
    TEST_PRESUPPOSITION_QUESTION,
    TEST_SIMPLE_QUESTION,
    TEST_SINGLE_ISSUE_TEXT,
    TEST_TRANSCRIPT,
)

load_dotenv()


def test_issue_agent():
    """Test the issue extraction agent."""
    print("=" * 60)
    print("Testing: Issue Agent")
    print("=" * 60)

    result = issue_extractor.invoke({"transcript": TEST_TRANSCRIPT})
    print(result)
    print()


def test_enrichment_agent():
    """Test the enrichment agent (sentiment, resolution, steps)."""
    print("=" * 60)
    print("Testing: Enrichment Agent")
    print("=" * 60)

    result = enrichment_extractor.invoke(
        {
            "transcript": TEST_TRANSCRIPT,
            "issue_report": TEST_SINGLE_ISSUE_TEXT,
        }
    )
    print(result)
    print()


def test_tagger_agent():
    """Test the tagger agent (tags with pooling)."""
    print("=" * 60)
    print("Testing: Tagger Agent")
    print("=" * 60)

    result = tagger_extractor.invoke(
        {
            "transcript": TEST_TRANSCRIPT,
            "issue_report": TEST_SINGLE_ISSUE_TEXT,
            "available_tags": "shipping, returns, account, billing",
        }
    )
    print(result)
    print()


def test_cluster_namer():
    """Test the cluster naming agent."""
    print("=" * 60)
    print("Testing: Cluster Naming Agent")
    print("=" * 60)

    result = cluster_namer.invoke(
        {
            "already_named_clusters": "None yet.",
            "issues_text": TEST_ISSUES_TEXT,
        }
    )
    print(result)
    print()


def test_executive_report_agent():
    """Test the executive report agent."""
    print("=" * 60)
    print("Testing: Executive Report Agent")
    print("=" * 60)

    result = executive_report_generator.invoke({"data_json": TEST_EXECUTIVE_DATA_JSON})
    print(result)
    print()


def test_faq_synthesizer():
    """Test the FAQ synthesizer agent."""
    print("=" * 60)
    print("Testing: FAQ Synthesizer Agent")
    print("=" * 60)

    result = faq_synthesizer.invoke(
        {
            "representative": TEST_FAQ_REPRESENTATIVE,
            "variants": TEST_FAQ_VARIANTS,
        }
    )
    print(result)
    print()


def test_question_decomposer():
    """Test the question decomposer agent (standard mode)."""
    print("=" * 60)
    print("Testing: Question Decomposer Agent (Standard)")
    print("=" * 60)

    decomposer = get_question_decomposer(with_presuppositions=False)

    print("Testing compound question:")
    result = decomposer.invoke({"question": TEST_COMPOUND_QUESTION})
    print(result)
    print()

    print("Testing simple question:")
    result = decomposer.invoke({"question": TEST_SIMPLE_QUESTION})
    print(result)
    print()


def test_question_decomposer_with_presuppositions():
    """Test the question decomposer agent (with presuppositions)."""
    print("=" * 60)
    print("Testing: Question Decomposer Agent (With Presuppositions)")
    print("=" * 60)

    decomposer = get_question_decomposer(with_presuppositions=True)

    result = decomposer.invoke({"question": TEST_PRESUPPOSITION_QUESTION})
    print(result)
    print()


def main():
    parser = argparse.ArgumentParser(description="Test LLM agents")
    parser.add_argument(
        "agent",
        choices=[
            "issue",
            "enrichment",
            "tagger",
            "cluster_namer",
            "executive_report",
            "faq_synthesizer",
            "decomposer",
            "decomposer_presup",
            "all",
        ],
        help="Which agent to test",
    )
    args = parser.parse_args()

    tests = {
        "issue": test_issue_agent,
        "enrichment": test_enrichment_agent,
        "tagger": test_tagger_agent,
        "cluster_namer": test_cluster_namer,
        "executive_report": test_executive_report_agent,
        "faq_synthesizer": test_faq_synthesizer,
        "decomposer": test_question_decomposer,
        "decomposer_presup": test_question_decomposer_with_presuppositions,
    }

    if args.agent == "all":
        for name, test_fn in tests.items():
            print(f"\n{'#' * 60}")
            print(f"# Running: {name}")
            print(f"{'#' * 60}\n")
            test_fn()
    else:
        tests[args.agent]()


if __name__ == "__main__":
    main()
