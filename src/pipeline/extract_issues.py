"""Step 2: Extract Issues - Analyze transcripts and extract structured issue reports.

Uses LLM agents to analyze conversation transcripts and extract structured
issue reports. Supports three processing modes based on config:

- Primary agent only: Fast parallel extraction of base fields
- With enrichment (default): Adds sentiment, resolution, steps (parallel)
- With tagger: Adds tags with pooling (sequential, must be explicitly enabled)

Features:
- Resume support: skips already-processed conversations
- Incremental saves: progress saved after each batch/item
- Retry logic: configurable retries for JSON parsing failures

Reads:
    config.paths.clean_conversations - Clean conversation data with transcript paths.

Writes:
    config.paths.issue_reports - Structured issue reports keyed by conversation_id.
"""

import asyncio
import json
import os

from dotenv import load_dotenv

from config import config
from src.agents.enrichment_agent import enrichment_extractor
from src.agents.issue_agent import issue_extractor
from src.agents.tagger_agent import tagger_extractor
from src.schemas import EnrichmentReport, IssueReport, TaggerReport
from src.utils.helpers import read_json_file

load_dotenv()


def read_transcript(transcript_path: str) -> str:
    """Read transcript text file.

    Args:
        transcript_path: Path to transcript text file.

    Returns:
        Transcript content as string.
    """
    with open(transcript_path, encoding="utf-8") as f:
        return f.read()


def save_issue_reports(data: dict, output_path: str) -> None:
    """Save issue reports to JSON file.

    Args:
        data: Dict of issue reports to save.
        output_path: Path to output JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =============================================================================
# Primary Agent Functions
# =============================================================================


async def async_get_base_report(
    conv_id: str,
    raw_transcript: str,
    max_retries: int | None = None,
) -> tuple[str, dict | None]:
    """Call primary LLM agent to extract base issue report from transcript.

    Uses structured output for guaranteed schema compliance - no JSON parsing needed.

    Args:
        conv_id: Conversation UUID.
        raw_transcript: Transcript text content.
        max_retries: Max retries for API/rate limit errors (JSON parsing no longer needed).

    Returns:
        Tuple of (conv_id, base_report) where base_report is None on failure.
    """
    if max_retries is None:
        max_retries = config["llm"]["max_retries"]

    for attempt in range(max_retries):
        try:
            # Structured output guarantees valid IssueReport schema
            result: IssueReport = await issue_extractor.ainvoke(
                {"transcript": raw_transcript}
            )
            # Convert Pydantic model to dict for compatibility with rest of pipeline
            base_report = result.model_dump()
            return (conv_id, base_report)
        except Exception as e:
            print(f"  {conv_id}: Retry {attempt + 1}/{max_retries} - Error: {e}")

    print(f"  {conv_id}: Failed after {max_retries} retries")
    return (conv_id, None)


# =============================================================================
# Enrichment Agent Functions (parallel)
# =============================================================================


async def async_get_enrichment_fields(
    conv_id: str,
    raw_transcript: str,
    base_report: dict,
    max_retries: int | None = None,
) -> dict | None:
    """Call enrichment agent to get sentiment, resolution, and steps.

    Uses structured output for guaranteed schema compliance.
    This agent runs in PARALLEL - no tag pooling needed.

    Args:
        conv_id: Conversation UUID.
        raw_transcript: Transcript text content.
        base_report: Base issue report from primary agent.
        max_retries: Max retries for API/rate limit errors.

    Returns:
        Enrichment fields dict on success, None if all retries fail.
    """
    if max_retries is None:
        max_retries = config["llm"]["max_retries"]

    for attempt in range(max_retries):
        try:
            # Structured output guarantees valid EnrichmentReport schema
            result: EnrichmentReport = await enrichment_extractor.ainvoke(
                {
                    "transcript": raw_transcript,
                    "issue_report": json.dumps(base_report, indent=2),
                }
            )
            return result.model_dump()  # type: ignore[no-any-return]
        except Exception as e:
            print(
                f"  {conv_id} (enrich): Retry {attempt + 1}/{max_retries} - Error: {e}"
            )

    print(f"  {conv_id} (enrich): Failed after {max_retries} retries")
    return None


# =============================================================================
# Tagger Agent Functions (sequential with tag pooling)
# =============================================================================


async def async_get_tags(
    conv_id: str,
    raw_transcript: str,
    base_report: dict,
    tag_pool: set,
    max_retries: int | None = None,
) -> dict | None:
    """Call tagger agent to get tags with pooling for consistency.

    Uses structured output for guaranteed schema compliance.
    This agent runs SEQUENTIALLY to maintain consistent tag vocabulary.

    Args:
        conv_id: Conversation UUID.
        raw_transcript: Transcript text content.
        base_report: Base issue report from primary agent.
        tag_pool: Existing tags to prefer for consistent tagging.
        max_retries: Max retries for API/rate limit errors.

    Returns:
        Tags dict on success, None if all retries fail.
    """
    if max_retries is None:
        max_retries = config["llm"]["max_retries"]

    available_tags = (
        ", ".join(sorted(tag_pool))
        if tag_pool
        else "None yet - create appropriate tags"
    )

    for attempt in range(max_retries):
        try:
            # Structured output guarantees valid TaggerReport schema
            result: TaggerReport = await tagger_extractor.ainvoke(
                {
                    "transcript": raw_transcript,
                    "issue_report": json.dumps(base_report, indent=2),
                    "available_tags": available_tags,
                }
            )
            return result.model_dump()  # type: ignore[no-any-return]
        except Exception as e:
            print(
                f"  {conv_id} (tagger): Retry {attempt + 1}/{max_retries} - Error: {e}"
            )

    print(f"  {conv_id} (tagger): Failed after {max_retries} retries")
    return None


# =============================================================================
# Parallel Processing (Primary + Enrichment)
# =============================================================================


async def process_all_parallel(
    items: list[tuple[str, str]],
    issue_reports: dict,
    output_path: str,
    max_concurrent: int,
    max_retries: int,
    batch_size: int,
    use_enrichment: bool = True,
) -> dict:
    """Process items in batches with limited concurrency (parallel mode).

    Processes primary agent and optionally enrichment agent in parallel.
    Saves incrementally after each batch.

    Args:
        items: List of (conv_id, raw_transcript) tuples to process.
        issue_reports: Existing reports dict (modified in place).
        output_path: Path to save incremental results.
        max_concurrent: Maximum concurrent API calls.
        max_retries: Max retries per item.
        batch_size: Number of items per batch.
        use_enrichment: Whether to include enrichment agent calls.

    Returns:
        Updated issue_reports dict.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(items)

    async def process_one(conv_id: str, raw_transcript: str):
        async with semaphore:
            # Step 1: Get base report
            conv_id_result, base_report = await async_get_base_report(
                conv_id, raw_transcript, max_retries=max_retries
            )

            if base_report is None:
                return (conv_id, None)

            # Step 2: Get enrichment fields if enabled
            if use_enrichment:
                enrichment_fields = await async_get_enrichment_fields(
                    conv_id, raw_transcript, base_report, max_retries=max_retries
                )
                if enrichment_fields:
                    base_report.update(enrichment_fields)
                    base_report["_enrichment_used"] = True
                else:
                    base_report["_enrichment_used"] = False
            else:
                base_report["_enrichment_used"] = False

            # Tags not done in parallel mode
            base_report["_tagger_used"] = False

            return (conv_id, base_report)

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}: Processing {len(batch)} items...")

        tasks = [process_one(conv_id, transcript) for conv_id, transcript in batch]
        results = await asyncio.gather(*tasks)

        batch_success = 0
        for conv_id, issue_report in results:
            if issue_report:
                issue_reports[conv_id] = issue_report
                batch_success += 1

        print(
            f"    Completed: {batch_success}/{len(batch)} successful, total: {len(issue_reports)}/{total}"
        )
        save_issue_reports(issue_reports, output_path)

        # Small delay between batches to avoid connection pressure
        if i + batch_size < len(items):
            await asyncio.sleep(1.0)

    return issue_reports


# =============================================================================
# Sequential Processing (Tagger with tag pooling)
# =============================================================================


async def process_tags_sequential(
    issue_reports: dict,
    clean_data: dict,
    output_path: str,
    max_retries: int,
) -> dict:
    """Process tags sequentially with dynamic tag pooling.

    Args:
        issue_reports: Existing reports dict (modified in place).
        clean_data: Clean conversation data with transcript paths.
        output_path: Path to save incremental results.
        max_retries: Max retries per item.

    Returns:
        Updated issue_reports dict.
    """
    # Build initial tag_pool from existing reports
    tag_pool: set[str] = set()
    for report in issue_reports.values():
        tag_pool.update(report.get("tags", []))

    # Find reports that need tagging
    pending_items = []
    for conv_id, report in issue_reports.items():
        if not report.get("_tagger_used", False):
            conversation = clean_data.get(conv_id)
            if conversation:
                transcript_path = conversation["metadata"]["transcript_path"]
                raw_transcript = read_transcript(transcript_path)
                pending_items.append((conv_id, raw_transcript, report))

    if not pending_items:
        print("  No reports need tagging")
        return issue_reports

    total = len(pending_items)
    processed = 0
    failed = 0

    print(f"  Tagging {total} reports sequentially (with tag pooling)...")

    for conv_id, raw_transcript, base_report in pending_items:
        tags_result = await async_get_tags(
            conv_id, raw_transcript, base_report, tag_pool, max_retries
        )
        # Small delay between requests
        await asyncio.sleep(0.5)

        if tags_result:
            # Update tag_pool with new tags
            new_tags = tags_result.get("tags", [])
            tag_pool.update(new_tags)
            # Merge tags into report
            issue_reports[conv_id]["tags"] = new_tags
            issue_reports[conv_id]["_tagger_used"] = True
            processed += 1
        else:
            failed += 1

        # Progress output
        total_done = processed + failed
        status = "OK" if tags_result else "FAILED"
        print(f"    [{total_done}/{total}] {conv_id[:8]}... {status}")

        # Save after EVERY item for resumability
        save_issue_reports(issue_reports, output_path)

    print(f"    Completed: {processed}/{total} successful")

    return issue_reports


# =============================================================================
# Main Extraction Function
# =============================================================================


def extract_all_issues(clean_data: dict, output_path: str) -> dict:
    """Extract issue reports for all conversations.

    Processing flow:
    1. Primary agent extracts base fields (parallel)
    2. Enrichment agent adds sentiment/resolution/steps if enabled (parallel)
    3. Tagger agent adds tags if enabled (sequential with pooling)

    Args:
        clean_data: Dict of clean conversations with transcript paths.
        output_path: Path to save issue reports JSON.

    Returns:
        Dict of issue reports keyed by conversation_id.
    """
    max_concurrent = config["llm"].get("max_concurrency", 10)
    max_retries = config["llm"]["max_retries"]
    use_enrichment = config["llm"].get("use_enrichment_agent", True)
    use_tagger = config["llm"].get("use_tagger_agent", False)
    recreate = config.get("recreate", False)

    # Load existing results if output file exists (for resumption)
    if recreate:
        issue_reports = {}
        print("  Starting fresh (--recreate flag)")
    elif os.path.exists(output_path):
        issue_reports = read_json_file(output_path)
        print(f"  Resuming: Found {len(issue_reports)} existing reports")
    else:
        issue_reports = {}

    # Filter out already-processed conversations
    pending_items = []
    for conv_id, conversation in clean_data.items():
        existing = issue_reports.get(conv_id)
        needs_processing = False

        if existing is None:
            needs_processing = True
        elif use_enrichment and not existing.get("_enrichment_used", False):
            # Enrichment enabled but report was made without it
            needs_processing = True

        if needs_processing:
            transcript_path = conversation["metadata"]["transcript_path"]
            raw_transcript = read_transcript(transcript_path)
            pending_items.append((conv_id, raw_transcript))

    total = len(clean_data)
    skipped = total - len(pending_items)

    if skipped > 0:
        print(f"  Skipped {skipped} already-processed conversations")

    # Phase 1: Primary + Enrichment (PARALLEL)
    if pending_items:
        mode_str = "with enrichment" if use_enrichment else "base only"
        print(
            f"  Processing {len(pending_items)} conversations ({mode_str}, concurrency: {max_concurrent})"
        )
        batch_size = max_concurrent * 2
        issue_reports = asyncio.run(
            process_all_parallel(
                pending_items,
                issue_reports,
                output_path,
                max_concurrent,
                max_retries,
                batch_size,
                use_enrichment=use_enrichment,
            )
        )
    else:
        print("  No new conversations to process")

    # Phase 2: Tagger (SEQUENTIAL, only if enabled)
    if use_tagger:
        issue_reports = asyncio.run(
            process_tags_sequential(issue_reports, clean_data, output_path, max_retries)
        )

    return issue_reports


def run() -> dict:
    """Run the extract_issues pipeline step.

    Loads clean conversations and extracts structured issue reports using LLM.
    Supports resume from existing results.

    Returns:
        Dict of issue reports keyed by conversation_id.
    """
    clean_data_path = config["paths"]["clean_conversations"]
    issue_reports_path = config["paths"]["issue_reports"]

    # Load clean conversations
    clean_data = read_json_file(clean_data_path)
    print(f"  Loaded {len(clean_data)} conversations")

    # Extract issue reports
    issue_reports = extract_all_issues(clean_data, issue_reports_path)
    print(f"  Extracted {len(issue_reports)} issue reports")

    return issue_reports


# =============================================================================
# Enrich Only Mode (parallel)
# =============================================================================


async def enrich_reports_parallel(
    items: list[tuple[str, str, dict]],
    issue_reports: dict,
    output_path: str,
    max_concurrent: int,
    max_retries: int,
) -> dict:
    """Enrich existing reports with enrichment agent fields (parallel).

    Args:
        items: List of (conv_id, raw_transcript, base_report) tuples.
        issue_reports: Existing reports dict (modified in place).
        output_path: Path to save incremental results.
        max_concurrent: Maximum concurrent API calls.
        max_retries: Max retries per item.

    Returns:
        Updated issue_reports dict.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    batch_size = max_concurrent * 2

    async def process_one(conv_id: str, raw_transcript: str, base_report: dict):
        async with semaphore:
            enrichment_fields = await async_get_enrichment_fields(
                conv_id, raw_transcript, base_report, max_retries
            )
            return (conv_id, enrichment_fields)

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}: Enriching {len(batch)} reports...")

        tasks = [
            process_one(conv_id, transcript, report)
            for conv_id, transcript, report in batch
        ]
        results = await asyncio.gather(*tasks)

        batch_success = 0
        for conv_id, enrichment_fields in results:
            if enrichment_fields:
                issue_reports[conv_id].update(enrichment_fields)
                issue_reports[conv_id]["_enrichment_used"] = True
                batch_success += 1

        print(f"    Completed: {batch_success}/{len(batch)} successful")
        save_issue_reports(issue_reports, output_path)

        if i + batch_size < len(items):
            await asyncio.sleep(1.0)

    return issue_reports


def enrich_only() -> dict:
    """Enrich existing issue reports with enrichment agent fields only.

    Does NOT re-run the primary agent. Only adds:
    - resolution_status
    - user_sentiment
    - steps_taken_by_agent

    Runs in PARALLEL for fast processing.

    Returns:
        Dict of enriched issue reports.
    """
    clean_data_path = config["paths"]["clean_conversations"]
    issue_reports_path = config["paths"]["issue_reports"]
    max_retries = config["llm"]["max_retries"]
    max_concurrent = config["llm"].get("max_concurrency", 10)

    # Load existing issue reports
    if not os.path.exists(issue_reports_path):
        print(f"  ERROR: No existing issue reports found at {issue_reports_path}")
        print("  Run the full pipeline first to extract issues.")
        return {}

    issue_reports = read_json_file(issue_reports_path)
    print(f"  Loaded {len(issue_reports)} existing issue reports")

    # Load clean conversations (for transcript paths)
    clean_data = read_json_file(clean_data_path)

    # Find reports that need enrichment
    pending_items = []
    for conv_id, report in issue_reports.items():
        if not report.get("_enrichment_used", False):
            conversation = clean_data.get(conv_id)
            if conversation:
                transcript_path = conversation["metadata"]["transcript_path"]
                raw_transcript = read_transcript(transcript_path)
                pending_items.append((conv_id, raw_transcript, report))
            else:
                print(f"  Warning: No conversation data for {conv_id}, skipping")

    already_enriched = len(issue_reports) - len(pending_items)
    if already_enriched > 0:
        print(f"  Skipped {already_enriched} already-enriched reports")

    if not pending_items:
        print("  No reports need enrichment")
        return issue_reports

    print(
        f"  Enriching {len(pending_items)} reports (parallel, concurrency: {max_concurrent})..."
    )

    issue_reports = asyncio.run(
        enrich_reports_parallel(
            pending_items,
            issue_reports,
            issue_reports_path,
            max_concurrent,
            max_retries,
        )
    )

    print(f"  Enrichment complete: {len(issue_reports)} total reports")

    return issue_reports


# =============================================================================
# Tag Only Mode (sequential)
# =============================================================================


def tag_only() -> dict:
    """Add tags to existing issue reports using tagger agent.

    Does NOT re-run primary or enrichment agents. Only adds:
    - tags (with tag pooling for consistency)

    Runs SEQUENTIALLY to maintain consistent tag vocabulary.

    Returns:
        Dict of tagged issue reports.
    """
    clean_data_path = config["paths"]["clean_conversations"]
    issue_reports_path = config["paths"]["issue_reports"]
    max_retries = config["llm"]["max_retries"]

    # Load existing issue reports
    if not os.path.exists(issue_reports_path):
        print(f"  ERROR: No existing issue reports found at {issue_reports_path}")
        print("  Run the full pipeline first to extract issues.")
        return {}

    issue_reports = read_json_file(issue_reports_path)
    print(f"  Loaded {len(issue_reports)} existing issue reports")

    # Load clean conversations (for transcript paths)
    clean_data = read_json_file(clean_data_path)

    # Process tags sequentially
    issue_reports = asyncio.run(
        process_tags_sequential(
            issue_reports, clean_data, issue_reports_path, max_retries
        )
    )

    print(f"  Tagging complete: {len(issue_reports)} total reports")

    return issue_reports


if __name__ == "__main__":
    issue_reports = run()

    # Print sample for debugging
    if issue_reports:
        sample_id = list(issue_reports.keys())[0]
        print("\n### Sample issue report ###\n")
        print(json.dumps(issue_reports[sample_id], indent=2, ensure_ascii=False))
