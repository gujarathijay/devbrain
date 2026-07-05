"""
Fetch GitHub Issues from a repository.

This module calls the GitHub Issues API to:
1. Fetch the most recent issues (configurable, default 500)
2. For each issue, format the title + body + comments into one document
3. Save them locally to data/raw/issues/ for processing

Why issues matter for DevBrain:
- Docs tell you how things SHOULD work
- Issues tell you what happens when things DON'T work
- Real error messages, stack traces, workarounds — gold for a support tool
- Often contain solutions the docs haven't been updated with yet

Pagination explained:
GitHub returns max 100 items per request. If we want 500 issues,
we need 5 requests (page=1, page=2, ... page=5). Each response
includes a Link header telling us if there's a next page.
"""

import json
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from src.config import (
    GITHUB_TOKEN,
    MAX_ISSUES,
    RAW_ISSUES_DIR,
    TARGET_REPO_NAME,
    TARGET_REPO_OWNER,
)

console = Console()

BASE_URL = "https://api.github.com"


def _get_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def fetch_issues(
    owner: str = TARGET_REPO_OWNER,
    repo: str = TARGET_REPO_NAME,
    max_issues: int = MAX_ISSUES,
) -> list[dict]:
    """
    Fetch recent issues from a GitHub repository.

    Parameters:
    - max_issues: how many issues to fetch (default 500)

    How pagination works:
        GitHub caps each response at 100 items. To get 500 issues:
        - Request page 1 → items 1-100
        - Request page 2 → items 101-200
        - ... until we have enough or run out of pages

    We fetch issues sorted by "updated" (most recently active first)
    because recently updated issues are more likely to be relevant
    and contain current information.

    We also filter to state="all" to get both open AND closed issues.
    Closed issues are valuable — they often contain the solution!

    Returns a list of raw issue dicts from the GitHub API.
    """
    url = f"{BASE_URL}/repos/{owner}/{repo}/issues"
    all_issues = []
    page = 1
    per_page = 100  # GitHub's maximum per request

    with httpx.Client(headers=_get_headers(), timeout=30.0) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching issues...", total=max_issues)

            while len(all_issues) < max_issues:
                params = {
                    "state": "all",        # open + closed
                    "sort": "updated",     # most recently active first
                    "direction": "desc",   # newest updates first
                    "per_page": per_page,
                    "page": page,
                }

                response = client.get(url, params=params)
                response.raise_for_status()
                batch = response.json()

                # If GitHub returns empty list, we've exhausted all issues
                if not batch:
                    break

                # GitHub's Issues API also returns pull requests.
                # We filter them out — PRs have a "pull_request" key.
                issues_only = [
                    item for item in batch
                    if "pull_request" not in item
                ]

                all_issues.extend(issues_only)
                progress.update(task, completed=min(len(all_issues), max_issues))

                # Rate limiting
                time.sleep(0.2)
                page += 1

    # Trim to exact count requested
    all_issues = all_issues[:max_issues]
    console.print(f"[green]Fetched {len(all_issues)} issues[/green]")
    return all_issues


def format_issue_as_document(issue: dict) -> dict:
    """
    Convert a raw GitHub issue into our standard document format.

    An issue has:
    - title: "Bug: 422 error with nested model"
    - body: detailed description with code examples
    - labels: ["bug", "question", "answered"]
    - state: "open" or "closed"
    - number: 12345
    - html_url: link to the issue on GitHub

    We format it as a single text document:
    ┌─────────────────────────────────────────────┐
    │ Issue #12345: Bug: 422 error with nested... │
    │ Status: closed | Labels: bug, answered      │
    │                                             │
    │ <original issue body>                       │
    └─────────────────────────────────────────────┘

    Why this format?
    - The title in the content helps embedding models understand
      what the issue is about
    - Labels and status provide context ("closed" + "answered"
      means there's likely a solution in here)
    - The structured header makes retrieved chunks more readable
    """
    number = issue["number"]
    title = issue["title"]
    body = issue.get("body") or ""  # some issues have null body
    state = issue["state"]
    labels = [label["name"] for label in issue.get("labels", [])]
    labels_str = ", ".join(labels) if labels else "none"
    url = issue["html_url"]
    created = issue["created_at"][:10]   # "2024-03-15T..." → "2024-03-15"
    updated = issue["updated_at"][:10]

    # Build the document content
    content = (
        f"Issue #{number}: {title}\n"
        f"Status: {state} | Labels: {labels_str} | "
        f"Created: {created} | Updated: {updated}\n\n"
        f"{body}"
    )

    metadata = {
        "source": "issues",
        "issue_number": number,
        "title": title,
        "url": url,
        "state": state,
        "labels": labels,
        "created_at": created,
        "updated_at": updated,
    }

    return {"content": content, "metadata": metadata}


def process_issues(raw_issues: list[dict]) -> list[dict]:
    """
    Convert raw GitHub API responses into our document format and save to disk.

    Steps:
    1. Format each issue into a document
    2. Skip issues with very short content (< 50 chars — usually spam/empty)
    3. Save each issue as a JSON file for debugging
    4. Save a manifest summarizing everything downloaded

    Returns a list of document dicts ready for chunking.
    """
    documents = []

    for issue in raw_issues:
        doc = format_issue_as_document(issue)

        # Skip trivially short issues
        if len(doc["content"].strip()) < 50:
            continue

        documents.append(doc)

        # Save individual issue as JSON (for debugging/inspection)
        issue_path = RAW_ISSUES_DIR / f"issue_{doc['metadata']['issue_number']}.json"
        issue_path.write_text(
            json.dumps(doc, indent=2, default=str),
            encoding="utf-8",
        )

    console.print(f"[green]✅ Processed {len(documents)} issues[/green]")

    # Save manifest
    manifest_path = RAW_ISSUES_DIR / "_manifest.json"
    manifest = [
        {
            "number": d["metadata"]["issue_number"],
            "title": d["metadata"]["title"],
            "state": d["metadata"]["state"],
        }
        for d in documents
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return documents


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run directly to test: uv run python src/ingestion/github_issues_loader.py

    Fetches a small batch (20 issues) and shows a sample.
    """
    console.print("\n[bold]🐛 Fetching FastAPI issues...[/bold]\n")

    # Fetch only 20 for testing (override the default 500)
    raw = fetch_issues(max_issues=20)
    docs = process_issues(raw)

    console.print(f"\n[bold green]Done! {len(docs)} issues processed.[/bold green]")
    console.print(f"Saved to: {RAW_ISSUES_DIR}")

    # Show a sample
    if docs:
        sample = docs[0]
        console.print(f"\n[bold]Sample issue:[/bold]")
        console.print(f"  #{sample['metadata']['issue_number']}: {sample['metadata']['title']}")
        console.print(f"  State: {sample['metadata']['state']}")
        console.print(f"  Labels: {sample['metadata']['labels']}")
        console.print(f"  Content length: {len(sample['content'])} chars")
        console.print(f"  First 300 chars:\n  {sample['content'][:300]}...")