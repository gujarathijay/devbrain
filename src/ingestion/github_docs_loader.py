"""
Fetch markdown documentation files from a GitHub repository.

This module calls the GitHub Contents API to:
1. List all .md files recursively inside the docs directory
2. Download each file's raw content
3. Save them locally to data/raw/docs/ for processing

Why save locally first (instead of chunking directly)?
- Separates "fetching" from "processing" — if chunking logic changes,
  you don't re-download 200+ files from GitHub
- Lets you inspect raw files manually in VS Code
- Makes debugging easier — you can see exactly what was downloaded
"""

import json
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.config import (
    GITHUB_TOKEN,
    RAW_DOCS_DIR,
    TARGET_DOCS_PATH,
    TARGET_REPO_NAME,
    TARGET_REPO_OWNER,
)

console = Console()

# ── GitHub API setup ──────────────────────────────────────────────
BASE_URL = "https://api.github.com"

def _get_headers() -> dict[str, str]:
    """
    Build HTTP headers for GitHub API requests.

    The Accept header asks for v3 JSON responses.
    The Authorization header (if token exists) upgrades us from
    60 → 5,000 requests/hour.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


# ── Core functions ────────────────────────────────────────────────

def list_markdown_files(
    owner: str = TARGET_REPO_OWNER,
    repo: str = TARGET_REPO_NAME,
    path: str = TARGET_DOCS_PATH,
) -> list[dict]:
    """
    Recursively list all .md files in a GitHub repo directory.

    How it works:
    - GitHub's Contents API returns items in a directory
    - Each item is either a "file" or a "dir"
    - If it's a dir, we recurse into it
    - If it's a .md file, we collect it

    Returns a list of dicts, each with:
      {
        "path": "docs/en/docs/tutorial/first-steps.md",
        "download_url": "https://raw.githubusercontent.com/...",
        "name": "first-steps.md"
      }
    """
    url = f"{BASE_URL}/repos/{owner}/{repo}/contents/{path}"
    md_files = []

    with httpx.Client(headers=_get_headers(), timeout=30.0) as client:
        _recurse_directory(client, url, md_files)

    console.print(f"[green]Found {len(md_files)} markdown files[/green]")
    return md_files


def _recurse_directory(
    client: httpx.Client,
    url: str,
    collector: list[dict],
) -> None:
    """
    Recursively walk a GitHub directory, collecting .md files.

    Why recursive? Because FastAPI docs have nested folders:
      docs/en/docs/
      ├── tutorial/
      │   ├── first-steps.md
      │   ├── path-params.md
      │   └── security/
      │       └── oauth2.md
      ├── advanced/
      │   └── middleware.md
      └── index.md

    We need to go into every subfolder to find all files.
    """
    response = client.get(url)
    response.raise_for_status()     # crash loudly if GitHub returns an error
    items = response.json()

    # GitHub API rate limit: be polite, wait 0.1s between requests
    time.sleep(0.1)

    for item in items:
        if item["type"] == "dir":
            # It's a folder — go deeper
            _recurse_directory(client, item["url"], collector)
        elif item["type"] == "file" and item["name"].endswith(".md"):
            # It's a markdown file — collect it
            collector.append({
                "path": item["path"],
                "download_url": item["download_url"],
                "name": item["name"],
            })


def download_docs(md_files: list[dict]) -> list[dict]:
    """
    Download the raw content of each markdown file and save to disk.

    For each file, we:
    1. GET the raw content from download_url
    2. Save to data/raw/docs/<relative_path>.md
    3. Build a document dict with content + metadata

    The metadata we attach here will travel with this document through
    the entire pipeline — chunking, embedding, storage, retrieval.
    When the user gets an answer, we use this metadata to say
    "Source: FastAPI Docs — Tutorial > First Steps"

    Returns a list of document dicts:
      {
        "content": "# First Steps\n\nYou can create...",
        "metadata": {
            "source": "docs",
            "file_path": "docs/en/docs/tutorial/first-steps.md",
            "title": "first-steps",
            "url": "https://github.com/fastapi/fastapi/blob/master/docs/en/docs/tutorial/first-steps.md",
        }
      }
    """
    documents = []

    with httpx.Client(headers=_get_headers(), timeout=30.0) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading docs...", total=len(md_files))

            for file_info in md_files:
                # Download raw markdown content
                response = client.get(file_info["download_url"])
                response.raise_for_status()
                content = response.text

                # Skip empty files or very short files (< 50 chars)
                if len(content.strip()) < 50:
                    progress.advance(task)
                    continue

                # Save raw file to disk
                # Convert "docs/en/docs/tutorial/first-steps.md"
                # → save as "data/raw/docs/tutorial/first-steps.md"
                relative_path = file_info["path"].replace(TARGET_DOCS_PATH + "/", "")
                save_path = RAW_DOCS_DIR / relative_path
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_text(content, encoding="utf-8")

                # Build the GitHub URL for this file (for citations later)
                github_url = (
                    f"https://github.com/{TARGET_REPO_OWNER}/{TARGET_REPO_NAME}"
                    f"/blob/master/{file_info['path']}"
                )

                # Build document dict
                # "title" = filename without .md extension
                title = file_info["name"].replace(".md", "")

                documents.append({
                    "content": content,
                    "metadata": {
                        "source": "docs",
                        "file_path": file_info["path"],
                        "title": title,
                        "url": github_url,
                    },
                })

                # Rate limiting: be nice to GitHub
                time.sleep(0.05)
                progress.advance(task)

    console.print(f"[green]✅ Downloaded {len(documents)} doc files[/green]")

    # Save a manifest (summary of what we downloaded) for debugging
    manifest_path = RAW_DOCS_DIR / "_manifest.json"
    manifest = [
        {"title": d["metadata"]["title"], "path": d["metadata"]["file_path"]}
        for d in documents
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return documents


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    """
    Run directly to test: uv run python src/ingestion/github_docs_loader.py
    Downloads all docs and prints stats.
    """
    console.print("\n[bold]📄 Fetching FastAPI documentation...[/bold]\n")
    files = list_markdown_files()
    docs = download_docs(files)
    console.print(f"\n[bold green]Done! {len(docs)} documents downloaded.[/bold green]")
    console.print(f"Saved to: {RAW_DOCS_DIR}")

    # Show a sample
    if docs:
        sample = docs[0]
        console.print(f"\n[bold]Sample document:[/bold]")
        console.print(f"  Title: {sample['metadata']['title']}")
        console.print(f"  Path:  {sample['metadata']['file_path']}")
        console.print(f"  Size:  {len(sample['content'])} chars")
        console.print(f"  First 200 chars:\n  {sample['content'][:200]}...")