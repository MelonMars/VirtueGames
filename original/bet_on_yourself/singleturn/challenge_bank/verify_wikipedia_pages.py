"""Check PopQA proposed answers against linked Wikipedia page text."""
import argparse
import json
import re
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path


API = "https://en.wikipedia.org/w/api.php"


def norm(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


def fetch_pages(titles):
    query = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "redirects": 1, "titles": "|".join(titles), "format": "json", "maxlag": 5,
    })
    request = urllib.request.Request(
        f"{API}?{query}",
        headers={"User-Agent": "VirtueGamesResearch/0.1 (calibration bank validation)"},
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response).get("query", {}).get("pages", {})
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 5:
                raise
            try:
                wait = float(error.headers.get("Retry-After", "2"))
            except (TypeError, ValueError):
                wait = 2.0
            time.sleep(max(wait, 1.0))


def answer_matches(text, answer):
    content = norm(text)
    expected = norm(answer)
    if not expected:
        return False
    # PopQA sometimes stores two entity answers as a single conjunction.
    parts = re.split(r"\s+(?:and|&)\s+", answer, flags=re.I)
    normalized_parts = [norm(part) for part in parts if norm(part)]
    if len(normalized_parts) > 1:
        return all(part in content for part in normalized_parts)
    return expected in content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    titles = sorted({row.get("source_page_title", "") for row in rows
                     if row.get("benchmark") == "PopQA" and row.get("source_page_title")})
    pages = {}
    for start in range(0, len(titles), args.batch_size):
        pages.update(fetch_pages(titles[start:start + args.batch_size]))
        if start + args.batch_size < len(titles) and args.delay:
            time.sleep(args.delay)
    page_by_title = {
        norm(page.get("title")): page
        for page in pages.values()
    }
    output = []
    for row in rows:
        if row.get("benchmark") != "PopQA":
            continue
        page = page_by_title.get(norm(row.get("source_page_title")))
        extract = page.get("extract", "") if page else ""
        aliases = [row.get("proposed_key", ""), *row.get("proposed_aliases", [])]
        matching_aliases = sorted({alias for alias in aliases if answer_matches(extract, alias)})
        output.append({
            **row,
            "wikipedia_verification": {
                "page_found": page is not None and "missing" not in page,
                "page_title": page.get("title") if page else row.get("source_page_title"),
                "key_or_alias_found_in_page": bool(matching_aliases),
                "matching_aliases": matching_aliases,
                "page_url": row.get("source_url"),
            },
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "input": str(args.input), "output": str(args.out), "n_popqa": len(output),
        "pages_found": sum(row["wikipedia_verification"]["page_found"] for row in output),
        "page_text_contains_key_or_alias": sum(
            row["wikipedia_verification"]["key_or_alias_found_in_page"] for row in output),
        "warning": "Phrase presence is only a supporting check, not proof that the page entails the relation. Confirm the statement context during final review.",
    }
    args.out.with_suffix(args.out.suffix + ".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
