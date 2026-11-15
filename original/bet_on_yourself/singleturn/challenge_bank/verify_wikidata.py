"""Check PopQA subject-property-object claims against Wikidata's current API."""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


API = "https://www.wikidata.org/w/api.php"


def qid(value):
    match = re.search(r"(?:entity/)?(Q\d+)$", str(value or ""))
    return match.group(1) if match else None


def retrieve_claims(entity_ids):
    query = urllib.parse.urlencode({
        "action": "wbgetentities",
        "ids": "|".join(sorted(entity_ids)),
        "props": "claims",
        "format": "json",
        "languages": "en",
    })
    request = urllib.request.Request(
        f"{API}?{query}",
        headers={"User-Agent": "VirtueGamesResearch/0.1 (calibration bank validation)"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response).get("entities", {})


def claim_object_id(statement):
    snak = statement.get("mainsnak", {})
    if snak.get("snaktype") != "value":
        return None
    value = snak.get("datavalue", {}).get("value")
    return value.get("id") if isinstance(value, dict) else None


def reference_urls(statement):
    urls = set()
    for reference in statement.get("references", []):
        for snak in reference.get("snaks", {}).get("P854", []):
            value = snak.get("datavalue", {}).get("value")
            if isinstance(value, str) and value.startswith(("http://", "https://")):
                urls.add(value)
    return sorted(urls)


def reference_items(statement):
    ids = set()
    for reference in statement.get("references", []):
        for snak in reference.get("snaks", {}).get("P248", []):
            value = snak.get("datavalue", {}).get("value")
            if isinstance(value, dict) and value.get("id"):
                ids.add(value["id"])
    return sorted(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.15)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    popqa = [row for row in rows if row.get("benchmark") == "PopQA"]
    ids = sorted({qid(row.get("source_entity_uri")) for row in popqa} - {None})
    entities = {}
    for start in range(0, len(ids), args.batch_size):
        batch = ids[start:start + args.batch_size]
        entities.update(retrieve_claims(batch))
        if start + args.batch_size < len(ids) and args.delay:
            time.sleep(args.delay)

    results = []
    for row in popqa:
        subject = qid(row.get("source_entity_uri"))
        obj = qid(row.get("source_object_uri"))
        prop = row.get("wikidata_property_id")
        entity = entities.get(subject, {})
        statements = entity.get("claims", {}).get(prop, [])
        live = [statement for statement in statements if statement.get("rank") != "deprecated"]
        matching = [statement for statement in live if claim_object_id(statement) == obj]
        values = sorted({claim_object_id(statement) for statement in live if claim_object_id(statement)})
        refs = sorted({url for statement in matching for url in reference_urls(statement)})
        ref_items = sorted({item for statement in matching for item in reference_items(statement)})
        ref_count = max((len(statement.get("references", [])) for statement in matching), default=0)
        status = ("exact_match_single_value" if matching and len(values) == 1 else
                  "exact_match_multiple_values" if matching else
                  "claim_value_missing" if entity else "subject_missing")
        results.append({
            **row,
            "wikidata_verification": {
                "status": status,
                "subject_id": subject,
                "property_id": prop,
                "expected_object_id": obj,
                "live_object_ids": values,
                "matching_statement_count": len(matching),
                "reference_count": ref_count,
                "reference_urls": refs,
                "reference_item_ids": ref_items,
                "api_url": f"https://www.wikidata.org/wiki/{subject}",
            },
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as stream:
        for row in results:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = {}
    for row in results:
        status = row["wikidata_verification"]["status"]
        counts[status] = counts.get(status, 0) + 1
    report = {
        "input": str(args.input),
        "output": str(args.out),
        "n_popqa": len(results),
        "status_counts": counts,
        "warning": "This checks the current Wikidata triple, not independent truth. Each retained fact still needs a reliable source citation and question review.",
    }
    args.out.with_suffix(args.out.suffix + ".report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
