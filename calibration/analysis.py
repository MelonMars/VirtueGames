"""Count parsed and unparsed calibration responses by difficulty."""
import sys
from pathlib import Path

# Direct script execution needs the repository root for shared helpers.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from collections import Counter
from pathlib import Path
from runs import read_jsonl


def analyze(run_dir):
    rows = read_jsonl(Path(run_dir) / "results.jsonl")
    counts = {}
    for ok in (True, False):
        label = "parsed" if ok else "UNPARSED"
        counts[label] = dict(Counter(r.get("difficulty") for r in rows if r["parse_ok"] == ok))
        print(label, counts[label])
    return counts


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return analyze(parser.parse_args(argv).run_dir)


if __name__ == "__main__":
    main()
