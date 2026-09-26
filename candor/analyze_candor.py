"""Compatibility entry point for the family-safe response direction extractor."""
import runpy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from response_direction_analysis import analyze, pool, split_groups, resolve_device, key, read_rows

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("analyze_response_directions.py")), run_name="__main__")
