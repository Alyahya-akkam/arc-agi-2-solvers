"""Stand-in for scripts/fn_kvq/bench.py in run P5.5-01's CPU smoke (phase K's path, no kernels).

Usage::

    python tests/testdata/fn_kvab/bench_stub.py --out kvq_bench.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    """Write a small JSON report where the real benchmark writes its own."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps({"stand_in": True, "ok": True}) + "\n", encoding="utf-8")
    print("bench stand-in: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
