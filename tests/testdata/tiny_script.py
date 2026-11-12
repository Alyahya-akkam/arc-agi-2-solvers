"""Stand-in script of the simulated script notebook (tests/test_notebook_local.py, task P0.5d).

It behaves like a BRIEF A8 script: ``--config <yaml>`` names its config, whose ``out_dir`` is its
run-log folder relative to the working directory. It imports ``arc26`` (the wheel the notebook
installed) and ``yaml``, writes ``<out_dir>/result.json`` with what it saw, and then, on request,
starts a child that would outlive it (``--spawn``), sleeps (``--sleep``), or exits with a code
(``--exit``).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

import arc26

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--note")
parser.add_argument("--spawn", action="store_true")
parser.add_argument("--sleep", type=float, default=0.0)
parser.add_argument("--exit", type=int, default=0)
args = parser.parse_args()
config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
out = Path(config["out_dir"])
out.mkdir(parents=True, exist_ok=True)
result = {
    "argv": sys.argv[1:],
    "config": config,
    "arc26": arc26.__file__,
    "cwd": os.getcwd(),
    "unbuffered": os.environ.get("PYTHONUNBUFFERED"),
    "child": None,
}
if args.spawn:
    # A grandchild of the notebook in the script's process group; the hard stop must end it too.
    child = subprocess.Popen([sys.executable, "-S", "-c", "import time; time.sleep(120)"])
    result["child"] = child.pid
(out / "result.json").write_text(json.dumps(result), encoding="utf-8")
print("tiny script: wrote result.json", flush=True)
if args.sleep:
    print(f"tiny script: sleeping {args.sleep} s", flush=True)
    time.sleep(args.sleep)
sys.exit(args.exit)
