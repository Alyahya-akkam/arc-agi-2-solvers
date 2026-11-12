"""A stand-in for the wheels notebook's ``install_wheelhouse.py`` in the hybrid simulation (P6.5).

``tests/test_hybrid_notebook.py`` mounts it as ``install_wheelhouse.py`` in the simulated
wheelhouse output (task P6.3). It takes the installer's arguments (``--wheelhouse``, ``--venv``,
``--report``, ``--log``, ``--imports``), records the environment it got (the notebook must cut
``PYTHONPATH`` and ``PIP_*``), and behaves as ``ARC26_FAKE_INSTALL`` says: ``ok`` (default) makes
the venv with ``bin/python`` and ``bin/vllm`` and reports ok; ``fail`` reports a failed install
and exits 1; ``no-engine`` reports ok but leaves no ``bin/vllm``; ``hang`` sleeps until the
notebook's timeout stops it.

The ``bin/vllm`` it writes starts task P6.4's fake engine (``arc26.agent.server fake``) on the
port of its ``vllm serve`` command line when ``ARC26_FAKE_REPLIES`` names a replies file, as the
real-agent simulation needs; otherwise it exits 1, an engine that cannot boot.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

VLLM = """#!{python} -S
import os
import sys

argv = sys.argv[1:]
replies = os.environ.get("ARC26_FAKE_REPLIES")
if not replies:
    print("fake vllm: no engine on this machine", file=sys.stderr)
    sys.exit(1)
sys.path[:0] = [os.environ["ARC26_SHIM_TARGET"], *os.environ["ARC26_FAKE_PATHS"].split(os.pathsep)]
from arc26.agent.server import main

port = argv[argv.index("--port") + 1]
name = argv[argv.index("--served-model-name") + 1]
sys.exit(main(["fake", "--port", port, "--replies", replies, "--model", name]))
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--venv", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--log")
    parser.add_argument("--imports", default="")
    args, _ = parser.parse_known_args()
    mode = os.environ.get("ARC26_FAKE_INSTALL", "ok")
    print(f"fake installer: {mode}", flush=True)
    if mode == "hang":
        time.sleep(600)
    venv = Path(args.venv)
    (venv / "bin").mkdir(parents=True, exist_ok=True)
    python = venv / "bin" / "python"
    if not python.exists():
        python.symlink_to(sys.executable)
    if mode != "no-engine":
        engine = venv / "bin" / "vllm"
        engine.write_text(VLLM.format(python=sys.executable), encoding="utf-8")
        engine.chmod(0o755)
    ok = mode != "fail"
    report = {
        "ok": ok,
        "problems": [] if ok else ["install exited 1"],
        "imports": {name: {"ok": ok} for name in args.imports.split(",") if name},
        "seconds": 0.1,
        "wheelhouse": args.wheelhouse,
        "env": {
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "pip": sorted(name for name in os.environ if name.startswith("PIP_")),
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
        },
    }
    Path(args.report).write_text(json.dumps(report), encoding="utf-8")
    if args.log:
        Path(args.log).write_text(f"fake install {mode}\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
