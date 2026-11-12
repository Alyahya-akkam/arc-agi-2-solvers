"""Stand-in for tests/test_fn_kvq.py in run P5.5-01's CPU smoke: one gpu-marked check that runs.

Named outside pytest's test_*.py pattern, so the repository's own run never collects it; the
smoke's phase K names it on pytest's command line with ``-m gpu``, as the run names the real file.
"""

import pytest


@pytest.mark.gpu
def test_stand_in_runs_under_the_gpu_marker():
    assert 1 + 1 == 2
