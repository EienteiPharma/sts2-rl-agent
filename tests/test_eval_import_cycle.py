"""Regression: import sts2_env.eval / HOLD --help must not complete combat_jev cycle."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    return env


def test_import_sts2_env_eval_does_not_load_combat_jev():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, sts2_env.eval as e\n"
            "assert 'sts2_env.eval.combat_jev' not in sys.modules\n"
            "assert 'sts2_env.core.combat' not in sys.modules\n"
            "assert 'CombatJevTelemetry' in dir(e)\n"
            "print('eval_ok')\n",
        ],
        cwd=str(REPO),
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "eval_ok" in proc.stdout


def test_eval_getattr_combat_jev_after_lazy_load():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sts2_env.eval as e\n"
            "Tel = e.CombatJevTelemetry\n"
            "assert Tel().as_report()['jev_calls'] == 0\n"
            "print('lazy_ok')\n",
        ],
        cwd=str(REPO),
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "lazy_ok" in proc.stdout


def test_eval_combat_suite_help_succeeds_without_combat_jev():
    proc = subprocess.run(
        [sys.executable, "scripts/eval_combat_suite.py", "--help"],
        cwd=str(REPO),
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "--workers" in proc.stdout
    assert "--combat-policy" in proc.stdout
    assert "experimental" in proc.stdout
    assert "failed HOLD bypass" in proc.stdout


def test_eval_combat_suite_help_does_not_import_combat_jev_module():
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys\n"
            "sys.argv = ['eval_combat_suite.py', '--help']\n"
            "try:\n"
            "    runpy.run_path('scripts/eval_combat_suite.py', run_name='__main__')\n"
            "except SystemExit as e:\n"
            "    assert e.code in (0, None)\n"
            "assert 'sts2_env.eval.combat_jev' not in sys.modules\n"
            "print('help_ok')\n",
        ],
        cwd=str(REPO),
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "help_ok" in proc.stdout
