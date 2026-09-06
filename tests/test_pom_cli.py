"""Tests for pom unified CLI (INV-6, INV-1)."""

import argparse
from unittest.mock import patch
from pom import cmd_run


def test_pom_refuses_missing_consent(capsys):
    args = argparse.Namespace(
        photo="fake.jpg",
        consent_confirmed=False,
        adaface_root=None,
        output="receipt.json",
    )
    code = cmd_run(args)
    assert code == 1
    captured = capsys.readouterr()
    assert "--consent-confirmed is REQUIRED" in captured.err


def test_pom_refuses_directory_input(tmp_path, capsys):
    test_dir = str(tmp_path / "photos")
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "photos").mkdir(exist_ok=True)

    args = argparse.Namespace(
        photo=test_dir,
        consent_confirmed=True,
        adaface_root=None,
        output="receipt.json",
    )
    code = cmd_run(args)
    assert code == 1
    captured = capsys.readouterr()
    assert "Refusing directory input" in captured.err
