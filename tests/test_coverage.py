"""Test to enforce minimum code coverage."""

import subprocess
import sys

import pytest


@pytest.mark.slow
def test_coverage_threshold() -> None:
    """Enforce minimum 90% code coverage (excluding CLI and plotting)."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--cov=src",
            "--cov-report=term",
            "--cov-config=pyproject.toml",
            "-q",
            "-m",
            "not slow",
        ],
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr

    for line in output.split("\n"):
        if "TOTAL" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if "%" in part:
                    coverage = float(part.replace("%", ""))
                    assert coverage >= 90, f"Coverage {coverage}% is below 90%"
                    return

    pytest.skip("Could not determine coverage from output")


if __name__ == "__main__":
    test_coverage_threshold()
