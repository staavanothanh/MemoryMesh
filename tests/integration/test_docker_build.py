"""Test that the Dockerfile builds successfully.

This is a shell-level integration test. It is skipped on Windows
because Docker builds are primarily tested on Linux/CI.
"""

import os
import sys
import subprocess
import pytest

# Skip this test on Windows — Docker builds are tested on Linux/CI
skip_windows = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="Docker build test skipped on Windows (requires Docker daemon on Linux/CI)",
)


@skip_windows
class TestDockerBuild:
    """Docker build verification — requires Docker daemon."""

    def test_docker_build_succeeds(self):
        """Run 'docker build -f docker/Dockerfile -t memorymesh-test .' — exit code 0."""
        # Change to project root directory
        project_root = os.path.join(os.path.dirname(__file__), "..", "..")

        result = subprocess.run(
            ["docker", "build", "-f", "docker/Dockerfile", "-t", "memorymesh-test", "."],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for Docker build
        )

        if result.returncode != 0:
            # Print output for debugging
            print(f"STDOUT: {result.stdout[:2000]}")
            print(f"STDERR: {result.stderr[:2000]}")

        assert result.returncode == 0, (
            f"Docker build failed with exit code {result.returncode}.\n"
            f"STDERR: {result.stderr[:500]}"
        )

    @pytest.mark.skip(reason="Docker build may not be available in all environments")
    def test_docker_build_daemon_not_available(self):
        """Graceful skip if Docker daemon is not running."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                pytest.skip("Docker daemon is not running")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("Docker command not available")
