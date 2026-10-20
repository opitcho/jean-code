"""Tests for the shells being usable again after `close()`: a subagent's agent is closed after every turn, and a
resumed one runs commands on the same shell, in a new container. The Docker test starts a container, so it's skipped
without Docker.

Each test names the bug it would catch.
"""

import io
import shutil
import subprocess
import uuid

import pytest

from jean_code.tools import shell as shell_module
from jean_code.tools.shell import DockerShell, LocalShell


def docker_ready() -> bool:
    """Whether the daemon runs and the sandbox image exists."""
    docker = shutil.which("docker")
    if not docker:
        return False
    image = subprocess.run([docker, "image", "inspect", "jean-code-sandbox"], capture_output=True, check=False)
    return image.returncode == 0


def test_a_closed_shell_runs_background_jobs_again(tmp_path):
    # Catches a background job after close failing on the deleted jobs folder, or numbering on from the old jobs.
    shell = LocalShell(workdir=tmp_path)
    assert "started job 1" in shell.bash("echo first", background=True)
    assert "exited with code 0" in shell.job("1", wait=5)
    shell.close()
    assert not shell.jobs_dir.exists() and shell.jobs == {}
    try:
        assert "started job 1" in shell.bash("echo second", background=True)
        assert shell.jobs_dir.stat().st_mode & 0o777 == 0o700  # it was a mkdtemp folder: only its owner may write
        out = shell.job("1", wait=5)
        assert "exited with code 0" in out and "second" in out
    finally:
        shell.close()


@pytest.mark.skipif(not docker_ready(), reason="needs Docker and the jean-code-sandbox image")
def test_a_closed_docker_shell_starts_a_new_container(tmp_path):
    # Catches a closed sandbox that can't run commands again, loses the workspace's files, or keeps old job ids.
    shell = DockerShell(workdir=tmp_path, network=False)
    try:
        marker = uuid.uuid4().hex
        shell.bash(f"echo {marker} > kept.txt")
        assert "started job 1" in shell.bash("echo first", background=True)
        first = shell.container
        shell.close()
        assert shell.container is None and shell.jobs == {}
        assert marker in shell.bash("cat kept.txt")
        assert shell.container not in (None, first)
        assert "started job 1" in shell.bash("echo second", background=True)
        assert "second" in shell.job("1", wait=10)
    finally:
        shell.close()


def test_a_new_sandbox_finds_the_package_caches_of_the_last(monkeypatch, tmp_path):
    # Catches every new container downloading every package again: each run, and each resume, gets a new one.
    started: list[list[str]] = []

    class Container:
        stdout = io.StringIO("ready\n")

        def communicate(self):
            return "", ""

    def popen(command, **kwargs):
        started.append(command)
        return Container()

    monkeypatch.setattr(shell_module.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
    monkeypatch.setattr(shell_module.subprocess, "Popen", popen)
    shell = DockerShell(tmp_path)
    shell.start()
    shell.close()
    [command] = started
    assert "XDG_CACHE_HOME=/workspace/.cache" in command
