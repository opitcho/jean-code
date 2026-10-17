"""Shells for the agent's `bash` and `job` tools: on the host, or in Docker.

Commands are non-interactive (no TTY, stdin closed) and stateless: each call is a fresh `bash -c` in the working
directory, so `cd` and `export` don't carry over but files do. `LocalShell` runs them on the host, with no
sandboxing; `DockerShell` runs them in a container, in the mounted `workspace/` folder.
"""

import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
import weakref
from pathlib import Path

from jean_code.config import WORKSPACE_DIR
from jean_code.core.tool_schema import tool

JOBS_DIR = ".jobs"
MAX_OUTPUT_CHARS = 10000
ENV = {"PAGER": "cat", "GIT_PAGER": "cat", "TERM": "dumb", "DEBIAN_FRONTEND": "noninteractive"}
TIMEOUT_HINT = "For commands that take longer, rerun with background=True and check on them with job()."


def truncate_middle(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """Keeps the head and tail of long output: errors are usually at the end.

    Args:
        text: The output.
        max_chars: The most to keep.

    Returns:
        The text, with its middle replaced by a note if it's longer than `max_chars`.
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n[truncated {len(text) - 2 * half} chars]\n{text[-half:]}"


def format_result(code: int, output: str) -> str:
    """Returns what the model sees for a finished command: exit code, then output.

    Args:
        code: The exit code.
        output: Its stdout and stderr together.
    """
    return f"exit code: {code}\n{truncate_middle(output)}".rstrip()


class Shell:
    """The `bash` and `job` tools, on a backend that runs the commands.

    A backend implements `describe`, `_exec` and `_spawn`. The arguments it keeps are attributes of the same name.

    Args:
        workdir: Where commands run.
        jobs_dir: Where background jobs' files go, on the host.
        max_timeout: The most seconds a `bash` call may wait.

    Attributes:
        requires_approval: Whether the agent should ask the user before each `bash` call.
        jobs: Each background job's id, and how much of its log was already returned.
    """

    requires_approval = True

    def __init__(self, workdir: str | Path, jobs_dir: str | Path, max_timeout: int = 600):
        self.workdir = Path(workdir).resolve()
        self.jobs_dir = Path(jobs_dir).resolve()
        self._make_jobs_dir()
        self.max_timeout = max_timeout
        self.jobs: dict[str, int] = {}

    @tool
    def bash(self, command: str, timeout: int = 120, background: bool = False) -> str:
        """Run a shell command in the working directory and return its exit code and output.

        Commands are not interactive: there is no terminal and no stdin, so use flags like -y
        and avoid pagers and editors. Each call starts a fresh shell in the working directory, so `cd`
        and `export` don't carry over to the next call; files do.

        Args:
            command: The bash command to run, e.g. "python script.py" or "ls -la".
            timeout: Seconds before the command is killed, up to 600.
            background: Start the command without waiting, for servers or long builds. Returns
                a job id; check its output and status with the job tool.

        Returns:
            The exit code followed by stdout and stderr, or the id of the background job.
        """
        if background:
            self._make_jobs_dir()  # close() deletes it, and a resumed agent runs on the same shell
            job_id = str(len(self.jobs) + 1)
            for kind in ("pid", "log", "exit"):  # ids restart each session: clear files from an older job
                self._job_file(job_id, kind).unlink(missing_ok=True)
            self._spawn(self._job_script(job_id, command))
            self.jobs[job_id] = 0
            return f"started job {job_id} (output in {self._jobs_path()}/{job_id}.log); check it with job('{job_id}')"

        timeout = max(1, min(timeout, self.max_timeout))
        code, output, timed_out = self._exec(command, timeout)
        if timed_out:
            return f"exit code: 124 (timed out after {timeout}s)\n{truncate_middle(output)}\n\n{TIMEOUT_HINT}"
        return format_result(code, output)

    @tool
    def job(self, job_id: str, wait: int = 0, kill: bool = False) -> str:
        """Check on a background job started with bash(background=True).

        Args:
            job_id: The id bash returned when it started the job.
            wait: Seconds to wait for the job to finish before answering, up to 600.
            kill: Stop the job.

        Returns:
            Whether the job is running or its exit code, then the output it printed since the last check.
        """
        job_id = str(job_id)
        if job_id not in self.jobs:
            raise ValueError(f"unknown job '{job_id}'; jobs: {', '.join(self.jobs) or 'none'}")
        if kill:
            self._kill_job(job_id)
        deadline = time.monotonic() + max(0, min(wait, self.max_timeout))
        while self._exit_code(job_id) is None and time.monotonic() < deadline:
            time.sleep(0.5)

        log_file = self._job_file(job_id, "log")
        log = log_file.read_text(errors="replace") if log_file.exists() else ""
        new_output, self.jobs[job_id] = log[self.jobs[job_id] :], len(log)
        code = self._exit_code(job_id)
        status = "running" if code is None else "killed" if code == "killed" else f"exited with code {code}"
        return f"job {job_id}: {status}\n{truncate_middle(new_output)}".rstrip()

    def describe(self) -> str:
        """Returns one line for the system prompt, saying where commands run."""
        raise NotImplementedError

    def close(self) -> None:
        """Kills background jobs that are still running, and forgets them.

        The shell can be used again afterwards.
        """
        for job_id in self.jobs:
            if self._exit_code(job_id) is None:
                self._kill_job(job_id)
        self.jobs.clear()  # their files go with the jobs folder, so new jobs count from 1 again

    def _exec(self, command: str, timeout: int) -> tuple[int, str, bool]:
        """Runs `command` to completion.

        Returns:
            The exit code, stdout and stderr together, and whether it timed out.
        """
        raise NotImplementedError

    def _spawn(self, command: str) -> None:
        """Starts `command` detached, and returns at once."""
        raise NotImplementedError

    def _make_jobs_dir(self) -> None:
        # Owner-only, as LocalShell's mkdtemp folder was: its path in /tmp is known in advance
        self.jobs_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    # A job writes `<id>.pid`, `.log` and `.exit` into `jobs_dir`, on the host, so `job` reads the files directly:
    # a temp folder for the host shell (to keep them out of the user's project), and `workspace/.jobs/<shell id>`
    # for Docker (so the container can write them).
    def _job_file(self, job_id: str, kind: str) -> Path:
        return self.jobs_dir / f"{job_id}.{kind}"

    def _jobs_path(self) -> str:
        """Returns the jobs folder as commands see it."""
        return str(self.jobs_dir)

    def _job_script(self, job_id: str, command: str) -> str:
        """Returns a script that runs `command` as its own process group's leader.

        The script records the job's pid, output and exit code.
        """
        base = shlex.quote(f"{self._jobs_path()}/{job_id}")
        # a subshell, so an `exit` in the command still lets the exit code be recorded
        inner = f"echo $$ > {base}.pid; ( {command}\n) > {base}.log 2>&1 < /dev/null; echo $? > {base}.exit"
        return f"setsid bash -c {shlex.quote(inner)}"

    def _exit_code(self, job_id: str) -> str | None:
        path = self._job_file(job_id, "exit")
        return path.read_text().strip() if path.exists() else None

    def _kill_job(self, job_id: str) -> None:
        """Kills the job's whole process group, and records the job as killed.

        It kills from where the job runs, so pids match.
        """
        pid_file = self._job_file(job_id, "pid")
        deadline = time.monotonic() + 2
        while not pid_file.exists() and time.monotonic() < deadline:  # a just-started job may not have written it yet
            time.sleep(0.05)
        if pid_file.exists() and self._exit_code(job_id) is None:
            self._exec(f"kill -TERM -- -{pid_file.read_text().strip()}", timeout=10)
            time.sleep(0.2)
            if self._exit_code(job_id) is None:
                self._job_file(job_id, "exit").write_text("killed\n")


class LocalShell(Shell):
    """A shell on the host, in the directory the agent was started from.

    Asks for approval by default. There is no sandboxing: commands can `cd` anywhere, so approval is the only
    safety net.

    Args:
        workdir: Where commands run; the current directory by default.
        max_timeout: The most seconds a `bash` call may wait.
    """

    requires_approval = True

    def __init__(self, workdir: str | Path | None = None, max_timeout: int = 600):
        super().__init__(workdir or Path.cwd(), tempfile.mkdtemp(prefix="jean-code-jobs-"), max_timeout)

    def close(self) -> None:
        """Kills background jobs that are still running, and removes their files."""
        super().close()
        shutil.rmtree(self.jobs_dir, ignore_errors=True)

    def describe(self) -> str:
        return (
            f"Shell commands run on the user's machine in {self.workdir}, "
            "and the user approves each one before it runs."
        )

    def _exec(self, command: str, timeout: int) -> tuple[int, str, bool]:
        process = subprocess.Popen(
            ["bash", "-c", command],
            cwd=self.workdir,
            env=os.environ | ENV,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=True,  # own process group, so a timeout kills its children too
        )
        try:
            output, _ = process.communicate(timeout=timeout)
            return process.returncode, output, False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            try:
                output, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:  # a child left the group and still holds the pipe
                process.kill()
                output = ""
            return process.returncode, output, True

    def _spawn(self, command: str) -> None:
        subprocess.Popen(
            ["bash", "-c", command],
            cwd=self.workdir,
            env=os.environ | ENV,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _remove_container(name: str, process: subprocess.Popen) -> None:
    """Removes a sandbox container, and reaps the `docker run` that started it."""
    subprocess.run(["docker", "rm", "--force", name], capture_output=True, check=False)
    process.communicate()


class DockerShell(Shell):
    """A shell in a long-lived Docker container with only the workspace mounted.

    The container starts on the first command and lives until `close()`, so installed packages persist. It also
    dies with this object or this process, even on a crash. It runs as the host user, so files it writes to the
    workspace are yours. Commands run without approval by default: the sandbox is the safety net.

    Several can share one workspace (an agent and its subagents), each in its own container and with its own
    background jobs. The arguments it keeps are attributes of the same name.

    Args:
        workdir: The workspace folder, mounted at /workspace.
        image: The container image.
        network: Whether the container gets a network.
        max_timeout: The most seconds a `bash` call may wait.

    Attributes:
        container: The running container's name; None until the first command.
    """

    requires_approval = False

    def __init__(
        self,
        workdir: str | Path = WORKSPACE_DIR,
        image: str = "jean-code-sandbox",
        network: bool = True,
        max_timeout: int = 600,
    ):
        # One per shell, relative to the workspace: every shell numbers its jobs from 1
        self._jobs_folder = f"{JOBS_DIR}/{uuid.uuid4().hex[:8]}"
        super().__init__(workdir, Path(workdir) / self._jobs_folder, max_timeout)
        self.image = image
        self.network = network
        self.container: str | None = None
        self._remove: weakref.finalize | None = None

    def describe(self) -> str:
        return (
            "Shell commands run in a Docker container (Debian, Python 3.12, uv, git) in /workspace, "
            f"a folder shared with the user's machine. Network access is {'on' if self.network else 'off'}."
        )

    def start(self) -> str:
        """Starts the container, if it isn't running yet, and returns its name.

        Raises:
            RuntimeError: Docker isn't running, or the container couldn't start.
        """
        if self.container:
            return self.container
        info = subprocess.run(
            ["docker", "info", "--format", "{{.HTTPProxy}}\n{{.HTTPSProxy}}\n{{.NoProxy}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if info.returncode != 0:
            raise RuntimeError(
                "Docker isn't running; start it with `systemctl --user start docker-desktop` (Docker Desktop) "
                "or `sudo systemctl start docker` (Docker Engine)"
            )
        name = f"jean-code-{uuid.uuid4().hex[:8]}"
        command = [
            "docker", "run", "--interactive", "--rm", "--init", "--name", name,
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--volume", f"{self.workdir}:/workspace", "--workdir", "/workspace",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--memory", "2g", "--cpus", "2", "--pids-limit", "256",
            *[arg for key, value in ENV.items() for arg in ("--env", f"{key}={value}")],
            *(self._proxy_env(info.stdout) if self.network else ["--network", "none"]),
            self.image, "bash", "-c", "echo ready && exec cat > /dev/null",
        ]  # fmt: skip
        # The container lives as long as `cat` reads our end of its stdin: when this process
        # exits, however it exits, the OS closes the pipe and the container stops and is removed.
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        first_line = process.stdout.readline() if process.stdout else ""
        if first_line.strip() != "ready":
            error = process.communicate()[1].strip()
            if "Unable to find image" in error or "pull access denied" in error:
                error += f"\n(build the image with `docker build -t {self.image} sandbox/`)"
            raise RuntimeError(f"couldn't start the sandbox: {error}")
        self.container = name
        # also on garbage collection (a dropped Streamlit session) and at interpreter exit
        self._remove = weakref.finalize(self, _remove_container, name, process)
        return name

    @staticmethod
    def _proxy_env(info: str) -> list[str]:
        """Returns `--env` flags passing the daemon's proxy on to the container.

        Each variable comes in both spellings tools look for.

        Docker Desktop sends container traffic through its own proxy: plain HTTP works without
        it, but HTTPS (pip, uv, curl) only works when the container is told to use it.
        """
        http, https, no_proxy = [*info.splitlines(), "", "", ""][:3]
        flags = []
        for key, raw in {"HTTP_PROXY": http, "HTTPS_PROXY": https, "NO_PROXY": no_proxy}.items():
            value = raw.strip()
            if value:
                url = value if "://" in value or key == "NO_PROXY" else f"http://{value}"
                flags += ["--env", f"{key}={url}", "--env", f"{key.lower()}={url}"]
        return flags

    def _jobs_path(self) -> str:
        return f"/workspace/{self._jobs_folder}"

    def close(self) -> None:
        """Removes the container, which also ends its background jobs, and their files.

        The next command starts a new container, with the same workspace.
        """
        if self._remove:
            self._remove()
            self.container = self._remove = None
        shutil.rmtree(self.jobs_dir, ignore_errors=True)
        # Not Shell.close: killing each job would start a new container just to reach the old one's processes
        self.jobs.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()

    def _exec(self, command: str, timeout: int) -> tuple[int, str, bool]:
        container = self.start()
        # `timeout` inside the container: killing the `docker exec` client wouldn't stop the command
        argv = ["docker", "exec", container, "timeout", "-k", "5", str(timeout), "bash", "-c", command]
        try:
            done = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                timeout=timeout + 10,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            output = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else e.stdout or ""
            return -1, output, True
        return done.returncode, done.stdout, done.returncode == 124

    def _spawn(self, command: str) -> None:
        container = self.start()
        subprocess.run(["docker", "exec", "--detach", container, "bash", "-c", command], check=True)
