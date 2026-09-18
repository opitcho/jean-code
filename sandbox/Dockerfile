# Sandbox for jean-code's `bash` tool (src/shell.py: DockerShell).
# Build: docker build -t jean-code-sandbox sandbox/
# If your user id isn't 1000: docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t jean-code-sandbox sandbox/
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# The container runs as the host user (--user uid:gid), so files written to /workspace are theirs.
# A venv owned by that user on PATH lets `pip install` work without root.
ARG UID=1000
ARG GID=1000
RUN groupadd --gid $GID agent && useradd --create-home --uid $UID --gid $GID agent
USER agent
ENV HOME=/home/agent \
    VIRTUAL_ENV=/home/agent/.venv \
    PATH=/home/agent/.venv/bin:$PATH
RUN python -m venv $VIRTUAL_ENV

WORKDIR /workspace
