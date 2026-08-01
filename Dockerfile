# syntax=docker/dockerfile:1

# ── SynapseKit official image ────────────────────────────────────────────────
# Two variants are produced from this single Dockerfile via the EXTRAS build arg:
#   EXTRAS=""      → core library + CLI            (small, published as :latest / :<version>)
#   EXTRAS="[all]" → batteries-included, all extras (published as :all / :<version>-all)
#
# Build locally:
#   docker build -t synapsekit:latest .
#   docker build --build-arg EXTRAS="[all]" -t synapsekit:all .
#
# The optional Rust-accelerated chunker (synapsekit._rust_core) is built
# separately with maturin and is NOT included here — SynapseKit falls back to
# its pure-Python chunker automatically.

# Python version to build on — overridable so images can target 3.11–3.14.
# SynapseKit requires-python is >=3.10; verified to import on 3.12/3.13/3.14.
ARG PYTHON_VERSION=3.12

# ── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

# uv provides fast, reproducible installs (matches the project's package manager).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Which optional-dependency group to bake in ("" = core only, "[all]" = everything).
ARG EXTRAS=""

# Copy only what the build backend needs to resolve + install the package.
COPY pyproject.toml README.md ./
COPY src ./src

# Install into an isolated venv so the runtime stage stays lean.
RUN uv venv /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip install --no-cache ".${EXTRAS}"

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
# Redeclare the global ARG so it is in scope for this stage's FROM.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim AS runtime

LABEL org.opencontainers.image.title="SynapseKit" \
      org.opencontainers.image.description="Async-first Python framework for RAG, agents, and LLM apps" \
      org.opencontainers.image.source="https://github.com/SynapseKit/synapsekit" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run as a non-root user.
RUN useradd --create-home --uid 1000 synapse
COPY --from=builder --chown=synapse:synapse /opt/venv /opt/venv

USER synapse
WORKDIR /home/synapse

# `synapsekit` is the console entrypoint (see [project.scripts]).
# Default to showing help; override with e.g. `docker run ... synapsekit serve app:rag --host 0.0.0.0`.
ENTRYPOINT ["synapsekit"]
CMD ["--help"]
