"""Serve SynapseKit agents as REST endpoints (``synapsekit[serve]`` extra)."""

from __future__ import annotations

from .server import ServeMixin, build_agent_app, serve_agent

__all__ = ["ServeMixin", "build_agent_app", "serve_agent"]
