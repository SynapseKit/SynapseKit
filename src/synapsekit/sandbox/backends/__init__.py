"""Runtime-selectable sandbox backends."""

from .base import BackendHandle, SandboxBackend, build_backend
from .docker import DockerBackend, OrbStackBackend
from .fake import FakeBackend
from .firecracker import FirecrackerBackend
from .lima import LimaBackend

__all__ = [
    "BackendHandle",
    "DockerBackend",
    "FakeBackend",
    "FirecrackerBackend",
    "LimaBackend",
    "OrbStackBackend",
    "SandboxBackend",
    "build_backend",
]
