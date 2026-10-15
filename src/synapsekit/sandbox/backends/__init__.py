"""Runtime-selectable sandbox backends."""

from .base import BackendHandle, SandboxBackend, build_backend
from .bwrap import BubblewrapBackend
from .docker import DockerBackend, OrbStackBackend
from .fake import FakeBackend
from .firecracker import FirecrackerBackend
from .lima import LimaBackend
from .podman import PodmanBackend

__all__ = [
    "BackendHandle",
    "BubblewrapBackend",
    "DockerBackend",
    "FakeBackend",
    "FirecrackerBackend",
    "LimaBackend",
    "OrbStackBackend",
    "PodmanBackend",
    "SandboxBackend",
    "build_backend",
]
