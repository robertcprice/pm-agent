"""
PM Agent Capabilities

This package contains modular capability extensions for the PM Agent.
Each capability provides specific functionality that can be plugged into
the agent's execution pipeline.

Available Capabilities:
- docker: Container management (build, run, stop, logs)
- database: Database operations (migrations, backups, seeding)
- cicd: CI/CD integration (GitHub Actions, GitLab CI)
- testing: Test automation (unit, integration, E2E)
- monitoring: Error monitoring and metrics (Sentry, Prometheus)
- deployment: Deployment automation (Vercel, AWS)
- filewatch: File watching and hot reload
"""

from .docker import DockerCapability
from .database import DatabaseCapability
from .cicd import CICDCapability
from .testing import TestCapability
from .monitoring import MonitoringCapability
from .deployment import DeploymentCapability
from .filewatch import FileWatchCapability
from .analyzer import ProjectAnalyzer, ProjectAnalysis, analyze_project

__all__ = [
    "DockerCapability",
    "DatabaseCapability",
    "CICDCapability",
    "TestCapability",
    "MonitoringCapability",
    "DeploymentCapability",
    "FileWatchCapability",
    "ProjectAnalyzer",
    "ProjectAnalysis",
    "analyze_project",
]
