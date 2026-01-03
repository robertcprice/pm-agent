"""
PM Agent - Autonomous Project Manager

An AI-powered project manager that orchestrates Claude Code CLI
and GLM 4.7 to manage software development projects.

Usage:
    from pm_agent import PMAgent, PMConfig

    config = PMConfig(project_path="/path/to/project")
    agent = PMAgent(config)
    await agent.run_cycle()
"""

__version__ = "0.1.0"
__author__ = "Bobby Price"
__license__ = "MIT"

# Core imports
from .agent import PMAgent, PMConfig, PMAgentState, CycleResult
from .task_queue import (
    TaskQueue,
    TaskStatus,
    TaskPriority,
    GoalStatus,
    Project,
    Goal,
    Task,
)
from .logger import ThoughtLogger
from .goal_analyzer import GoalAnalyzer
from .notifications import NotificationManager, NotificationChannel

# Backends
from .glm_backend import GLMBackend
from .hybrid_backend import HybridBackend, TaskRouter, TaskRole

# Advanced features
from .adaptive_learner import AdaptiveLearner
from .intelligent_retry import IntelligentRetry
from .model_selector import ModelSelector
from .claude_mentor import ClaudeMentor

__all__ = [
    # Core
    "PMAgent",
    "PMConfig",
    "PMAgentState",
    "CycleResult",
    # Task Queue
    "TaskQueue",
    "TaskStatus",
    "TaskPriority",
    "GoalStatus",
    "Project",
    "Goal",
    "Task",
    # Utilities
    "ThoughtLogger",
    "GoalAnalyzer",
    "NotificationManager",
    "NotificationChannel",
    # Backends
    "GLMBackend",
    "HybridBackend",
    "TaskRouter",
    "TaskRole",
    # Advanced features
    "AdaptiveLearner",
    "IntelligentRetry",
    "ModelSelector",
    "ClaudeMentor",
]

# Default configuration defaults
DEFAULT_WORK_HOURS = (9, 18)  # 9 AM to 6 PM
DEFAULT_MAX_PARALLEL_TASKS = 3
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_DB_PATH = "pm_agent.db"


def create_agent(
    project_path: str,
    claude_code_path: str = "claude",
    backend: str = "claude",  # claude, glm, or hybrid
    **kwargs
) -> PMAgent:
    """
    Convenience function to create a PM Agent.

    Args:
        project_path: Path to the project directory
        claude_code_path: Path to Claude Code CLI
        backend: Backend to use (claude, glm, or hybrid)
        **kwargs: Additional configuration options

    Returns:
        Configured PMAgent instance

    Example:
        agent = create_agent(
            project_path="/path/to/project",
            backend="glm"  # Use cost-effective GLM backend
        )
    """
    config = PMConfig(
        project_path=project_path,
        claude_code_path=claude_code_path,
        **kwargs
    )

    return PMAgent(config)
