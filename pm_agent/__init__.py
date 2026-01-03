"""
PM Agent - Autonomous Project Manager

An AI-powered project manager that orchestrates Claude Code CLI
and GLM 4.7 to manage software development projects.
"""

__version__ = "0.1.0"
__author__ = "Bobby Price"
__license__ = "MIT"

# Core imports - only what's definitely working
from .task_queue import (
    TaskQueue,
    TaskStatus,
    TaskPriority,
    GoalStatus,
    Project,
    Goal,
    Task,
)
from .logger import PMLogger

__all__ = [
    # Task Queue
    "TaskQueue",
    "TaskStatus",
    "TaskPriority",
    "GoalStatus",
    "Project",
    "Goal",
    "Task",
    # Logger
    "PMLogger",
    "__version__",
]

# Default configuration defaults
DEFAULT_WORK_HOURS = (9, 18)  # 9 AM to 6 PM
DEFAULT_MAX_PARALLEL_TASKS = 3
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_DB_PATH = "pm_agent.db"


def create_queue(db_path: str = DEFAULT_DB_PATH) -> TaskQueue:
    """
    Convenience function to create a task queue.

    Args:
        db_path: Path to SQLite database

    Returns:
        Configured TaskQueue instance
    """
    return TaskQueue(db_path=db_path)
