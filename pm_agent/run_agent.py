#!/usr/bin/env python3
"""
Entry point for running the PM Agent.

Usage:
    python -m conch_dna.pm.run_agent --project /path/to/project
    python -m conch_dna.pm.run_agent --config pm_config.yaml
"""

import asyncio
import argparse
import logging
import signal
from pathlib import Path
import yaml

from .agent import PMAgent, PMConfig
from .task_queue import TaskQueue, Project
from .claude_code import ClaudeCodeTool


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_agent(args) -> PMAgent:
    """Create and configure the PM Agent."""

    # Determine paths
    project_root = Path(args.project).resolve()
    data_dir = Path(args.data_dir).resolve() if args.data_dir else project_root / ".pm_agent"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create config
    config = PMConfig(
        project_root=project_root,
        data_dir=data_dir,
        max_concurrent_tasks=args.max_concurrent or 1,
        task_timeout_seconds=args.timeout or 600,
    )

    # Create task queue
    task_queue = TaskQueue(data_dir / "tasks.db")

    # Create or get project
    project_id = args.project_id or project_root.name
    existing = task_queue.get_project(project_id)
    if not existing:
        task_queue.add_project(Project(
            id=project_id,
            name=project_root.name,
            root_path=str(project_root),
            description=f"Project at {project_root}",
        ))

    # Create Claude Code tool
    claude_code = ClaudeCodeTool(
        project_root=project_root,
        auto_commit=args.auto_commit,
    )

    # Create agent (without EGO initially for simplicity)
    agent = PMAgent(
        config=config,
        task_queue=task_queue,
        claude_code=claude_code,
    )

    return agent


async def main():
    parser = argparse.ArgumentParser(description="Run the PM Agent")
    parser.add_argument("--project", "-p", required=True, help="Project root path")
    parser.add_argument("--data-dir", "-d", help="Data directory for PM state")
    parser.add_argument("--project-id", help="Project ID (defaults to directory name)")
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--max-concurrent", type=int, help="Max concurrent tasks")
    parser.add_argument("--timeout", type=int, help="Task timeout in seconds")
    parser.add_argument("--auto-commit", action="store_true", help="Auto-commit changes")
    parser.add_argument("--goal", "-g", help="Initial goal to add")
    parser.add_argument("--cycles", type=int, help="Run for N cycles then stop")

    args = parser.parse_args()

    # Create agent
    agent = create_agent(args)

    # Handle shutdown
    def shutdown(sig, frame):
        logger.info("Shutdown signal received...")
        agent.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Add initial goal if specified
    if args.goal:
        project_id = args.project_id or Path(args.project).name
        goal_id = agent.add_goal(args.goal, project_id)
        logger.info(f"Added goal: {goal_id}")

    # Run
    if args.cycles:
        # Run for specific number of cycles
        for i in range(args.cycles):
            if not agent.running:
                agent.running = True
            result = await agent.run_cycle()
            logger.info(f"Cycle {i+1}/{args.cycles}: {result.action_taken}")
            if result.sleep_duration > 0:
                await asyncio.sleep(min(result.sleep_duration, 5))
    else:
        # Run continuously
        await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
