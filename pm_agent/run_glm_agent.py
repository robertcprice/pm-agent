#!/usr/bin/env python3
"""
Run PM Agent with GLM 4.7 Backend

This script runs the PM Agent using GLM 4.7 as the coding backend instead of
Claude Code. This is a cost-effective alternative that provides similar
capabilities at lower cost.

Usage:
    # With environment variable
    GLM_API_KEY=your_key python -m conch_dna.pm.run_glm_agent

    # With argument
    python -m conch_dna.pm.run_glm_agent --api-key your_key --project /path/to/project

    # In Docker
    docker run -e GLM_API_KEY=your_key -v $(pwd):/workspace pm-agent-glm
"""

import os
import sys
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from conch_dna.pm import (
    PMAgent, PMConfig, TaskQueue,
    GoalAnalyzer, AdaptiveLearner, create_adaptive_learner,
)
from conch_dna.pm.glm_backend import GLMPMBackend, create_glm_backend, GLMModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


class GLMClaudeCodeAdapter:
    """
    Adapter that makes GLM backend compatible with PM Agent's Claude Code interface.

    This allows seamless switching between Claude Code and GLM without changing
    the PM Agent code.
    """

    def __init__(self, glm_backend: GLMPMBackend):
        self.backend = glm_backend
        self.default_model = "sonnet"  # Maps to glm-4.7
        self.default_max_turns = 50

    def execute_task(self, task):
        """Execute a task using GLM backend.

        Args:
            task: ClaudeCodeTask-like object

        Returns:
            ClaudeCodeResult-like object
        """
        from dataclasses import dataclass

        # Extract task parameters
        description = getattr(task, 'description', str(task))
        working_directory = getattr(task, 'working_directory', None)
        context_files = getattr(task, 'context_files', [])
        constraints = getattr(task, 'constraints', [])
        acceptance_criteria = getattr(task, 'acceptance_criteria', [])
        task_id = getattr(task, 'task_id', None)
        timeout = getattr(task, 'timeout_seconds', 600)

        # Execute via GLM
        result = self.backend.execute_task(
            description=description,
            working_directory=Path(working_directory) if working_directory else None,
            context_files=context_files,
            constraints=constraints,
            acceptance_criteria=acceptance_criteria,
            task_id=task_id,
            timeout_seconds=timeout,
        )

        # Create result object compatible with ClaudeCodeResult
        @dataclass
        class CompatibleResult:
            success: bool
            summary: str
            files_modified: list
            files_created: list
            error_message: str
            error_type: str
            stdout: str
            duration_seconds: float

        return CompatibleResult(
            success=result.success,
            summary=result.output[:500] if result.output else "",
            files_modified=result.files_modified,
            files_created=result.files_created,
            error_message=result.error_message or "",
            error_type="glm_error" if not result.success and result.error_message else "",
            stdout=result.output,
            duration_seconds=result.duration_seconds,
        )


async def run_pm_agent_with_glm(
    api_key: str,
    project_path: Path,
    data_dir: Path,
    goal: str = None,
    use_proxy: bool = False,
):
    """Run PM Agent with GLM backend.

    Args:
        api_key: GLM API key
        project_path: Path to the project to work on
        data_dir: Directory for PM Agent data
        goal: Optional initial goal to work on
        use_proxy: Whether to use Z.ai proxy for Claude Code tools
    """
    logger.info("=" * 60)
    logger.info("PM Agent with GLM 4.7 Backend")
    logger.info("=" * 60)
    logger.info(f"Project: {project_path}")
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Using proxy: {use_proxy}")
    logger.info("")

    # Create data directories
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "tasks.db"

    # Initialize GLM backend
    logger.info("Initializing GLM 4.7 backend...")
    glm_backend = create_glm_backend(
        api_key=api_key,
        use_proxy=use_proxy,
        data_dir=data_dir / "glm",
    )

    if not glm_backend.is_available():
        logger.error("GLM backend is not available. Check API key and dependencies.")
        return

    logger.info("✅ GLM backend ready")

    # Create adapter for PM Agent compatibility
    claude_code_adapter = GLMClaudeCodeAdapter(glm_backend)

    # Initialize task queue
    logger.info("Initializing task queue...")
    task_queue = TaskQueue(db_path)

    # Initialize adaptive learner
    learner = create_adaptive_learner(data_dir / "learning")

    # Create PM Agent config
    config = PMConfig(
        project_root=project_path,
        data_dir=data_dir,
        max_concurrent_tasks=1,
        task_timeout_seconds=600,
        max_task_attempts=3,
        idle_sleep_seconds=30,
        active_sleep_seconds=5,
    )

    # Create PM Agent with GLM backend
    logger.info("Initializing PM Agent...")
    agent = PMAgent(
        config=config,
        task_queue=task_queue,
        claude_code=claude_code_adapter,  # GLM adapter instead of Claude Code
        ego=None,  # Can add EGO later
        memory=None,  # Can add memory later
    )

    logger.info("✅ PM Agent ready with GLM 4.7 backend")
    logger.info("")

    # Create or get project
    project_name = project_path.name
    projects = task_queue.get_projects()
    project = next((p for p in projects if p.name == project_name), None)

    if not project:
        project_id = task_queue.create_project(
            name=project_name,
            root_path=str(project_path),
            description=f"Project managed by PM Agent with GLM backend",
        )
        logger.info(f"Created project: {project_name} ({project_id})")
    else:
        project_id = project.id
        logger.info(f"Using existing project: {project_name} ({project_id})")

    # Add initial goal if provided
    if goal:
        logger.info(f"\n📋 Adding goal: {goal}")
        goal_id = agent.add_goal(goal, project_id, priority="high")
        logger.info(f"   Goal ID: {goal_id}")

    # Run the agent
    logger.info("\n🚀 Starting PM Agent main loop...")
    logger.info("   Press Ctrl+C to stop\n")

    try:
        await agent.run()
    except KeyboardInterrupt:
        logger.info("\n⏹️  Stopping PM Agent...")
        agent.stop()

    # Print final stats
    stats = glm_backend.get_stats()
    logger.info("\n📊 Session Statistics:")
    logger.info(f"   Tasks executed: {stats.get('tasks', 0)}")
    logger.info(f"   Success rate: {stats.get('success_rate', 0):.0%}")
    logger.info(f"   Total duration: {stats.get('total_duration_seconds', 0):.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Run PM Agent with GLM 4.7 backend (cost-effective alternative to Claude Code)"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GLM_API_KEY"),
        help="GLM API key (or set GLM_API_KEY env var)"
    )
    parser.add_argument(
        "--project", "-p",
        default=os.getcwd(),
        help="Path to project directory (default: current directory)"
    )
    parser.add_argument(
        "--data-dir", "-d",
        default=os.environ.get("PM_DATA_DIR", "./pm_data"),
        help="Directory for PM Agent data"
    )
    parser.add_argument(
        "--goal", "-g",
        help="Initial goal to work on"
    )
    parser.add_argument(
        "--use-proxy",
        action="store_true",
        help="Use Z.ai proxy for Claude Code tool compatibility"
    )

    args = parser.parse_args()

    if not args.api_key:
        print("Error: GLM API key required. Set GLM_API_KEY env var or use --api-key")
        sys.exit(1)

    project_path = Path(args.project).resolve()
    if not project_path.exists():
        print(f"Error: Project path does not exist: {project_path}")
        sys.exit(1)

    data_dir = Path(args.data_dir).resolve()

    asyncio.run(run_pm_agent_with_glm(
        api_key=args.api_key,
        project_path=project_path,
        data_dir=data_dir,
        goal=args.goal,
        use_proxy=args.use_proxy,
    ))


if __name__ == "__main__":
    main()
