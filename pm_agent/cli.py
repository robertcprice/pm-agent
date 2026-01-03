#!/usr/bin/env python3
"""
PM Agent CLI

Command-line interface for the PM Agent.
"""

import asyncio
import sys
from pathlib import Path

import click


@click.group()
@click.version_option(version="0.1.0")
def main():
    """PM Agent - Autonomous Project Manager"""
    pass


@main.command()
@click.option("--project", "-p", required=True, type=click.Path(exists=True), help="Project path")
@click.option("--backend", "-b", default="claude", type=click.Choice(["claude", "glm", "hybrid"]), help="Backend to use")
@click.option("--dashboard", "-d", type=click.Choice ["cli", "web", "none"], default="none", help="Dashboard type")
@click.option("--goal", "-g", help="Add a goal and exit")
@click.option("--max-cycles", "-n", type=int, default=0, help="Max cycles to run (0 = infinite)")
@click.option("--interval", "-i", type=int, default=30, help="Seconds between cycles")
def run(project, backend, dashboard, goal, max_cycles, interval):
    """Run the PM Agent"""
    from pm_agent import PMAgent, PMConfig

    if backend == "glm":
        from pm_agent.glm_backend import GLMBackend
        backend_obj = GLMBackend()
    elif backend == "hybrid":
        from pm_agent.hybrid_backend import HybridBackend
        backend_obj = HybridBackend()
    else:
        backend_obj = None

    config = PMConfig(
        project_path=str(project),
        backend=backend_obj,
    )

    async def run_agent():
        agent = PMAgent(config)

        # Add goal if specified
        if goal:
            await agent.add_goal(name=goal, description=goal)

        # Start dashboard if requested
        if dashboard == "cli":
            from pm_agent.cli_dashboard import run_cli_dashboard
            asyncio.create_task(run_cli_dashboard(agent))
        elif dashboard == "web":
            from pm_agent.web_dashboard import start_web_dashboard
            asyncio.create_task(start_web_dashboard(agent))

        # Run agent
        cycle_count = 0
        try:
            while max_cycles == 0 or cycle_count < max_cycles:
                result = await agent.run_cycle()
                cycle_count += 1

                if result.stop_requested:
                    break

                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            click.echo("\nShutting down...")

    asyncio.run(run_agent())


@main.command()
@click.argument("name")
@click.option("--description", "-d", help="Goal description")
@click.option("--project", "-p", default=".", type=click.Path(exists=True), help="Project path")
def add_goal(name, description, project):
    """Add a new goal"""
    from pm_agent import PMAgent, PMConfig

    config = PMConfig(project_path=str(project))
    agent = PMAgent(config)

    async def add():
        goal_id = await agent.add_goal(name=name, description=description or name)
        click.echo(f"Added goal: {name} (ID: {goal_id})")

    asyncio.run(add())


@main.command()
@click.option("--project", "-p", default=".", type=click.Path(exists=True), help="Project path")
@click.option("--status", "-s", type=click.Choice(["pending", "in_progress", "completed", "failed"]), help="Filter by status")
def list_tasks(project, status):
    """List tasks"""
    from pm_agent import TaskQueue, TaskStatus

    queue = TaskQueue(db_path=str(Path(project) / "pm_agent.db"))

    if status:
        tasks = queue.get_tasks(status=TaskStatus(status))
    else:
        tasks = queue.get_tasks()

    click.echo(f"{'ID':<6} {'Status':<12} {'Priority':<10} {'Task'}")
    click.echo("-" * 80)
    for task in tasks:
        click.echo(f"{task.id:<6} {task.status.value:<12} {task.priority.value:<10} {task.description[:60]}")


@main.command()
@click.option("--project", "-p", default=".", type=click.Path(exists=True), help="Project path")
def status(project):
    """Show agent status"""
    from pm_agent import TaskQueue

    queue = TaskQueue(db_path=str(Path(project) / "pm_agent.db"))

    projects = queue.get_projects()
    goals = queue.get_goals()
    tasks = queue.get_tasks()

    click.echo(f"Projects: {len(projects)}")
    click.echo(f"Goals: {len(goals)}")
    click.echo(f"Tasks: {len(tasks)}")

    if tasks:
        status_counts = {}
        for task in tasks:
            status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1

        click.echo("\nTask Status:")
        for status, count in sorted(status_counts.items()):
            click.echo(f"  {status}: {count}")


if __name__ == "__main__":
    main()
