#!/usr/bin/env python3
"""
PM Agent CLI Dashboard - Rich terminal interface for monitoring and interaction.

Features:
- Real-time status display
- Live log streaming
- Thought visualization
- Interactive commands
- Task queue overview
- Session statistics
"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.style import Style
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.markdown import Markdown
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: 'rich' library not installed. Install with: pip install rich")

from .task_queue import TaskQueue, TaskStatus, GoalStatus
from .logger import PMLogger, PMLogEntry, ThoughtEntry, LogLevel


@dataclass
class DashboardState:
    """Current state of the dashboard."""
    agent_state: str = "idle"
    current_task: Optional[str] = None
    current_goal: Optional[str] = None
    cycle_count: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    escalations: int = 0
    uptime_seconds: float = 0
    last_action: str = ""
    logs: List[PMLogEntry] = None
    thoughts: List[ThoughtEntry] = None

    def __post_init__(self):
        if self.logs is None:
            self.logs = []
        if self.thoughts is None:
            self.thoughts = []


class CLIDashboard:
    """
    Rich terminal dashboard for PM Agent.

    Provides a live-updating view of:
    - Agent status and current work
    - Task queue overview
    - Recent logs and thoughts
    - Session statistics
    """

    def __init__(
        self,
        task_queue: TaskQueue,
        logger: PMLogger,
        refresh_rate: float = 1.0,
    ):
        if not RICH_AVAILABLE:
            raise ImportError("rich library required: pip install rich")

        self.queue = task_queue
        self.logger = logger
        self.refresh_rate = refresh_rate

        self.console = Console()
        self.state = DashboardState()
        self.running = False
        self.start_time = None

        # Subscribe to logger events
        self.logger.subscribe(self._on_log)
        self.logger.subscribe_thoughts(self._on_thought)

    def _on_log(self, entry: PMLogEntry):
        """Handle new log entry."""
        self.state.logs.append(entry)
        # Keep only recent logs
        if len(self.state.logs) > 50:
            self.state.logs = self.state.logs[-50:]

    def _on_thought(self, thought: ThoughtEntry):
        """Handle new thought."""
        self.state.thoughts.append(thought)
        if len(self.state.thoughts) > 20:
            self.state.thoughts = self.state.thoughts[-20:]

    def update_state(
        self,
        agent_state: str = None,
        current_task: str = None,
        current_goal: str = None,
        cycle_count: int = None,
        last_action: str = None,
        **stats
    ):
        """Update dashboard state."""
        if agent_state is not None:
            self.state.agent_state = agent_state
        if current_task is not None:
            self.state.current_task = current_task
        if current_goal is not None:
            self.state.current_goal = current_goal
        if cycle_count is not None:
            self.state.cycle_count = cycle_count
        if last_action is not None:
            self.state.last_action = last_action

        for key, value in stats.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)

    def _create_layout(self) -> Layout:
        """Create the dashboard layout."""
        layout = Layout()

        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=1),
        )

        layout["left"].split(
            Layout(name="status", size=8),
            Layout(name="logs"),
        )

        layout["right"].split(
            Layout(name="tasks", ratio=1),
            Layout(name="thoughts", ratio=1),
        )

        return layout

    def _render_header(self) -> Panel:
        """Render header panel."""
        uptime = time.time() - self.start_time if self.start_time else 0
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)

        title = Text()
        title.append("🤖 ", style="bold blue")
        title.append("PM AGENT DASHBOARD", style="bold white")
        title.append(" | ", style="dim")
        title.append(f"Cycle: {self.state.cycle_count}", style="cyan")
        title.append(" | ", style="dim")
        title.append(f"Uptime: {hours:02d}:{minutes:02d}:{seconds:02d}", style="green")
        title.append(" | ", style="dim")
        title.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), style="dim")

        return Panel(title, style="blue", box=box.ROUNDED)

    def _render_status(self) -> Panel:
        """Render status panel."""
        # State indicator with color
        state_colors = {
            "idle": ("⏸️ ", "dim"),
            "planning": ("📋", "yellow"),
            "delegating": ("📤", "cyan"),
            "waiting": ("⏳", "blue"),
            "reviewing": ("🔍", "magenta"),
            "escalating": ("🚨", "red"),
        }
        emoji, color = state_colors.get(self.state.agent_state, ("❓", "white"))

        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Key", style="bold")
        table.add_column("Value")

        table.add_row("State", Text(f"{emoji} {self.state.agent_state.upper()}", style=color))
        table.add_row("Current Task", Text(
            self.state.current_task[:40] + "..." if self.state.current_task and len(self.state.current_task) > 40
            else self.state.current_task or "None",
            style="cyan" if self.state.current_task else "dim"
        ))
        table.add_row("Last Action", Text(self.state.last_action[:50] if self.state.last_action else "—", style="dim"))

        # Stats row
        stats = Table(box=None, show_header=False, padding=(0, 2))
        stats.add_column()
        stats.add_column()
        stats.add_column()
        stats.add_column()

        stats.add_row(
            Text(f"✅ {self.state.tasks_completed}", style="green"),
            Text(f"❌ {self.state.tasks_failed}", style="red"),
            Text(f"🚨 {self.state.escalations}", style="yellow"),
            Text(f"🔄 {self.state.cycle_count} cycles", style="cyan"),
        )

        content = Table.grid()
        content.add_row(table)
        content.add_row(Text(""))
        content.add_row(stats)

        return Panel(content, title="[bold]Status[/bold]", border_style="green")

    def _render_logs(self) -> Panel:
        """Render logs panel."""
        table = Table(box=None, show_header=True, padding=(0, 1))
        table.add_column("Time", style="dim", width=8)
        table.add_column("Lvl", width=4)
        table.add_column("Message")

        level_styles = {
            LogLevel.DEBUG: ("D", "dim"),
            LogLevel.INFO: ("I", "blue"),
            LogLevel.THOUGHT: ("T", "magenta"),
            LogLevel.ACTION: ("A", "cyan"),
            LogLevel.RESULT: ("R", "green"),
            LogLevel.WARNING: ("W", "yellow"),
            LogLevel.ERROR: ("E", "red"),
            LogLevel.MILESTONE: ("M", "bold green"),
        }

        # Show most recent logs
        for log in self.state.logs[-15:]:
            lvl_char, lvl_style = level_styles.get(log.level, ("?", "white"))
            time_str = log.timestamp.strftime("%H:%M:%S")
            msg = log.message[:60] + "..." if len(log.message) > 60 else log.message

            table.add_row(
                Text(time_str, style="dim"),
                Text(lvl_char, style=lvl_style),
                Text(msg),
            )

        return Panel(table, title="[bold]Logs[/bold]", border_style="blue")

    def _render_tasks(self) -> Panel:
        """Render tasks panel."""
        try:
            stats = self.queue.get_stats()
        except Exception:
            stats = {"pending": 0, "in_progress": 0, "completed": 0, "failed": 0}

        table = Table(box=None, show_header=False, padding=(0, 1))
        table.add_column("Status", width=12)
        table.add_column("Count", justify="right")

        table.add_row(Text("⏳ Pending", style="yellow"), str(stats.get("pending", 0)))
        table.add_row(Text("🔄 In Progress", style="cyan"), str(stats.get("in_progress", 0)))
        table.add_row(Text("✅ Completed", style="green"), str(stats.get("completed", 0)))
        table.add_row(Text("❌ Failed", style="red"), str(stats.get("failed", 0)))
        table.add_row(Text("🚨 Escalated", style="magenta"), str(stats.get("escalated", 0)))

        return Panel(table, title="[bold]Tasks[/bold]", border_style="yellow")

    def _render_thoughts(self) -> Panel:
        """Render thoughts panel."""
        content = Text()

        thought_icons = {
            "analysis": "🔍",
            "decision": "⚖️",
            "reflection": "💭",
            "plan": "📋",
        }

        for thought in self.state.thoughts[-5:]:
            icon = thought_icons.get(thought.thought_type, "💡")
            time_str = thought.timestamp.strftime("%H:%M")
            preview = thought.content[:40] + "..." if len(thought.content) > 40 else thought.content

            content.append(f"{icon} ", style="bold")
            content.append(f"[{time_str}] ", style="dim")
            content.append(preview + "\n", style="italic")

        if not self.state.thoughts:
            content.append("No thoughts recorded yet...", style="dim italic")

        return Panel(content, title="[bold]Thoughts[/bold]", border_style="magenta")

    def _render_footer(self) -> Panel:
        """Render footer panel."""
        commands = Text()
        commands.append(" [q] Quit  ", style="dim")
        commands.append("[p] Pause  ", style="dim")
        commands.append("[g] Add Goal  ", style="dim")
        commands.append("[e] View Escalations  ", style="dim")
        commands.append("[h] Help ", style="dim")

        return Panel(commands, style="dim", box=box.ROUNDED)

    def _render(self) -> Layout:
        """Render the full dashboard."""
        layout = self._create_layout()

        layout["header"].update(self._render_header())
        layout["status"].update(self._render_status())
        layout["logs"].update(self._render_logs())
        layout["tasks"].update(self._render_tasks())
        layout["thoughts"].update(self._render_thoughts())
        layout["footer"].update(self._render_footer())

        return layout

    def run(self):
        """Run the dashboard in blocking mode."""
        self.running = True
        self.start_time = time.time()

        with Live(self._render(), console=self.console, refresh_per_second=1/self.refresh_rate) as live:
            try:
                while self.running:
                    live.update(self._render())
                    time.sleep(self.refresh_rate)
            except KeyboardInterrupt:
                self.running = False

    async def run_async(self):
        """Run the dashboard in async mode."""
        self.running = True
        self.start_time = time.time()

        with Live(self._render(), console=self.console, refresh_per_second=1/self.refresh_rate) as live:
            try:
                while self.running:
                    live.update(self._render())
                    await asyncio.sleep(self.refresh_rate)
            except asyncio.CancelledError:
                self.running = False

    def stop(self):
        """Stop the dashboard."""
        self.running = False

    def print_status(self):
        """Print a single status snapshot (non-live)."""
        self.console.print(self._render_header())
        self.console.print(self._render_status())
        self.console.print(self._render_tasks())


def run_standalone_dashboard(db_path: Path, log_dir: Path):
    """Run dashboard as standalone viewer."""
    if not RICH_AVAILABLE:
        print("Error: rich library required. Install with: pip install rich")
        return

    queue = TaskQueue(db_path)
    logger = PMLogger(log_dir, console_output=False)

    dashboard = CLIDashboard(queue, logger)
    dashboard.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PM Agent CLI Dashboard")
    parser.add_argument("--db", required=True, help="Path to tasks database")
    parser.add_argument("--logs", required=True, help="Path to logs directory")
    args = parser.parse_args()

    run_standalone_dashboard(Path(args.db), Path(args.logs))
