"""
PM Agent Logger - Comprehensive logging and work tracking system.

Provides:
- Structured logging with multiple outputs (file, console, websocket)
- Thought tracking and documentation
- Work session persistence
- Real-time event streaming for dashboards
"""

import logging
import json
import asyncio
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Set
from enum import Enum
from queue import Queue
from threading import Lock
import uuid


class LogLevel(Enum):
    """Log levels for PM Agent."""
    DEBUG = "debug"
    INFO = "info"
    THOUGHT = "thought"      # Agent thinking/reasoning
    ACTION = "action"        # Agent taking action
    RESULT = "result"        # Action result
    WARNING = "warning"
    ERROR = "error"
    MILESTONE = "milestone"  # Major accomplishment


@dataclass
class PMLogEntry:
    """A structured log entry."""
    id: str
    timestamp: datetime
    level: LogLevel
    category: str           # "planning", "execution", "review", "escalation"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None
    goal_id: Optional[str] = None
    cycle: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "category": self.category,
            "message": self.message,
            "details": self.details,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "cycle": self.cycle,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class ThoughtEntry:
    """A recorded thought/reasoning step."""
    id: str
    timestamp: datetime
    thought_type: str       # "analysis", "decision", "reflection", "plan"
    content: str
    context: Dict[str, Any] = field(default_factory=dict)
    related_task: Optional[str] = None
    related_goal: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "thought_type": self.thought_type,
            "content": self.content,
            "context": self.context,
            "related_task": self.related_task,
            "related_goal": self.related_goal,
            "confidence": self.confidence,
        }


@dataclass
class WorkSession:
    """A work session tracking container."""
    id: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    goals_completed: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    escalations: int = 0
    cycles: int = 0
    thoughts: List[ThoughtEntry] = field(default_factory=list)
    logs: List[PMLogEntry] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "goals_completed": self.goals_completed,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "escalations": self.escalations,
            "cycles": self.cycles,
            "thought_count": len(self.thoughts),
            "log_count": len(self.logs),
            "summary": self.summary,
        }


class PMLogger:
    """
    Comprehensive logging system for PM Agent.

    Features:
    - Multiple output handlers (file, console, websocket)
    - Thought tracking and documentation
    - Work session management
    - Real-time event streaming
    - Persistent history
    """

    def __init__(
        self,
        log_dir: Path,
        console_output: bool = True,
        file_output: bool = True,
        max_memory_logs: int = 1000,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.console_output = console_output
        self.file_output = file_output
        self.max_memory_logs = max_memory_logs

        # Current state
        self.current_session: Optional[WorkSession] = None
        self.current_cycle = 0

        # In-memory buffers
        self._logs: List[PMLogEntry] = []
        self._thoughts: List[ThoughtEntry] = []
        self._lock = Lock()

        # Event subscribers (for real-time streaming)
        self._subscribers: Set[Callable[[PMLogEntry], None]] = set()
        self._thought_subscribers: Set[Callable[[ThoughtEntry], None]] = set()

        # Async event queue for websocket streaming
        self._event_queue: asyncio.Queue = None

        # Setup Python logging
        self._setup_python_logger()

        # Log file handles
        self._log_file = None
        self._thought_file = None

    def _setup_python_logger(self):
        """Setup Python's logging module."""
        self.logger = logging.getLogger("pm_agent")
        self.logger.setLevel(logging.DEBUG)

        # Console handler
        if self.console_output:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(message)s',
                datefmt='%H:%M:%S'
            )
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    # =========================================================================
    # Session Management
    # =========================================================================

    def start_session(self) -> WorkSession:
        """Start a new work session."""
        self.current_session = WorkSession(
            id=str(uuid.uuid4()),
            started_at=datetime.now(),
        )

        # Open log files
        session_dir = self.log_dir / self.current_session.id[:8]
        session_dir.mkdir(exist_ok=True)

        if self.file_output:
            self._log_file = open(session_dir / "session.jsonl", "a")
            self._thought_file = open(session_dir / "thoughts.jsonl", "a")

        self.log(LogLevel.MILESTONE, "session", "Work session started",
                details={"session_id": self.current_session.id})

        return self.current_session

    def end_session(self, summary: str = "") -> WorkSession:
        """End the current work session."""
        if not self.current_session:
            return None

        self.current_session.ended_at = datetime.now()
        self.current_session.summary = summary

        self.log(LogLevel.MILESTONE, "session", "Work session ended",
                details={
                    "duration_seconds": (
                        self.current_session.ended_at -
                        self.current_session.started_at
                    ).total_seconds(),
                    "summary": summary,
                })

        # Close files
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        if self._thought_file:
            self._thought_file.close()
            self._thought_file = None

        # Save session summary
        self._save_session_summary()

        session = self.current_session
        self.current_session = None
        return session

    def _save_session_summary(self):
        """Save session summary to file."""
        if not self.current_session:
            return

        summary_path = self.log_dir / f"session_{self.current_session.id[:8]}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.current_session.to_dict(), f, indent=2)

    # =========================================================================
    # Logging
    # =========================================================================

    def log(
        self,
        level: LogLevel,
        category: str,
        message: str,
        details: Dict[str, Any] = None,
        task_id: str = None,
        goal_id: str = None,
    ) -> PMLogEntry:
        """Log a structured entry."""
        entry = PMLogEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            details=details or {},
            task_id=task_id,
            goal_id=goal_id,
            cycle=self.current_cycle,
        )

        # Add to memory buffer
        with self._lock:
            self._logs.append(entry)
            if len(self._logs) > self.max_memory_logs:
                self._logs = self._logs[-self.max_memory_logs:]

            if self.current_session:
                self.current_session.logs.append(entry)

        # Write to file
        if self._log_file:
            self._log_file.write(entry.to_json() + "\n")
            self._log_file.flush()

        # Console output
        if self.console_output:
            self._log_to_console(entry)

        # Notify subscribers
        for subscriber in self._subscribers:
            try:
                subscriber(entry)
            except Exception:
                pass

        # Push to async queue
        if self._event_queue:
            try:
                self._event_queue.put_nowait(("log", entry))
            except asyncio.QueueFull:
                pass

        return entry

    def _log_to_console(self, entry: PMLogEntry):
        """Log entry to console with formatting."""
        # Map to Python logging levels
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.THOUGHT: logging.INFO,
            LogLevel.ACTION: logging.INFO,
            LogLevel.RESULT: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.MILESTONE: logging.INFO,
        }

        # Format message with emoji
        emoji_map = {
            LogLevel.DEBUG: "🔍",
            LogLevel.INFO: "ℹ️ ",
            LogLevel.THOUGHT: "🧠",
            LogLevel.ACTION: "⚡",
            LogLevel.RESULT: "✅",
            LogLevel.WARNING: "⚠️ ",
            LogLevel.ERROR: "❌",
            LogLevel.MILESTONE: "🎯",
        }

        emoji = emoji_map.get(entry.level, "")
        formatted = f"{emoji} [{entry.category}] {entry.message}"

        self.logger.log(level_map.get(entry.level, logging.INFO), formatted)

    # Convenience methods
    def debug(self, category: str, message: str, **kwargs):
        return self.log(LogLevel.DEBUG, category, message, **kwargs)

    def info(self, category: str, message: str, **kwargs):
        return self.log(LogLevel.INFO, category, message, **kwargs)

    def action(self, category: str, message: str, **kwargs):
        return self.log(LogLevel.ACTION, category, message, **kwargs)

    def result(self, category: str, message: str, **kwargs):
        return self.log(LogLevel.RESULT, category, message, **kwargs)

    def warning(self, category: str, message: str, **kwargs):
        return self.log(LogLevel.WARNING, category, message, **kwargs)

    def error(self, category: str, message: str, **kwargs):
        return self.log(LogLevel.ERROR, category, message, **kwargs)

    def milestone(self, category: str, message: str, **kwargs):
        return self.log(LogLevel.MILESTONE, category, message, **kwargs)

    # =========================================================================
    # Thought Tracking
    # =========================================================================

    def think(
        self,
        thought_type: str,
        content: str,
        context: Dict[str, Any] = None,
        task_id: str = None,
        goal_id: str = None,
        confidence: float = 1.0,
    ) -> ThoughtEntry:
        """Record a thought/reasoning step."""
        thought = ThoughtEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            thought_type=thought_type,
            content=content,
            context=context or {},
            related_task=task_id,
            related_goal=goal_id,
            confidence=confidence,
        )

        # Add to memory
        with self._lock:
            self._thoughts.append(thought)
            if self.current_session:
                self.current_session.thoughts.append(thought)

        # Write to file
        if self._thought_file:
            self._thought_file.write(json.dumps(thought.to_dict()) + "\n")
            self._thought_file.flush()

        # Log the thought
        self.log(LogLevel.THOUGHT, "thinking", f"[{thought_type}] {content[:100]}...",
                details={"thought_id": thought.id, "confidence": confidence},
                task_id=task_id, goal_id=goal_id)

        # Notify subscribers
        for subscriber in self._thought_subscribers:
            try:
                subscriber(thought)
            except Exception:
                pass

        # Push to async queue
        if self._event_queue:
            try:
                self._event_queue.put_nowait(("thought", thought))
            except asyncio.QueueFull:
                pass

        return thought

    def analyze(self, content: str, **kwargs):
        """Record an analysis thought."""
        return self.think("analysis", content, **kwargs)

    def decide(self, content: str, **kwargs):
        """Record a decision."""
        return self.think("decision", content, **kwargs)

    def reflect(self, content: str, **kwargs):
        """Record a reflection."""
        return self.think("reflection", content, **kwargs)

    def plan(self, content: str, **kwargs):
        """Record a planning thought."""
        return self.think("plan", content, **kwargs)

    # =========================================================================
    # Subscriptions
    # =========================================================================

    def subscribe(self, callback: Callable[[PMLogEntry], None]):
        """Subscribe to log events."""
        self._subscribers.add(callback)

    def unsubscribe(self, callback: Callable[[PMLogEntry], None]):
        """Unsubscribe from log events."""
        self._subscribers.discard(callback)

    def subscribe_thoughts(self, callback: Callable[[ThoughtEntry], None]):
        """Subscribe to thought events."""
        self._thought_subscribers.add(callback)

    def unsubscribe_thoughts(self, callback: Callable[[ThoughtEntry], None]):
        """Unsubscribe from thought events."""
        self._thought_subscribers.discard(callback)

    def create_async_queue(self) -> asyncio.Queue:
        """Create async queue for websocket streaming."""
        self._event_queue = asyncio.Queue(maxsize=100)
        return self._event_queue

    # =========================================================================
    # Queries
    # =========================================================================

    def get_recent_logs(self, limit: int = 50, level: LogLevel = None) -> List[PMLogEntry]:
        """Get recent log entries."""
        with self._lock:
            logs = self._logs[-limit:]
            if level:
                logs = [l for l in logs if l.level == level]
            return logs

    def get_recent_thoughts(self, limit: int = 20) -> List[ThoughtEntry]:
        """Get recent thoughts."""
        with self._lock:
            return self._thoughts[-limit:]

    def get_logs_for_task(self, task_id: str) -> List[PMLogEntry]:
        """Get all logs for a specific task."""
        with self._lock:
            return [l for l in self._logs if l.task_id == task_id]

    def get_thoughts_for_task(self, task_id: str) -> List[ThoughtEntry]:
        """Get all thoughts for a specific task."""
        with self._lock:
            return [t for t in self._thoughts if t.related_task == task_id]

    # =========================================================================
    # Cycle Management
    # =========================================================================

    def set_cycle(self, cycle: int):
        """Update current cycle number."""
        self.current_cycle = cycle
        if self.current_session:
            self.current_session.cycles = cycle

    def increment_stats(self, **kwargs):
        """Increment session statistics."""
        if not self.current_session:
            return

        for key, value in kwargs.items():
            if hasattr(self.current_session, key):
                setattr(self.current_session, key,
                       getattr(self.current_session, key) + value)
