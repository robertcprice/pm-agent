"""
Production Hardening for PM Agent.

Provides enterprise-grade reliability features:
- Multi-project management with context switching
- Health checks and heartbeat monitoring
- Crash recovery and state persistence
- Metrics collection and monitoring
- Log aggregation and structured logging
"""

import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque
import sqlite3

logger = logging.getLogger(__name__)


# =============================================================================
# Health Check System
# =============================================================================

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    checks: List[HealthCheck]
    uptime_seconds: float
    last_check: datetime
    degraded_components: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "uptime_seconds": self.uptime_seconds,
            "last_check": self.last_check.isoformat(),
            "degraded_components": self.degraded_components,
            "warnings": self.warnings,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "latency_ms": c.latency_ms,
                }
                for c in self.checks
            ],
        }


class HealthMonitor:
    """
    Monitors system health with configurable checks.

    Features:
    - Heartbeat monitoring
    - Component health checks
    - Automatic alerting
    - Health history tracking
    """

    def __init__(
        self,
        check_interval_seconds: int = 30,
        history_size: int = 100,
        unhealthy_threshold: int = 3,
    ):
        self.check_interval = check_interval_seconds
        self.history_size = history_size
        self.unhealthy_threshold = unhealthy_threshold

        self._start_time = datetime.now()
        self._checks: Dict[str, Callable[[], HealthCheck]] = {}
        self._history: deque = deque(maxlen=history_size)
        self._running = False
        self._lock = threading.Lock()
        self._alert_callbacks: List[Callable[[SystemHealth], None]] = []
        self._consecutive_failures: Dict[str, int] = {}

    def register_check(
        self,
        name: str,
        check_fn: Callable[[], HealthCheck],
    ) -> None:
        """Register a health check function."""
        with self._lock:
            self._checks[name] = check_fn
            self._consecutive_failures[name] = 0

    def unregister_check(self, name: str) -> None:
        """Unregister a health check."""
        with self._lock:
            self._checks.pop(name, None)
            self._consecutive_failures.pop(name, None)

    def add_alert_callback(
        self,
        callback: Callable[[SystemHealth], None],
    ) -> None:
        """Add callback for health alerts."""
        self._alert_callbacks.append(callback)

    def run_checks(self) -> SystemHealth:
        """Run all health checks and return system health."""
        checks: List[HealthCheck] = []
        degraded = []
        warnings = []

        with self._lock:
            for name, check_fn in self._checks.items():
                try:
                    start = time.time()
                    result = check_fn()
                    result.latency_ms = (time.time() - start) * 1000
                    checks.append(result)

                    if result.status == HealthStatus.UNHEALTHY:
                        self._consecutive_failures[name] += 1
                        if self._consecutive_failures[name] >= self.unhealthy_threshold:
                            degraded.append(name)
                    elif result.status == HealthStatus.DEGRADED:
                        warnings.append(f"{name}: {result.message}")
                        self._consecutive_failures[name] = 0
                    else:
                        self._consecutive_failures[name] = 0

                except Exception as e:
                    logger.error(f"Health check {name} failed: {e}")
                    checks.append(HealthCheck(
                        name=name,
                        status=HealthStatus.UNKNOWN,
                        message=str(e),
                    ))
                    self._consecutive_failures[name] += 1

        # Determine overall status
        if degraded:
            overall_status = HealthStatus.UNHEALTHY
        elif warnings:
            overall_status = HealthStatus.DEGRADED
        elif not checks:
            overall_status = HealthStatus.UNKNOWN
        else:
            overall_status = HealthStatus.HEALTHY

        uptime = (datetime.now() - self._start_time).total_seconds()

        health = SystemHealth(
            status=overall_status,
            checks=checks,
            uptime_seconds=uptime,
            last_check=datetime.now(),
            degraded_components=degraded,
            warnings=warnings,
        )

        self._history.append(health)

        # Trigger alerts if unhealthy
        if overall_status == HealthStatus.UNHEALTHY:
            for callback in self._alert_callbacks:
                try:
                    callback(health)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

        return health

    async def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        self._running = True
        while self._running:
            self.run_checks()
            await asyncio.sleep(self.check_interval)

    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._running = False

    def get_history(self) -> List[SystemHealth]:
        """Get health history."""
        return list(self._history)


# =============================================================================
# Crash Recovery System
# =============================================================================

@dataclass
class AgentState:
    """Serializable agent state for crash recovery."""
    agent_id: str
    project_id: str
    state: str
    current_goal_id: Optional[str] = None
    current_task_id: Optional[str] = None
    cycle_count: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)
    checkpoint_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "project_id": self.project_id,
            "state": self.state,
            "current_goal_id": self.current_goal_id,
            "current_task_id": self.current_task_id,
            "cycle_count": self.cycle_count,
            "stats": self.stats,
            "checkpoint_time": self.checkpoint_time.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        data = data.copy()
        if isinstance(data.get("checkpoint_time"), str):
            data["checkpoint_time"] = datetime.fromisoformat(data["checkpoint_time"])
        return cls(**data)


class CrashRecoveryManager:
    """
    Manages crash recovery with state persistence.

    Features:
    - Periodic state checkpointing
    - Crash detection on startup
    - State restoration
    - Graceful shutdown handling
    """

    def __init__(
        self,
        data_dir: Path,
        checkpoint_interval_seconds: int = 60,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = checkpoint_interval_seconds

        self._state_file = self.data_dir / "agent_state.json"
        self._lock_file = self.data_dir / "agent.lock"
        self._running = False
        self._current_state: Optional[AgentState] = None

    def acquire_lock(self) -> bool:
        """Acquire exclusive lock for this agent instance."""
        try:
            if self._lock_file.exists():
                # Check if lock is stale (older than 5 minutes)
                lock_age = time.time() - self._lock_file.stat().st_mtime
                if lock_age < 300:
                    logger.warning("Another agent instance may be running")
                    return False

            # Create lock file with PID
            self._lock_file.write_text(str(os.getpid()))
            return True

        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return False

    def release_lock(self) -> None:
        """Release the lock file."""
        try:
            if self._lock_file.exists():
                self._lock_file.unlink()
        except Exception as e:
            logger.error(f"Failed to release lock: {e}")

    def check_crash(self) -> Optional[AgentState]:
        """Check if agent crashed and return last state if available."""
        try:
            if not self._state_file.exists():
                return None

            # Load last state
            data = json.loads(self._state_file.read_text())
            state = AgentState.from_dict(data)

            # Check if this was a clean shutdown
            if state.metadata.get("clean_shutdown"):
                return None

            logger.warning(f"Detected crash - last checkpoint: {state.checkpoint_time}")
            return state

        except Exception as e:
            logger.error(f"Failed to check crash state: {e}")
            return None

    def checkpoint(self, state: AgentState) -> None:
        """Save current state to disk."""
        try:
            state.checkpoint_time = datetime.now()
            state.metadata["clean_shutdown"] = False
            self._current_state = state
            self._state_file.write_text(json.dumps(state.to_dict(), indent=2))
            logger.debug(f"Checkpoint saved: {state.checkpoint_time}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def mark_clean_shutdown(self) -> None:
        """Mark that the agent is shutting down cleanly."""
        try:
            if self._current_state:
                self._current_state.metadata["clean_shutdown"] = True
                self._state_file.write_text(
                    json.dumps(self._current_state.to_dict(), indent=2)
                )
        except Exception as e:
            logger.error(f"Failed to mark clean shutdown: {e}")

    async def start_checkpointing(
        self,
        state_provider: Callable[[], AgentState],
    ) -> None:
        """Start periodic checkpointing."""
        self._running = True
        while self._running:
            try:
                state = state_provider()
                self.checkpoint(state)
            except Exception as e:
                logger.error(f"Checkpointing failed: {e}")
            await asyncio.sleep(self.checkpoint_interval)

    def stop_checkpointing(self) -> None:
        """Stop periodic checkpointing."""
        self._running = False
        self.mark_clean_shutdown()


# =============================================================================
# Metrics System
# =============================================================================

@dataclass
class Metric:
    """A single metric measurement."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    unit: str = ""


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"      # Monotonically increasing
    GAUGE = "gauge"          # Current value
    HISTOGRAM = "histogram"  # Distribution of values


class MetricsCollector:
    """
    Collects and exposes metrics for monitoring.

    Provides metrics in Prometheus-compatible format.
    """

    def __init__(self, prefix: str = "pm_agent"):
        self.prefix = prefix
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._labels: Dict[str, Dict[str, str]] = {}
        self._lock = threading.Lock()

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Increment a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value
            if labels:
                self._labels[key] = labels

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Set a gauge value."""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value
            if labels:
                self._labels[key] = labels

    def observe(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Observe a value for histogram."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)
            if labels:
                self._labels[key] = labels
            # Keep last 1000 observations
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]

    def _make_key(
        self,
        name: str,
        labels: Optional[Dict[str, str]],
    ) -> str:
        """Create a unique key for a metric with labels."""
        key = f"{self.prefix}_{name}"
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            key = f"{key}{{{label_str}}}"
        return key

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key, 0)

    def get_histogram_stats(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, float]:
        """Get histogram statistics."""
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])
        if not values:
            return {"count": 0, "sum": 0, "avg": 0, "min": 0, "max": 0}

        return {
            "count": len(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    k: self._compute_histogram_stats(v)
                    for k, v in self._histograms.items()
                },
            }

    def _compute_histogram_stats(self, values: List[float]) -> Dict[str, float]:
        """Compute histogram statistics."""
        if not values:
            return {"count": 0, "sum": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0}

        sorted_values = sorted(values)
        count = len(sorted_values)

        return {
            "count": count,
            "sum": sum(values),
            "avg": sum(values) / count,
            "p50": sorted_values[int(count * 0.5)],
            "p95": sorted_values[int(count * 0.95)] if count >= 20 else sorted_values[-1],
            "p99": sorted_values[int(count * 0.99)] if count >= 100 else sorted_values[-1],
        }

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        with self._lock:
            # Counters
            for key, value in self._counters.items():
                lines.append(f"# TYPE {key.split('{')[0]} counter")
                lines.append(f"{key} {value}")

            # Gauges
            for key, value in self._gauges.items():
                lines.append(f"# TYPE {key.split('{')[0]} gauge")
                lines.append(f"{key} {value}")

            # Histograms (as summary)
            for key, values in self._histograms.items():
                base_key = key.split('{')[0]
                lines.append(f"# TYPE {base_key} summary")
                stats = self._compute_histogram_stats(values)
                lines.append(f"{base_key}_count {stats['count']}")
                lines.append(f"{base_key}_sum {stats['sum']}")

        return "\n".join(lines)


# =============================================================================
# Multi-Project Manager
# =============================================================================

@dataclass
class ProjectContext:
    """Context for a managed project."""
    project_id: str
    name: str
    root_path: Path
    data_dir: Path
    config: Dict[str, Any] = field(default_factory=dict)
    active: bool = False
    last_active: Optional[datetime] = None
    stats: Dict[str, Any] = field(default_factory=dict)


class MultiProjectManager:
    """
    Manages multiple projects with context switching.

    Features:
    - Project registration and configuration
    - Context switching between projects
    - Project-specific state isolation
    - Resource cleanup on switch
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._projects: Dict[str, ProjectContext] = {}
        self._active_project: Optional[str] = None
        self._switch_callbacks: List[Callable[[str, str], None]] = []
        self._db_path = self.data_dir / "projects.db"
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the projects database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    root_path TEXT NOT NULL,
                    data_dir TEXT NOT NULL,
                    config TEXT DEFAULT '{}',
                    active INTEGER DEFAULT 0,
                    last_active TEXT,
                    stats TEXT DEFAULT '{}',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def register_project(
        self,
        project_id: str,
        name: str,
        root_path: Path,
        config: Optional[Dict[str, Any]] = None,
    ) -> ProjectContext:
        """Register a new project."""
        root_path = Path(root_path).resolve()
        data_dir = self.data_dir / project_id
        data_dir.mkdir(parents=True, exist_ok=True)

        context = ProjectContext(
            project_id=project_id,
            name=name,
            root_path=root_path,
            data_dir=data_dir,
            config=config or {},
        )

        self._projects[project_id] = context

        # Persist to database
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO projects
                (id, name, root_path, data_dir, config)
                VALUES (?, ?, ?, ?, ?)
            """, (
                project_id,
                name,
                str(root_path),
                str(data_dir),
                json.dumps(config or {}),
            ))
            conn.commit()

        logger.info(f"Registered project: {name} ({project_id})")
        return context

    def unregister_project(self, project_id: str) -> bool:
        """Unregister a project."""
        if project_id == self._active_project:
            logger.error("Cannot unregister active project")
            return False

        if project_id in self._projects:
            del self._projects[project_id]

        with sqlite3.connect(self._db_path) as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.commit()

        logger.info(f"Unregistered project: {project_id}")
        return True

    def get_project(self, project_id: str) -> Optional[ProjectContext]:
        """Get project context."""
        if project_id in self._projects:
            return self._projects[project_id]

        # Try loading from database
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM projects WHERE id = ?",
                (project_id,)
            ).fetchone()

            if row:
                context = ProjectContext(
                    project_id=row[0],
                    name=row[1],
                    root_path=Path(row[2]),
                    data_dir=Path(row[3]),
                    config=json.loads(row[4] or "{}"),
                    active=bool(row[5]),
                    last_active=datetime.fromisoformat(row[6]) if row[6] else None,
                    stats=json.loads(row[7] or "{}"),
                )
                self._projects[project_id] = context
                return context

        return None

    def list_projects(self) -> List[ProjectContext]:
        """List all registered projects."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute("SELECT id FROM projects").fetchall()
            return [self.get_project(row[0]) for row in rows if self.get_project(row[0])]

    def switch_project(self, project_id: str) -> bool:
        """Switch to a different project."""
        context = self.get_project(project_id)
        if not context:
            logger.error(f"Project not found: {project_id}")
            return False

        old_project = self._active_project

        # Deactivate old project
        if old_project and old_project in self._projects:
            self._projects[old_project].active = False
            self._update_project_status(old_project, active=False)

        # Activate new project
        self._active_project = project_id
        context.active = True
        context.last_active = datetime.now()
        self._update_project_status(project_id, active=True)

        # Notify callbacks
        for callback in self._switch_callbacks:
            try:
                callback(old_project or "", project_id)
            except Exception as e:
                logger.error(f"Switch callback failed: {e}")

        logger.info(f"Switched to project: {context.name}")
        return True

    def _update_project_status(
        self,
        project_id: str,
        active: bool,
    ) -> None:
        """Update project status in database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                UPDATE projects
                SET active = ?, last_active = ?
                WHERE id = ?
            """, (int(active), datetime.now().isoformat(), project_id))
            conn.commit()

    def get_active_project(self) -> Optional[ProjectContext]:
        """Get currently active project."""
        if self._active_project:
            return self.get_project(self._active_project)
        return None

    def on_project_switch(
        self,
        callback: Callable[[str, str], None],
    ) -> None:
        """Register callback for project switches."""
        self._switch_callbacks.append(callback)

    def update_project_stats(
        self,
        project_id: str,
        stats: Dict[str, Any],
    ) -> None:
        """Update project statistics."""
        context = self.get_project(project_id)
        if context:
            context.stats.update(stats)
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    UPDATE projects SET stats = ? WHERE id = ?
                """, (json.dumps(context.stats), project_id))
                conn.commit()


# =============================================================================
# Production Manager (combines all components)
# =============================================================================

class ProductionManager:
    """
    Unified production management for PM Agent.

    Combines:
    - Health monitoring
    - Crash recovery
    - Metrics collection
    - Multi-project management
    """

    def __init__(
        self,
        data_dir: Path,
        agent_id: str = "pm_agent",
        check_interval: int = 30,
        checkpoint_interval: int = 60,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.agent_id = agent_id

        # Initialize components
        self.health = HealthMonitor(check_interval_seconds=check_interval)
        self.recovery = CrashRecoveryManager(
            data_dir=self.data_dir / "recovery",
            checkpoint_interval_seconds=checkpoint_interval,
        )
        self.metrics = MetricsCollector(prefix="pm_agent")
        self.projects = MultiProjectManager(self.data_dir / "projects")

        # Register default health checks
        self._register_default_checks()

        # State
        self._running = False
        self._tasks: List[asyncio.Task] = []

    def _register_default_checks(self) -> None:
        """Register default health checks."""
        # Database check
        def check_database() -> HealthCheck:
            try:
                db_path = self.data_dir / "projects" / "projects.db"
                if db_path.exists():
                    with sqlite3.connect(db_path) as conn:
                        conn.execute("SELECT 1").fetchone()
                    return HealthCheck(
                        name="database",
                        status=HealthStatus.HEALTHY,
                        message="Database accessible",
                    )
                return HealthCheck(
                    name="database",
                    status=HealthStatus.DEGRADED,
                    message="Database not initialized",
                )
            except Exception as e:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                )

        self.health.register_check("database", check_database)

        # Disk space check
        def check_disk() -> HealthCheck:
            try:
                import shutil
                total, used, free = shutil.disk_usage(self.data_dir)
                free_percent = (free / total) * 100

                if free_percent < 5:
                    return HealthCheck(
                        name="disk",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Very low disk space: {free_percent:.1f}% free",
                    )
                elif free_percent < 15:
                    return HealthCheck(
                        name="disk",
                        status=HealthStatus.DEGRADED,
                        message=f"Low disk space: {free_percent:.1f}% free",
                    )
                return HealthCheck(
                    name="disk",
                    status=HealthStatus.HEALTHY,
                    message=f"Disk OK: {free_percent:.1f}% free",
                )
            except Exception as e:
                return HealthCheck(
                    name="disk",
                    status=HealthStatus.UNKNOWN,
                    message=str(e),
                )

        self.health.register_check("disk", check_disk)

    async def start(
        self,
        state_provider: Callable[[], AgentState],
    ) -> None:
        """Start production monitoring."""
        self._running = True

        # Acquire lock
        if not self.recovery.acquire_lock():
            logger.error("Failed to acquire agent lock")
            return

        # Check for crash recovery
        crash_state = self.recovery.check_crash()
        if crash_state:
            logger.warning(f"Recovering from crash - cycle {crash_state.cycle_count}")
            self.metrics.increment("crash_recovery_count")

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self.health.start_monitoring()),
            asyncio.create_task(self.recovery.start_checkpointing(state_provider)),
        ]

        logger.info("Production monitoring started")

    async def stop(self) -> None:
        """Stop production monitoring."""
        self._running = False

        # Stop components
        self.health.stop_monitoring()
        self.recovery.stop_checkpointing()

        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Release lock
        self.recovery.release_lock()

        logger.info("Production monitoring stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get overall production status."""
        health = self.health.run_checks()
        active_project = self.projects.get_active_project()

        return {
            "agent_id": self.agent_id,
            "health": health.to_dict(),
            "metrics": self.metrics.get_all_metrics(),
            "active_project": active_project.project_id if active_project else None,
            "project_count": len(self.projects.list_projects()),
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_production_manager(
    data_dir: Path,
    agent_id: str = "pm_agent",
) -> ProductionManager:
    """Create a configured production manager."""
    return ProductionManager(
        data_dir=data_dir,
        agent_id=agent_id,
    )


def setup_signal_handlers(
    cleanup_fn: Callable[[], None],
) -> None:
    """Setup graceful shutdown signal handlers."""
    def handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        cleanup_fn()
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
