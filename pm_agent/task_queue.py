"""
Task Management System for PM Agent.

Provides persistent task queue with:
- Goals broken into tasks
- Task dependencies
- Status tracking
- Audit logging
- Human escalations
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum
from datetime import datetime
import sqlite3
import json
import uuid


class GoalStatus(Enum):
    """Status of a high-level goal."""
    PENDING = "pending"          # Not started
    PLANNING = "planning"        # Being broken into tasks
    IN_PROGRESS = "in_progress"  # Tasks being executed
    BLOCKED = "blocked"          # Waiting on something
    COMPLETED = "completed"      # All tasks done
    FAILED = "failed"            # Could not complete
    CANCELLED = "cancelled"      # User cancelled


class TaskStatus(Enum):
    """Status of an individual task."""
    PENDING = "pending"          # Waiting to be picked up
    QUEUED = "queued"            # In queue, ready to execute
    IN_PROGRESS = "in_progress"  # Currently being executed
    WAITING_REVIEW = "waiting_review"  # Done, awaiting PM review
    NEEDS_REVISION = "needs_revision"  # Review failed, needs another attempt
    COMPLETED = "completed"      # Successfully completed
    FAILED = "failed"            # Failed after max attempts
    BLOCKED = "blocked"          # Waiting on dependency
    ESCALATED = "escalated"      # Escalated to human
    CANCELLED = "cancelled"      # Cancelled


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1   # Do immediately
    HIGH = 2       # Do soon
    MEDIUM = 3     # Normal priority
    LOW = 4        # Do when nothing else
    BACKLOG = 5    # Maybe someday


class EventType(Enum):
    """Types of events logged for tasks."""
    CREATED = "created"
    QUEUED = "queued"
    STARTED = "started"
    PROGRESS = "progress"
    DELEGATED = "delegated"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    ESCALATED = "escalated"
    REVIEWED = "reviewed"
    CANCELLED = "cancelled"


@dataclass
class Project:
    """A project the PM manages."""
    id: str
    name: str
    root_path: str
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None


@dataclass
class Goal:
    """A high-level goal to achieve."""
    id: str
    description: str
    project_id: str
    status: GoalStatus = GoalStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: str = "human"  # "human" or "pm_agent"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Task:
    """A concrete task to execute."""
    id: str
    goal_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    assigned_to: str = "claude_code"  # "claude_code" or "local_coder"

    # Execution tracking
    attempt_count: int = 0
    max_attempts: int = 3

    # Context for execution
    context_files: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)

    # Results
    result_summary: str = ""
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    error_message: str = ""

    # Timestamps
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Dependencies (task IDs that must complete first)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TaskLog:
    """An event in a task's lifecycle."""
    id: str
    task_id: str
    event_type: EventType
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Escalation:
    """An escalation to human oversight."""
    id: str
    task_id: str
    reason: str
    status: str = "pending"  # pending, resolved, dismissed
    human_response: str = ""
    created_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


class TaskQueue:
    """
    Persistent task queue with SQLite backend.

    Manages the full lifecycle of goals and tasks:
    - Goals are high-level objectives from humans
    - Tasks are concrete work items broken from goals
    - Dependencies ensure proper execution order
    - Logs provide full audit trail
    - Escalations handle human-in-the-loop
    """

    def __init__(self, db_path: Path):
        """
        Initialize the task queue.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.executescript('''
                -- Projects table
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    root_path TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    config_json TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Goals table
                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    priority INTEGER DEFAULT 3,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    created_by TEXT DEFAULT 'human',
                    metadata_json TEXT DEFAULT '{}',
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                );

                -- Tasks table
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    priority INTEGER DEFAULT 3,
                    assigned_to TEXT DEFAULT 'claude_code',
                    attempt_count INTEGER DEFAULT 0,
                    max_attempts INTEGER DEFAULT 3,
                    context_files_json TEXT DEFAULT '[]',
                    constraints_json TEXT DEFAULT '[]',
                    acceptance_criteria_json TEXT DEFAULT '[]',
                    result_summary TEXT DEFAULT '',
                    files_modified_json TEXT DEFAULT '[]',
                    files_created_json TEXT DEFAULT '[]',
                    error_message TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (goal_id) REFERENCES goals(id)
                );

                -- Task dependencies
                CREATE TABLE IF NOT EXISTS task_dependencies (
                    task_id TEXT NOT NULL,
                    depends_on_task_id TEXT NOT NULL,
                    PRIMARY KEY (task_id, depends_on_task_id),
                    FOREIGN KEY (task_id) REFERENCES tasks(id),
                    FOREIGN KEY (depends_on_task_id) REFERENCES tasks(id)
                );

                -- Task logs
                CREATE TABLE IF NOT EXISTS task_logs (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata_json TEXT DEFAULT '{}',
                    FOREIGN KEY (task_id) REFERENCES tasks(id)
                );

                -- Escalations
                CREATE TABLE IF NOT EXISTS escalations (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    human_response TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved_at TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES tasks(id)
                );

                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_tasks_goal ON tasks(goal_id);
                CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
                CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status);
                CREATE INDEX IF NOT EXISTS idx_goals_project ON goals(project_id);
                CREATE INDEX IF NOT EXISTS idx_logs_task ON task_logs(task_id);
                CREATE INDEX IF NOT EXISTS idx_escalations_status ON escalations(status);
            ''')

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # =========================================================================
    # Project Management
    # =========================================================================

    def add_project(self, project: Project) -> str:
        """Add a new project."""
        if not project.id:
            project.id = str(uuid.uuid4())

        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO projects (id, name, root_path, description, config_json)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                project.id,
                project.name,
                project.root_path,
                project.description,
                json.dumps(project.config),
            ))

        return project.id

    def get_project(self, project_id: str) -> Optional[Project]:
        """Get a project by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                'SELECT * FROM projects WHERE id = ?',
                (project_id,)
            ).fetchone()

            if row:
                return Project(
                    id=row['id'],
                    name=row['name'],
                    root_path=row['root_path'],
                    description=row['description'],
                    config=json.loads(row['config_json']),
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                )
            return None

    def list_projects(self) -> List[Project]:
        """List all projects."""
        with self._get_conn() as conn:
            rows = conn.execute('SELECT * FROM projects ORDER BY created_at DESC').fetchall()
            return [
                Project(
                    id=row['id'],
                    name=row['name'],
                    root_path=row['root_path'],
                    description=row['description'],
                    config=json.loads(row['config_json']),
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                )
                for row in rows
            ]

    # =========================================================================
    # Goal Management
    # =========================================================================

    def add_goal(self, goal: Goal) -> str:
        """Add a new goal."""
        if not goal.id:
            goal.id = str(uuid.uuid4())

        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO goals (id, project_id, description, status, priority, created_by, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                goal.id,
                goal.project_id,
                goal.description,
                goal.status.value,
                goal.priority.value,
                goal.created_by,
                json.dumps(goal.metadata),
            ))

        return goal.id

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                'SELECT * FROM goals WHERE id = ?',
                (goal_id,)
            ).fetchone()

            if row:
                return self._row_to_goal(row)
            return None

    def _row_to_goal(self, row: sqlite3.Row) -> Goal:
        """Convert database row to Goal object."""
        return Goal(
            id=row['id'],
            project_id=row['project_id'],
            description=row['description'],
            status=GoalStatus(row['status']),
            priority=TaskPriority(row['priority']),
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
            created_by=row['created_by'],
            metadata=json.loads(row['metadata_json']),
        )

    def update_goal_status(self, goal_id: str, status: GoalStatus) -> None:
        """Update a goal's status."""
        with self._get_conn() as conn:
            if status == GoalStatus.COMPLETED:
                conn.execute('''
                    UPDATE goals SET status = ?, completed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (status.value, goal_id))
            else:
                conn.execute(
                    'UPDATE goals SET status = ? WHERE id = ?',
                    (status.value, goal_id)
                )

    def get_active_goals(self, project_id: str = None) -> List[Goal]:
        """Get all active (non-completed, non-cancelled) goals."""
        with self._get_conn() as conn:
            if project_id:
                rows = conn.execute('''
                    SELECT * FROM goals
                    WHERE project_id = ? AND status NOT IN ('completed', 'cancelled', 'failed')
                    ORDER BY priority, created_at
                ''', (project_id,)).fetchall()
            else:
                rows = conn.execute('''
                    SELECT * FROM goals
                    WHERE status NOT IN ('completed', 'cancelled', 'failed')
                    ORDER BY priority, created_at
                ''').fetchall()

            return [self._row_to_goal(row) for row in rows]

    # =========================================================================
    # Task Management
    # =========================================================================

    def add_task(self, task: Task) -> str:
        """Add a new task."""
        if not task.id:
            task.id = str(uuid.uuid4())

        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO tasks (
                    id, goal_id, description, status, priority, assigned_to,
                    attempt_count, max_attempts, context_files_json, constraints_json,
                    acceptance_criteria_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.id,
                task.goal_id,
                task.description,
                task.status.value,
                task.priority.value,
                task.assigned_to,
                task.attempt_count,
                task.max_attempts,
                json.dumps(task.context_files),
                json.dumps(task.constraints),
                json.dumps(task.acceptance_criteria),
            ))

            # Add dependencies
            for dep_id in task.dependencies:
                conn.execute('''
                    INSERT INTO task_dependencies (task_id, depends_on_task_id)
                    VALUES (?, ?)
                ''', (task.id, dep_id))

            # Log creation
            self._log_event(conn, task.id, EventType.CREATED, "Task created")

        return task.id

    def add_tasks_batch(self, tasks: List[Task]) -> List[str]:
        """Add multiple tasks in a batch."""
        task_ids = []
        for task in tasks:
            task_id = self.add_task(task)
            task_ids.append(task_id)
        return task_ids

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                'SELECT * FROM tasks WHERE id = ?',
                (task_id,)
            ).fetchone()

            if row:
                return self._row_to_task(conn, row)
            return None

    def _row_to_task(self, conn: sqlite3.Connection, row: sqlite3.Row) -> Task:
        """Convert database row to Task object."""
        # Get dependencies
        dep_rows = conn.execute(
            'SELECT depends_on_task_id FROM task_dependencies WHERE task_id = ?',
            (row['id'],)
        ).fetchall()
        dependencies = [r['depends_on_task_id'] for r in dep_rows]

        return Task(
            id=row['id'],
            goal_id=row['goal_id'],
            description=row['description'],
            status=TaskStatus(row['status']),
            priority=TaskPriority(row['priority']),
            assigned_to=row['assigned_to'],
            attempt_count=row['attempt_count'],
            max_attempts=row['max_attempts'],
            context_files=json.loads(row['context_files_json']),
            constraints=json.loads(row['constraints_json']),
            acceptance_criteria=json.loads(row['acceptance_criteria_json']),
            result_summary=row['result_summary'],
            files_modified=json.loads(row['files_modified_json']),
            files_created=json.loads(row['files_created_json']),
            error_message=row['error_message'],
            created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
            started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
            completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
            dependencies=dependencies,
        )

    def get_next_task(self, assigned_to: str = None) -> Optional[Task]:
        """
        Get the next task to work on.

        Returns highest priority task that:
        - Is in PENDING or QUEUED status
        - Has all dependencies completed
        - Matches assigned_to if specified
        """
        with self._get_conn() as conn:
            # Build query
            query = '''
                SELECT t.* FROM tasks t
                WHERE t.status IN ('pending', 'queued')
            '''
            params = []

            if assigned_to:
                query += ' AND t.assigned_to = ?'
                params.append(assigned_to)

            # Exclude tasks with incomplete dependencies
            query += '''
                AND NOT EXISTS (
                    SELECT 1 FROM task_dependencies td
                    JOIN tasks dep ON td.depends_on_task_id = dep.id
                    WHERE td.task_id = t.id
                    AND dep.status NOT IN ('completed')
                )
            '''

            query += ' ORDER BY t.priority, t.created_at LIMIT 1'

            row = conn.execute(query, params).fetchone()

            if row:
                return self._row_to_task(conn, row)
            return None

    def get_tasks_for_goal(self, goal_id: str) -> List[Task]:
        """Get all tasks for a goal."""
        with self._get_conn() as conn:
            rows = conn.execute(
                'SELECT * FROM tasks WHERE goal_id = ? ORDER BY priority, created_at',
                (goal_id,)
            ).fetchall()
            return [self._row_to_task(conn, row) for row in rows]

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        message: str = "",
    ) -> None:
        """Update a task's status with logging."""
        with self._get_conn() as conn:
            now = datetime.now().isoformat()

            if status == TaskStatus.IN_PROGRESS:
                conn.execute('''
                    UPDATE tasks SET status = ?, started_at = ?
                    WHERE id = ?
                ''', (status.value, now, task_id))
            elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                conn.execute('''
                    UPDATE tasks SET status = ?, completed_at = ?
                    WHERE id = ?
                ''', (status.value, now, task_id))
            else:
                conn.execute(
                    'UPDATE tasks SET status = ? WHERE id = ?',
                    (status.value, task_id)
                )

            # Log the event
            event_type = {
                TaskStatus.PENDING: EventType.CREATED,
                TaskStatus.QUEUED: EventType.QUEUED,
                TaskStatus.IN_PROGRESS: EventType.STARTED,
                TaskStatus.COMPLETED: EventType.COMPLETED,
                TaskStatus.FAILED: EventType.FAILED,
                TaskStatus.ESCALATED: EventType.ESCALATED,
                TaskStatus.CANCELLED: EventType.CANCELLED,
            }.get(status, EventType.PROGRESS)

            self._log_event(conn, task_id, event_type, message or f"Status changed to {status.value}")

    def record_task_result(
        self,
        task_id: str,
        success: bool,
        summary: str,
        files_modified: List[str] = None,
        files_created: List[str] = None,
        error_message: str = "",
    ) -> None:
        """Record the result of a task execution."""
        with self._get_conn() as conn:
            task_row = conn.execute(
                'SELECT attempt_count, max_attempts FROM tasks WHERE id = ?',
                (task_id,)
            ).fetchone()

            new_attempt_count = task_row['attempt_count'] + 1

            if success:
                status = TaskStatus.WAITING_REVIEW
            elif new_attempt_count >= task_row['max_attempts']:
                status = TaskStatus.FAILED
            else:
                status = TaskStatus.NEEDS_REVISION

            conn.execute('''
                UPDATE tasks SET
                    status = ?,
                    attempt_count = ?,
                    result_summary = ?,
                    files_modified_json = ?,
                    files_created_json = ?,
                    error_message = ?
                WHERE id = ?
            ''', (
                status.value,
                new_attempt_count,
                summary,
                json.dumps(files_modified or []),
                json.dumps(files_created or []),
                error_message,
                task_id,
            ))

            self._log_event(
                conn, task_id,
                EventType.COMPLETED if success else EventType.FAILED,
                summary or error_message,
                {"files_modified": files_modified, "files_created": files_created}
            )

    def mark_task_for_retry(self, task_id: str, reason: str = "") -> None:
        """Mark a task for retry."""
        with self._get_conn() as conn:
            conn.execute(
                'UPDATE tasks SET status = ? WHERE id = ?',
                (TaskStatus.QUEUED.value, task_id)
            )
            self._log_event(conn, task_id, EventType.RETRYING, reason or "Marked for retry")

    # =========================================================================
    # Logging
    # =========================================================================

    def _log_event(
        self,
        conn: sqlite3.Connection,
        task_id: str,
        event_type: EventType,
        message: str,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Log an event for a task."""
        conn.execute('''
            INSERT INTO task_logs (id, task_id, event_type, message, metadata_json)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()),
            task_id,
            event_type.value,
            message,
            json.dumps(metadata or {}),
        ))

    def get_task_logs(self, task_id: str) -> List[TaskLog]:
        """Get all logs for a task."""
        with self._get_conn() as conn:
            rows = conn.execute(
                'SELECT * FROM task_logs WHERE task_id = ? ORDER BY timestamp',
                (task_id,)
            ).fetchall()

            return [
                TaskLog(
                    id=row['id'],
                    task_id=row['task_id'],
                    event_type=EventType(row['event_type']),
                    message=row['message'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    metadata=json.loads(row['metadata_json']),
                )
                for row in rows
            ]

    # =========================================================================
    # Escalations
    # =========================================================================

    def create_escalation(self, task_id: str, reason: str) -> str:
        """Create an escalation for a task."""
        escalation_id = str(uuid.uuid4())

        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO escalations (id, task_id, reason)
                VALUES (?, ?, ?)
            ''', (escalation_id, task_id, reason))

            conn.execute(
                'UPDATE tasks SET status = ? WHERE id = ?',
                (TaskStatus.ESCALATED.value, task_id)
            )

            self._log_event(conn, task_id, EventType.ESCALATED, reason)

        return escalation_id

    def get_pending_escalations(self) -> List[Escalation]:
        """Get all pending escalations."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM escalations WHERE status = 'pending' ORDER BY created_at"
            ).fetchall()

            return [
                Escalation(
                    id=row['id'],
                    task_id=row['task_id'],
                    reason=row['reason'],
                    status=row['status'],
                    human_response=row['human_response'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
                )
                for row in rows
            ]

    def resolve_escalation(
        self,
        escalation_id: str,
        response: str,
        new_task_status: TaskStatus = TaskStatus.QUEUED,
    ) -> None:
        """Resolve an escalation with human input."""
        with self._get_conn() as conn:
            # Get the task ID
            row = conn.execute(
                'SELECT task_id FROM escalations WHERE id = ?',
                (escalation_id,)
            ).fetchone()

            if not row:
                raise ValueError(f"Escalation not found: {escalation_id}")

            task_id = row['task_id']

            # Update escalation
            conn.execute('''
                UPDATE escalations SET
                    status = 'resolved',
                    human_response = ?,
                    resolved_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (response, escalation_id))

            # Update task
            conn.execute(
                'UPDATE tasks SET status = ? WHERE id = ?',
                (new_task_status.value, task_id)
            )

            self._log_event(
                conn, task_id, EventType.PROGRESS,
                f"Escalation resolved: {response}"
            )

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self, project_id: str = None) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._get_conn() as conn:
            base_query = "SELECT status, COUNT(*) as count FROM tasks"
            if project_id:
                base_query += " WHERE goal_id IN (SELECT id FROM goals WHERE project_id = ?)"
                params = (project_id,)
            else:
                params = ()
            base_query += " GROUP BY status"

            rows = conn.execute(base_query, params).fetchall()

            status_counts = {row['status']: row['count'] for row in rows}

            return {
                "total_tasks": sum(status_counts.values()),
                "pending": status_counts.get('pending', 0) + status_counts.get('queued', 0),
                "in_progress": status_counts.get('in_progress', 0),
                "completed": status_counts.get('completed', 0),
                "failed": status_counts.get('failed', 0),
                "escalated": status_counts.get('escalated', 0),
                "status_breakdown": status_counts,
            }
