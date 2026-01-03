"""
Project Memory System for PM Agent.

Provides learning and context management by:
- Tracking project-specific patterns and conventions
- Recording task history: successes, failures, and lessons learned
- Storing decisions and their outcomes
- Providing relevant context for new tasks
- Learning from past work to improve future decisions
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import sqlite3
from contextlib import contextmanager

from conch.memory.store import MemoryStore, Memory, MemoryType


logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns the PM can learn."""
    CODING_CONVENTION = "coding_convention"      # Code style, naming, structure
    ARCHITECTURE = "architecture"                # System design patterns
    TASK_BREAKDOWN = "task_breakdown"            # How to split work effectively
    FAILURE_MODE = "failure_mode"                # Common failure patterns
    SUCCESS_STRATEGY = "success_strategy"        # What worked well
    DEPENDENCY = "dependency"                    # Common task dependencies
    ESTIMATION = "estimation"                    # Time/complexity estimates
    TOOL_USAGE = "tool_usage"                    # Effective tool/library usage


class DecisionOutcome(Enum):
    """Outcome of a decision."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


@dataclass
class Pattern:
    """A learned pattern from project history."""
    id: str
    pattern_type: PatternType
    project_id: str

    # Pattern content
    title: str
    description: str
    context: str                              # When this pattern applies

    # Evidence
    examples: List[str] = field(default_factory=list)  # Task IDs demonstrating pattern
    confidence: float = 0.5                   # 0-1, how confident we are
    frequency: int = 1                        # How often we've seen this

    # Metadata
    discovered_at: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)

    # Impact
    success_correlation: float = 0.0          # -1 to 1, correlation with success

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.value,
            "project_id": self.project_id,
            "title": self.title,
            "description": self.description,
            "context": self.context,
            "examples": self.examples,
            "confidence": self.confidence,
            "frequency": self.frequency,
            "discovered_at": self.discovered_at.isoformat() if self.discovered_at else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "tags": self.tags,
            "success_correlation": self.success_correlation,
        }


@dataclass
class Decision:
    """A decision made by the PM Agent."""
    id: str
    task_id: str
    goal_id: Optional[str]
    project_id: str

    # Decision content
    decision_type: str                        # "planning", "review", "escalation", etc.
    description: str
    reasoning: str
    alternatives_considered: List[str] = field(default_factory=list)

    # Outcome
    outcome: DecisionOutcome = DecisionOutcome.UNKNOWN
    outcome_description: str = ""

    # Metadata
    made_at: Optional[datetime] = None
    evaluated_at: Optional[datetime] = None
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "project_id": self.project_id,
            "decision_type": self.decision_type,
            "description": self.description,
            "reasoning": self.reasoning,
            "alternatives_considered": self.alternatives_considered,
            "outcome": self.outcome.value,
            "outcome_description": self.outcome_description,
            "made_at": self.made_at.isoformat() if self.made_at else None,
            "evaluated_at": self.evaluated_at.isoformat() if self.evaluated_at else None,
            "confidence": self.confidence,
        }


@dataclass
class TaskHistory:
    """Historical record of a task execution."""
    task_id: str
    goal_id: str
    project_id: str

    # Task details
    description: str
    success: bool
    duration_seconds: Optional[float] = None

    # What happened
    approach_taken: str = ""
    files_modified: List[str] = field(default_factory=list)
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Lessons
    what_worked: List[str] = field(default_factory=list)
    what_failed: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)

    # Metadata
    executed_at: Optional[datetime] = None
    assigned_to: str = "claude_code"


class ProjectMemory:
    """
    Project-specific memory system for the PM Agent.

    Extends the base MemoryStore with PM-specific functionality:
    - Pattern recognition and learning
    - Decision tracking and outcome analysis
    - Task history and lessons learned
    - Context-aware retrieval for new tasks
    """

    def __init__(self, memory_store: MemoryStore, db_path: Path):
        """
        Initialize project memory.

        Args:
            memory_store: Base memory store for general memories
            db_path: Path to SQLite database for PM-specific data
        """
        self.memory_store = memory_store
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()
        logger.info(f"ProjectMemory initialized at {db_path}")

    def _init_database(self) -> None:
        """Initialize PM-specific database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Patterns table
                CREATE TABLE IF NOT EXISTS patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    context TEXT DEFAULT '',
                    examples_json TEXT DEFAULT '[]',
                    confidence REAL DEFAULT 0.5,
                    frequency INTEGER DEFAULT 1,
                    discovered_at TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    tags_json TEXT DEFAULT '[]',
                    success_correlation REAL DEFAULT 0.0
                );

                CREATE INDEX IF NOT EXISTS idx_patterns_project ON patterns(project_id);
                CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
                CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON patterns(confidence);

                -- Decisions table
                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    goal_id TEXT,
                    project_id TEXT NOT NULL,
                    decision_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    alternatives_json TEXT DEFAULT '[]',
                    outcome TEXT DEFAULT 'unknown',
                    outcome_description TEXT DEFAULT '',
                    made_at TEXT NOT NULL,
                    evaluated_at TEXT,
                    confidence REAL DEFAULT 0.5
                );

                CREATE INDEX IF NOT EXISTS idx_decisions_task ON decisions(task_id);
                CREATE INDEX IF NOT EXISTS idx_decisions_project ON decisions(project_id);
                CREATE INDEX IF NOT EXISTS idx_decisions_outcome ON decisions(outcome);

                -- Task history table
                CREATE TABLE IF NOT EXISTS task_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    goal_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    duration_seconds REAL,
                    approach_taken TEXT DEFAULT '',
                    files_modified_json TEXT DEFAULT '[]',
                    error_type TEXT,
                    error_message TEXT,
                    what_worked_json TEXT DEFAULT '[]',
                    what_failed_json TEXT DEFAULT '[]',
                    lessons_learned_json TEXT DEFAULT '[]',
                    executed_at TEXT NOT NULL,
                    assigned_to TEXT DEFAULT 'claude_code'
                );

                CREATE INDEX IF NOT EXISTS idx_history_task ON task_history(task_id);
                CREATE INDEX IF NOT EXISTS idx_history_project ON task_history(project_id);
                CREATE INDEX IF NOT EXISTS idx_history_success ON task_history(success);
            """)

    @contextmanager
    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # =========================================================================
    # Pattern Management
    # =========================================================================

    def store_pattern(self, pattern: Pattern) -> str:
        """Store or update a learned pattern."""
        with self._get_connection() as conn:
            # Check if pattern exists
            existing = conn.execute(
                "SELECT id FROM patterns WHERE id = ?",
                (pattern.id,)
            ).fetchone()

            if existing:
                # Update existing
                conn.execute("""
                    UPDATE patterns SET
                        title = ?, description = ?, context = ?,
                        examples_json = ?, confidence = ?, frequency = ?,
                        last_seen = ?, tags_json = ?, success_correlation = ?
                    WHERE id = ?
                """, (
                    pattern.title,
                    pattern.description,
                    pattern.context,
                    json.dumps(pattern.examples),
                    pattern.confidence,
                    pattern.frequency,
                    pattern.last_seen.isoformat() if pattern.last_seen else datetime.now().isoformat(),
                    json.dumps(pattern.tags),
                    pattern.success_correlation,
                    pattern.id,
                ))
            else:
                # Insert new
                conn.execute("""
                    INSERT INTO patterns (
                        id, pattern_type, project_id, title, description, context,
                        examples_json, confidence, frequency, discovered_at, last_seen,
                        tags_json, success_correlation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.id,
                    pattern.pattern_type.value,
                    pattern.project_id,
                    pattern.title,
                    pattern.description,
                    pattern.context,
                    json.dumps(pattern.examples),
                    pattern.confidence,
                    pattern.frequency,
                    pattern.discovered_at.isoformat() if pattern.discovered_at else datetime.now().isoformat(),
                    pattern.last_seen.isoformat() if pattern.last_seen else datetime.now().isoformat(),
                    json.dumps(pattern.tags),
                    pattern.success_correlation,
                ))

            # Also store as memory for search
            self.memory_store.store(Memory(
                content=f"Pattern: {pattern.title} - {pattern.description}",
                memory_type=MemoryType.PATTERN,
                source="pm_agent",
                importance=min(1.0, pattern.confidence * 0.8 + 0.2),
                tags=["pattern", pattern.pattern_type.value, pattern.project_id],
                metadata={
                    "pattern_id": pattern.id,
                    "pattern_type": pattern.pattern_type.value,
                    "project_id": pattern.project_id,
                },
            ))

        logger.debug(f"Stored pattern: {pattern.title}")
        return pattern.id

    def get_patterns(
        self,
        project_id: str,
        pattern_type: Optional[PatternType] = None,
        min_confidence: float = 0.3,
        limit: int = 20,
    ) -> List[Pattern]:
        """Retrieve learned patterns for a project."""
        with self._get_connection() as conn:
            query = "SELECT * FROM patterns WHERE project_id = ? AND confidence >= ?"
            params = [project_id, min_confidence]

            if pattern_type:
                query += " AND pattern_type = ?"
                params.append(pattern_type.value)

            query += " ORDER BY confidence DESC, frequency DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()

            patterns = []
            for row in rows:
                patterns.append(Pattern(
                    id=row["id"],
                    pattern_type=PatternType(row["pattern_type"]),
                    project_id=row["project_id"],
                    title=row["title"],
                    description=row["description"],
                    context=row["context"],
                    examples=json.loads(row["examples_json"]) if row["examples_json"] else [],
                    confidence=row["confidence"],
                    frequency=row["frequency"],
                    discovered_at=datetime.fromisoformat(row["discovered_at"]) if row["discovered_at"] else None,
                    last_seen=datetime.fromisoformat(row["last_seen"]) if row["last_seen"] else None,
                    tags=json.loads(row["tags_json"]) if row["tags_json"] else [],
                    success_correlation=row["success_correlation"],
                ))

            return patterns

    def update_pattern_confidence(
        self,
        pattern_id: str,
        success: bool,
        weight: float = 0.1,
    ) -> None:
        """Update pattern confidence based on new evidence."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT confidence, frequency FROM patterns WHERE id = ?",
                (pattern_id,)
            ).fetchone()

            if row:
                # Bayesian update: adjust confidence based on success/failure
                current_confidence = row["confidence"]
                adjustment = weight if success else -weight
                new_confidence = max(0.0, min(1.0, current_confidence + adjustment))

                conn.execute("""
                    UPDATE patterns SET
                        confidence = ?,
                        frequency = frequency + 1,
                        last_seen = ?
                    WHERE id = ?
                """, (
                    new_confidence,
                    datetime.now().isoformat(),
                    pattern_id,
                ))

    # =========================================================================
    # Decision Tracking
    # =========================================================================

    def store_decision(
        self,
        task_id: str,
        decision_type: str,
        description: str,
        reasoning: str,
        project_id: str,
        goal_id: Optional[str] = None,
        alternatives: Optional[List[str]] = None,
        confidence: float = 0.5,
    ) -> str:
        """Record a decision made by the PM."""
        decision = Decision(
            id=f"dec_{task_id}_{decision_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            task_id=task_id,
            goal_id=goal_id,
            project_id=project_id,
            decision_type=decision_type,
            description=description,
            reasoning=reasoning,
            alternatives_considered=alternatives or [],
            confidence=confidence,
            made_at=datetime.now(),
        )

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO decisions (
                    id, task_id, goal_id, project_id, decision_type,
                    description, reasoning, alternatives_json, outcome,
                    outcome_description, made_at, confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision.id,
                decision.task_id,
                decision.goal_id,
                decision.project_id,
                decision.decision_type,
                decision.description,
                decision.reasoning,
                json.dumps(decision.alternatives_considered),
                decision.outcome.value,
                decision.outcome_description,
                decision.made_at.isoformat(),
                decision.confidence,
            ))

        # Store as memory
        self.memory_store.store(Memory(
            content=f"Decision ({decision_type}): {description} - Reasoning: {reasoning}",
            memory_type=MemoryType.LEARNING,
            source="pm_agent",
            importance=0.7,
            tags=["decision", decision_type, task_id],
            metadata={
                "decision_id": decision.id,
                "task_id": task_id,
                "project_id": project_id,
            },
        ))

        logger.debug(f"Stored decision: {description[:50]}...")
        return decision.id

    def record_decision_outcome(
        self,
        decision_id: str,
        outcome: DecisionOutcome,
        description: str = "",
    ) -> None:
        """Record the outcome of a decision."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE decisions SET
                    outcome = ?,
                    outcome_description = ?,
                    evaluated_at = ?
                WHERE id = ?
            """, (
                outcome.value,
                description,
                datetime.now().isoformat(),
                decision_id,
            ))

        logger.debug(f"Recorded outcome for decision {decision_id}: {outcome.value}")

    def get_decisions_for_task(self, task_id: str) -> List[Decision]:
        """Get all decisions made for a task."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM decisions WHERE task_id = ? ORDER BY made_at",
                (task_id,)
            ).fetchall()

            decisions = []
            for row in rows:
                decisions.append(Decision(
                    id=row["id"],
                    task_id=row["task_id"],
                    goal_id=row["goal_id"],
                    project_id=row["project_id"],
                    decision_type=row["decision_type"],
                    description=row["description"],
                    reasoning=row["reasoning"],
                    alternatives_considered=json.loads(row["alternatives_json"]) if row["alternatives_json"] else [],
                    outcome=DecisionOutcome(row["outcome"]),
                    outcome_description=row["outcome_description"],
                    made_at=datetime.fromisoformat(row["made_at"]) if row["made_at"] else None,
                    evaluated_at=datetime.fromisoformat(row["evaluated_at"]) if row["evaluated_at"] else None,
                    confidence=row["confidence"],
                ))

            return decisions

    def get_successful_decisions(
        self,
        project_id: str,
        decision_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Decision]:
        """Get successful decisions for learning."""
        with self._get_connection() as conn:
            query = """
                SELECT * FROM decisions
                WHERE project_id = ? AND outcome = 'success'
            """
            params = [project_id]

            if decision_type:
                query += " AND decision_type = ?"
                params.append(decision_type)

            query += " ORDER BY confidence DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()

            decisions = []
            for row in rows:
                decisions.append(Decision(
                    id=row["id"],
                    task_id=row["task_id"],
                    goal_id=row["goal_id"],
                    project_id=row["project_id"],
                    decision_type=row["decision_type"],
                    description=row["description"],
                    reasoning=row["reasoning"],
                    alternatives_considered=json.loads(row["alternatives_json"]) if row["alternatives_json"] else [],
                    outcome=DecisionOutcome(row["outcome"]),
                    outcome_description=row["outcome_description"],
                    made_at=datetime.fromisoformat(row["made_at"]) if row["made_at"] else None,
                    evaluated_at=datetime.fromisoformat(row["evaluated_at"]) if row["evaluated_at"] else None,
                    confidence=row["confidence"],
                ))

            return decisions

    # =========================================================================
    # Task History
    # =========================================================================

    def record_task_execution(
        self,
        task_id: str,
        goal_id: str,
        project_id: str,
        description: str,
        success: bool,
        duration_seconds: Optional[float] = None,
        approach_taken: str = "",
        files_modified: Optional[List[str]] = None,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        what_worked: Optional[List[str]] = None,
        what_failed: Optional[List[str]] = None,
        lessons_learned: Optional[List[str]] = None,
        assigned_to: str = "claude_code",
    ) -> None:
        """Record the execution of a task for learning."""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO task_history (
                    task_id, goal_id, project_id, description, success,
                    duration_seconds, approach_taken, files_modified_json,
                    error_type, error_message, what_worked_json, what_failed_json,
                    lessons_learned_json, executed_at, assigned_to
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task_id,
                goal_id,
                project_id,
                description,
                1 if success else 0,
                duration_seconds,
                approach_taken,
                json.dumps(files_modified or []),
                error_type,
                error_message,
                json.dumps(what_worked or []),
                json.dumps(what_failed or []),
                json.dumps(lessons_learned or []),
                datetime.now().isoformat(),
                assigned_to,
            ))

        # Store as memory
        status = "succeeded" if success else "failed"
        self.memory_store.store(Memory(
            content=f"Task {status}: {description}. {'; '.join(lessons_learned or [])}",
            memory_type=MemoryType.LEARNING,
            source="pm_agent",
            importance=0.8 if success else 0.9,  # Failures are more important to remember
            tags=["task_execution", status, project_id],
            metadata={
                "task_id": task_id,
                "success": success,
                "project_id": project_id,
            },
        ))

        logger.debug(f"Recorded task execution: {task_id} ({status})")

    def get_similar_tasks(
        self,
        description: str,
        project_id: str,
        limit: int = 5,
    ) -> List[TaskHistory]:
        """Find similar tasks from history."""
        # Use memory store's search for similarity
        memories = self.memory_store.search(
            query=description,
            tags=["task_execution", project_id],
            limit=limit * 2,  # Get more candidates
        )

        # Get task IDs from memories
        task_ids = [m.metadata.get("task_id") for m in memories if m.metadata.get("task_id")]

        if not task_ids:
            return []

        # Fetch full task history
        with self._get_connection() as conn:
            placeholders = ",".join("?" * len(task_ids))
            rows = conn.execute(
                f"SELECT * FROM task_history WHERE task_id IN ({placeholders}) LIMIT ?",
                (*task_ids, limit)
            ).fetchall()

            history = []
            for row in rows:
                history.append(TaskHistory(
                    task_id=row["task_id"],
                    goal_id=row["goal_id"],
                    project_id=row["project_id"],
                    description=row["description"],
                    success=bool(row["success"]),
                    duration_seconds=row["duration_seconds"],
                    approach_taken=row["approach_taken"],
                    files_modified=json.loads(row["files_modified_json"]) if row["files_modified_json"] else [],
                    error_type=row["error_type"],
                    error_message=row["error_message"],
                    what_worked=json.loads(row["what_worked_json"]) if row["what_worked_json"] else [],
                    what_failed=json.loads(row["what_failed_json"]) if row["what_failed_json"] else [],
                    lessons_learned=json.loads(row["lessons_learned_json"]) if row["lessons_learned_json"] else [],
                    executed_at=datetime.fromisoformat(row["executed_at"]) if row["executed_at"] else None,
                    assigned_to=row["assigned_to"],
                ))

            return history

    # =========================================================================
    # Context Retrieval
    # =========================================================================

    def get_relevant_context(
        self,
        task_description: str,
        project_id: str,
        include_patterns: bool = True,
        include_decisions: bool = True,
        include_history: bool = True,
    ) -> str:
        """
        Get relevant context for a task.

        This is the main interface for the PM Agent to retrieve
        useful information when planning or executing tasks.
        """
        context_parts = []

        # Get relevant patterns
        if include_patterns:
            patterns = self.get_patterns(project_id, min_confidence=0.4, limit=5)
            if patterns:
                context_parts.append("# Relevant Patterns\n")
                for p in patterns:
                    context_parts.append(
                        f"- **{p.title}** ({p.pattern_type.value}, confidence: {p.confidence:.2f})\n"
                        f"  {p.description}\n"
                        f"  Context: {p.context}\n"
                    )

        # Get successful decisions
        if include_decisions:
            decisions = self.get_successful_decisions(project_id, limit=3)
            if decisions:
                context_parts.append("\n# Successful Past Decisions\n")
                for d in decisions:
                    context_parts.append(
                        f"- **{d.decision_type}**: {d.description}\n"
                        f"  Reasoning: {d.reasoning}\n"
                    )

        # Get similar task history
        if include_history:
            similar_tasks = self.get_similar_tasks(task_description, project_id, limit=3)
            if similar_tasks:
                context_parts.append("\n# Similar Past Tasks\n")
                for t in similar_tasks:
                    status = "✓ Success" if t.success else "✗ Failed"
                    context_parts.append(f"- {status}: {t.description}\n")

                    if t.what_worked:
                        context_parts.append(f"  What worked: {', '.join(t.what_worked)}\n")
                    if t.what_failed:
                        context_parts.append(f"  What failed: {', '.join(t.what_failed)}\n")
                    if t.lessons_learned:
                        context_parts.append(f"  Lessons: {', '.join(t.lessons_learned)}\n")

        if not context_parts:
            return "No relevant historical context found."

        return "".join(context_parts)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self, project_id: str) -> Dict[str, Any]:
        """Get memory statistics for a project."""
        with self._get_connection() as conn:
            # Pattern stats
            pattern_count = conn.execute(
                "SELECT COUNT(*) as count FROM patterns WHERE project_id = ?",
                (project_id,)
            ).fetchone()["count"]

            # Decision stats
            decision_stats = conn.execute("""
                SELECT outcome, COUNT(*) as count
                FROM decisions
                WHERE project_id = ?
                GROUP BY outcome
            """, (project_id,)).fetchall()

            decision_counts = {row["outcome"]: row["count"] for row in decision_stats}

            # Task history stats
            task_stats = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(success) as successful,
                    AVG(duration_seconds) as avg_duration
                FROM task_history
                WHERE project_id = ?
            """, (project_id,)).fetchone()

            return {
                "patterns_learned": pattern_count,
                "decisions_made": sum(decision_counts.values()),
                "successful_decisions": decision_counts.get("success", 0),
                "failed_decisions": decision_counts.get("failure", 0),
                "tasks_recorded": task_stats["total"] or 0,
                "successful_tasks": task_stats["successful"] or 0,
                "average_task_duration": task_stats["avg_duration"],
            }
