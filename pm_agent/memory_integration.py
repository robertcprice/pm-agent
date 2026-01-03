"""
Memory Integration Layer for PM Agent.

Automatically captures decisions and outcomes from the agent's execution
and stores them in project memory for learning.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from .project_memory import (
    ProjectMemory,
    Pattern,
    PatternType,
    DecisionOutcome,
)
from .task_queue import Task, TaskStatus, Goal
from .logger import PMLogger, PMLogEntry, ThoughtEntry, LogLevel


logger = logging.getLogger(__name__)


class MemoryIntegration:
    """
    Bridges the PM Agent's logger and project memory system.

    Automatically captures:
    - Planning decisions from thought logs
    - Task execution outcomes
    - Pattern recognition from repeated behaviors
    - Lessons learned from failures
    """

    def __init__(
        self,
        project_memory: ProjectMemory,
        pm_logger: PMLogger,
    ):
        self.memory = project_memory
        self.logger = pm_logger

        # Subscribe to events
        self.logger.subscribe(self._on_log_entry)
        self.logger.subscribe_thoughts(self._on_thought)

        # State tracking for pattern recognition
        self._task_start_times: Dict[str, datetime] = {}
        self._planning_decisions: Dict[str, str] = {}  # task_id -> decision_id

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_log_entry(self, entry: PMLogEntry):
        """Handle log entries and extract learnings."""
        # Capture task completions
        if entry.level == LogLevel.RESULT and entry.task_id:
            self._capture_task_result(entry)

        # Capture errors for learning
        if entry.level == LogLevel.ERROR and entry.task_id:
            self._capture_error_pattern(entry)

        # Capture milestones
        if entry.level == LogLevel.MILESTONE:
            self._capture_milestone(entry)

    def _on_thought(self, thought: ThoughtEntry):
        """Handle thought entries and extract decisions."""
        # Capture planning decisions
        if thought.thought_type == "plan" and thought.related_task:
            self._capture_planning_decision(thought)

        # Capture review decisions
        if thought.thought_type == "decision" and thought.related_task:
            self._capture_review_decision(thought)

    # =========================================================================
    # Capture Methods
    # =========================================================================

    def _capture_task_result(self, entry: PMLogEntry):
        """Capture task completion for learning."""
        task_id = entry.task_id
        if not task_id:
            return

        # Get task details from entry
        success = "success" in entry.message.lower() or "completed" in entry.message.lower()
        duration = None

        # Calculate duration if we tracked start time
        if task_id in self._task_start_times:
            duration = (entry.timestamp - self._task_start_times[task_id]).total_seconds()
            del self._task_start_times[task_id]

        # Extract project_id and goal_id from details
        project_id = entry.details.get("project_id", "unknown")
        goal_id = entry.goal_id or entry.details.get("goal_id")

        # Record in memory
        try:
            self.memory.record_task_execution(
                task_id=task_id,
                goal_id=goal_id or "unknown",
                project_id=project_id,
                description=entry.message,
                success=success,
                duration_seconds=duration,
                approach_taken=entry.details.get("approach", ""),
                files_modified=entry.details.get("files_modified", []),
                error_type=entry.details.get("error_type"),
                error_message=entry.details.get("error_message"),
                lessons_learned=entry.details.get("lessons", []),
            )

            # Update decision outcomes if we have them
            if task_id in self._planning_decisions:
                decision_id = self._planning_decisions[task_id]
                outcome = DecisionOutcome.SUCCESS if success else DecisionOutcome.FAILURE
                self.memory.record_decision_outcome(
                    decision_id=decision_id,
                    outcome=outcome,
                    description=entry.message,
                )
                del self._planning_decisions[task_id]

        except Exception as e:
            logger.warning(f"Failed to capture task result in memory: {e}")

    def _capture_error_pattern(self, entry: PMLogEntry):
        """Capture error patterns for learning."""
        if not entry.task_id or not entry.details.get("error_type"):
            return

        error_type = entry.details.get("error_type", "unknown_error")
        project_id = entry.details.get("project_id", "unknown")

        # Create or update failure pattern
        pattern_id = f"failure_{error_type}_{project_id}"

        try:
            # Check if pattern exists
            patterns = self.memory.get_patterns(
                project_id=project_id,
                pattern_type=PatternType.FAILURE_MODE,
                limit=100,
            )

            existing = next((p for p in patterns if p.id == pattern_id), None)

            if existing:
                # Update frequency
                existing.frequency += 1
                existing.examples.append(entry.task_id)
                self.memory.store_pattern(existing)
            else:
                # Create new pattern
                pattern = Pattern(
                    id=pattern_id,
                    pattern_type=PatternType.FAILURE_MODE,
                    project_id=project_id,
                    title=f"Common failure: {error_type}",
                    description=entry.message,
                    context=f"Error type: {error_type}",
                    examples=[entry.task_id],
                    confidence=0.6,
                    frequency=1,
                    discovered_at=datetime.now(),
                    last_seen=datetime.now(),
                    tags=["failure", error_type],
                    success_correlation=-0.5,
                )
                self.memory.store_pattern(pattern)

        except Exception as e:
            logger.warning(f"Failed to capture error pattern: {e}")

    def _capture_planning_decision(self, thought: ThoughtEntry):
        """Capture planning decisions."""
        task_id = thought.related_task
        if not task_id:
            return

        project_id = thought.context.get("project_id", "unknown")
        goal_id = thought.related_goal

        try:
            decision_id = self.memory.store_decision(
                task_id=task_id,
                decision_type="planning",
                description=f"Task breakdown: {thought.content[:100]}",
                reasoning=thought.content,
                project_id=project_id,
                goal_id=goal_id,
                alternatives=thought.context.get("alternatives", []),
                confidence=thought.confidence,
            )

            # Track for later outcome recording
            self._planning_decisions[task_id] = decision_id

        except Exception as e:
            logger.warning(f"Failed to capture planning decision: {e}")

    def _capture_review_decision(self, thought: ThoughtEntry):
        """Capture review decisions."""
        task_id = thought.related_task
        if not task_id:
            return

        project_id = thought.context.get("project_id", "unknown")
        goal_id = thought.related_goal

        try:
            self.memory.store_decision(
                task_id=task_id,
                decision_type="review",
                description=f"Review decision: {thought.content[:100]}",
                reasoning=thought.content,
                project_id=project_id,
                goal_id=goal_id,
                alternatives=thought.context.get("alternatives", []),
                confidence=thought.confidence,
            )

        except Exception as e:
            logger.warning(f"Failed to capture review decision: {e}")

    def _capture_milestone(self, entry: PMLogEntry):
        """Capture milestones as success patterns."""
        if not entry.details.get("pattern_learned"):
            return

        project_id = entry.details.get("project_id", "unknown")

        try:
            pattern = Pattern(
                id=f"milestone_{entry.id}",
                pattern_type=PatternType.SUCCESS_STRATEGY,
                project_id=project_id,
                title=entry.message,
                description=entry.details.get("description", entry.message),
                context=entry.details.get("context", ""),
                confidence=0.8,
                frequency=1,
                discovered_at=entry.timestamp,
                last_seen=entry.timestamp,
                tags=["milestone", "success"],
                success_correlation=0.8,
            )
            self.memory.store_pattern(pattern)

        except Exception as e:
            logger.warning(f"Failed to capture milestone pattern: {e}")

    # =========================================================================
    # Manual Capture Methods
    # =========================================================================

    def record_task_start(self, task_id: str):
        """Manually record when a task starts."""
        self._task_start_times[task_id] = datetime.now()

    def record_pattern_discovery(
        self,
        pattern_type: PatternType,
        project_id: str,
        title: str,
        description: str,
        context: str = "",
        examples: Optional[List[str]] = None,
        confidence: float = 0.6,
    ):
        """Manually record a discovered pattern."""
        pattern = Pattern(
            id=f"manual_{pattern_type.value}_{datetime.now().timestamp()}",
            pattern_type=pattern_type,
            project_id=project_id,
            title=title,
            description=description,
            context=context,
            examples=examples or [],
            confidence=confidence,
            frequency=1,
            discovered_at=datetime.now(),
            last_seen=datetime.now(),
        )
        self.memory.store_pattern(pattern)

    def record_lesson_learned(
        self,
        task_id: str,
        goal_id: str,
        project_id: str,
        lesson: str,
        success: bool = True,
    ):
        """Manually record a lesson learned."""
        self.memory.record_task_execution(
            task_id=task_id,
            goal_id=goal_id,
            project_id=project_id,
            description=f"Lesson learned: {lesson}",
            success=success,
            lessons_learned=[lesson],
        )

    # =========================================================================
    # Context Helpers
    # =========================================================================

    def get_context_for_task(
        self,
        task: Task,
        project_id: str,
    ) -> str:
        """Get relevant context for a task."""
        return self.memory.get_relevant_context(
            task_description=task.description,
            project_id=project_id,
            include_patterns=True,
            include_decisions=True,
            include_history=True,
        )

    def get_context_for_goal(
        self,
        goal: Goal,
        project_id: str,
    ) -> str:
        """Get relevant context for goal planning."""
        return self.memory.get_relevant_context(
            task_description=goal.description,
            project_id=project_id,
            include_patterns=True,
            include_decisions=True,
            include_history=True,
        )

    # =========================================================================
    # Pattern Recognition Helpers
    # =========================================================================

    def analyze_task_patterns(
        self,
        project_id: str,
        min_frequency: int = 3,
    ) -> List[Pattern]:
        """
        Analyze task history to discover patterns.

        This can be run periodically to mine patterns from history.
        """
        discovered_patterns = []

        # TODO: Implement pattern mining algorithms
        # - Clustering similar tasks
        # - Identifying common failure modes
        # - Detecting architectural patterns from file changes
        # - Recognizing successful task breakdown strategies

        return discovered_patterns

    def update_pattern_from_task(
        self,
        task: Task,
        project_id: str,
        success: bool,
    ):
        """Update patterns based on task outcome."""
        # Get relevant patterns
        patterns = self.memory.get_patterns(project_id, limit=50)

        # Update confidence for patterns that apply to this task
        for pattern in patterns:
            # Simple heuristic: check if pattern tags match task context
            if any(tag in task.description.lower() for tag in pattern.tags):
                self.memory.update_pattern_confidence(
                    pattern_id=pattern.id,
                    success=success,
                    weight=0.05,
                )
