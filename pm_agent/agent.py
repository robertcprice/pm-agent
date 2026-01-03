"""
PM Agent - The autonomous project manager.

This is the brain of the system. It:
- Receives goals from humans
- Breaks them into tasks (using EGO)
- Delegates tasks to Claude Code
- Reviews results
- Handles failures and escalations
- Maintains project memory
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
from enum import Enum
from datetime import datetime
import asyncio
import logging
import json

from .task_queue import (
    TaskQueue, Task, Goal, Project,
    TaskStatus, GoalStatus, TaskPriority, Escalation
)
from .notifications import NotificationManager, NotificationConfig
from .claude_code import (
    ClaudeCodeTool, ClaudeCodeTask, ClaudeCodeResult,
    ClaudeCodeModel, ClaudeCodeCreditsExhaustedError
)
from .ego_integration import PMEgoAdapter
from .goal_analyzer import GoalAnalyzer, ComplexityLevel


logger = logging.getLogger(__name__)


class PMAgentState(Enum):
    """Current state of the PM Agent."""
    IDLE = "idle"                    # Waiting for work
    PLANNING = "planning"            # Breaking down a goal
    DELEGATING = "delegating"        # Sending task to coder
    WAITING = "waiting"              # Waiting for coder to finish
    REVIEWING = "reviewing"          # Reviewing completed work
    ESCALATING = "escalating"        # Handling an escalation


@dataclass
class PMConfig:
    """Configuration for the PM Agent."""
    # Paths
    project_root: Path
    data_dir: Path

    # Execution settings
    max_concurrent_tasks: int = 1          # How many tasks to run in parallel
    task_timeout_seconds: int = 600        # Max time per task
    max_task_attempts: int = 3             # Retries before escalation

    # Cycle settings
    idle_sleep_seconds: int = 60           # Sleep when nothing to do
    active_sleep_seconds: int = 5          # Sleep between active cycles

    # Review settings
    auto_approve_threshold: float = 0.9    # Auto-approve if EGO confidence > this

    # Safety
    require_human_approval_for: List[str] = None  # Task types needing approval

    def __post_init__(self):
        if self.require_human_approval_for is None:
            self.require_human_approval_for = [
                "delete_files",
                "modify_config",
                "external_api",
                "database_migration",
            ]


@dataclass
class CycleResult:
    """Result of a single PM cycle."""
    state: PMAgentState
    action_taken: str
    task_id: Optional[str] = None
    goal_id: Optional[str] = None
    success: bool = True
    error: Optional[str] = None
    sleep_duration: int = 60


class PMAgent:
    """
    The autonomous Project Manager Agent.

    This agent runs continuously, managing projects by:
    1. Receiving high-level goals from humans
    2. Using EGO to break goals into concrete tasks
    3. Delegating coding tasks to Claude Code
    4. Reviewing and validating completed work
    5. Escalating when stuck or uncertain

    The PM never writes code itself - it orchestrates.
    """

    def __init__(
        self,
        config: PMConfig,
        task_queue: TaskQueue,
        claude_code: ClaudeCodeTool,
        ego: Optional[Any] = None,
        superego: Optional[Any] = None,
        memory: Optional[Any] = None,
        fallback_coder: Optional[Any] = None,
        notification_callback: Optional[Callable] = None,
        notification_manager: Optional[NotificationManager] = None,
        logger: Optional[Any] = None,
    ):
        """
        Initialize the PM Agent.

        Args:
            config: Agent configuration
            task_queue: Persistent task management
            claude_code: Tool for delegating to Claude Code
            ego: EGO model for planning and review (optional initially)
            superego: Safety layer for validating actions
            memory: Project memory store
            fallback_coder: Local coder for when Claude Code unavailable
            notification_callback: Function to call for human notifications (deprecated, use notification_manager)
            notification_manager: NotificationManager instance for multi-channel notifications
            logger: PMLogger instance for structured logging and thought tracking
        """
        self.config = config
        self.task_queue = task_queue
        self.claude_code = claude_code
        self.ego = ego
        self.superego = superego
        self.memory = memory
        self.fallback_coder = fallback_coder
        self.logger = logger

        # Initialize EGO adapter if EGO is available
        self.ego_adapter = PMEgoAdapter(ego) if ego else None

        # Initialize GoalAnalyzer for intelligent task breakdown
        self.goal_analyzer = GoalAnalyzer()

        # Notification system
        self.notification_manager = notification_manager or NotificationManager()
        # Backward compatibility: support legacy notification_callback
        self._legacy_notify = notification_callback or self._default_notify

        # State
        self.state = PMAgentState.IDLE
        self.current_task: Optional[Task] = None
        self.cycle_count = 0
        self.running = False

        # Stats
        self.stats = {
            "cycles": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "escalations": 0,
            "started_at": None,
        }

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def run(self) -> None:
        """
        Main execution loop.

        Runs continuously until stopped.
        """
        self.running = True
        self.stats["started_at"] = datetime.now()

        logger.info("PM Agent starting...")

        while self.running:
            try:
                result = await self.run_cycle()

                self.cycle_count += 1
                self.stats["cycles"] = self.cycle_count

                # Log cycle result
                logger.info(
                    f"Cycle {self.cycle_count}: {result.state.value} - {result.action_taken}"
                )

                # Sleep before next cycle
                await asyncio.sleep(result.sleep_duration)

            except Exception as e:
                logger.error(f"Error in PM cycle: {e}", exc_info=True)

                # Notify about critical error
                self.notification_manager.notify_error(
                    f"PM Agent cycle error: {str(e)}",
                    {
                        "cycle": self.cycle_count,
                        "state": self.state.value,
                        "current_task": self.current_task.id if self.current_task else "None",
                    }
                )

                await asyncio.sleep(self.config.idle_sleep_seconds)

        logger.info("PM Agent stopped.")

    def stop(self) -> None:
        """Stop the agent gracefully."""
        self.running = False

    async def run_cycle(self) -> CycleResult:
        """
        Execute one PM cycle.

        This is the core logic of the PM Agent.
        """

        # =====================================================================
        # Step 1: Check for pending escalations (priority)
        # =====================================================================

        pending_escalations = self.task_queue.get_pending_escalations()
        if pending_escalations:
            # Don't proceed until escalations are resolved
            self.state = PMAgentState.ESCALATING
            return CycleResult(
                state=self.state,
                action_taken=f"Waiting on {len(pending_escalations)} escalation(s)",
                sleep_duration=self.config.idle_sleep_seconds,
            )

        # =====================================================================
        # Step 2: Check for goals that need planning
        # =====================================================================

        goals_to_plan = self._get_goals_needing_planning()
        if goals_to_plan:
            goal = goals_to_plan[0]
            self.state = PMAgentState.PLANNING

            try:
                await self._plan_goal(goal)
                return CycleResult(
                    state=self.state,
                    action_taken=f"Planned goal: {goal.description[:50]}",
                    goal_id=goal.id,
                    sleep_duration=self.config.active_sleep_seconds,
                )
            except Exception as e:
                return CycleResult(
                    state=self.state,
                    action_taken=f"Failed to plan goal: {e}",
                    goal_id=goal.id,
                    success=False,
                    error=str(e),
                    sleep_duration=self.config.idle_sleep_seconds,
                )

        # =====================================================================
        # Step 3: Check for tasks waiting for review
        # =====================================================================

        tasks_to_review = self._get_tasks_needing_review()
        if tasks_to_review:
            task = tasks_to_review[0]
            self.state = PMAgentState.REVIEWING

            try:
                review_result = await self._review_task(task)
                return CycleResult(
                    state=self.state,
                    action_taken=f"Reviewed task: {review_result}",
                    task_id=task.id,
                    sleep_duration=self.config.active_sleep_seconds,
                )
            except Exception as e:
                return CycleResult(
                    state=self.state,
                    action_taken=f"Failed to review task: {e}",
                    task_id=task.id,
                    success=False,
                    error=str(e),
                    sleep_duration=self.config.idle_sleep_seconds,
                )

        # =====================================================================
        # Step 4: Get next task to work on
        # =====================================================================

        next_task = self.task_queue.get_next_task()

        if not next_task:
            # Nothing to do
            self.state = PMAgentState.IDLE
            return CycleResult(
                state=self.state,
                action_taken="No tasks available",
                sleep_duration=self.config.idle_sleep_seconds,
            )

        # =====================================================================
        # Step 5: Execute the task
        # =====================================================================

        self.state = PMAgentState.DELEGATING
        self.current_task = next_task

        try:
            result = await self._execute_task(next_task)

            self.current_task = None

            if result.success:
                self.stats["tasks_completed"] += 1
            else:
                self.stats["tasks_failed"] += 1

            return CycleResult(
                state=self.state,
                action_taken=f"Executed task: {'success' if result.success else 'failed'}",
                task_id=next_task.id,
                success=result.success,
                error=result.error_message if not result.success else None,
                sleep_duration=self.config.active_sleep_seconds,
            )

        except ClaudeCodeCreditsExhaustedError:
            # Fall back to local coder or escalate
            if self.fallback_coder:
                logger.warning("Claude Code credits exhausted, using fallback coder")
                return await self._execute_with_fallback(next_task)
            else:
                self._escalate_task(next_task, "Claude Code credits exhausted")
                self.stats["escalations"] += 1
                return CycleResult(
                    state=PMAgentState.ESCALATING,
                    action_taken="Escalated: Credits exhausted, no fallback",
                    task_id=next_task.id,
                    success=False,
                    error="Credits exhausted",
                    sleep_duration=self.config.idle_sleep_seconds,
                )

    # =========================================================================
    # Planning
    # =========================================================================

    def _get_goals_needing_planning(self) -> List[Goal]:
        """Get goals that haven't been broken into tasks yet."""
        active_goals = self.task_queue.get_active_goals()
        return [g for g in active_goals if g.status == GoalStatus.PENDING]

    async def _plan_goal(self, goal: Goal) -> List[Task]:
        """
        Break a goal into concrete tasks using GoalAnalyzer and optionally EGO.

        This is where the PM's intelligence matters. The GoalAnalyzer provides
        intelligent task breakdown based on complexity analysis, then EGO can
        optionally enhance it with project-specific context.
        """
        logger.info(f"Planning goal: {goal.description}")

        # Update goal status
        self.task_queue.update_goal_status(goal.id, GoalStatus.PLANNING)

        # Get project context
        project = self.task_queue.get_project(goal.project_id)
        project_structure = self._get_project_structure(Path(project.root_path))

        # Get relevant context from memory
        memory_context = ""
        if self.memory:
            memory_context = self.memory.get_relevant_context(
                task_description=goal.description,
                project_id=goal.project_id,
            )

        # Use GoalAnalyzer for intelligent breakdown
        analysis = self.goal_analyzer.analyze_goal(goal.description)

        logger.info(
            f"Goal analysis: complexity={analysis.complexity.value}, "
            f"subtasks={len(analysis.subtasks)}, "
            f"estimated_time={analysis.estimated_total_minutes}min"
        )

        # For MEDIUM+ complexity, use GoalAnalyzer's breakdown
        # For LOW complexity or when EGO is available and complexity is HIGH+,
        # optionally enhance with EGO
        if analysis.complexity in (ComplexityLevel.HIGH, ComplexityLevel.COMPLEX) and self.ego:
            # Use EGO for complex goals with GoalAnalyzer insights
            tasks = await self._plan_with_ego(
                goal, project, project_structure, memory_context, analysis
            )
        elif analysis.complexity != ComplexityLevel.LOW and len(analysis.subtasks) > 1:
            # Use GoalAnalyzer breakdown directly
            tasks = self._plan_with_analyzer(goal, analysis)
        else:
            # Simple single-task fallback for LOW complexity
            tasks = [Task(
                id="",
                goal_id=goal.id,
                description=goal.description,
                priority=goal.priority,
                acceptance_criteria=[f"Goal '{goal.description}' is achieved"],
            )]

        # Validate and fix dependencies
        dependencies = self.goal_analyzer.identify_dependencies(
            [s for s in analysis.subtasks]
        )

        # Add tasks to queue
        task_ids = self.task_queue.add_tasks_batch(tasks)

        # Update goal status
        self.task_queue.update_goal_status(goal.id, GoalStatus.IN_PROGRESS)

        logger.info(
            f"Created {len(tasks)} tasks for goal {goal.id} "
            f"(complexity: {analysis.complexity.value})"
        )

        return tasks

    def _plan_with_analyzer(self, goal: Goal, analysis) -> List[Task]:
        """
        Create tasks from GoalAnalyzer breakdown.

        Args:
            goal: The goal being planned
            analysis: GoalAnalysis from GoalAnalyzer

        Returns:
            List of Task objects
        """
        tasks = []

        for i, subtask in enumerate(analysis.subtasks):
            # Convert dependency indices to task references (will be filled later)
            # For now, store as empty since tasks don't have IDs yet
            task = Task(
                id="",  # Will be assigned by task queue
                goal_id=goal.id,
                description=subtask.description,
                priority=self._parse_priority(subtask.priority),
                acceptance_criteria=subtask.acceptance_criteria,
                constraints=[],
                context_files=[],
                dependencies=[],  # Dependencies will be handled by task queue
            )
            tasks.append(task)

        # Log the analysis reasoning
        logger.info(f"GoalAnalyzer reasoning: {analysis.reasoning}")
        if analysis.risks:
            logger.warning(f"Identified risks: {', '.join(analysis.risks)}")

        return tasks

    async def _plan_with_ego(
        self,
        goal: Goal,
        project: Project,
        project_structure: str,
        memory_context: str,
        analyzer_analysis = None,
    ) -> List[Task]:
        """
        Use EGO to intelligently break down a goal.

        Can optionally use GoalAnalyzer insights to guide EGO's planning.
        """

        # Build analysis context if available
        analyzer_context = ""
        if analyzer_analysis:
            analyzer_context = f"""
# GoalAnalyzer Insights
Complexity: {analyzer_analysis.complexity.value}
Estimated Time: {analyzer_analysis.estimated_total_minutes} minutes
Suggested Breakdown: {len(analyzer_analysis.subtasks)} subtasks
Key Entities: {', '.join(analyzer_analysis.key_entities) if analyzer_analysis.key_entities else 'None'}
Risks: {', '.join(analyzer_analysis.risks) if analyzer_analysis.risks else 'None'}

Suggested Subtasks:
{chr(10).join([f"{i+1}. {s.description}" for i, s in enumerate(analyzer_analysis.subtasks)])}
"""

        planning_prompt = f"""You are the planning component of an autonomous PM Agent.

# Goal to Accomplish
{goal.description}

# Project Information
Name: {project.name}
Path: {project.root_path}

# Project Structure
{project_structure}

# Relevant Past Context
{memory_context}

{analyzer_context}

# Your Task
Break this goal into concrete, independent coding tasks that can be delegated to Claude Code.

Each task should be:
1. Specific and actionable
2. Completable in 5-30 minutes
3. Independently testable
4. Clear about what files are involved

Output as JSON:
{{
    "analysis": "Brief analysis of what needs to be done",
    "tasks": [
        {{
            "description": "Clear, specific task description",
            "context_files": ["path/to/relevant/file.py"],
            "constraints": ["Don't modify X", "Follow Y pattern"],
            "acceptance_criteria": ["Test X passes", "Function Y exists"],
            "priority": "high|medium|low",
            "dependencies": []  // Indices of tasks that must complete first
        }}
    ],
    "execution_strategy": "parallel|sequential|mixed",
    "risks": ["Potential risk 1", "Potential risk 2"],
    "estimated_total_time_minutes": 30
}}
"""

        response = self.ego.generate(
            prompt=planning_prompt,
            role="planner",
        )

        # Parse response
        try:
            # Extract JSON from response (might have markdown)
            json_str = self._extract_json(response)
            plan = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse EGO planning response: {e}")
            # Fall back to single task
            return [Task(
                id="",
                goal_id=goal.id,
                description=goal.description,
                priority=goal.priority,
            )]

        # Convert to Task objects
        tasks = []
        for i, t in enumerate(plan.get("tasks", [])):
            # Handle dependencies (convert indices to task IDs)
            deps = []
            for dep_idx in t.get("dependencies", []):
                if 0 <= dep_idx < len(tasks):
                    deps.append(tasks[dep_idx].id)

            task = Task(
                id="",  # Will be assigned by task queue
                goal_id=goal.id,
                description=t["description"],
                priority=self._parse_priority(t.get("priority", "medium")),
                context_files=t.get("context_files", []),
                constraints=t.get("constraints", []),
                acceptance_criteria=t.get("acceptance_criteria", []),
                dependencies=deps,
            )
            tasks.append(task)

        # Store planning decision in memory
        if self.memory and self.logger:
            try:
                self.memory.store_decision(
                    task_id=f"goal_{goal.id}",
                    decision_type="goal_planning",
                    description=f"Broke goal into {len(tasks)} tasks",
                    reasoning=plan.get("analysis", ""),
                    project_id=goal.project_id,
                    goal_id=goal.id,
                    alternatives=[],
                    confidence=0.7,
                )
            except Exception as e:
                logger.warning(f"Failed to store planning decision in memory: {e}")

        return tasks

    # =========================================================================
    # Task Execution
    # =========================================================================

    async def _execute_task(self, task: Task) -> ClaudeCodeResult:
        """Execute a task by delegating to Claude Code."""

        logger.info(f"Executing task: {task.description[:50]}...")

        # Update status
        self.task_queue.update_task_status(
            task.id,
            TaskStatus.IN_PROGRESS,
            "Delegating to Claude Code"
        )

        # Safety check
        if self.superego:
            safety_result = self.superego.check_task(task)
            if not safety_result.is_approved:
                self._escalate_task(task, f"Safety check failed: {safety_result.reason}")
                return ClaudeCodeResult(
                    success=False,
                    error_message=f"Blocked by safety: {safety_result.reason}",
                    error_type="safety_block",
                )

        # Build Claude Code task
        cc_task = ClaudeCodeTask(
            description=task.description,
            working_directory=self.config.project_root,
            context_files=task.context_files,
            constraints=task.constraints,
            acceptance_criteria=task.acceptance_criteria,
            task_id=task.id,
            max_turns=50,
            timeout_seconds=self.config.task_timeout_seconds,
        )

        # Add project context from memory
        if self.memory:
            try:
                project = self.task_queue.get_project(
                    self.task_queue.get_goal(task.goal_id).project_id
                )
                memory_context = self.memory.get_relevant_context(
                    task_description=task.description,
                    project_id=project.id if project else "unknown",
                )
                if memory_context:
                    cc_task.context_summary = memory_context
            except Exception as e:
                logger.warning(f"Failed to get memory context for task: {e}")

        # Execute
        result = self.claude_code.execute_task(cc_task)

        # Record result
        self.task_queue.record_task_result(
            task_id=task.id,
            success=result.success,
            summary=result.summary,
            files_modified=result.files_modified,
            files_created=result.files_created,
            error_message=result.error_message or "",
        )

        # Record in project memory
        if self.memory:
            try:
                goal = self.task_queue.get_goal(task.goal_id)
                self.memory.record_task_execution(
                    task_id=task.id,
                    goal_id=task.goal_id,
                    project_id=goal.project_id if goal else "unknown",
                    description=task.description,
                    success=result.success,
                    files_modified=result.files_modified,
                    files_created=result.files_created,
                    error_type=result.error_type if hasattr(result, 'error_type') else None,
                    error_message=result.error_message,
                )
            except Exception as e:
                logger.warning(f"Failed to record task execution in memory: {e}")

        return result

    async def _execute_with_fallback(self, task: Task) -> CycleResult:
        """Execute task with local fallback coder."""
        if not self.fallback_coder:
            raise ValueError("No fallback coder configured")

        logger.info(f"Using fallback coder for task: {task.id}")

        # Change assignment
        task.assigned_to = "local_coder"

        # Execute with fallback
        result = self.fallback_coder.execute(task)

        # Record result
        self.task_queue.record_task_result(
            task_id=task.id,
            success=result.success,
            summary=result.summary,
            files_modified=result.files_modified,
            files_created=result.files_created,
            error_message=result.error_message or "",
        )

        return CycleResult(
            state=PMAgentState.DELEGATING,
            action_taken=f"Executed with fallback: {'success' if result.success else 'failed'}",
            task_id=task.id,
            success=result.success,
            sleep_duration=self.config.active_sleep_seconds,
        )

    # =========================================================================
    # Review
    # =========================================================================

    def _get_tasks_needing_review(self) -> List[Task]:
        """Get tasks that are waiting for PM review."""
        # Query tasks with WAITING_REVIEW status
        with self.task_queue._get_conn() as conn:
            rows = conn.execute('''
                SELECT * FROM tasks WHERE status = ?
                ORDER BY priority, created_at
            ''', (TaskStatus.WAITING_REVIEW.value,)).fetchall()
            return [self.task_queue._row_to_task(conn, row) for row in rows]

    async def _review_task(self, task: Task) -> str:
        """Review a completed task."""

        logger.info(f"Reviewing task: {task.id}")

        if not self.ego:
            # Without EGO, auto-approve
            self.task_queue.update_task_status(task.id, TaskStatus.COMPLETED)
            self._check_goal_completion(task.goal_id)
            return "auto_approved (no EGO)"

        # Get the changes made
        changes = self._get_file_changes(
            task.files_modified + task.files_created
        )

        # Use EGO to review
        review_prompt = f"""You are the review component of an autonomous PM Agent.

# Original Task
{task.description}

# Acceptance Criteria
{json.dumps(task.acceptance_criteria, indent=2)}

# Changes Made
{changes}

# Result Summary
{task.result_summary}

# Review Task
Evaluate whether the changes correctly and completely address the task.

Output as JSON:
{{
    "approved": true/false,
    "confidence": 0.0-1.0,
    "criteria_met": {{"criterion": true/false}},
    "issues": ["issue 1", "issue 2"],
    "feedback": "Specific feedback if revision needed",
    "recommendation": "approve|revise|escalate"
}}
"""

        response = self.ego.generate(prompt=review_prompt, role="reviewer")

        try:
            json_str = self._extract_json(response)
            review = json.loads(json_str)
        except json.JSONDecodeError:
            logger.error("Failed to parse review response")
            # Conservative: escalate if can't parse
            self._escalate_task(task, "Could not parse review response")
            return "escalated (parse failure)"

        # Decide based on review
        if review.get("approved") and review.get("confidence", 0) >= self.config.auto_approve_threshold:
            self.task_queue.update_task_status(task.id, TaskStatus.COMPLETED)
            self._check_goal_completion(task.goal_id)
            return "approved"

        elif review.get("recommendation") == "escalate":
            self._escalate_task(task, review.get("feedback", "Review recommended escalation"))
            return "escalated"

        elif review.get("recommendation") == "revise" or not review.get("approved"):
            if task.attempt_count < task.max_attempts:
                # Add feedback for next attempt
                task.constraints.append(f"Previous feedback: {review.get('feedback', 'Needs revision')}")
                self.task_queue.mark_task_for_retry(task.id, review.get("feedback", ""))
                return "needs_revision"
            else:
                self._escalate_task(task, f"Max attempts reached. Last feedback: {review.get('feedback')}")
                return "escalated (max attempts)"

        # Default: approve with lower confidence
        self.task_queue.update_task_status(task.id, TaskStatus.COMPLETED)
        self._check_goal_completion(task.goal_id)
        return "approved (default)"

    def _check_goal_completion(self, goal_id: str) -> None:
        """Check if all tasks for a goal are complete."""
        tasks = self.task_queue.get_tasks_for_goal(goal_id)

        all_complete = all(t.status == TaskStatus.COMPLETED for t in tasks)
        any_failed = any(t.status == TaskStatus.FAILED for t in tasks)

        if all_complete:
            self.task_queue.update_goal_status(goal_id, GoalStatus.COMPLETED)
            goal = self.task_queue.get_goal(goal_id)
            logger.info(f"Goal {goal_id} completed!")

            # Calculate duration if possible
            duration_minutes = None
            if goal and goal.created_at and goal.completed_at:
                duration = goal.completed_at - goal.created_at
                duration_minutes = duration.total_seconds() / 60

            # Notify about goal completion
            if goal:
                self.notification_manager.notify_goal_completed(
                    goal=goal,
                    task_count=len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
                    duration_minutes=duration_minutes,
                )

        elif any_failed:
            # Check if all non-failed are complete
            non_failed = [t for t in tasks if t.status != TaskStatus.FAILED]
            if all(t.status == TaskStatus.COMPLETED for t in non_failed):
                self.task_queue.update_goal_status(goal_id, GoalStatus.COMPLETED)
                goal = self.task_queue.get_goal(goal_id)
                logger.info(f"Goal {goal_id} completed with some failed tasks")

                # Notify about partial completion
                if goal:
                    self.notification_manager.notify_milestone(
                        f"Goal completed with some failures: {goal.description}",
                        {
                            "completed_tasks": len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
                            "failed_tasks": len([t for t in tasks if t.status == TaskStatus.FAILED]),
                            "total_tasks": len(tasks),
                        }
                    )

    # =========================================================================
    # Escalation
    # =========================================================================

    def _escalate_task(self, task: Task, reason: str) -> str:
        """Escalate a task to human oversight."""
        logger.warning(f"Escalating task {task.id}: {reason}")

        escalation_id = self.task_queue.create_escalation(task.id, reason)

        # Get the escalation object and related goal
        escalation = None
        goal = None
        with self.task_queue._get_conn() as conn:
            row = conn.execute(
                'SELECT * FROM escalations WHERE id = ?',
                (escalation_id,)
            ).fetchone()
            if row:
                escalation = Escalation(
                    id=row['id'],
                    task_id=row['task_id'],
                    reason=row['reason'],
                    status=row['status'],
                    human_response=row['human_response'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
                )

        if task.goal_id:
            goal = self.task_queue.get_goal(task.goal_id)

        # Notify via new notification system
        if escalation:
            self.notification_manager.notify_escalation(escalation, task, goal)

        # Also call legacy notification callback for backward compatibility
        self._legacy_notify(
            f"Task Escalated: {task.description[:50]}",
            f"Reason: {reason}\n\nTask ID: {task.id}\nEscalation ID: {escalation_id}"
        )

        self.stats["escalations"] += 1

        return escalation_id

    # =========================================================================
    # Utilities
    # =========================================================================

    def _get_project_structure(self, root: Path, max_depth: int = 3) -> str:
        """Get a string representation of project structure."""
        lines = []

        def walk(path: Path, depth: int, prefix: str = ""):
            if depth > max_depth:
                return

            try:
                items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            except PermissionError:
                return

            for i, item in enumerate(items):
                if item.name.startswith('.') or item.name == '__pycache__':
                    continue
                if item.name == 'node_modules' or item.name == 'venv':
                    continue

                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                lines.append(f"{prefix}{current_prefix}{item.name}")

                if item.is_dir():
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    walk(item, depth + 1, next_prefix)

        lines.append(str(root))
        walk(root, 0)

        return "\n".join(lines[:100])  # Limit output

    def _get_file_changes(self, file_paths: List[str]) -> str:
        """Get the content of changed files."""
        changes = []
        for path in file_paths[:5]:  # Limit to 5 files
            full_path = self.config.project_root / path
            if full_path.exists():
                try:
                    content = full_path.read_text()[:2000]  # Limit content
                    changes.append(f"### {path}\n```\n{content}\n```")
                except Exception:
                    changes.append(f"### {path}\n(Could not read)")
        return "\n\n".join(changes)

    def _format_memories(self, memories: List[Any]) -> str:
        """Format memories for prompts."""
        if not memories:
            return "No relevant past context."
        return "\n".join([f"- {m.content if hasattr(m, 'content') else str(m)}" for m in memories])

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might have markdown."""
        # Try to find JSON block
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            return text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            return text[start:end].strip()
        elif "{" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            return text[start:end]
        return text

    def _parse_priority(self, priority_str: str) -> TaskPriority:
        """Parse priority string to enum."""
        mapping = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "medium": TaskPriority.MEDIUM,
            "low": TaskPriority.LOW,
            "backlog": TaskPriority.BACKLOG,
        }
        return mapping.get(priority_str.lower(), TaskPriority.MEDIUM)

    def _default_notify(self, title: str, message: str) -> None:
        """Default notification - just log."""
        logger.info(f"NOTIFICATION: {title}\n{message}")

    # =========================================================================
    # Public Interface
    # =========================================================================

    def add_goal(self, description: str, project_id: str, priority: str = "medium") -> str:
        """Add a new goal for the PM to work on."""
        goal = Goal(
            id="",
            description=description,
            project_id=project_id,
            priority=self._parse_priority(priority),
            created_by="human",
        )
        return self.task_queue.add_goal(goal)

    def get_status(self) -> Dict[str, Any]:
        """Get current PM status."""
        return {
            "state": self.state.value,
            "current_task": self.current_task.id if self.current_task else None,
            "cycle_count": self.cycle_count,
            "stats": self.stats,
            "queue_stats": self.task_queue.get_stats(),
        }
