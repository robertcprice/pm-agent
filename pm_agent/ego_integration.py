"""
EGO Integration for PM Agent.

This module provides the bridge between the PM Agent and the EGO model,
enabling the PM to leverage EGO's reasoning capabilities for:
- Breaking goals into concrete tasks (planning)
- Evaluating completed work (review)
- Deciding when to escalate (judgment)

The integration maintains loose coupling through a clean adapter interface.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .task_queue import Task, Goal, TaskPriority
from ..tools.claude_code import ClaudeCodeResult

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Templates
# =============================================================================

PLANNING_PROMPT_TEMPLATE = """You are the planning component of an autonomous PM Agent.

# Goal to Accomplish
{goal_description}

# Project Information
Name: {project_name}
Path: {project_path}

# Project Structure
{project_structure}

# Relevant Past Context
{memories}

# Your Task
Break this goal into concrete, independent coding tasks that can be delegated to Claude Code.

Each task should be:
1. Specific and actionable
2. Completable in 5-30 minutes by an expert coder
3. Independently testable
4. Clear about what files are involved
5. Have clear acceptance criteria

Guidelines:
- Prefer smaller, focused tasks over large ones
- Identify dependencies between tasks
- Consider project patterns and conventions
- Think about testing and validation
- Flag any safety-critical operations

Output as JSON:
{{
    "analysis": "Brief analysis of what needs to be done and approach",
    "tasks": [
        {{
            "description": "Clear, specific task description",
            "context_files": ["path/to/relevant/file.py"],
            "constraints": ["Don't modify X", "Follow Y pattern"],
            "acceptance_criteria": ["Test X passes", "Function Y exists"],
            "priority": "critical|high|medium|low|backlog",
            "dependencies": [],  // Indices of tasks that must complete first
            "estimated_minutes": 15,
            "safety_notes": "Any safety considerations"
        }}
    ],
    "execution_strategy": "parallel|sequential|mixed",
    "risks": ["Potential risk 1", "Potential risk 2"],
    "estimated_total_time_minutes": 30,
    "confidence": 0.85  // 0.0-1.0 confidence in this plan
}}
"""


REVIEW_PROMPT_TEMPLATE = """You are the review component of an autonomous PM Agent.

# Original Task
{task_description}

# Acceptance Criteria
{acceptance_criteria}

# Changes Made
{changes}

# Result Summary
{result_summary}

# Execution Metrics
- Success: {success}
- Duration: {duration_seconds}s
- Turns used: {turns_used}
- Files modified: {files_modified_count}
- Files created: {files_created_count}

# Your Task
Evaluate whether the changes correctly and completely address the task.

Consider:
1. Do the changes meet all acceptance criteria?
2. Is the implementation correct and complete?
3. Are there any obvious bugs or issues?
4. Does it follow the project's patterns and conventions?
5. Are there any safety or quality concerns?

Output as JSON:
{{
    "approved": true/false,
    "confidence": 0.85,  // 0.0-1.0 confidence in this review
    "criteria_met": {{
        "criterion 1": {{"met": true, "notes": "explanation"}},
        "criterion 2": {{"met": false, "notes": "explanation"}}
    }},
    "issues": [
        {{"severity": "critical|high|medium|low", "description": "issue description"}}
    ],
    "strengths": ["positive aspect 1", "positive aspect 2"],
    "feedback": "Specific, actionable feedback if revision needed",
    "recommendation": "approve|revise|escalate",
    "reasoning": "Explain your recommendation"
}}
"""


ESCALATION_PROMPT_TEMPLATE = """You are the escalation judgment component of an autonomous PM Agent.

# Task Information
Description: {task_description}
Priority: {task_priority}
Attempt: {attempt_count}/{max_attempts}

# History
{attempt_history}

# Current Situation
Last error: {last_error}
Last feedback: {last_feedback}

# Your Task
Decide whether this task should be escalated to human oversight or retried.

Escalate if:
- Task is safety-critical and showing persistent issues
- Fundamental approach is wrong and needs human guidance
- Technical dependencies or blockers require human intervention
- Task complexity exceeds autonomous capability

Retry if:
- Error is transient or recoverable
- Feedback is clear and actionable
- Previous attempts show progress
- One more attempt has good chance of success

Output as JSON:
{{
    "should_escalate": true/false,
    "confidence": 0.85,  // 0.0-1.0 confidence in this decision
    "reasoning": "Detailed explanation of decision",
    "escalation_reason": "Brief reason for human (if escalating)",
    "retry_guidance": "Specific guidance for next attempt (if retrying)",
    "urgency": "critical|high|medium|low"
}}
"""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PlanningResult:
    """Result from EGO's task planning."""
    tasks: List[Task]
    analysis: str
    execution_strategy: str
    risks: List[str]
    estimated_total_minutes: int
    confidence: float
    raw_response: str


@dataclass
class ReviewResult:
    """Result from EGO's task review."""
    approved: bool
    confidence: float
    criteria_met: Dict[str, Dict[str, Any]]
    issues: List[Dict[str, str]]
    strengths: List[str]
    feedback: str
    recommendation: str  # "approve", "revise", "escalate"
    reasoning: str
    raw_response: str


@dataclass
class EscalationDecision:
    """Result from EGO's escalation judgment."""
    should_escalate: bool
    confidence: float
    reasoning: str
    escalation_reason: str
    retry_guidance: str
    urgency: str
    raw_response: str


# =============================================================================
# PM EGO Adapter
# =============================================================================

class PMEgoAdapter:
    """
    Adapter that bridges PM Agent and EGO model.

    This provides PM-specific interfaces to EGO while keeping the
    EGO model itself generic and reusable.
    """

    def __init__(self, ego_model):
        """
        Initialize the adapter.

        Args:
            ego_model: Instance of EgoModel from conch_dna.ego.model
        """
        self.ego = ego_model
        self.logger = logger

    def plan_tasks(
        self,
        goal: Goal,
        project_info: Dict[str, Any],
        project_structure: str,
        memories: List[Any]
    ) -> PlanningResult:
        """
        Use EGO to break a goal into concrete tasks.

        Args:
            goal: The goal to plan for
            project_info: Dict with project metadata (name, path, etc)
            project_structure: String representation of project structure
            memories: Relevant memories from project history

        Returns:
            PlanningResult with tasks and metadata
        """
        self.logger.info(f"[PMEgoAdapter] Planning tasks for goal: {goal.description[:50]}...")

        # Format memories
        memories_text = self._format_memories(memories)

        # Build prompt from template
        prompt = PLANNING_PROMPT_TEMPLATE.format(
            goal_description=goal.description,
            project_name=project_info.get("name", "Unknown"),
            project_path=project_info.get("path", "Unknown"),
            project_structure=project_structure,
            memories=memories_text
        )

        # Generate with EGO
        try:
            response = self.ego.generate(
                prompt=prompt,
                cycle_count=0,
                mood="focused",
                dominant_need="achievement"
            )
        except Exception as e:
            self.logger.error(f"[PMEgoAdapter] EGO generation failed: {e}")
            # Return fallback single task
            return self._fallback_single_task_plan(goal, str(e))

        # Parse response
        parsed = self._parse_json_from_response(response)

        if not parsed:
            self.logger.error("[PMEgoAdapter] Failed to parse planning response")
            return self._fallback_single_task_plan(goal, "JSON parsing failed")

        # Convert to Task objects
        tasks = self._convert_to_tasks(
            parsed.get("tasks", []),
            goal.id,
            goal.priority
        )

        result = PlanningResult(
            tasks=tasks,
            analysis=parsed.get("analysis", ""),
            execution_strategy=parsed.get("execution_strategy", "sequential"),
            risks=parsed.get("risks", []),
            estimated_total_minutes=parsed.get("estimated_total_time_minutes", 0),
            confidence=parsed.get("confidence", 0.5),
            raw_response=response
        )

        self.logger.info(
            f"[PMEgoAdapter] Planning complete: {len(tasks)} tasks, "
            f"confidence={result.confidence:.2f}"
        )

        return result

    def review_result(
        self,
        task: Task,
        result: ClaudeCodeResult
    ) -> ReviewResult:
        """
        Use EGO to review a completed task.

        Args:
            task: The task that was executed
            result: The result from Claude Code execution

        Returns:
            ReviewResult with approval decision and feedback
        """
        self.logger.info(f"[PMEgoAdapter] Reviewing task: {task.id}")

        # Format acceptance criteria
        criteria_text = json.dumps(task.acceptance_criteria, indent=2)

        # Build changes summary
        changes = self._format_file_changes(
            result.files_modified + result.files_created,
            max_files=5
        )

        # Build prompt
        prompt = REVIEW_PROMPT_TEMPLATE.format(
            task_description=task.description,
            acceptance_criteria=criteria_text,
            changes=changes,
            result_summary=result.summary,
            success=result.success,
            duration_seconds=result.duration_seconds,
            turns_used=result.turns_used,
            files_modified_count=len(result.files_modified),
            files_created_count=len(result.files_created)
        )

        # Generate with EGO
        try:
            response = self.ego.generate(
                prompt=prompt,
                cycle_count=0,
                mood="analytical",
                dominant_need="competence"
            )
        except Exception as e:
            self.logger.error(f"[PMEgoAdapter] EGO review failed: {e}")
            return self._fallback_review(task, result, str(e))

        # Parse response
        parsed = self._parse_json_from_response(response)

        if not parsed:
            self.logger.error("[PMEgoAdapter] Failed to parse review response")
            return self._fallback_review(task, result, "JSON parsing failed")

        review = ReviewResult(
            approved=parsed.get("approved", False),
            confidence=parsed.get("confidence", 0.5),
            criteria_met=parsed.get("criteria_met", {}),
            issues=parsed.get("issues", []),
            strengths=parsed.get("strengths", []),
            feedback=parsed.get("feedback", ""),
            recommendation=parsed.get("recommendation", "revise"),
            reasoning=parsed.get("reasoning", ""),
            raw_response=response
        )

        self.logger.info(
            f"[PMEgoAdapter] Review complete: {review.recommendation}, "
            f"confidence={review.confidence:.2f}"
        )

        return review

    def should_escalate(
        self,
        task: Task,
        attempt_history: List[Dict[str, Any]]
    ) -> EscalationDecision:
        """
        Use EGO judgment to decide if a task should be escalated.

        Args:
            task: The task with issues
            attempt_history: List of previous attempt summaries

        Returns:
            EscalationDecision with recommendation
        """
        self.logger.info(
            f"[PMEgoAdapter] Evaluating escalation for task: {task.id} "
            f"(attempt {task.attempt_count}/{task.max_attempts})"
        )

        # Format attempt history
        history_text = self._format_attempt_history(attempt_history)

        # Get last error and feedback
        last_error = task.error_message or "Unknown error"
        last_feedback = "No feedback available"
        if attempt_history:
            last_attempt = attempt_history[-1]
            last_feedback = last_attempt.get("feedback", "No feedback")

        # Build prompt
        prompt = ESCALATION_PROMPT_TEMPLATE.format(
            task_description=task.description,
            task_priority=task.priority.name,
            attempt_count=task.attempt_count,
            max_attempts=task.max_attempts,
            attempt_history=history_text,
            last_error=last_error,
            last_feedback=last_feedback
        )

        # Generate with EGO
        try:
            response = self.ego.generate(
                prompt=prompt,
                cycle_count=0,
                mood="concerned",
                dominant_need="safety"
            )
        except Exception as e:
            self.logger.error(f"[PMEgoAdapter] EGO escalation judgment failed: {e}")
            # Conservative fallback: escalate if max attempts reached
            return self._fallback_escalation_decision(task, str(e))

        # Parse response
        parsed = self._parse_json_from_response(response)

        if not parsed:
            self.logger.error("[PMEgoAdapter] Failed to parse escalation response")
            return self._fallback_escalation_decision(task, "JSON parsing failed")

        decision = EscalationDecision(
            should_escalate=parsed.get("should_escalate", task.attempt_count >= task.max_attempts),
            confidence=parsed.get("confidence", 0.5),
            reasoning=parsed.get("reasoning", ""),
            escalation_reason=parsed.get("escalation_reason", "Max attempts reached"),
            retry_guidance=parsed.get("retry_guidance", ""),
            urgency=parsed.get("urgency", "medium"),
            raw_response=response
        )

        self.logger.info(
            f"[PMEgoAdapter] Escalation decision: "
            f"{'ESCALATE' if decision.should_escalate else 'RETRY'}, "
            f"confidence={decision.confidence:.2f}"
        )

        return decision

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON from EGO response, handling markdown code blocks.

        Args:
            response: Raw response from EGO

        Returns:
            Parsed dict or None if parsing fails
        """
        # Try direct JSON parsing first
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                try:
                    return json.loads(response[start:end].strip())
                except json.JSONDecodeError:
                    pass

        # Try finding JSON object
        if "{" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            if end > start:
                try:
                    return json.loads(response[start:end])
                except json.JSONDecodeError:
                    pass

        return None

    def _convert_to_tasks(
        self,
        task_dicts: List[Dict[str, Any]],
        goal_id: str,
        default_priority: TaskPriority
    ) -> List[Task]:
        """
        Convert dict representations to Task objects.

        Args:
            task_dicts: List of task dictionaries from EGO
            goal_id: Parent goal ID
            default_priority: Priority to use if not specified

        Returns:
            List of Task objects
        """
        tasks = []

        for i, td in enumerate(task_dicts):
            # Parse priority
            priority_str = td.get("priority", "medium")
            priority = self._parse_priority(priority_str, default_priority)

            # Handle dependencies (convert indices to task IDs later)
            # For now, store as metadata
            dependencies = td.get("dependencies", [])

            task = Task(
                id="",  # Will be assigned by task queue
                goal_id=goal_id,
                description=td["description"],
                priority=priority,
                context_files=td.get("context_files", []),
                constraints=td.get("constraints", []),
                acceptance_criteria=td.get("acceptance_criteria", []),
                dependencies=dependencies  # Store raw indices for now
            )

            tasks.append(task)

        return tasks

    def _parse_priority(
        self,
        priority_str: str,
        default: TaskPriority
    ) -> TaskPriority:
        """Parse priority string to enum."""
        mapping = {
            "critical": TaskPriority.CRITICAL,
            "high": TaskPriority.HIGH,
            "medium": TaskPriority.MEDIUM,
            "low": TaskPriority.LOW,
            "backlog": TaskPriority.BACKLOG,
        }
        return mapping.get(priority_str.lower(), default)

    def _format_memories(self, memories: List[Any]) -> str:
        """Format memories for prompts."""
        if not memories:
            return "No relevant past context available."

        lines = []
        for m in memories[:5]:  # Limit to 5 most relevant
            if hasattr(m, 'content'):
                lines.append(f"- {m.content}")
            else:
                lines.append(f"- {str(m)}")

        return "\n".join(lines)

    def _format_file_changes(
        self,
        file_paths: List[str],
        max_files: int = 5
    ) -> str:
        """Format file changes for review prompt."""
        if not file_paths:
            return "No files were modified."

        changes = []
        for path in file_paths[:max_files]:
            changes.append(f"- {path}")

        if len(file_paths) > max_files:
            changes.append(f"... and {len(file_paths) - max_files} more files")

        return "\n".join(changes)

    def _format_attempt_history(
        self,
        history: List[Dict[str, Any]]
    ) -> str:
        """Format attempt history for escalation prompt."""
        if not history:
            return "No previous attempts."

        lines = []
        for i, attempt in enumerate(history, 1):
            lines.append(f"\nAttempt {i}:")
            lines.append(f"  Success: {attempt.get('success', False)}")
            lines.append(f"  Error: {attempt.get('error', 'None')}")
            lines.append(f"  Feedback: {attempt.get('feedback', 'None')}")

        return "\n".join(lines)

    # =========================================================================
    # Fallback Methods (when EGO fails)
    # =========================================================================

    def _fallback_single_task_plan(
        self,
        goal: Goal,
        error: str
    ) -> PlanningResult:
        """Fallback to single task when EGO planning fails."""
        self.logger.warning(f"[PMEgoAdapter] Using fallback plan: {error}")

        task = Task(
            id="",
            goal_id=goal.id,
            description=goal.description,
            priority=goal.priority,
            acceptance_criteria=[f"Goal '{goal.description}' is achieved"],
        )

        return PlanningResult(
            tasks=[task],
            analysis=f"Fallback plan due to: {error}",
            execution_strategy="sequential",
            risks=["Planning failed, using simple fallback"],
            estimated_total_minutes=30,
            confidence=0.3,
            raw_response=f"Error: {error}"
        )

    def _fallback_review(
        self,
        task: Task,
        result: ClaudeCodeResult,
        error: str
    ) -> ReviewResult:
        """Fallback review when EGO review fails."""
        self.logger.warning(f"[PMEgoAdapter] Using fallback review: {error}")

        # Conservative fallback: approve if successful, escalate if not
        if result.success:
            return ReviewResult(
                approved=True,
                confidence=0.5,
                criteria_met={},
                issues=[],
                strengths=["Task completed successfully"],
                feedback="",
                recommendation="approve",
                reasoning=f"Fallback approval due to: {error}",
                raw_response=f"Error: {error}"
            )
        else:
            return ReviewResult(
                approved=False,
                confidence=0.5,
                criteria_met={},
                issues=[{"severity": "high", "description": result.error_message or "Unknown error"}],
                strengths=[],
                feedback="Task failed, needs review",
                recommendation="escalate",
                reasoning=f"Fallback escalation due to: {error}",
                raw_response=f"Error: {error}"
            )

    def _fallback_escalation_decision(
        self,
        task: Task,
        error: str
    ) -> EscalationDecision:
        """Fallback escalation decision when EGO fails."""
        self.logger.warning(f"[PMEgoAdapter] Using fallback escalation: {error}")

        # Conservative: escalate if at max attempts
        should_escalate = task.attempt_count >= task.max_attempts

        return EscalationDecision(
            should_escalate=should_escalate,
            confidence=0.5,
            reasoning=f"Fallback decision due to: {error}",
            escalation_reason="Max attempts reached" if should_escalate else "",
            retry_guidance="Review previous errors and try again" if not should_escalate else "",
            urgency="medium",
            raw_response=f"Error: {error}"
        )
