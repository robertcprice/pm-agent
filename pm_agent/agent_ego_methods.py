"""
Additional EGO integration methods for PMAgent.
These should be added to the PMAgent class in agent.py
"""

async def _plan_with_ego_adapter(
    self,
    goal,
    project,
    project_structure,
    memory_context,
):
    """
    Use EGO adapter to intelligently break down a goal.

    This is the new preferred method that uses the PMEgoAdapter.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"[PMAgent] Using EGO adapter for planning")

    # Build project info dict
    project_info = {
        "name": project.name,
        "path": project.root_path,
    }

    # Convert memory context string to list (simplified)
    memories = [{"content": memory_context}] if memory_context else []

    # Use EGO adapter to plan
    try:
        result = self.ego_adapter.plan_tasks(
            goal=goal,
            project_info=project_info,
            project_structure=project_structure,
            memories=memories
        )

        # Store planning decision in memory
        if self.memory and self.logger:
            try:
                self.memory.store_decision(
                    task_id=f"goal_{goal.id}",
                    decision_type="goal_planning",
                    description=f"Broke goal into {len(result.tasks)} tasks",
                    reasoning=result.analysis,
                    project_id=goal.project_id,
                    goal_id=goal.id,
                    alternatives=[],
                    confidence=result.confidence,
                )
            except Exception as e:
                logger.warning(f"Failed to store planning decision in memory: {e}")

        logger.info(
            f"[PMAgent] EGO adapter created {len(result.tasks)} tasks "
            f"(confidence={result.confidence:.2f})"
        )

        return result.tasks

    except Exception as e:
        logger.error(f"[PMAgent] EGO adapter planning failed: {e}")
        from .task_queue import Task
        # Fallback to single task
        return [Task(
            id="",
            goal_id=goal.id,
            description=goal.description,
            priority=goal.priority,
            acceptance_criteria=[f"Goal '{goal.description}' is achieved"],
        )]


async def _review_task_with_ego_adapter(self, task):
    """Review a completed task using EGO adapter."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"[PMAgent] Reviewing task with EGO adapter: {task.id}")

    # Get the changes made
    changes_files = task.files_modified + task.files_created

    # Get Claude Code result from task
    # Reconstruct result from task data
    from ..tools.claude_code import ClaudeCodeResult
    result = ClaudeCodeResult(
        success=task.status.value not in ["failed", "escalated"],
        files_modified=task.files_modified,
        files_created=task.files_created,
        summary=task.result_summary,
        error_message=task.error_message,
        duration_seconds=0.0,  # Not stored in task
        turns_used=0,  # Not stored in task
    )

    # Use EGO adapter to review
    try:
        review = self.ego_adapter.review_result(task, result)

        logger.info(
            f"[PMAgent] EGO adapter review: {review.recommendation}, "
            f"confidence={review.confidence:.2f}"
        )

        # Process recommendation
        from .task_queue import TaskStatus
        if review.recommendation == "approve" and review.confidence >= self.config.auto_approve_threshold:
            self.task_queue.update_task_status(task.id, TaskStatus.COMPLETED)
            self._check_goal_completion(task.goal_id)
            return "approved"

        elif review.recommendation == "escalate":
            self._escalate_task(task, review.feedback)
            return "escalated"

        elif review.recommendation == "revise":
            if task.attempt_count < task.max_attempts:
                task.constraints.append(f"Previous feedback: {review.feedback}")
                self.task_queue.mark_task_for_retry(task.id, review.feedback)
                return "needs_revision"
            else:
                self._escalate_task(task, f"Max attempts reached. Last feedback: {review.feedback}")
                return "escalated (max attempts)"

        # Default: approve with lower confidence
        self.task_queue.update_task_status(task.id, TaskStatus.COMPLETED)
        self._check_goal_completion(task.goal_id)
        return "approved (default)"

    except Exception as e:
        logger.error(f"[PMAgent] EGO adapter review failed: {e}")
        # Conservative fallback: escalate
        self._escalate_task(task, f"Review failed: {e}")
        return "escalated (review failure)"
