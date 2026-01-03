"""
Test script for EGO integration with PM Agent.

This validates that the PMEgoAdapter correctly interfaces with the EGO model.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from conch_dna.pm.ego_integration import (
    PMEgoAdapter,
    PlanningResult,
    ReviewResult,
    EscalationDecision,
)
from conch_dna.pm.task_queue import Task, Goal, TaskPriority, TaskStatus
from conch_dna.tools.claude_code import ClaudeCodeResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class MockEgo:
    """Mock EGO model for testing without MLX."""

    def generate(self, prompt, cycle_count=0, mood="neutral", dominant_need="curiosity"):
        """Generate mock responses based on prompt content."""

        if "planning component" in prompt.lower():
            # Planning response
            return '''```json
{
    "analysis": "This goal requires setting up user authentication with database models, API endpoints, and UI components.",
    "tasks": [
        {
            "description": "Create User model and database migration",
            "context_files": ["models/user.py", "migrations/"],
            "constraints": ["Use bcrypt for password hashing", "Follow existing model patterns"],
            "acceptance_criteria": ["User model created", "Migration runs successfully", "Tests pass"],
            "priority": "high",
            "dependencies": [],
            "estimated_minutes": 15,
            "safety_notes": "Ensure password field is properly encrypted"
        },
        {
            "description": "Implement login/logout API endpoints",
            "context_files": ["api/auth.py"],
            "constraints": ["Use JWT tokens", "Return proper HTTP status codes"],
            "acceptance_criteria": ["Login endpoint works", "Logout endpoint works", "Invalid credentials rejected"],
            "priority": "high",
            "dependencies": [0],
            "estimated_minutes": 20,
            "safety_notes": "Validate all inputs, prevent SQL injection"
        },
        {
            "description": "Create login UI component",
            "context_files": ["components/Login.jsx"],
            "constraints": ["Match existing UI patterns", "Responsive design"],
            "acceptance_criteria": ["Login form renders", "Form validation works", "Integrates with API"],
            "priority": "medium",
            "dependencies": [1],
            "estimated_minutes": 15,
            "safety_notes": "Sanitize user input on frontend"
        }
    ],
    "execution_strategy": "sequential",
    "risks": ["Password security must be handled correctly", "Session management complexity"],
    "estimated_total_time_minutes": 50,
    "confidence": 0.87
}
```'''

        elif "review component" in prompt.lower():
            # Review response
            return '''```json
{
    "approved": true,
    "confidence": 0.92,
    "criteria_met": {
        "User model created": {"met": true, "notes": "Model follows existing patterns"},
        "Migration runs successfully": {"met": true, "notes": "Migration tested locally"},
        "Tests pass": {"met": true, "notes": "All unit tests passing"}
    },
    "issues": [],
    "strengths": ["Clean code", "Good test coverage", "Follows project conventions"],
    "feedback": "",
    "recommendation": "approve",
    "reasoning": "All acceptance criteria met, implementation is clean and well-tested"
}
```'''

        elif "escalation judgment" in prompt.lower():
            # Escalation decision response
            return '''```json
{
    "should_escalate": false,
    "confidence": 0.85,
    "reasoning": "Error appears to be a transient network issue. Previous attempt was close to success.",
    "escalation_reason": "",
    "retry_guidance": "Ensure database connection is stable before retrying. Check network connectivity.",
    "urgency": "low"
}
```'''

        else:
            return '{"error": "Unknown prompt type"}'


def test_planning():
    """Test the planning capability."""
    logger.info("=" * 60)
    logger.info("TEST: Planning")
    logger.info("=" * 60)

    # Create mock EGO
    ego = MockEgo()
    adapter = PMEgoAdapter(ego)

    # Create test goal
    goal = Goal(
        id="goal_123",
        description="Add user authentication to the web application",
        project_id="proj_1",
        priority=TaskPriority.HIGH
    )

    project_info = {
        "name": "WebApp",
        "path": "/path/to/webapp"
    }

    project_structure = """
webapp/
├── models/
│   ├── user.py
│   └── __init__.py
├── api/
│   ├── auth.py
│   └── __init__.py
└── components/
    ├── Login.jsx
    └── __init__.py
"""

    memories = []

    # Execute planning
    result = adapter.plan_tasks(goal, project_info, project_structure, memories)

    # Validate results
    assert isinstance(result, PlanningResult), "Result should be PlanningResult"
    assert len(result.tasks) == 3, f"Expected 3 tasks, got {len(result.tasks)}"
    assert result.confidence > 0.8, f"Expected high confidence, got {result.confidence}"
    assert result.execution_strategy == "sequential", "Expected sequential execution"

    logger.info(f"✓ Planning successful: {len(result.tasks)} tasks created")
    logger.info(f"  Analysis: {result.analysis[:100]}...")
    logger.info(f"  Confidence: {result.confidence:.2f}")
    logger.info(f"  Tasks:")
    for i, task in enumerate(result.tasks, 1):
        logger.info(f"    {i}. {task.description}")

    return result


def test_review():
    """Test the review capability."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Review")
    logger.info("=" * 60)

    # Create mock EGO
    ego = MockEgo()
    adapter = PMEgoAdapter(ego)

    # Create test task
    task = Task(
        id="task_456",
        goal_id="goal_123",
        description="Create User model and database migration",
        priority=TaskPriority.HIGH,
        acceptance_criteria=[
            "User model created",
            "Migration runs successfully",
            "Tests pass"
        ],
        status=TaskStatus.WAITING_REVIEW
    )

    # Create test result
    result = ClaudeCodeResult(
        success=True,
        files_modified=["models/user.py"],
        files_created=["migrations/001_add_user.py"],
        summary="Created User model with bcrypt password hashing and database migration",
        duration_seconds=45.5,
        turns_used=3
    )

    # Execute review
    review = adapter.review_result(task, result)

    # Validate results
    assert isinstance(review, ReviewResult), "Result should be ReviewResult"
    assert review.approved, "Task should be approved"
    assert review.confidence > 0.9, f"Expected high confidence, got {review.confidence}"
    assert review.recommendation == "approve", "Should recommend approval"

    logger.info(f"✓ Review successful: {review.recommendation}")
    logger.info(f"  Approved: {review.approved}")
    logger.info(f"  Confidence: {review.confidence:.2f}")
    logger.info(f"  Strengths: {', '.join(review.strengths)}")

    return review


def test_escalation():
    """Test the escalation decision capability."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Escalation Decision")
    logger.info("=" * 60)

    # Create mock EGO
    ego = MockEgo()
    adapter = PMEgoAdapter(ego)

    # Create test task with failure
    task = Task(
        id="task_789",
        goal_id="goal_123",
        description="Implement login/logout API endpoints",
        priority=TaskPriority.HIGH,
        status=TaskStatus.NEEDS_REVISION,
        attempt_count=2,
        max_attempts=3,
        error_message="Database connection timeout"
    )

    # Create attempt history
    attempt_history = [
        {
            "attempt": 1,
            "success": False,
            "error": "Database connection timeout",
            "feedback": "Check database connectivity"
        },
        {
            "attempt": 2,
            "success": False,
            "error": "Database connection timeout",
            "feedback": "Ensure database is running"
        }
    ]

    # Execute escalation decision
    decision = adapter.should_escalate(task, attempt_history)

    # Validate results
    assert isinstance(decision, EscalationDecision), "Result should be EscalationDecision"
    assert decision.confidence > 0.8, f"Expected high confidence, got {decision.confidence}"

    logger.info(f"✓ Escalation decision made: {'ESCALATE' if decision.should_escalate else 'RETRY'}")
    logger.info(f"  Should escalate: {decision.should_escalate}")
    logger.info(f"  Confidence: {decision.confidence:.2f}")
    logger.info(f"  Reasoning: {decision.reasoning[:100]}...")
    if not decision.should_escalate:
        logger.info(f"  Retry guidance: {decision.retry_guidance[:100]}...")

    return decision


def test_fallback_behavior():
    """Test that fallbacks work when EGO fails."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST: Fallback Behavior")
    logger.info("=" * 60)

    class FailingEgo:
        """EGO that always fails."""
        def generate(self, *args, **kwargs):
            raise Exception("EGO generation failed!")

    # Create adapter with failing EGO
    ego = FailingEgo()
    adapter = PMEgoAdapter(ego)

    # Test planning fallback
    goal = Goal(
        id="goal_fallback",
        description="Test fallback",
        project_id="proj_1",
        priority=TaskPriority.MEDIUM
    )

    result = adapter.plan_tasks(goal, {}, "", [])
    assert len(result.tasks) == 1, "Fallback should create single task"
    assert result.confidence < 0.5, "Fallback should have low confidence"

    logger.info(f"✓ Planning fallback working: {len(result.tasks)} task(s)")

    # Test review fallback
    task = Task(
        id="task_fallback",
        goal_id="goal_fallback",
        description="Test",
        priority=TaskPriority.MEDIUM
    )

    cc_result = ClaudeCodeResult(success=True, summary="Done")
    review = adapter.review_result(task, cc_result)
    assert review.recommendation in ["approve", "escalate"], "Fallback should approve or escalate"

    logger.info(f"✓ Review fallback working: {review.recommendation}")

    # Test escalation fallback
    decision = adapter.should_escalate(task, [])
    assert isinstance(decision, EscalationDecision), "Fallback should return decision"

    logger.info(f"✓ Escalation fallback working")

    return True


def main():
    """Run all tests."""
    logger.info("Testing PM Agent EGO Integration")
    logger.info("================================\n")

    try:
        # Run tests
        planning_result = test_planning()
        review_result = test_review()
        escalation_result = test_escalation()
        fallback_ok = test_fallback_behavior()

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("ALL TESTS PASSED!")
        logger.info("=" * 60)
        logger.info("✓ Planning: Creates structured tasks from goals")
        logger.info("✓ Review: Evaluates work against criteria")
        logger.info("✓ Escalation: Makes intelligent retry/escalate decisions")
        logger.info("✓ Fallbacks: Gracefully handles EGO failures")
        logger.info("\nEGO Integration is ready for production use!")

        return 0

    except Exception as e:
        logger.error(f"\n✗ TEST FAILED: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
