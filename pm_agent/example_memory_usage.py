#!/usr/bin/env python3
"""
Example: Using the PM Agent Memory System

This demonstrates how to:
1. Initialize the memory system
2. Manually record patterns and decisions
3. Query memory for relevant context
4. Track lessons learned
"""

from pathlib import Path
from datetime import datetime

from conch.memory.store import MemoryStore
from conch_dna.pm.project_memory import (
    ProjectMemory,
    Pattern,
    PatternType,
    DecisionOutcome,
)
from conch_dna.pm.memory_integration import MemoryIntegration
from conch_dna.pm.logger import PMLogger
from conch_dna.pm.task_queue import Task, TaskPriority


def main():
    """Demonstrate memory system usage."""

    # Setup
    data_dir = Path("./data/pm_demo")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Initialize memory system
    print("Initializing memory system...")
    memory_store = MemoryStore(data_dir / "memories.db")
    project_memory = ProjectMemory(memory_store, data_dir / "pm_memory.db")

    # Initialize logger
    logger = PMLogger(
        log_dir=data_dir / "logs",
        console_output=True,
        file_output=True,
    )
    logger.start_session()

    # Initialize memory integration
    integration = MemoryIntegration(project_memory, logger)

    project_id = "demo_project"

    # =========================================================================
    # Example 1: Record a coding pattern
    # =========================================================================

    print("\n=== Recording a coding pattern ===")
    pattern = Pattern(
        id="pattern_1",
        pattern_type=PatternType.CODING_CONVENTION,
        project_id=project_id,
        title="Use dataclasses for data structures",
        description="Prefer Python dataclasses over manual __init__ methods for data containers",
        context="When creating classes that primarily hold data",
        examples=["task_123", "task_456"],
        confidence=0.8,
        frequency=5,
        discovered_at=datetime.now(),
        last_seen=datetime.now(),
        tags=["python", "dataclass", "best_practice"],
        success_correlation=0.7,
    )

    project_memory.store_pattern(pattern)
    print(f"Stored pattern: {pattern.title}")

    # =========================================================================
    # Example 2: Record a decision
    # =========================================================================

    print("\n=== Recording a decision ===")
    decision_id = project_memory.store_decision(
        task_id="task_789",
        decision_type="architecture",
        description="Use SQLite for persistent storage instead of JSON files",
        reasoning="SQLite provides better query performance and ACID guarantees for concurrent access",
        project_id=project_id,
        goal_id="goal_1",
        alternatives=[
            "JSON files with file locking",
            "PostgreSQL database",
            "In-memory storage only",
        ],
        confidence=0.85,
    )
    print(f"Stored decision: {decision_id}")

    # Later, record the outcome
    project_memory.record_decision_outcome(
        decision_id=decision_id,
        outcome=DecisionOutcome.SUCCESS,
        description="SQLite implementation worked well, no performance issues",
    )
    print("Recorded decision outcome: SUCCESS")

    # =========================================================================
    # Example 3: Record task execution with lessons
    # =========================================================================

    print("\n=== Recording task execution ===")
    project_memory.record_task_execution(
        task_id="task_101",
        goal_id="goal_1",
        project_id=project_id,
        description="Implement user authentication",
        success=True,
        duration_seconds=1800,  # 30 minutes
        approach_taken="Used JWT tokens with bcrypt password hashing",
        files_modified=["auth.py", "models.py", "api.py"],
        what_worked=[
            "JWT library was straightforward to use",
            "Bcrypt integration was simple",
        ],
        lessons_learned=[
            "Should have written tests first - had to refactor after",
            "Need to document token expiration policy better",
        ],
    )
    print("Recorded successful task execution")

    # Record a failed task for learning
    project_memory.record_task_execution(
        task_id="task_102",
        goal_id="goal_1",
        project_id=project_id,
        description="Add OAuth2 provider integration",
        success=False,
        duration_seconds=3600,  # 1 hour
        error_type="dependency_error",
        error_message="OAuth library incompatible with Python 3.12",
        what_failed=[
            "Library version too old",
            "No clear migration path documented",
        ],
        lessons_learned=[
            "Always check library compatibility before starting",
            "Have fallback plan for critical dependencies",
        ],
    )
    print("Recorded failed task execution")

    # =========================================================================
    # Example 4: Query relevant context
    # =========================================================================

    print("\n=== Querying relevant context ===")

    # Create a hypothetical new task
    new_task_description = "Implement password reset functionality"

    context = project_memory.get_relevant_context(
        task_description=new_task_description,
        project_id=project_id,
        include_patterns=True,
        include_decisions=True,
        include_history=True,
    )

    print(f"\nContext for task: '{new_task_description}'")
    print("-" * 60)
    print(context)
    print("-" * 60)

    # =========================================================================
    # Example 5: Retrieve patterns by type
    # =========================================================================

    print("\n=== Retrieving patterns ===")
    patterns = project_memory.get_patterns(
        project_id=project_id,
        pattern_type=PatternType.CODING_CONVENTION,
        min_confidence=0.5,
    )

    print(f"Found {len(patterns)} coding convention patterns:")
    for p in patterns:
        print(f"  - {p.title} (confidence: {p.confidence:.2f})")

    # =========================================================================
    # Example 6: Get successful decisions
    # =========================================================================

    print("\n=== Retrieving successful decisions ===")
    successful_decisions = project_memory.get_successful_decisions(
        project_id=project_id,
        limit=5,
    )

    print(f"Found {len(successful_decisions)} successful decisions:")
    for d in successful_decisions:
        print(f"  - {d.decision_type}: {d.description}")
        print(f"    Reasoning: {d.reasoning[:80]}...")

    # =========================================================================
    # Example 7: Memory statistics
    # =========================================================================

    print("\n=== Memory statistics ===")
    stats = project_memory.get_statistics(project_id)

    print(f"Patterns learned: {stats['patterns_learned']}")
    print(f"Decisions made: {stats['decisions_made']}")
    print(f"Successful decisions: {stats['successful_decisions']}")
    print(f"Tasks recorded: {stats['tasks_recorded']}")
    print(f"Successful tasks: {stats['successful_tasks']}")
    if stats['average_task_duration']:
        print(f"Average task duration: {stats['average_task_duration']:.0f}s")

    # =========================================================================
    # Example 8: Pattern confidence updates
    # =========================================================================

    print("\n=== Updating pattern confidence ===")
    print(f"Initial confidence: {pattern.confidence:.2f}")

    # Simulate seeing the pattern work in practice
    project_memory.update_pattern_confidence(
        pattern_id=pattern.id,
        success=True,
        weight=0.1,
    )

    updated_patterns = project_memory.get_patterns(project_id, limit=100)
    updated_pattern = next((p for p in updated_patterns if p.id == pattern.id), None)
    if updated_pattern:
        print(f"Updated confidence: {updated_pattern.confidence:.2f}")
        print(f"Frequency: {updated_pattern.frequency}")

    # =========================================================================
    # Example 9: Integration with PM Logger
    # =========================================================================

    print("\n=== Memory integration with logger ===")

    # The integration automatically captures events from the logger
    logger.think(
        thought_type="plan",
        content="Breaking this task into 3 subtasks: setup, implementation, testing",
        task_id="task_200",
        goal_id="goal_2",
        context={"project_id": project_id},
        confidence=0.8,
    )

    # The integration captures this as a planning decision automatically
    print("Logged planning thought - automatically captured in memory")

    # Log a result
    logger.result(
        category="execution",
        message="Task completed successfully",
        task_id="task_200",
        goal_id="goal_2",
        details={
            "project_id": project_id,
            "files_modified": ["feature.py"],
        }
    )

    print("Logged result - automatically captured in memory")

    # End session
    logger.end_session("Demonstrated memory system features")

    print("\n=== Demo complete ===")
    print(f"Memory data stored in: {data_dir}")


if __name__ == "__main__":
    main()
