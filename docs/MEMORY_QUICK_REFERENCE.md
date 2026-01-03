# Project Memory - Quick Reference

## Setup

```python
from pathlib import Path
from conch.memory.store import MemoryStore
from conch_dna.pm.project_memory import ProjectMemory
from conch_dna.pm.logger import PMLogger
from conch_dna.pm.memory_integration import MemoryIntegration

# Initialize
memory_store = MemoryStore(Path("data/memories.db"))
project_memory = ProjectMemory(memory_store, Path("data/pm_memory.db"))
logger = PMLogger(log_dir=Path("data/logs"))
integration = MemoryIntegration(project_memory, logger)
```

## Common Operations

### Record a Pattern

```python
from conch_dna.pm.project_memory import Pattern, PatternType

pattern = Pattern(
    id="pattern_unique_id",
    pattern_type=PatternType.CODING_CONVENTION,
    project_id="my_project",
    title="Short descriptive title",
    description="Detailed description",
    context="When this applies",
    confidence=0.7,
)
project_memory.store_pattern(pattern)
```

### Record a Decision

```python
decision_id = project_memory.store_decision(
    task_id="task_123",
    decision_type="planning",  # or "review", "architecture", etc.
    description="What was decided",
    reasoning="Why this was chosen",
    project_id="my_project",
    goal_id="goal_456",
    alternatives=["Option A", "Option B"],
    confidence=0.8,
)

# Later, record outcome
project_memory.record_decision_outcome(
    decision_id=decision_id,
    outcome=DecisionOutcome.SUCCESS,  # or FAILURE, PARTIAL_SUCCESS, UNKNOWN
    description="What happened",
)
```

### Record Task Execution

```python
project_memory.record_task_execution(
    task_id="task_123",
    goal_id="goal_456",
    project_id="my_project",
    description="What was done",
    success=True,
    duration_seconds=1800,
    files_modified=["file1.py", "file2.py"],
    what_worked=["Thing that worked"],
    what_failed=["Thing that failed"],
    lessons_learned=["Important lesson"],
)
```

### Get Relevant Context

```python
context = project_memory.get_relevant_context(
    task_description="New task description",
    project_id="my_project",
    include_patterns=True,
    include_decisions=True,
    include_history=True,
)

# Use in prompts
prompt = f"Task: {task.description}\n\nContext:\n{context}\n\nYour instructions..."
```

### Query Patterns

```python
# Get all patterns
patterns = project_memory.get_patterns(
    project_id="my_project",
    pattern_type=PatternType.SUCCESS_STRATEGY,  # Optional filter
    min_confidence=0.6,
    limit=10,
)

# Update pattern confidence
project_memory.update_pattern_confidence(
    pattern_id="pattern_id",
    success=True,  # or False
    weight=0.1,    # How much to adjust
)
```

### Query Decisions

```python
# Get decisions for a task
decisions = project_memory.get_decisions_for_task("task_123")

# Get successful decisions
successful = project_memory.get_successful_decisions(
    project_id="my_project",
    decision_type="planning",  # Optional filter
    limit=5,
)
```

### Query Task History

```python
similar_tasks = project_memory.get_similar_tasks(
    description="Task description",
    project_id="my_project",
    limit=5,
)

for task in similar_tasks:
    print(f"{task.description}: {'✓' if task.success else '✗'}")
    print(f"  Lessons: {task.lessons_learned}")
```

### Get Statistics

```python
stats = project_memory.get_statistics("my_project")

print(f"Patterns: {stats['patterns_learned']}")
print(f"Decisions: {stats['decisions_made']}")
print(f"Success rate: {stats['successful_tasks']/stats['tasks_recorded']:.1%}")
```

## Pattern Types

- `CODING_CONVENTION` - Code style, naming
- `ARCHITECTURE` - System design patterns
- `TASK_BREAKDOWN` - How to split work
- `FAILURE_MODE` - Common failures
- `SUCCESS_STRATEGY` - What works well
- `DEPENDENCY` - Task dependencies
- `ESTIMATION` - Time/complexity
- `TOOL_USAGE` - Library/tool usage

## Decision Outcomes

- `SUCCESS` - Worked as intended
- `PARTIAL_SUCCESS` - Mostly worked
- `FAILURE` - Didn't work
- `UNKNOWN` - Not yet evaluated

## Automatic Capture (via MemoryIntegration)

When using MemoryIntegration, these are captured automatically:

### Planning Decisions
```python
logger.think(
    thought_type="plan",
    content="Planning decision",
    task_id="task_123",
    context={"project_id": "my_project"},
)
# → Automatically stored as planning decision
```

### Review Decisions
```python
logger.think(
    thought_type="decision",
    content="Review decision",
    task_id="task_123",
    context={"project_id": "my_project"},
)
# → Automatically stored as review decision
```

### Task Results
```python
logger.result(
    category="execution",
    message="Task completed",
    task_id="task_123",
    details={"project_id": "my_project", "files_modified": [...]},
)
# → Automatically recorded in task history
```

### Error Patterns
```python
logger.error(
    category="execution",
    message="Error occurred",
    task_id="task_123",
    details={"project_id": "my_project", "error_type": "dependency_error"},
)
# → Automatically creates/updates failure pattern
```

## Tips

1. **Start with lower confidence** (0.5-0.6) and let evidence build it up
2. **Be specific in contexts** so patterns apply at the right times
3. **Record outcomes** to enable learning
4. **Use tags** to make retrieval more accurate
5. **Check statistics** periodically to monitor learning

## Common Workflows

### Learning from Success
```python
# 1. Task succeeds
logger.result("execution", "Success", task_id="t1", ...)

# 2. Integration captures automatically

# 3. Update relevant patterns
integration.update_pattern_from_task(task, "my_project", success=True)

# 4. Future tasks get this context
context = project_memory.get_relevant_context(...)
```

### Learning from Failure
```python
# 1. Task fails
logger.error("execution", "Failed", task_id="t1",
             details={"error_type": "timeout"})

# 2. Integration creates failure pattern

# 3. Future similar tasks get warning
context = project_memory.get_relevant_context(...)
# Contains: "Common failure: timeout"
```

### Decision Tracking
```python
# 1. Make decision
decision_id = project_memory.store_decision(...)

# 2. Execute based on decision
# ... task execution ...

# 3. Record outcome
project_memory.record_decision_outcome(decision_id, outcome)

# 4. Future similar decisions reference this
successful = project_memory.get_successful_decisions(...)
```

## Files

- `project_memory.py` - Core memory system (30K)
- `memory_integration.py` - Automatic capture (13K)
- `example_memory_usage.py` - Working examples (9.5K)
- `MEMORY_SYSTEM.md` - Full documentation
- `MEMORY_QUICK_REFERENCE.md` - This file

## See Also

- Full documentation: `MEMORY_SYSTEM.md`
- Working examples: `example_memory_usage.py`
- Implementation summary: `PHASE_5_SUMMARY.md`
