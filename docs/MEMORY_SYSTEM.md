# PM Agent Memory System

The PM Agent Memory System provides project-specific learning and context management, enabling the agent to improve its decision-making over time by learning from past work.

## Overview

The memory system consists of three main components:

1. **ProjectMemory** - Core memory storage and retrieval
2. **MemoryIntegration** - Automatic capture from agent execution
3. **Base MemoryStore** - Foundation from the Conch memory system

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      PM Agent                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Planning   │  │  Execution   │  │    Review    │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                  │                  │         │
│         └──────────────────┼──────────────────┘         │
│                            │                            │
│                    ┌───────▼────────┐                  │
│                    │   PM Logger    │                  │
│                    └───────┬────────┘                  │
└────────────────────────────┼───────────────────────────┘
                             │
                     ┌───────▼────────┐
                     │     Memory     │
                     │  Integration   │
                     └───────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
    │  Patterns  │   │  Decisions  │   │   History   │
    └────────────┘   └─────────────┘   └─────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                     ┌───────▼────────┐
                     │ Project Memory │
                     │   SQLite DB    │
                     └────────────────┘
```

## Core Concepts

### 1. Patterns

Patterns are learned behaviors and conventions discovered from project history.

**Pattern Types:**
- `CODING_CONVENTION` - Code style, naming, structure
- `ARCHITECTURE` - System design patterns
- `TASK_BREAKDOWN` - Effective task decomposition strategies
- `FAILURE_MODE` - Common failure patterns to avoid
- `SUCCESS_STRATEGY` - Approaches that work well
- `DEPENDENCY` - Common task dependencies
- `ESTIMATION` - Time/complexity estimates
- `TOOL_USAGE` - Effective tool/library usage

**Example Pattern:**
```python
pattern = Pattern(
    pattern_type=PatternType.CODING_CONVENTION,
    project_id="my_project",
    title="Use dataclasses for data structures",
    description="Prefer Python dataclasses over manual __init__ methods",
    context="When creating classes that primarily hold data",
    confidence=0.8,
    frequency=5,
    success_correlation=0.7,
)
```

### 2. Decisions

Decisions track choices made by the PM Agent and their outcomes.

**Decision Types:**
- `planning` - How to break down goals
- `review` - Whether to approve/revise work
- `escalation` - When to escalate to humans
- `architecture` - Technical design choices

**Example Decision:**
```python
decision_id = memory.store_decision(
    task_id="task_123",
    decision_type="architecture",
    description="Use SQLite for persistent storage",
    reasoning="SQLite provides ACID guarantees and good performance",
    alternatives=["JSON files", "PostgreSQL", "In-memory"],
    confidence=0.85,
)

# Later, record the outcome
memory.record_decision_outcome(
    decision_id=decision_id,
    outcome=DecisionOutcome.SUCCESS,
    description="SQLite implementation worked well",
)
```

### 3. Task History

Historical records of task executions with lessons learned.

**What's Captured:**
- Task description and outcome
- Duration and approach taken
- Files modified
- What worked / what failed
- Lessons learned

**Example:**
```python
memory.record_task_execution(
    task_id="task_101",
    goal_id="goal_1",
    project_id="my_project",
    description="Implement user authentication",
    success=True,
    duration_seconds=1800,
    what_worked=["JWT library was easy to use"],
    lessons_learned=["Should write tests first"],
)
```

## Usage

### Basic Setup

```python
from pathlib import Path
from conch.memory.store import MemoryStore
from conch_dna.pm.project_memory import ProjectMemory

# Initialize
memory_store = MemoryStore(Path("./data/memories.db"))
project_memory = ProjectMemory(
    memory_store=memory_store,
    db_path=Path("./data/pm_memory.db")
)
```

### With PM Agent

```python
from conch_dna.pm.agent import PMAgent, PMConfig
from conch_dna.pm.task_queue import TaskQueue
from conch_dna.pm.logger import PMLogger
from conch_dna.pm.memory_integration import MemoryIntegration

# Setup components
task_queue = TaskQueue(data_dir / "tasks.db")
logger = PMLogger(log_dir=data_dir / "logs")
memory_store = MemoryStore(data_dir / "memories.db")
project_memory = ProjectMemory(memory_store, data_dir / "pm_memory.db")

# Create integration
integration = MemoryIntegration(project_memory, logger)

# Initialize agent with memory
agent = PMAgent(
    config=config,
    task_queue=task_queue,
    claude_code=claude_code_tool,
    memory=project_memory,
    logger=logger,
)
```

### Querying Context

```python
# Get relevant context for a new task
context = project_memory.get_relevant_context(
    task_description="Add user profile feature",
    project_id="my_project",
    include_patterns=True,
    include_decisions=True,
    include_history=True,
)

# Use context in planning
planning_prompt = f"""
Goal: {goal.description}

# Relevant Past Context
{context}

# Your Task
Create a detailed plan...
"""
```

### Manual Pattern Recording

```python
from conch_dna.pm.project_memory import Pattern, PatternType

# Record a discovered pattern
pattern = Pattern(
    id="pattern_auth_jwt",
    pattern_type=PatternType.SUCCESS_STRATEGY,
    project_id="my_project",
    title="JWT tokens for stateless auth",
    description="Use JWT tokens instead of sessions for better scalability",
    context="When implementing authentication in microservices",
    confidence=0.9,
    frequency=3,
)

project_memory.store_pattern(pattern)
```

### Retrieving Patterns

```python
# Get patterns by type
patterns = project_memory.get_patterns(
    project_id="my_project",
    pattern_type=PatternType.CODING_CONVENTION,
    min_confidence=0.6,
    limit=10,
)

for pattern in patterns:
    print(f"{pattern.title}: {pattern.description}")
```

### Learning from Outcomes

```python
# Update pattern confidence based on new evidence
project_memory.update_pattern_confidence(
    pattern_id="pattern_auth_jwt",
    success=True,  # Pattern led to success
    weight=0.1,    # How much to adjust confidence
)
```

## Automatic Capture

When using `MemoryIntegration`, the system automatically captures:

**From Logger Events:**
- Task completions → Task history
- Errors → Failure patterns
- Planning thoughts → Planning decisions
- Review decisions → Review decisions

**Example:**
```python
# This is automatically captured
logger.think(
    thought_type="plan",
    content="Breaking into 3 subtasks: setup, impl, test",
    task_id="task_200",
    context={"project_id": "my_project"},
)

# Becomes a planning decision in memory
# No manual intervention needed!
```

## Memory Retrieval Strategy

The `get_relevant_context()` method uses a multi-faceted approach:

1. **Pattern Matching** - Finds patterns with high confidence and relevance
2. **Decision History** - Retrieves successful past decisions
3. **Similar Tasks** - Uses full-text search to find similar past work
4. **Contextual Filtering** - Considers project, tags, and recency

**Output Format:**
```markdown
# Relevant Patterns
- **Use dataclasses** (coding_convention, confidence: 0.80)
  Prefer Python dataclasses over manual __init__ methods
  Context: When creating classes that primarily hold data

# Successful Past Decisions
- **architecture**: Use SQLite for persistent storage
  Reasoning: SQLite provides ACID guarantees and good performance

# Similar Past Tasks
- ✓ Success: Implement user authentication
  What worked: JWT library was easy to use
  Lessons: Should write tests first
```

## Statistics

```python
stats = project_memory.get_statistics("my_project")

print(f"Patterns learned: {stats['patterns_learned']}")
print(f"Decisions made: {stats['decisions_made']}")
print(f"Successful decisions: {stats['successful_decisions']}")
print(f"Tasks recorded: {stats['tasks_recorded']}")
print(f"Success rate: {stats['successful_tasks'] / stats['tasks_recorded']:.2%}")
```

## Advanced Features

### Pattern Mining (Future)

The system is designed to support automatic pattern discovery:

```python
# Analyze task history to discover patterns
patterns = integration.analyze_task_patterns(
    project_id="my_project",
    min_frequency=3,  # Must appear at least 3 times
)
```

This could discover patterns like:
- Files that are frequently modified together
- Common sequences of task types
- Architectural patterns from file structures
- Optimal task granularity

### Confidence Evolution

Patterns and decisions start with moderate confidence and evolve based on outcomes:

```python
# Pattern starts at confidence 0.5
pattern = Pattern(confidence=0.5, ...)

# After successful use
update_pattern_confidence(pattern.id, success=True, weight=0.1)
# Confidence → 0.6

# After another success
update_pattern_confidence(pattern.id, success=True, weight=0.1)
# Confidence → 0.7

# After a failure
update_pattern_confidence(pattern.id, success=False, weight=0.1)
# Confidence → 0.6
```

### Cross-Project Learning

While currently scoped to individual projects, the system can be extended for cross-project learning:

```python
# Future: Get patterns from similar projects
similar_projects = identify_similar_projects("my_project")
for proj_id in similar_projects:
    patterns = project_memory.get_patterns(proj_id)
    # Adapt patterns to current project
```

## Database Schema

### Patterns Table
```sql
CREATE TABLE patterns (
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
```

### Decisions Table
```sql
CREATE TABLE decisions (
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
```

### Task History Table
```sql
CREATE TABLE task_history (
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
```

## Integration Points

### With Base MemoryStore

ProjectMemory extends the base Conch MemoryStore:
- Patterns stored as `PATTERN` type memories for full-text search
- Decisions stored as `LEARNING` type memories
- Task history stored as `LEARNING` type memories
- Enables unified search across all memory types

### With PM Logger

MemoryIntegration subscribes to logger events:
- `log()` events → Capture task results, errors, milestones
- `think()` events → Capture planning and review decisions
- Automatic extraction of relevant metadata

### With Task Queue

Memory system reads from task queue for context:
- Project IDs for scoping
- Goal IDs for decision tracking
- Task details for history recording

## Best Practices

1. **Let Integration Handle Automatic Capture**
   - Use MemoryIntegration for most recording
   - Manual recording for special cases only

2. **Be Specific with Context**
   - Pattern context should clearly describe when it applies
   - Decision reasoning should be detailed enough to learn from

3. **Track Outcomes**
   - Always record decision outcomes when available
   - Use outcomes to update pattern confidence

4. **Use Tags Effectively**
   - Tag patterns with relevant technologies, domains
   - Makes retrieval more accurate

5. **Review Statistics Periodically**
   - Monitor learning effectiveness
   - Identify areas needing more data

6. **Start with Lower Confidence**
   - New patterns start at 0.5-0.6
   - Let evidence build confidence over time

## Example: Full Workflow

```python
# 1. Initialize system
memory_store = MemoryStore(db_path)
project_memory = ProjectMemory(memory_store, pm_db_path)
logger = PMLogger(log_dir)
integration = MemoryIntegration(project_memory, logger)

# 2. Agent planning - automatically captured
logger.think(
    thought_type="plan",
    content="Breaking into 3 tasks based on past similar work",
    task_id="task_new",
    context={"project_id": "my_project"},
)

# 3. Get context for execution
context = project_memory.get_relevant_context(
    task_description="Add email notification feature",
    project_id="my_project",
)

# 4. Execute with context (agent does this)
# Claude Code receives context in its prompt

# 5. Record result - automatically captured
logger.result(
    category="execution",
    message="Task completed successfully",
    task_id="task_new",
    details={
        "project_id": "my_project",
        "files_modified": ["notifications.py", "email.py"],
    }
)

# 6. Learn from outcome - automatic
# Integration updates relevant patterns based on success

# 7. Query statistics
stats = project_memory.get_statistics("my_project")
print(f"Total patterns: {stats['patterns_learned']}")
print(f"Success rate: {stats['successful_tasks'] / stats['tasks_recorded']:.1%}")
```

## Future Enhancements

1. **Semantic Pattern Discovery**
   - Use embeddings to find similar code patterns
   - Cluster tasks by similarity
   - Automatic architecture pattern detection

2. **Predictive Success Estimation**
   - Predict task success probability based on history
   - Warn about approaches with low success correlation

3. **Cross-Project Transfer Learning**
   - Share patterns across similar projects
   - Build domain-specific pattern libraries

4. **Temporal Pattern Analysis**
   - Track how patterns evolve over time
   - Detect changes in project conventions

5. **Automated Decision Explanation**
   - Generate natural language explanations for decisions
   - Reference historical evidence

## Files

- `project_memory.py` - Core memory storage and retrieval
- `memory_integration.py` - Automatic capture from logger
- `example_memory_usage.py` - Usage examples
- `MEMORY_SYSTEM.md` - This documentation

## Dependencies

- `conch.memory.store` - Base memory system
- `sqlite3` - Database backend
- `datetime` - Timestamp handling
- `json` - Data serialization
