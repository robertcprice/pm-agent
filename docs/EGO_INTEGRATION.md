# PM Agent EGO Integration - Phase 4

## Overview

The PM Agent now integrates with the EGO model to provide intelligent reasoning capabilities for planning, reviewing, and escalation decisions. This integration is implemented through the `PMEgoAdapter` class, which maintains loose coupling between the PM Agent and EGO components.

## Architecture

```
PMAgent
   ├── ego: EgoModel (raw EGO instance)
   └── ego_adapter: PMEgoAdapter (PM-specific interface)
       ├── plan_tasks()
       ├── review_result()
       └── should_escalate()
```

## Components

### 1. PMEgoAdapter (`conch_dna/pm/ego_integration.py`)

The adapter provides three main capabilities:

#### **Planning: `plan_tasks(goal, project_info, project_structure, memories)`**
- Breaks high-level goals into concrete, executable tasks
- Uses EGO's reasoning to understand dependencies and priorities
- Returns `PlanningResult` with tasks, analysis, risks, and confidence

**Output Structure:**
```python
PlanningResult(
    tasks=[Task(...)],              # List of concrete tasks
    analysis="...",                 # EGO's analysis of the goal
    execution_strategy="parallel",  # Execution approach
    risks=["risk1", "risk2"],      # Identified risks
    estimated_total_minutes=45,     # Time estimate
    confidence=0.85                 # EGO's confidence (0-1)
)
```

#### **Review: `review_result(task, result)`**
- Evaluates completed work against acceptance criteria
- Identifies issues and provides actionable feedback
- Returns approval decision with reasoning

**Output Structure:**
```python
ReviewResult(
    approved=True,                  # Approval decision
    confidence=0.9,                 # Confidence in review
    criteria_met={...},             # Per-criterion analysis
    issues=[...],                   # Identified problems
    strengths=[...],                # Positive aspects
    feedback="...",                 # Actionable feedback
    recommendation="approve",       # "approve"|"revise"|"escalate"
    reasoning="..."                 # Decision reasoning
)
```

#### **Escalation: `should_escalate(task, attempt_history)`**
- Decides whether to retry or escalate failed tasks
- Analyzes error patterns and provides guidance
- Returns escalation decision with reasoning

**Output Structure:**
```python
EscalationDecision(
    should_escalate=False,          # Escalate or retry?
    confidence=0.8,                 # Confidence in decision
    reasoning="...",                # Decision reasoning
    escalation_reason="...",        # Reason for human (if escalating)
    retry_guidance="...",           # Guidance for next attempt
    urgency="medium"                # Priority level
)
```

### 2. Prompt Templates

All prompts are defined in `ego_integration.py`:
- `PLANNING_PROMPT_TEMPLATE` - For breaking goals into tasks
- `REVIEW_PROMPT_TEMPLATE` - For evaluating completed work
- `ESCALATION_PROMPT_TEMPLATE` - For escalation decisions

These templates provide:
- Clear context and instructions for EGO
- Structured JSON output format
- Quality guidelines and constraints

### 3. Integration Points in PMAgent

The PM Agent integrates EGO through several methods:

```python
# Initialization
self.ego_adapter = PMEgoAdapter(ego) if ego else None

# Planning (called in _plan_goal)
if self.ego_adapter:
    result = await self.ego_adapter.plan_tasks(
        goal, project_info, project_structure, memories
    )
    tasks = result.tasks

# Review (called in _review_task)
if self.ego_adapter:
    review = self.ego_adapter.review_result(task, result)
    # Process review.recommendation

# Escalation (called when task fails)
if self.ego_adapter:
    decision = self.ego_adapter.should_escalate(task, history)
    if decision.should_escalate:
        # Escalate to human
    else:
        # Retry with guidance
```

## Usage

### Basic Setup

```python
from conch_dna.ego.model import EgoModel, EgoConfig
from conch_dna.pm.agent import PMAgent, PMConfig
from conch_dna.pm.task_queue import TaskQueue
from conch_dna.tools.claude_code import ClaudeCodeTool

# Initialize EGO
ego_config = EgoConfig(
    backend="mlx_pacore",
    model_name="mlx-community/Qwen3-8B-4bit"
)
ego = EgoModel(ego_config)
ego.load()

# Initialize PM Agent with EGO
pm_config = PMConfig(
    project_root=Path("/path/to/project"),
    data_dir=Path("/path/to/data")
)
task_queue = TaskQueue(pm_config.data_dir / "tasks.db")
claude_code = ClaudeCodeTool()

pm_agent = PMAgent(
    config=pm_config,
    task_queue=task_queue,
    claude_code=claude_code,
    ego=ego  # EGO adapter initialized automatically
)

# Add a goal - PM will use EGO to plan it
goal_id = pm_agent.add_goal(
    description="Add user authentication to the web app",
    project_id="webapp_project",
    priority="high"
)

# Run the agent
await pm_agent.run()
```

### Advanced: Custom EGO Configuration

```python
# Configure EGO for specific reasoning needs
ego_config = EgoConfig(
    backend="mlx_pacore",
    pacore_parallel_trajectories=6,  # More parallel reasoning
    pacore_coordination_rounds=2,     # More coordination
    temperature=0.5,                  # Lower for more focused planning
    max_tokens=2048                   # Longer for complex analysis
)
ego = EgoModel(ego_config)
```

## Integration with Agent Methods

### Method: `_plan_with_ego_adapter()`

Located in `agent_ego_methods.py` (to be merged into `agent.py`).

This method replaces the legacy `_plan_with_ego()` and provides:
- Cleaner interface using the adapter
- Better error handling with fallbacks
- Memory integration for decision tracking
- Confidence-based quality metrics

**Usage in PMAgent:**
```python
# In _plan_goal():
if self.ego_adapter:
    tasks = await self._plan_with_ego_adapter(
        goal, project, project_structure, memory_context
    )
```

### Method: `_review_task_with_ego_adapter()`

Also in `agent_ego_methods.py`.

Provides intelligent task review with:
- Criteria-based evaluation
- Actionable feedback for revisions
- Automatic escalation on critical issues
- Confidence thresholds for auto-approval

**Usage in PMAgent:**
```python
# In _review_task():
if self.ego_adapter:
    review = self.ego_adapter.review_result(task, result)
    # Process review.recommendation
```

## Fallback Behavior

The adapter gracefully handles failures at each level:

1. **EGO Generation Fails:**
   - Falls back to conservative defaults
   - Single-task plan for planning failures
   - Auto-approve or escalate for review failures
   - Escalate at max attempts for judgment failures

2. **JSON Parsing Fails:**
   - Attempts multiple extraction strategies
   - Provides meaningful error messages
   - Returns safe fallback results

3. **No EGO Available:**
   - PM Agent functions without EGO
   - Uses simple planning (goal → single task)
   - Auto-approves completed tasks
   - Escalates all failures

## Benefits

### Loose Coupling
- PM Agent works with or without EGO
- EGO model remains generic and reusable
- Adapter provides PM-specific intelligence

### Intelligent Reasoning
- Goals broken into optimal task sequences
- Work validated against acceptance criteria
- Smart retry vs. escalate decisions

### Explainable Decisions
- All decisions include reasoning
- Confidence scores for uncertainty tracking
- Raw EGO responses preserved for debugging

### Production Ready
- Comprehensive error handling
- Fallbacks at every integration point
- Memory integration for learning

## Files Created

1. **`conch_dna/pm/ego_integration.py`** (769 lines)
   - PMEgoAdapter class
   - Prompt templates
   - Data classes (PlanningResult, ReviewResult, EscalationDecision)
   - Helper methods and fallbacks

2. **`conch_dna/pm/agent_ego_methods.py`** (165 lines)
   - `_plan_with_ego_adapter()` - New planning method
   - `_review_task_with_ego_adapter()` - New review method
   - Ready to merge into PMAgent class

3. **`conch_dna/pm/agent.py`** (updated)
   - Added import: `from .ego_integration import PMEgoAdapter`
   - Added initialization: `self.ego_adapter = PMEgoAdapter(ego) if ego else None`

4. **`conch_dna/pm/EGO_INTEGRATION.md`** (this file)
   - Complete documentation
   - Usage examples
   - Architecture overview

## Next Steps

### To Complete Integration:

1. **Merge methods into PMAgent:**
   ```bash
   # Copy methods from agent_ego_methods.py into PMAgent class in agent.py
   # Add them after the existing _plan_with_ego method (around line 513)
   ```

2. **Update planning call:**
   ```python
   # In _plan_goal() around line 382, change:
   if self.ego:
       tasks = await self._plan_with_ego(...)
   # To:
   if self.ego_adapter:
       tasks = await self._plan_with_ego_adapter(...)
   ```

3. **Update review call (optional but recommended):**
   ```python
   # In _review_task(), add support for new adapter method
   if self.ego_adapter:
       return await self._review_task_with_ego_adapter(task)
   # Otherwise fall through to existing logic
   ```

4. **Test the integration:**
   ```bash
   python -m conch_dna.pm.run_agent --project test_project
   ```

### Future Enhancements:

1. **Adaptive Planning:**
   - Learn from task completion times
   - Adjust task granularity based on success rates
   - Cache planning patterns for similar goals

2. **Review Learning:**
   - Track which criteria are most predictive
   - Adjust confidence thresholds over time
   - Learn project-specific quality patterns

3. **Escalation Intelligence:**
   - Identify systematic issues vs. one-offs
   - Suggest root cause fixes
   - Prioritize escalations by impact

## Example Session Flow

```
1. Human adds goal:
   "Add user authentication to the web app"

2. PM Agent (PLANNING):
   - Calls ego_adapter.plan_tasks()
   - EGO analyzes project structure
   - Returns 5 tasks:
     1. Create User model and database schema
     2. Implement password hashing
     3. Create login/logout endpoints
     4. Add session management
     5. Create login UI components

3. PM Agent (EXECUTION):
   - Delegates Task 1 to Claude Code
   - Claude Code creates User model
   - Task completes successfully

4. PM Agent (REVIEW):
   - Calls ego_adapter.review_result()
   - EGO checks acceptance criteria
   - Approves (confidence: 0.92)

5. Repeat for Tasks 2-5...

6. Goal complete!
   - All tasks approved
   - Authentication system working
   - PM Agent marks goal as COMPLETED
```

## Conclusion

Phase 4 EGO Integration successfully provides the PM Agent with intelligent reasoning capabilities while maintaining clean architecture and production-ready error handling. The adapter pattern ensures loose coupling and allows the system to gracefully degrade when EGO is unavailable.
