"""
Hybrid Backend - GLM for coding, Claude for planning/review.

This module provides a cost-optimized hybrid approach:

- GLM 4.7: Heavy lifting (code generation, implementation, refactoring)
- Claude Code: Planning, architecture, testing, validation, code review

This reduces costs significantly by routing expensive tasks to GLM
while using Claude's superior reasoning for planning and review.
"""

import os
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from enum import Enum
import httpx
import re

logger = logging.getLogger(__name__)


class TaskRole(Enum):
    """Role determines which backend handles the task."""
    PLANNING = "planning"           # Claude: architecture, design, breakdown
    CODING = "coding"               # GLM: implementation, heavy lifting
    TESTING = "testing"             # Claude: test generation, validation
    REVIEW = "review"               # Claude: code review, improvement suggestions
    REFACTORING = "refactoring"     # GLM: actual refactoring changes
    DOCUMENTATION = "documentation"  # Claude: documentation generation
    DEBUGGING = "debugging"         # Claude: analysis, GLM: fixes


@dataclass
class HybridResult:
    """Result from hybrid execution."""
    success: bool
    output: str
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    backend_used: str = "unknown"  # "glm", "claude", or "hybrid"
    cost_saved_estimate: float = 0.0  # Estimated cost savings


@dataclass
class HybridConfig:
    """Configuration for hybrid backend."""
    # GLM settings
    glm_api_key: str
    glm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    glm_model: str = "glm-4.7"

    # Claude Code settings (uses system claude CLI)
    claude_model: str = "sonnet"
    claude_timeout: int = 300

    # Routing settings
    code_gen_threshold: int = 50  # Lines of code expected = use GLM
    planning_always_claude: bool = True
    testing_always_claude: bool = True
    review_always_claude: bool = True

    # Cost tracking
    track_costs: bool = True
    glm_cost_per_1k_tokens: float = 0.001  # Much cheaper than Claude
    claude_cost_per_1k_tokens: float = 0.015


class TaskRouter:
    """Routes tasks to appropriate backend based on task characteristics."""

    # Keywords that suggest planning/architecture (→ Claude)
    PLANNING_KEYWORDS = [
        'plan', 'design', 'architect', 'structure', 'organize',
        'break down', 'decompose', 'analyze requirements', 'strategy',
        'approach', 'how should', 'what approach', 'best way to',
        'analyze', 'requirements', 'specification', 'scope'
    ]

    # Keywords that suggest code review (→ Claude)
    REVIEW_KEYWORDS = [
        'review', 'check', 'validate', 'verify', 'audit',
        'improve', 'optimize', 'suggestions', 'feedback',
        'quality', 'best practices', 'security review'
    ]

    # Keywords that suggest testing (→ Claude)
    TESTING_KEYWORDS = [
        'test', 'spec', 'unittest', 'pytest', 'coverage',
        'edge cases', 'test cases', 'integration test',
        'e2e', 'validation', 'assert'
    ]

    # Keywords that suggest documentation (→ Claude)
    DOCUMENTATION_KEYWORDS = [
        'document', 'readme', 'docstring', 'comment', 'explain',
        'describe', 'documentation', 'api docs', 'usage guide'
    ]

    # Keywords that suggest heavy coding (→ GLM)
    CODING_KEYWORDS = [
        'implement', 'create', 'build', 'add',
        'generate', 'code', 'function', 'class', 'module',
        'feature', 'endpoint', 'api', 'handler', 'component'
    ]

    # Keywords that suggest refactoring (→ GLM for changes, Claude for analysis)
    REFACTORING_KEYWORDS = [
        'refactor', 'restructure', 'reorganize', 'rename',
        'extract', 'move', 'split', 'merge', 'clean up'
    ]

    def route(self, task_description: str, constraints: List[str] = None) -> TaskRole:
        """Determine the best backend for a task.

        Args:
            task_description: Description of the task
            constraints: Task constraints

        Returns:
            TaskRole indicating which backend should handle this
        """
        desc_lower = task_description.lower()
        constraints_text = ' '.join(constraints or []).lower()
        combined = f"{desc_lower} {constraints_text}"

        # Check each category
        planning_score = sum(1 for kw in self.PLANNING_KEYWORDS if kw in combined)
        review_score = sum(1 for kw in self.REVIEW_KEYWORDS if kw in combined)
        testing_score = sum(1 for kw in self.TESTING_KEYWORDS if kw in combined)
        documentation_score = sum(1 for kw in self.DOCUMENTATION_KEYWORDS if kw in combined)
        coding_score = sum(1 for kw in self.CODING_KEYWORDS if kw in combined)
        refactoring_score = sum(1 for kw in self.REFACTORING_KEYWORDS if kw in combined)

        # Prioritize roles - Claude roles get higher weights for planning/review
        # But refactoring is GLM work so needs to override planning signals
        scores = {
            TaskRole.PLANNING: planning_score * 2.5,  # Weight planning highly
            TaskRole.REVIEW: review_score * 2.5,
            TaskRole.TESTING: testing_score * 2,
            TaskRole.DOCUMENTATION: documentation_score * 2,  # Documentation → Claude
            TaskRole.REFACTORING: refactoring_score * 3,  # Refactoring overrides planning
            TaskRole.CODING: coding_score,
        }

        best_role = max(scores.items(), key=lambda x: x[1])

        # Default to coding if no clear signal
        if best_role[1] == 0:
            return TaskRole.CODING

        return best_role[0]

    def should_use_glm(self, role: TaskRole) -> bool:
        """Determine if GLM should handle this role."""
        # GLM handles: coding, refactoring (the actual changes)
        return role in (TaskRole.CODING, TaskRole.REFACTORING)

    def should_use_claude(self, role: TaskRole) -> bool:
        """Determine if Claude should handle this role."""
        # Claude handles: planning, testing, review, documentation, debugging analysis
        return role in (
            TaskRole.PLANNING,
            TaskRole.TESTING,
            TaskRole.REVIEW,
            TaskRole.DOCUMENTATION,
            TaskRole.DEBUGGING,
        )


class GLMCoder:
    """GLM client for heavy coding tasks."""

    def __init__(self, config: HybridConfig):
        self.config = config
        self.client = httpx.Client(timeout=300)

    def generate_code(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> HybridResult:
        """Generate code using GLM."""
        start_time = time.time()

        system = system_prompt or """You are an expert software engineer. Generate clean,
well-documented code. Output code in markdown blocks with filename comments.
Example:
```python
# filename: src/utils.py
def helper():
    pass
```
"""

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.post(
                f"{self.config.glm_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.glm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.glm_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                }
            )

            response.raise_for_status()
            data = response.json()

            output = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)

            return HybridResult(
                success=True,
                output=output,
                duration_seconds=time.time() - start_time,
                backend_used="glm",
                cost_saved_estimate=tokens * (
                    self.config.claude_cost_per_1k_tokens -
                    self.config.glm_cost_per_1k_tokens
                ) / 1000,
            )

        except Exception as e:
            return HybridResult(
                success=False,
                output="",
                error_message=str(e),
                duration_seconds=time.time() - start_time,
                backend_used="glm",
            )


class ClaudePlanner:
    """Claude Code for planning, testing, and review tasks."""

    def __init__(self, config: HybridConfig):
        self.config = config

    def execute(
        self,
        prompt: str,
        working_directory: Optional[Path] = None,
        allowed_tools: str = "Read,Glob,Grep",  # Limited tools for planning
        timeout: int = 300,
    ) -> HybridResult:
        """Execute a planning/review task using Claude Code."""
        start_time = time.time()
        cwd = str(working_directory or Path.cwd())

        cmd = [
            "claude",
            "--print",
            "--dangerously-skip-permissions",
            "--model", self.config.claude_model,
            "--allowedTools", allowed_tools,
            "-p", prompt,
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output = result.stdout.strip()
            if result.stderr:
                output += f"\n\n[stderr]: {result.stderr.strip()}"

            return HybridResult(
                success=result.returncode == 0,
                output=output if output else "Completed with no output",
                duration_seconds=time.time() - start_time,
                backend_used="claude",
            )

        except subprocess.TimeoutExpired:
            return HybridResult(
                success=False,
                output="",
                error_message=f"Timed out after {timeout}s",
                duration_seconds=timeout,
                backend_used="claude",
            )
        except Exception as e:
            return HybridResult(
                success=False,
                output="",
                error_message=str(e),
                duration_seconds=time.time() - start_time,
                backend_used="claude",
            )


class HybridBackend:
    """
    Hybrid PM Agent backend combining GLM and Claude Code.

    Cost optimization strategy:
    - GLM 4.7: Heavy coding tasks (implementation, refactoring)
    - Claude Code: Planning, architecture, testing, code review

    This can reduce costs by 60-80% while maintaining quality for
    the tasks that matter most (planning and review).
    """

    def __init__(self, config: HybridConfig, data_dir: Optional[Path] = None):
        """Initialize hybrid backend.

        Args:
            config: Hybrid configuration
            data_dir: Directory for data storage
        """
        self.config = config
        self.router = TaskRouter()
        self.glm_coder = GLMCoder(config)
        self.claude_planner = ClaudePlanner(config)

        self.data_dir = data_dir or Path.cwd() / "data" / "hybrid_pm"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            "glm_tasks": 0,
            "claude_tasks": 0,
            "total_cost_saved": 0.0,
            "total_duration": 0.0,
        }

    def execute_task(
        self,
        description: str,
        working_directory: Optional[Path] = None,
        context_files: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        acceptance_criteria: Optional[List[str]] = None,
        force_backend: Optional[str] = None,  # "glm", "claude", or None for auto
        task_id: Optional[str] = None,
    ) -> HybridResult:
        """Execute a task using the appropriate backend.

        Args:
            description: Task description
            working_directory: Directory to work in
            context_files: Files to include as context
            constraints: Task constraints
            acceptance_criteria: Success criteria
            force_backend: Override automatic routing
            task_id: Optional task ID

        Returns:
            HybridResult with execution outcome
        """
        cwd = working_directory or Path.cwd()

        # Determine which backend to use
        if force_backend:
            use_glm = force_backend == "glm"
        else:
            role = self.router.route(description, constraints)
            use_glm = self.router.should_use_glm(role)
            logger.info(f"Task routed to {'GLM' if use_glm else 'Claude'} (role: {role.value})")

        # Build prompt with context
        prompt = self._build_prompt(description, context_files, constraints, acceptance_criteria, cwd)

        # Execute on appropriate backend
        if use_glm:
            result = self._execute_glm(prompt, cwd, context_files)
            self.stats["glm_tasks"] += 1
        else:
            result = self._execute_claude(prompt, cwd)
            self.stats["claude_tasks"] += 1

        # Update stats
        self.stats["total_duration"] += result.duration_seconds
        self.stats["total_cost_saved"] += result.cost_saved_estimate

        # Save stats periodically
        if (self.stats["glm_tasks"] + self.stats["claude_tasks"]) % 5 == 0:
            self._save_stats()

        return result

    def _build_prompt(
        self,
        description: str,
        context_files: Optional[List[str]],
        constraints: Optional[List[str]],
        acceptance_criteria: Optional[List[str]],
        cwd: Path,
    ) -> str:
        """Build the full task prompt."""
        parts = [f"# Task\n{description}"]

        # Add file context
        if context_files:
            context_parts = []
            for file_path in context_files[:5]:
                full_path = Path(file_path) if Path(file_path).is_absolute() else cwd / file_path
                if full_path.exists():
                    try:
                        content = full_path.read_text()[:15000]
                        context_parts.append(f"## {file_path}\n```\n{content}\n```")
                    except Exception:
                        pass

            if context_parts:
                parts.append("# Context Files\n" + "\n\n".join(context_parts))

        if constraints:
            parts.append("# Constraints\n" + "\n".join(f"- {c}" for c in constraints))

        if acceptance_criteria:
            parts.append("# Acceptance Criteria\n" + "\n".join(f"- {c}" for c in acceptance_criteria))

        return "\n\n".join(parts)

    def _execute_glm(
        self,
        prompt: str,
        cwd: Path,
        context_files: Optional[List[str]],
    ) -> HybridResult:
        """Execute coding task with GLM."""
        result = self.glm_coder.generate_code(prompt)

        # If successful, extract and write files
        if result.success and result.output:
            files_created = self._extract_and_write_files(result.output, cwd)
            result.files_created = files_created

        return result

    def _execute_claude(self, prompt: str, cwd: Path) -> HybridResult:
        """Execute planning/review task with Claude."""
        return self.claude_planner.execute(prompt, cwd)

    def _extract_and_write_files(self, output: str, cwd: Path) -> List[str]:
        """Extract code blocks and write to files."""
        files_created = []

        # Pattern: ```language\n# filename: path\ncontent```
        pattern = r'```(\w+)?\n#\s*filename:\s*([^\n]+)\n(.*?)```'
        matches = re.findall(pattern, output, re.DOTALL)

        for lang, filename, content in matches:
            filename = filename.strip()
            file_path = cwd / filename

            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content.strip())
                files_created.append(filename)
                logger.info(f"Created: {filename}")
            except Exception as e:
                logger.error(f"Failed to write {filename}: {e}")

        return files_created

    def plan_goal(self, goal_description: str, project_context: str = "") -> HybridResult:
        """Use Claude to plan a goal into tasks.

        This always uses Claude for its superior reasoning.

        Args:
            goal_description: The goal to plan
            project_context: Context about the project

        Returns:
            HybridResult with planning output
        """
        prompt = f"""You are planning a software development goal.

# Goal
{goal_description}

# Project Context
{project_context}

# Your Task
Break this goal into specific, actionable tasks. For each task:
1. Description (what needs to be done)
2. Acceptance criteria (how to know it's done)
3. Estimated complexity (low/medium/high)
4. Dependencies (which tasks must complete first)

Output as JSON:
{{
    "analysis": "Brief analysis of the goal",
    "tasks": [
        {{
            "description": "...",
            "acceptance_criteria": ["..."],
            "complexity": "low|medium|high",
            "dependencies": []
        }}
    ],
    "estimated_total_minutes": 60
}}
"""

        return self.claude_planner.execute(
            prompt,
            allowed_tools="Read,Glob,Grep",
            timeout=180,
        )

    def review_code(self, code_or_file: str, review_focus: str = "quality") -> HybridResult:
        """Use Claude to review code.

        Args:
            code_or_file: Code content or file path
            review_focus: What to focus on (quality, security, performance)

        Returns:
            HybridResult with review feedback
        """
        prompt = f"""Review the following code for {review_focus}.

{code_or_file}

Provide:
1. Overall assessment
2. Specific issues found
3. Improvement suggestions
4. Security concerns (if any)
5. Performance considerations (if any)

Be specific and actionable in your feedback.
"""

        return self.claude_planner.execute(
            prompt,
            allowed_tools="Read,Glob,Grep",
            timeout=180,
        )

    def generate_tests(self, code_or_file: str, test_framework: str = "pytest") -> HybridResult:
        """Use Claude to generate tests.

        Args:
            code_or_file: Code to test
            test_framework: Testing framework to use

        Returns:
            HybridResult with generated tests
        """
        prompt = f"""Generate comprehensive tests for the following code using {test_framework}.

{code_or_file}

Include:
1. Unit tests for each function/method
2. Edge case tests
3. Error handling tests
4. Integration tests where appropriate

Output the tests as code that can be run directly.
"""

        # This generates code, but we want high quality - use Claude
        result = self.claude_planner.execute(
            prompt,
            allowed_tools="Read,Glob,Grep,Write,Edit",  # Allow writing for tests
            timeout=300,
        )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        total_tasks = self.stats["glm_tasks"] + self.stats["claude_tasks"]

        return {
            **self.stats,
            "total_tasks": total_tasks,
            "glm_percentage": (self.stats["glm_tasks"] / total_tasks * 100) if total_tasks else 0,
            "claude_percentage": (self.stats["claude_tasks"] / total_tasks * 100) if total_tasks else 0,
        }

    def _save_stats(self):
        """Save statistics to disk."""
        stats_file = self.data_dir / "hybrid_stats.json"
        try:
            with open(stats_file, "w") as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save stats: {e}")


# ============================================================================
# Factory function
# ============================================================================

def create_hybrid_backend(
    glm_api_key: str,
    data_dir: Optional[Path] = None,
    claude_model: str = "sonnet",
) -> HybridBackend:
    """Create a hybrid backend.

    Args:
        glm_api_key: GLM API key for coding tasks
        data_dir: Directory for data storage
        claude_model: Claude model for planning/review

    Returns:
        Configured HybridBackend
    """
    config = HybridConfig(
        glm_api_key=glm_api_key,
        claude_model=claude_model,
    )
    return HybridBackend(config, data_dir)
