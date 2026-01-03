"""
Local Fallback Coder for PM Agent.

Provides coding capabilities when Claude Code CLI is unavailable by using
the local cortex neurons. Much less capable than Claude Code but works offline.

This enables the PM Agent to continue functioning even without external APIs,
using small local models for basic code generation and debugging.
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from ..tools.claude_code import ClaudeCodeResult, ClaudeCodeTask, TaskComplexity
from .task_queue import Task

logger = logging.getLogger(__name__)


@dataclass
class LocalCoderConfig:
    """Configuration for local coder fallback."""

    # Model settings
    base_models: Optional[Dict[str, str]] = None
    adapter_paths: Optional[Dict[str, str]] = None

    # Execution settings
    max_thinking_rounds: int = 3
    max_retries: int = 2
    timeout_seconds: int = 300

    # Safety settings
    allow_file_writes: bool = True
    allow_file_deletes: bool = False
    allowed_paths: List[str] = field(default_factory=list)
    blocked_patterns: List[str] = field(default_factory=lambda: [
        r"\.env$", r"\.env\.", r"secrets?", r"credentials?",
        r"\.key$", r"\.pem$", r"\.p12$", r"password"
    ])

    # Debug settings
    verbose: bool = False
    record_experiences: bool = True


@dataclass
class ThinkingStep:
    """A single thinking step in local coder reasoning."""
    round_num: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LocalCoderResult:
    """Result from local coder execution."""
    success: bool
    output: str
    error: Optional[str] = None
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    thinking_steps: List[ThinkingStep] = field(default_factory=list)
    duration_seconds: float = 0.0
    fallback_reason: str = "Claude Code unavailable"
    confidence: float = 0.5

    def to_claude_code_result(self) -> ClaudeCodeResult:
        """Convert to ClaudeCodeResult for compatibility with PM Agent."""
        return ClaudeCodeResult(
            success=self.success,
            output=self.output,
            error=self.error,
            files_modified=self.files_modified,
            files_created=self.files_created,
            duration_seconds=self.duration_seconds,
            turn_count=len(self.thinking_steps),
            model_used="local-cortex",
            tokens_used=0,  # Local models don't track API tokens
            cost_estimate=0.0,
            metadata={
                "fallback_reason": self.fallback_reason,
                "thinking_steps": [
                    {
                        "round": s.round_num,
                        "thought": s.thought[:200],
                        "action": s.action,
                        "confidence": s.confidence,
                    }
                    for s in self.thinking_steps
                ],
            },
        )


class LocalCoderAgent:
    """
    Fallback coder using local cortex neurons.

    Uses ThinkCortex, ActionCortex, and DebugCortex to provide basic
    coding capabilities when Claude Code CLI is unavailable.

    Capabilities:
    - Read and analyze code files
    - Generate simple code modifications
    - Debug basic errors
    - Run shell commands (with safety checks)

    Limitations:
    - Much less capable than Claude Code
    - Limited context window
    - No web search or external tools
    - Best for small, focused tasks
    """

    def __init__(
        self,
        project_root: Path,
        config: Optional[LocalCoderConfig] = None,
        cortex: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize local coder.

        Args:
            project_root: Root directory of the project
            config: Configuration options
            cortex: Pre-initialized cortex neurons (optional)
        """
        self.project_root = Path(project_root).resolve()
        self.config = config or LocalCoderConfig()
        self._cortex = cortex
        self._initialized = False

    def _init_cortex(self) -> None:
        """Lazy initialization of cortex neurons."""
        if self._initialized:
            return

        if self._cortex is None:
            try:
                from ..cortex import create_cortex_suite
                self._cortex = create_cortex_suite(
                    base_models=self.config.base_models,
                    adapter_paths=self.config.adapter_paths,
                )
                logger.info("Local coder cortex initialized")
            except ImportError as e:
                logger.warning(f"Cortex modules not available: {e}")
                self._cortex = {}
            except Exception as e:
                logger.error(f"Failed to initialize cortex: {e}")
                self._cortex = {}

        self._initialized = True

    @property
    def think(self):
        """Get think cortex neuron."""
        self._init_cortex()
        return self._cortex.get("think")

    @property
    def action(self):
        """Get action cortex neuron."""
        self._init_cortex()
        return self._cortex.get("action")

    @property
    def debug(self):
        """Get debug cortex neuron."""
        self._init_cortex()
        return self._cortex.get("debug")

    @property
    def task_cortex(self):
        """Get task cortex neuron."""
        self._init_cortex()
        return self._cortex.get("task")

    def is_available(self) -> bool:
        """Check if local coder is available."""
        self._init_cortex()
        return self.think is not None and self.action is not None

    def execute(self, task: Task, context: Optional[Dict[str, Any]] = None) -> LocalCoderResult:
        """
        Execute a coding task using local neurons.

        Args:
            task: The task to execute
            context: Additional context (project info, memories, etc.)

        Returns:
            LocalCoderResult with execution details
        """
        start_time = datetime.now()
        context = context or {}
        thinking_steps: List[ThinkingStep] = []
        files_modified: List[str] = []
        files_created: List[str] = []

        try:
            # Check availability
            if not self.is_available():
                return LocalCoderResult(
                    success=False,
                    output="",
                    error="Local cortex neurons not available",
                    fallback_reason="Cortex not initialized",
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                )

            # Build initial context
            task_context = self._build_task_context(task, context)

            # Thinking loop
            for round_num in range(1, self.config.max_thinking_rounds + 1):
                step = self._thinking_round(
                    round_num=round_num,
                    task=task,
                    context=task_context,
                    previous_steps=thinking_steps,
                )
                thinking_steps.append(step)

                # Execute action if determined
                if step.action:
                    observation, modified, created = self._execute_action(
                        step.action,
                        step.action_input or {},
                    )
                    step.observation = observation
                    files_modified.extend(modified)
                    files_created.extend(created)

                    # Update context with observation
                    task_context["last_observation"] = observation

                    # Check if task is complete
                    if step.action == "finish":
                        break

                    # Check for errors and try to debug
                    if "error" in observation.lower() and self.debug:
                        debug_result = self._debug_error(
                            error=observation,
                            task=task,
                            context=task_context,
                        )
                        task_context["debug_insight"] = debug_result

            # Calculate overall confidence
            avg_confidence = sum(s.confidence for s in thinking_steps) / len(thinking_steps)

            # Build output
            output = self._build_output(thinking_steps, files_modified, files_created)

            return LocalCoderResult(
                success=True,
                output=output,
                files_modified=list(set(files_modified)),
                files_created=list(set(files_created)),
                thinking_steps=thinking_steps,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
                confidence=avg_confidence,
            )

        except Exception as e:
            logger.exception(f"Local coder execution failed: {e}")
            return LocalCoderResult(
                success=False,
                output="",
                error=str(e),
                thinking_steps=thinking_steps,
                files_modified=files_modified,
                files_created=files_created,
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )

    def _build_task_context(
        self,
        task: Task,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build context for thinking."""
        task_context = {
            "task_description": task.description,
            "constraints": task.constraints or [],
            "acceptance_criteria": task.acceptance_criteria or [],
            "project_root": str(self.project_root),
            "context_files": task.context_files or [],
        }

        # Add file contents for context files
        file_contents = {}
        for file_path in task.context_files or []:
            full_path = self.project_root / file_path
            if full_path.exists() and full_path.is_file():
                try:
                    content = full_path.read_text()
                    # Truncate large files
                    if len(content) > 5000:
                        content = content[:5000] + "\n... [truncated]"
                    file_contents[file_path] = content
                except Exception as e:
                    file_contents[file_path] = f"[Error reading: {e}]"

        task_context["file_contents"] = file_contents
        task_context.update(context)

        return task_context

    def _thinking_round(
        self,
        round_num: int,
        task: Task,
        context: Dict[str, Any],
        previous_steps: List[ThinkingStep],
    ) -> ThinkingStep:
        """Execute one round of thinking."""
        # Build prompt for think neuron
        prompt = self._build_thinking_prompt(task, context, previous_steps)

        # Generate thought
        thought_output = None
        thought_text = ""
        confidence = 0.5

        if self.think:
            try:
                thought_output = self.think.think(
                    context=prompt,
                    needs={"task": task.description},
                )
                thought_text = self.think.extract_thought_text(thought_output)
                confidence = thought_output.confidence
            except Exception as e:
                logger.warning(f"Think neuron failed: {e}")
                thought_text = f"Analyzing task: {task.description}"

        # Determine action
        action = None
        action_input = None

        if self.action:
            try:
                action_prompt = f"""
Task: {task.description}
Thought: {thought_text}
Previous observations: {context.get('last_observation', 'None')}

Available actions:
- read_file(path): Read a file
- write_file(path, content): Write to a file
- modify_file(path, changes): Modify a file with specific changes
- run_command(cmd): Run a shell command
- finish(summary): Complete the task

What action should be taken?
"""
                action_output = self.action.select_action(
                    context=action_prompt,
                    available_tools=["read_file", "write_file", "modify_file", "run_command", "finish"],
                )

                # Parse action from output
                action, action_input = self._parse_action(action_output)

            except Exception as e:
                logger.warning(f"Action neuron failed: {e}")
                # Default to finish on last round
                if round_num == self.config.max_thinking_rounds:
                    action = "finish"
                    action_input = {"summary": thought_text}

        return ThinkingStep(
            round_num=round_num,
            thought=thought_text,
            action=action,
            action_input=action_input,
            confidence=confidence,
        )

    def _build_thinking_prompt(
        self,
        task: Task,
        context: Dict[str, Any],
        previous_steps: List[ThinkingStep],
    ) -> str:
        """Build prompt for thinking round."""
        parts = [
            f"# Task\n{task.description}",
            f"\n# Project\n{context.get('project_root', 'Unknown')}",
        ]

        if context.get("constraints"):
            parts.append(f"\n# Constraints\n" + "\n".join(f"- {c}" for c in context["constraints"]))

        if context.get("file_contents"):
            parts.append("\n# Relevant Files")
            for path, content in context["file_contents"].items():
                parts.append(f"\n## {path}\n```\n{content}\n```")

        if previous_steps:
            parts.append("\n# Previous Thinking")
            for step in previous_steps[-3:]:  # Last 3 steps
                parts.append(f"\nRound {step.round_num}:")
                parts.append(f"  Thought: {step.thought[:200]}")
                if step.action:
                    parts.append(f"  Action: {step.action}")
                if step.observation:
                    parts.append(f"  Observation: {step.observation[:200]}")

        if context.get("last_observation"):
            parts.append(f"\n# Last Result\n{context['last_observation'][:500]}")

        if context.get("debug_insight"):
            parts.append(f"\n# Debug Insight\n{context['debug_insight']}")

        return "\n".join(parts)

    def _parse_action(self, action_output) -> tuple:
        """Parse action from neuron output."""
        try:
            if hasattr(action_output, "action_type") and action_output.action_type:
                action_type = action_output.action_type.value
                params = action_output.tool_params or {}
                return action_type, params

            # Try to parse from text
            text = str(action_output.content if hasattr(action_output, "content") else action_output)

            # Look for action patterns
            patterns = [
                (r"read_file\s*\(\s*['\"]?([^'\")\s]+)['\"]?\s*\)", "read_file", lambda m: {"path": m.group(1)}),
                (r"write_file\s*\(\s*['\"]?([^'\")\s,]+)['\"]?\s*,\s*(.+)\)", "write_file", lambda m: {"path": m.group(1), "content": m.group(2)}),
                (r"run_command\s*\(\s*['\"]?(.+?)['\"]?\s*\)", "run_command", lambda m: {"cmd": m.group(1)}),
                (r"finish\s*\(\s*['\"]?(.+?)['\"]?\s*\)", "finish", lambda m: {"summary": m.group(1)}),
            ]

            for pattern, action_name, param_extractor in patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    return action_name, param_extractor(match)

            # Default to finish if no action found
            return "finish", {"summary": text[:200]}

        except Exception as e:
            logger.warning(f"Failed to parse action: {e}")
            return "finish", {"summary": "Action parsing failed"}

    def _execute_action(
        self,
        action: str,
        params: Dict[str, Any],
    ) -> tuple:
        """
        Execute an action.

        Returns:
            (observation, files_modified, files_created)
        """
        files_modified = []
        files_created = []

        try:
            if action == "read_file":
                path = self.project_root / params.get("path", "")
                if not self._is_safe_path(path):
                    return f"Error: Path {path} is not allowed", [], []

                if path.exists():
                    content = path.read_text()
                    if len(content) > 5000:
                        content = content[:5000] + "\n... [truncated]"
                    return f"File contents:\n{content}", [], []
                else:
                    return f"Error: File {path} does not exist", [], []

            elif action == "write_file":
                if not self.config.allow_file_writes:
                    return "Error: File writes are disabled", [], []

                path = self.project_root / params.get("path", "")
                if not self._is_safe_path(path):
                    return f"Error: Path {path} is not allowed", [], []

                content = params.get("content", "")
                is_new = not path.exists()

                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)

                if is_new:
                    files_created.append(str(path.relative_to(self.project_root)))
                else:
                    files_modified.append(str(path.relative_to(self.project_root)))

                return f"Successfully wrote to {path}", files_modified, files_created

            elif action == "modify_file":
                if not self.config.allow_file_writes:
                    return "Error: File writes are disabled", [], []

                path = self.project_root / params.get("path", "")
                if not self._is_safe_path(path):
                    return f"Error: Path {path} is not allowed", [], []

                if not path.exists():
                    return f"Error: File {path} does not exist", [], []

                # For now, just replace entire content
                # A more sophisticated implementation would parse changes
                changes = params.get("changes", "")
                if isinstance(changes, str):
                    path.write_text(changes)
                    files_modified.append(str(path.relative_to(self.project_root)))
                    return f"Successfully modified {path}", files_modified, []
                else:
                    return "Error: Invalid changes format", [], []

            elif action == "run_command":
                cmd = params.get("cmd", "")
                if not cmd:
                    return "Error: No command provided", [], []

                # Safety check for dangerous commands
                dangerous_patterns = [
                    r"rm\s+-rf\s+/", r"rm\s+-rf\s+\*", r"rm\s+-rf\s+~",
                    r"mkfs", r"dd\s+if=", r">\s*/dev/",
                    r"curl.*\|\s*sh", r"wget.*\|\s*sh",
                ]
                for pattern in dangerous_patterns:
                    if re.search(pattern, cmd, re.IGNORECASE):
                        return f"Error: Command blocked for safety: {cmd}", [], []

                result = subprocess.run(
                    cmd,
                    shell=True,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                output = result.stdout
                if result.returncode != 0:
                    output += f"\nError (exit {result.returncode}): {result.stderr}"

                return output[:2000], [], []

            elif action == "finish":
                return f"Task completed: {params.get('summary', 'Done')}", [], []

            else:
                return f"Unknown action: {action}", [], []

        except subprocess.TimeoutExpired:
            return "Error: Command timed out", files_modified, files_created
        except Exception as e:
            return f"Error executing action: {e}", files_modified, files_created

    def _is_safe_path(self, path: Path) -> bool:
        """Check if path is safe to access."""
        try:
            # Resolve to absolute path
            resolved = path.resolve()

            # Must be within project root
            if not str(resolved).startswith(str(self.project_root)):
                return False

            # Check against blocked patterns
            for pattern in self.config.blocked_patterns:
                if re.search(pattern, str(resolved), re.IGNORECASE):
                    return False

            # Check allowed paths if specified
            if self.config.allowed_paths:
                rel_path = resolved.relative_to(self.project_root)
                return any(
                    str(rel_path).startswith(allowed)
                    for allowed in self.config.allowed_paths
                )

            return True

        except (ValueError, RuntimeError):
            return False

    def _debug_error(
        self,
        error: str,
        task: Task,
        context: Dict[str, Any],
    ) -> str:
        """Use debug neuron to analyze an error."""
        if not self.debug:
            return "Debug neuron not available"

        try:
            debug_result = self.debug.analyze_error(
                error=error,
                context=f"Task: {task.description}\nContext: {json.dumps(context.get('file_contents', {}), indent=2)[:1000]}",
            )

            if hasattr(debug_result, "root_cause"):
                return f"Root cause: {debug_result.root_cause}\nSuggested fix: {debug_result.fix_suggestion}"
            else:
                return str(debug_result.content if hasattr(debug_result, "content") else debug_result)

        except Exception as e:
            logger.warning(f"Debug analysis failed: {e}")
            return f"Debug failed: {e}"

    def _build_output(
        self,
        steps: List[ThinkingStep],
        files_modified: List[str],
        files_created: List[str],
    ) -> str:
        """Build final output summary."""
        parts = ["# Local Coder Execution Summary\n"]

        parts.append(f"## Thinking Steps: {len(steps)}")
        for step in steps:
            parts.append(f"\n### Round {step.round_num}")
            parts.append(f"**Thought**: {step.thought[:300]}")
            if step.action:
                parts.append(f"**Action**: {step.action}")
            if step.observation:
                parts.append(f"**Result**: {step.observation[:200]}")

        if files_modified:
            parts.append(f"\n## Files Modified\n" + "\n".join(f"- {f}" for f in files_modified))

        if files_created:
            parts.append(f"\n## Files Created\n" + "\n".join(f"- {f}" for f in files_created))

        return "\n".join(parts)


def create_local_coder(
    project_root: Path,
    config: Optional[LocalCoderConfig] = None,
) -> LocalCoderAgent:
    """
    Factory function to create a local coder agent.

    Args:
        project_root: Root directory of the project
        config: Configuration options

    Returns:
        Configured LocalCoderAgent instance
    """
    return LocalCoderAgent(
        project_root=project_root,
        config=config or LocalCoderConfig(),
    )
