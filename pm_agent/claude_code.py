"""
Claude Code Tool - Interface between PM Agent and Claude Code CLI.

This module provides programmatic access to Claude Code for the PM Agent
to delegate coding tasks.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum
from datetime import datetime
import subprocess
import json
import os
import tempfile
import shutil


class TaskComplexity(Enum):
    """Estimated complexity of a coding task."""
    LOW = "low"          # < 5 minutes, simple changes
    MEDIUM = "medium"    # 5-15 minutes, moderate changes
    HIGH = "high"        # 15-30 minutes, significant changes
    COMPLEX = "complex"  # 30+ minutes, may need breakdown


class ClaudeCodeModel(Enum):
    """Available Claude Code models."""
    SONNET = "sonnet"    # Fast, good for most tasks
    OPUS = "opus"        # Slower, better for complex reasoning
    HAIKU = "haiku"      # Fastest, good for simple tasks


@dataclass
class ClaudeCodeTask:
    """
    A task to be executed by Claude Code.

    This is the primary interface for telling Claude Code what to do.
    The PM Agent constructs these and sends them to Claude Code.
    """

    # Required fields
    description: str                    # Clear description of what to do
    working_directory: Path             # Where to execute (project root)

    # Context fields
    context_files: List[str] = field(default_factory=list)  # Files CC should read
    context_summary: str = ""           # Additional context from PM memory

    # Constraints
    constraints: List[str] = field(default_factory=list)    # Things to avoid
    patterns_to_follow: List[str] = field(default_factory=list)  # Patterns to use

    # Acceptance criteria
    acceptance_criteria: List[str] = field(default_factory=list)

    # Execution parameters
    model: ClaudeCodeModel = ClaudeCodeModel.SONNET
    max_turns: int = 50                 # Prevent runaway sessions
    timeout_seconds: int = 600          # 10 minute default timeout

    # Metadata
    task_id: str = ""                   # Set by task queue
    parent_goal_id: str = ""            # Which goal this serves
    complexity: TaskComplexity = TaskComplexity.MEDIUM

    def to_prompt(self) -> str:
        """Convert task to a well-structured prompt for Claude Code."""
        sections = []

        # Main task
        sections.append(f"# Task\n{self.description}")

        # Context files
        if self.context_files:
            sections.append("\n# Relevant Files")
            sections.append("Read and understand these files before making changes:")
            for f in self.context_files:
                sections.append(f"- `{f}`")

        # Additional context from PM
        if self.context_summary:
            sections.append(f"\n# Project Context\n{self.context_summary}")

        # Constraints
        if self.constraints:
            sections.append("\n# Constraints")
            sections.append("You MUST follow these constraints:")
            for c in self.constraints:
                sections.append(f"- {c}")

        # Patterns
        if self.patterns_to_follow:
            sections.append("\n# Patterns to Follow")
            sections.append("Use these patterns from the existing codebase:")
            for p in self.patterns_to_follow:
                sections.append(f"- {p}")

        # Acceptance criteria
        if self.acceptance_criteria:
            sections.append("\n# Acceptance Criteria")
            sections.append("The task is complete when ALL of these are true:")
            for i, ac in enumerate(self.acceptance_criteria, 1):
                sections.append(f"{i}. {ac}")

        # Standard instructions
        sections.append("\n# Instructions")
        sections.append("1. Read the relevant files to understand the context")
        sections.append("2. Make the necessary changes to complete the task")
        sections.append("3. If tests exist, run them to verify your changes")
        sections.append("4. Summarize what you did and any decisions you made")

        return "\n".join(sections)


@dataclass
class ClaudeCodeResult:
    """
    Result from a Claude Code session.

    Contains everything the PM Agent needs to evaluate the work.
    """

    # Execution status
    success: bool                       # Did CC complete without errors?
    exit_code: int = 0                  # Process exit code

    # What changed
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)

    # Output
    stdout: str = ""                    # Full stdout from CC
    stderr: str = ""                    # Full stderr from CC
    summary: str = ""                   # Extracted summary of work done

    # Errors
    error_message: Optional[str] = None
    error_type: Optional[str] = None   # "timeout", "auth", "rate_limit", etc.

    # Metrics
    duration_seconds: float = 0.0
    turns_used: int = 0                 # How many conversation turns
    estimated_cost_usd: Optional[float] = None

    # Metadata
    task_id: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    git_commit_before: str = ""         # Git state before execution
    git_commit_after: str = ""          # Git state after execution


@dataclass
class GitState:
    """Captured git repository state for before/after comparison."""
    head_commit: str
    branch: str
    status: str                         # Output of git status --porcelain
    staged_files: List[str] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    untracked_files: List[str] = field(default_factory=list)


class ClaudeCodeError(Exception):
    """Base exception for Claude Code errors."""
    pass


class ClaudeCodeAuthError(ClaudeCodeError):
    """Authentication failed."""
    pass


class ClaudeCodeRateLimitError(ClaudeCodeError):
    """Rate limit exceeded."""
    pass


class ClaudeCodeTimeoutError(ClaudeCodeError):
    """Execution timed out."""
    pass


class ClaudeCodeCreditsExhaustedError(ClaudeCodeError):
    """No credits remaining."""
    pass


class ClaudeCodeTool:
    """
    Primary interface for the PM Agent to delegate tasks to Claude Code.

    This class handles:
    - Building prompts from tasks
    - Invoking the Claude Code CLI
    - Capturing git state before/after
    - Parsing results
    - Error handling and classification

    Usage:
        tool = ClaudeCodeTool(project_root=Path("/path/to/project"))
        result = tool.execute_task(task)
        if result.success:
            print(f"Modified: {result.files_modified}")
    """

    def __init__(
        self,
        project_root: Path,
        default_model: ClaudeCodeModel = ClaudeCodeModel.SONNET,
        default_max_turns: int = 50,
        default_timeout: int = 600,
        auto_commit: bool = False,
        commit_message_prefix: str = "[PM Agent]",
    ):
        """
        Initialize Claude Code Tool.

        Args:
            project_root: Root directory of the project to work on
            default_model: Default model to use for tasks
            default_max_turns: Default max conversation turns
            default_timeout: Default timeout in seconds
            auto_commit: Whether to auto-commit after successful tasks
            commit_message_prefix: Prefix for auto-commit messages
        """
        self.project_root = Path(project_root).resolve()
        self.default_model = default_model
        self.default_max_turns = default_max_turns
        self.default_timeout = default_timeout
        self.auto_commit = auto_commit
        self.commit_message_prefix = commit_message_prefix

        # Session tracking
        self.session_history: List[ClaudeCodeResult] = []
        self._credits_exhausted = False

        # Validate setup
        self._validate_setup()

    def _validate_setup(self) -> None:
        """Validate that Claude Code is available and configured."""
        # Check project root exists
        if not self.project_root.exists():
            raise ValueError(f"Project root does not exist: {self.project_root}")

        # Check Claude Code is installed
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise ClaudeCodeError("Claude Code CLI not working properly")
        except FileNotFoundError:
            raise ClaudeCodeError(
                "Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code"
            )

    def execute_task(self, task: ClaudeCodeTask) -> ClaudeCodeResult:
        """
        Execute a coding task via Claude Code.

        This is the main entry point for the PM Agent.

        Args:
            task: The task to execute

        Returns:
            ClaudeCodeResult with all execution details
        """
        if self._credits_exhausted:
            raise ClaudeCodeCreditsExhaustedError(
                "Credits exhausted. Use fallback coder or wait for reset."
            )

        started_at = datetime.now()

        # Capture git state before
        git_before = self._capture_git_state()

        # Build the prompt
        prompt = task.to_prompt()

        # Prepare the command
        cmd = self._build_command(task, prompt)

        try:
            # Execute Claude Code
            process_result = subprocess.run(
                cmd,
                cwd=task.working_directory or self.project_root,
                capture_output=True,
                text=True,
                timeout=task.timeout_seconds or self.default_timeout,
                env=self._get_env(),
            )

            completed_at = datetime.now()

            # Capture git state after
            git_after = self._capture_git_state()

            # Analyze what changed
            files_modified, files_created, files_deleted = self._analyze_changes(
                git_before, git_after
            )

            # Check for specific error conditions in output
            error_type, error_message = self._check_for_errors(
                process_result.stdout,
                process_result.stderr,
                process_result.returncode,
            )

            # Build result
            result = ClaudeCodeResult(
                success=process_result.returncode == 0 and error_type is None,
                exit_code=process_result.returncode,
                files_modified=files_modified,
                files_created=files_created,
                files_deleted=files_deleted,
                stdout=process_result.stdout,
                stderr=process_result.stderr,
                summary=self._extract_summary(process_result.stdout),
                error_message=error_message,
                error_type=error_type,
                duration_seconds=(completed_at - started_at).total_seconds(),
                task_id=task.task_id,
                started_at=started_at,
                completed_at=completed_at,
                git_commit_before=git_before.head_commit,
                git_commit_after=git_after.head_commit,
            )

            # Handle credits exhausted
            if error_type == "credits_exhausted":
                self._credits_exhausted = True

            # Auto-commit if enabled and successful
            if self.auto_commit and result.success and (files_modified or files_created):
                self._auto_commit(task, result)

            # Track history
            self.session_history.append(result)

            return result

        except subprocess.TimeoutExpired:
            return ClaudeCodeResult(
                success=False,
                exit_code=-1,
                error_message=f"Task timed out after {task.timeout_seconds}s",
                error_type="timeout",
                task_id=task.task_id,
                started_at=started_at,
                completed_at=datetime.now(),
                duration_seconds=(datetime.now() - started_at).total_seconds(),
            )

    def _build_command(self, task: ClaudeCodeTask, prompt: str) -> List[str]:
        """Build the Claude Code CLI command."""
        cmd = [
            "claude",
            "--print",  # Non-interactive mode
            "--dangerously-skip-permissions",  # Auto-approve tool use
            "--model", (task.model or self.default_model).value,
            "--max-turns", str(task.max_turns or self.default_max_turns),
        ]

        # Add the prompt
        # For long prompts, use a temp file
        if len(prompt) > 10000:
            prompt_file = self._write_prompt_to_temp_file(prompt)
            cmd.extend(["--prompt-file", str(prompt_file)])
        else:
            cmd.extend(["-p", prompt])

        return cmd

    def _write_prompt_to_temp_file(self, prompt: str) -> Path:
        """Write long prompts to a temp file."""
        fd, path = tempfile.mkstemp(suffix=".txt", prefix="pm_prompt_")
        with os.fdopen(fd, 'w') as f:
            f.write(prompt)
        return Path(path)

    def _get_env(self) -> Dict[str, str]:
        """Get environment variables for subprocess."""
        env = os.environ.copy()
        # Add any additional env vars needed
        return env

    def _capture_git_state(self) -> GitState:
        """Capture current git state for comparison."""
        try:
            # Get HEAD commit
            head_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            head_commit = head_result.stdout.strip() if head_result.returncode == 0 else ""

            # Get current branch
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            branch = branch_result.stdout.strip() if branch_result.returncode == 0 else ""

            # Get status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )
            status = status_result.stdout if status_result.returncode == 0 else ""

            # Parse status into categories
            staged = []
            modified = []
            untracked = []

            for line in status.strip().split("\n"):
                if not line:
                    continue
                status_code = line[:2]
                filepath = line[3:]

                if status_code[0] in "MADRC":  # Staged changes
                    staged.append(filepath)
                if status_code[1] in "MD":  # Unstaged modifications
                    modified.append(filepath)
                if status_code == "??":  # Untracked
                    untracked.append(filepath)

            return GitState(
                head_commit=head_commit,
                branch=branch,
                status=status,
                staged_files=staged,
                modified_files=modified,
                untracked_files=untracked,
            )

        except Exception as e:
            # Return empty state if git fails (maybe not a git repo)
            return GitState(
                head_commit="",
                branch="",
                status="",
            )

    def _analyze_changes(
        self,
        before: GitState,
        after: GitState,
    ) -> tuple[List[str], List[str], List[str]]:
        """Analyze what files changed between two git states."""

        # Get diff if we have commits to compare
        if before.head_commit and after.head_commit and before.head_commit != after.head_commit:
            diff_result = subprocess.run(
                ["git", "diff", "--name-status", before.head_commit, after.head_commit],
                cwd=self.project_root,
                capture_output=True,
                text=True,
            )

            modified = []
            created = []
            deleted = []

            for line in diff_result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    status, filepath = parts[0], parts[1]
                    if status == "A":
                        created.append(filepath)
                    elif status == "D":
                        deleted.append(filepath)
                    elif status in ("M", "R"):
                        modified.append(filepath)

            return modified, created, deleted

        # Fall back to comparing status
        before_files = set(before.modified_files + before.staged_files)
        after_files = set(after.modified_files + after.staged_files + after.untracked_files)

        # New untracked files are "created"
        created = [f for f in after.untracked_files if f not in before.untracked_files]

        # Files now modified that weren't before
        modified = [f for f in after_files if f in before_files or f not in created]

        return modified, created, []

    def _check_for_errors(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> tuple[Optional[str], Optional[str]]:
        """Check output for specific error conditions."""

        combined = (stdout + stderr).lower()

        # Check for rate limiting
        if "rate limit" in combined or "too many requests" in combined:
            return "rate_limit", "Rate limit exceeded. Wait before retrying."

        # Check for auth errors
        if "authentication" in combined or "unauthorized" in combined:
            return "auth", "Authentication error. Try: claude auth login"

        # Check for credits exhausted
        if "credits" in combined and ("exhausted" in combined or "exceeded" in combined):
            return "credits_exhausted", "Claude Code credits exhausted."

        # Check for model unavailable
        if "model" in combined and "unavailable" in combined:
            return "model_unavailable", "Requested model is unavailable."

        # Generic error from exit code
        if exit_code != 0:
            return "execution_error", f"Claude Code exited with code {exit_code}"

        return None, None

    def _extract_summary(self, stdout: str) -> str:
        """Extract a summary from Claude Code's output."""
        lines = stdout.strip().split("\n")

        # Look for summary section markers
        summary_markers = ["## summary", "# summary", "summary:", "in summary"]

        for i, line in enumerate(lines):
            if any(marker in line.lower() for marker in summary_markers):
                # Return everything after the marker
                return "\n".join(lines[i:]).strip()

        # Fall back to last meaningful chunk
        # Skip empty lines and take last 10 non-empty lines
        meaningful_lines = [l for l in lines if l.strip()]
        if meaningful_lines:
            return "\n".join(meaningful_lines[-10:])

        return ""

    def _auto_commit(self, task: ClaudeCodeTask, result: ClaudeCodeResult) -> None:
        """Auto-commit changes if enabled."""
        try:
            # Stage all changes
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.project_root,
                capture_output=True,
            )

            # Build commit message
            message = f"{self.commit_message_prefix} {task.description[:50]}"
            if len(task.description) > 50:
                message += "..."

            # Commit
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.project_root,
                capture_output=True,
            )
        except Exception:
            # Don't fail the task if commit fails
            pass

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def quick_task(
        self,
        description: str,
        context_files: List[str] = None,
    ) -> ClaudeCodeResult:
        """
        Execute a simple task with minimal configuration.

        Convenience method for quick tasks.

        Args:
            description: What to do
            context_files: Optional list of relevant files

        Returns:
            ClaudeCodeResult
        """
        task = ClaudeCodeTask(
            description=description,
            working_directory=self.project_root,
            context_files=context_files or [],
        )
        return self.execute_task(task)

    def is_available(self) -> bool:
        """Check if Claude Code is available for use."""
        return not self._credits_exhausted

    def reset_credits_flag(self) -> None:
        """Reset the credits exhausted flag (e.g., after credits refill)."""
        self._credits_exhausted = False

    def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about this session's Claude Code usage."""
        if not self.session_history:
            return {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "total_duration_seconds": 0,
            }

        successful = [r for r in self.session_history if r.success]
        failed = [r for r in self.session_history if not r.success]

        return {
            "total_tasks": len(self.session_history),
            "successful_tasks": len(successful),
            "failed_tasks": len(failed),
            "success_rate": len(successful) / len(self.session_history),
            "total_duration_seconds": sum(r.duration_seconds for r in self.session_history),
            "average_duration_seconds": sum(r.duration_seconds for r in self.session_history) / len(self.session_history),
            "total_files_modified": sum(len(r.files_modified) for r in self.session_history),
            "total_files_created": sum(len(r.files_created) for r in self.session_history),
        }
