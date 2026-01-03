"""
Conch DNA - Tools Layer: Git Tool

Safe git operations with protection against destructive commands.
Supports status, log, diff, branch, add, and commit operations.
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

from .base import Tool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


class GitTool(Tool):
    """
    Safe git operations with protection against destructive commands.

    Features:
    - Read-only operations: status, log, diff, branch
    - Safe write operations: add, commit
    - Blocks dangerous operations: force push, hard reset
    - Repository validation
    - Timeout support
    """

    # Blocked git operations
    BLOCKED_OPERATIONS = [
        "push --force",
        "push -f",
        "reset --hard",
        "clean -fd",
        "clean -fdx",
        "branch -D",
        "rebase --force",
        "filter-branch",
        "update-ref -d",
    ]

    def __init__(
        self,
        name: str = "git",
        description: str = "Perform safe git operations",
        timeout: float = 30.0,
    ):
        """
        Initialize git tool.

        Args:
            name: Tool name
            description: Tool description
            timeout: Default timeout for git operations
        """
        super().__init__(name, description, timeout)
        logger.info("GitTool initialized")

    def _is_git_repo(self, path: Path) -> bool:
        """Check if path is a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _is_operation_safe(self, operation: str) -> tuple[bool, Optional[str]]:
        """
        Check if git operation is safe.

        Args:
            operation: Git operation string

        Returns:
            (is_safe, error_message)
        """
        for blocked in self.BLOCKED_OPERATIONS:
            if blocked in operation.lower():
                return False, f"Blocked dangerous operation: {blocked}"
        return True, None

    def _run_git_command(
        self,
        args: List[str],
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ToolResult:
        """
        Run a git command.

        Args:
            args: Git command arguments
            cwd: Working directory
            timeout: Command timeout

        Returns:
            ToolResult with command output
        """
        # Validate repository
        repo_path = Path(cwd) if cwd else Path.cwd()
        if not self._is_git_repo(repo_path):
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Not a git repository: {repo_path}",
                metadata={"path": str(repo_path)},
            )

        # Check safety
        command_str = " ".join(args)
        is_safe, error = self._is_operation_safe(command_str)
        if not is_safe:
            return ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error=error,
                metadata={"command": command_str},
            )

        # Execute command
        exec_timeout = timeout if timeout is not None else self.timeout

        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=exec_timeout,
            )

            # Combine stdout and stderr
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr}")

            output = "\n".join(output_parts)

            if result.returncode != 0:
                return ToolResult(
                    status=ToolStatus.ERROR,
                    output=output,
                    error=f"Git command failed with code {result.returncode}",
                    metadata={
                        "command": command_str,
                        "return_code": result.returncode,
                    },
                )

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata={
                    "command": command_str,
                    "return_code": result.returncode,
                },
            )

        except subprocess.TimeoutExpired:
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                output="",
                error=f"Git command timed out after {exec_timeout}s",
                metadata={"command": command_str},
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Git command failed: {str(e)}",
                metadata={"command": command_str},
            )

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """
        Execute a git operation.

        Args:
            operation: Operation type (status, log, diff, branch, add, commit)
            **kwargs: Operation-specific parameters

        Returns:
            ToolResult with operation outcome
        """
        if not self.enabled:
            return ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error="Git tool is disabled",
            )

        operations = {
            "status": self._status,
            "log": self._log,
            "diff": self._diff,
            "branch": self._branch,
            "add": self._add,
            "commit": self._commit,
        }

        if operation not in operations:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Unknown operation: {operation}. "
                      f"Available: {', '.join(operations.keys())}",
            )

        try:
            return operations[operation](**kwargs)
        except Exception as e:
            logger.exception(f"Error in git operation '{operation}'")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Operation failed: {str(e)}",
                metadata={"operation": operation},
            )

    def _status(self, cwd: Optional[str] = None) -> ToolResult:
        """
        Get git status.

        Args:
            cwd: Repository path

        Returns:
            ToolResult with status output
        """
        return self._run_git_command(["status"], cwd=cwd)

    def _log(
        self,
        cwd: Optional[str] = None,
        max_count: int = 10,
        oneline: bool = False,
    ) -> ToolResult:
        """
        Get git log.

        Args:
            cwd: Repository path
            max_count: Maximum number of commits to show
            oneline: Use oneline format

        Returns:
            ToolResult with log output
        """
        args = ["log", f"--max-count={max_count}"]
        if oneline:
            args.append("--oneline")

        return self._run_git_command(args, cwd=cwd)

    def _diff(
        self,
        cwd: Optional[str] = None,
        cached: bool = False,
        file_path: Optional[str] = None,
    ) -> ToolResult:
        """
        Get git diff.

        Args:
            cwd: Repository path
            cached: Show staged changes
            file_path: Specific file to diff

        Returns:
            ToolResult with diff output
        """
        args = ["diff"]
        if cached:
            args.append("--cached")
        if file_path:
            args.append(file_path)

        return self._run_git_command(args, cwd=cwd)

    def _branch(
        self,
        cwd: Optional[str] = None,
        list_all: bool = False,
    ) -> ToolResult:
        """
        Get git branches.

        Args:
            cwd: Repository path
            list_all: List all branches including remote

        Returns:
            ToolResult with branch list
        """
        args = ["branch"]
        if list_all:
            args.append("-a")

        return self._run_git_command(args, cwd=cwd)

    def _add(
        self,
        cwd: Optional[str] = None,
        paths: Optional[List[str]] = None,
        all_files: bool = False,
    ) -> ToolResult:
        """
        Stage files for commit.

        Args:
            cwd: Repository path
            paths: Specific paths to add
            all_files: Add all modified files

        Returns:
            ToolResult with add status
        """
        args = ["add"]

        if all_files:
            args.append("-A")
        elif paths:
            args.extend(paths)
        else:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error="Must specify either paths or all_files=True",
            )

        return self._run_git_command(args, cwd=cwd)

    def _commit(
        self,
        cwd: Optional[str] = None,
        message: Optional[str] = None,
    ) -> ToolResult:
        """
        Create a commit.

        Args:
            cwd: Repository path
            message: Commit message

        Returns:
            ToolResult with commit status
        """
        if not message:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error="Commit message is required",
            )

        args = ["commit", "-m", message]
        return self._run_git_command(args, cwd=cwd)
