"""
Conch DNA - Tools Layer: Shell Tool

Safe shell command execution with timeout and security validation.
Blocks dangerous patterns and captures stdout/stderr.
"""

import logging
import re
import subprocess
from typing import List, Optional, Set

from .base import Tool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


class ShellTool(Tool):
    """
    Execute shell commands safely with security checks.

    Features:
    - Configurable blocked patterns (dangerous commands)
    - Timeout support (default 30s)
    - Stdout/stderr capture
    - Working directory support
    - Environment variable control
    """

    # Default blocked command patterns (regex)
    DEFAULT_BLOCKED_PATTERNS = [
        r"^rm\s+-rf\s+/",          # Recursive delete from root
        r"^dd\s+if=/dev/.*of=/",   # Direct disk writes
        r"^mkfs\.",                # Format filesystem
        r":\(\)\{.*\}",            # Fork bomb pattern
        r".*>\s*/dev/sd[a-z]",     # Write to raw device
        r"^sudo\s+rm",             # Sudo dangerous deletes
        r"shutdown|reboot|halt",   # System shutdown commands
        r"chmod\s+777",            # Overly permissive permissions
        r"curl.*\|\s*sh",          # Pipe to shell (dangerous)
        r"wget.*\|\s*sh",          # Pipe to shell (dangerous)
        r"exec\s+/bin/",           # Execute interpreter directly
    ]

    def __init__(
        self,
        name: str = "shell",
        description: str = "Execute shell commands safely",
        timeout: float = 30.0,
        blocked_patterns: Optional[List[str]] = None,
        allowed_commands: Optional[Set[str]] = None,
    ):
        """
        Initialize shell tool.

        Args:
            name: Tool name
            description: Tool description
            timeout: Command timeout in seconds
            blocked_patterns: Additional regex patterns to block
            allowed_commands: If set, only these commands are allowed
        """
        super().__init__(name, description, timeout)

        # Compile blocked patterns for efficiency
        patterns = self.DEFAULT_BLOCKED_PATTERNS.copy()
        if blocked_patterns:
            patterns.extend(blocked_patterns)

        self._blocked_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self._allowed_commands = allowed_commands

        logger.info(
            f"ShellTool initialized with {len(self._blocked_patterns)} blocked patterns"
        )

    def _is_command_safe(self, command: str) -> tuple[bool, Optional[str]]:
        """
        Check if command is safe to execute.

        Args:
            command: Command string to validate

        Returns:
            (is_safe, error_message)
        """
        # Check against blocked patterns
        for pattern in self._blocked_patterns:
            if pattern.search(command):
                return False, f"Command matches blocked pattern: {pattern.pattern}"

        # Check allowed commands if configured
        if self._allowed_commands is not None:
            # Extract base command (first word)
            base_cmd = command.split()[0] if command.split() else ""
            if base_cmd not in self._allowed_commands:
                return False, f"Command '{base_cmd}' not in allowed list"

        return True, None

    def execute(
        self,
        command: str,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        shell: bool = True,
    ) -> ToolResult:
        """
        Execute a shell command.

        Args:
            command: Command string to execute
            timeout: Override default timeout (seconds)
            cwd: Working directory for command
            env: Environment variables (extends os.environ)
            shell: Whether to use shell execution (default True)

        Returns:
            ToolResult with command output or error
        """
        if not self.enabled:
            return ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error="Shell tool is disabled",
            )

        # Validate command safety
        is_safe, error_msg = self._is_command_safe(command)
        if not is_safe:
            logger.warning(f"Blocked unsafe command: {command[:100]}")
            return ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error=error_msg or "Command blocked by security policy",
                metadata={"command": command},
            )

        # Use provided timeout or default
        exec_timeout = timeout if timeout is not None else self.timeout

        try:
            logger.debug(f"Executing command: {command[:100]}")

            # Execute command with timeout
            result = subprocess.run(
                command,
                shell=shell,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=exec_timeout,
            )

            # Combine stdout and stderr for output
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout)
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr}")

            output = "\n".join(output_parts)

            # Check return code
            if result.returncode != 0:
                logger.warning(
                    f"Command failed with code {result.returncode}: {command[:50]}"
                )
                return ToolResult(
                    status=ToolStatus.ERROR,
                    output=output,
                    error=f"Command exited with code {result.returncode}",
                    metadata={
                        "command": command,
                        "return_code": result.returncode,
                        "cwd": cwd,
                    },
                )

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata={
                    "command": command,
                    "return_code": result.returncode,
                    "cwd": cwd,
                },
            )

        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {exec_timeout}s: {command[:50]}")
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                output="",
                error=f"Command timed out after {exec_timeout} seconds",
                metadata={"command": command, "timeout": exec_timeout},
            )

        except FileNotFoundError as e:
            logger.error(f"Command not found: {command[:50]}")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Command not found: {str(e)}",
                metadata={"command": command},
            )

        except Exception as e:
            logger.exception(f"Unexpected error executing command: {command[:50]}")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Unexpected error: {str(e)}",
                metadata={"command": command},
            )

    def add_blocked_pattern(self, pattern: str) -> None:
        """
        Add a new blocked command pattern.

        Args:
            pattern: Regex pattern to block
        """
        compiled = re.compile(pattern, re.IGNORECASE)
        self._blocked_patterns.append(compiled)
        logger.info(f"Added blocked pattern: {pattern}")

    def remove_blocked_pattern(self, pattern: str) -> None:
        """
        Remove a blocked command pattern.

        Args:
            pattern: Regex pattern to remove
        """
        self._blocked_patterns = [
            p for p in self._blocked_patterns if p.pattern != pattern
        ]
        logger.info(f"Removed blocked pattern: {pattern}")

    def list_blocked_patterns(self) -> List[str]:
        """Get list of blocked patterns."""
        return [p.pattern for p in self._blocked_patterns]
