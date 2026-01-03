"""
Conch DNA - Tools Layer: Base Classes

Foundation for all tool implementations with registry and result types.
Production-ready with comprehensive error handling and logging.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class ToolStatus(Enum):
    """Status codes for tool execution results."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


@dataclass
class ToolResult:
    """
    Result from tool execution.

    Attributes:
        status: Execution status
        output: String output from the tool
        error: Error message if status is not SUCCESS
        execution_time: Time taken in seconds
        metadata: Additional tool-specific information
    """
    status: ToolStatus
    output: str
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata,
        }


class Tool(ABC):
    """
    Abstract base class for all tools.

    Tools are stateless executors that perform specific operations.
    They should validate inputs, handle errors gracefully, and log activity.
    """

    def __init__(
        self,
        name: str,
        description: str,
        timeout: float = 30.0,
        enabled: bool = True
    ):
        """
        Initialize tool.

        Args:
            name: Unique tool identifier
            description: Human-readable description
            timeout: Default timeout in seconds
            enabled: Whether tool is currently enabled
        """
        self.name = name
        self.description = description
        self.timeout = timeout
        self.enabled = enabled
        self._execution_count = 0
        self._error_count = 0
        logger.info(f"Tool '{name}' initialized")

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with execution outcome
        """
        pass

    def _record_execution(self, result: ToolResult) -> None:
        """Record execution statistics."""
        self._execution_count += 1
        if not result.success:
            self._error_count += 1
            logger.warning(
                f"Tool '{self.name}' failed: {result.error} "
                f"(errors: {self._error_count}/{self._execution_count})"
            )
        else:
            logger.debug(f"Tool '{self.name}' executed successfully")

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "name": self.name,
            "executions": self._execution_count,
            "errors": self._error_count,
            "success_rate": (
                (self._execution_count - self._error_count) / self._execution_count
                if self._execution_count > 0
                else 0.0
            ),
            "enabled": self.enabled,
        }

    def __str__(self) -> str:
        return f"Tool({self.name})"

    def __repr__(self) -> str:
        return (
            f"Tool(name={self.name!r}, enabled={self.enabled}, "
            f"executions={self._execution_count})"
        )


class ToolRegistry:
    """
    Registry for managing available tools.

    Provides centralized tool registration, lookup, and execution.
    Enforces tool uniqueness and provides batch operations.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._tools: Dict[str, Tool] = {}
        logger.info("ToolRegistry initialized")

    def register(self, tool: Tool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If tool name already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")

        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        """
        Unregister a tool.

        Args:
            name: Name of tool to remove

        Raises:
            KeyError: If tool not found
        """
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found")

        del self._tools[name]
        logger.info(f"Unregistered tool: {name}")

    def get(self, name: str) -> Optional[Tool]:
        """
        Get tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def execute(
        self,
        tool_name: str,
        **kwargs
    ) -> ToolResult:
        """
        Execute a registered tool.

        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult from execution
        """
        tool = self.get(tool_name)

        if tool is None:
            logger.error(f"Tool '{tool_name}' not found")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Tool '{tool_name}' not found in registry",
            )

        if not tool.enabled:
            logger.warning(f"Tool '{tool_name}' is disabled")
            return ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error=f"Tool '{tool_name}' is currently disabled",
            )

        try:
            start_time = datetime.now()
            result = tool.execute(**kwargs)
            result.execution_time = (datetime.now() - start_time).total_seconds()
            tool._record_execution(result)
            return result
        except Exception as e:
            logger.exception(f"Unexpected error executing '{tool_name}'")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Unexpected error: {str(e)}",
            )

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools.

        Returns:
            List of tool information dictionaries
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "enabled": tool.enabled,
                "timeout": tool.timeout,
            }
            for tool in self._tools.values()
        ]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Statistics for all tools
        """
        return {
            "total_tools": len(self._tools),
            "enabled_tools": sum(1 for t in self._tools.values() if t.enabled),
            "tools": {name: tool.get_stats() for name, tool in self._tools.items()},
        }

    def enable(self, name: str) -> None:
        """Enable a tool."""
        tool = self.get(name)
        if tool:
            tool.enabled = True
            logger.info(f"Enabled tool: {name}")

    def disable(self, name: str) -> None:
        """Disable a tool."""
        tool = self.get(name)
        if tool:
            tool.enabled = False
            logger.info(f"Disabled tool: {name}")

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    def __iter__(self):
        """Iterate over registered tools."""
        return iter(self._tools.values())
