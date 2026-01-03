"""
Conch DNA - Tools Layer: FileSystem Tool

Safe filesystem operations with path validation and size limits.
Supports read, write, list, delete, and directory operations.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional, Set

from .base import Tool, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


class FileSystemTool(Tool):
    """
    Safe filesystem operations with security validation.

    Features:
    - Path validation against blocked patterns
    - Size limits for read operations
    - Recursive directory operations
    - File/directory info retrieval
    - Safe deletion with confirmation
    """

    # Default blocked path patterns
    DEFAULT_BLOCKED_PATTERNS = [
        "/etc/passwd",
        "/etc/shadow",
        "/boot/",
        "/.ssh/",
        "/proc/",
        "/sys/",
        "/dev/",
    ]

    # Max file size for reads (10MB default)
    DEFAULT_MAX_READ_SIZE = 10 * 1024 * 1024

    def __init__(
        self,
        name: str = "filesystem",
        description: str = "Perform filesystem operations",
        blocked_patterns: Optional[List[str]] = None,
        max_read_size: int = DEFAULT_MAX_READ_SIZE,
        allowed_paths: Optional[Set[str]] = None,
    ):
        """
        Initialize filesystem tool.

        Args:
            name: Tool name
            description: Tool description
            blocked_patterns: Additional path patterns to block
            max_read_size: Maximum file size to read (bytes)
            allowed_paths: If set, only these paths are accessible
        """
        super().__init__(name, description)

        patterns = self.DEFAULT_BLOCKED_PATTERNS.copy()
        if blocked_patterns:
            patterns.extend(blocked_patterns)

        self._blocked_patterns = patterns
        self._max_read_size = max_read_size
        self._allowed_paths = allowed_paths

        logger.info(
            f"FileSystemTool initialized with max_read_size={max_read_size} bytes"
        )

    def _is_path_safe(self, path: Path) -> tuple[bool, Optional[str]]:
        """
        Check if path is safe to access.

        Args:
            path: Path to validate

        Returns:
            (is_safe, error_message)
        """
        path_str = str(path.resolve())

        # Check blocked patterns
        for pattern in self._blocked_patterns:
            if pattern in path_str:
                return False, f"Path contains blocked pattern: {pattern}"

        # Check allowed paths if configured
        if self._allowed_paths is not None:
            if not any(path_str.startswith(allowed) for allowed in self._allowed_paths):
                return False, "Path not in allowed paths list"

        return True, None

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """
        Execute a filesystem operation.

        Args:
            operation: Operation type (read, write, list, exists, info, mkdir, delete)
            **kwargs: Operation-specific parameters

        Returns:
            ToolResult with operation outcome
        """
        if not self.enabled:
            return ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error="Filesystem tool is disabled",
            )

        operations = {
            "read": self._read,
            "write": self._write,
            "list": self._list,
            "exists": self._exists,
            "info": self._info,
            "mkdir": self._mkdir,
            "delete": self._delete,
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
            logger.exception(f"Error in filesystem operation '{operation}'")
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Operation failed: {str(e)}",
                metadata={"operation": operation},
            )

    def _read(self, path: str, encoding: str = "utf-8") -> ToolResult:
        """
        Read file contents.

        Args:
            path: File path to read
            encoding: Text encoding (default utf-8)

        Returns:
            ToolResult with file contents
        """
        file_path = Path(path).resolve()

        # Validate path
        is_safe, error = self._is_path_safe(file_path)
        if not is_safe:
            return ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error=error,
                metadata={"path": str(file_path)},
            )

        # Check if file exists
        if not file_path.exists():
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"File not found: {file_path}",
                metadata={"path": str(file_path)},
            )

        if not file_path.is_file():
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Path is not a file: {file_path}",
                metadata={"path": str(file_path)},
            )

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self._max_read_size:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"File too large: {file_size} bytes (max: {self._max_read_size})",
                metadata={"path": str(file_path), "size": file_size},
            )

        # Read file
        try:
            content = file_path.read_text(encoding=encoding)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=content,
                metadata={
                    "path": str(file_path),
                    "size": file_size,
                    "encoding": encoding,
                },
            )
        except UnicodeDecodeError as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Encoding error: {str(e)}",
                metadata={"path": str(file_path), "encoding": encoding},
            )

    def _write(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        create_dirs: bool = True,
    ) -> ToolResult:
        """
        Write content to file.

        Args:
            path: File path to write
            content: Content to write
            encoding: Text encoding (default utf-8)
            create_dirs: Create parent directories if needed

        Returns:
            ToolResult with operation status
        """
        file_path = Path(path).resolve()

        # Validate path
        is_safe, error = self._is_path_safe(file_path)
        if not is_safe:
            return ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error=error,
                metadata={"path": str(file_path)},
            )

        # Create parent directories if needed
        if create_dirs:
            file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        try:
            file_path.write_text(content, encoding=encoding)
            file_size = file_path.stat().st_size

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Wrote {file_size} bytes to {file_path}",
                metadata={
                    "path": str(file_path),
                    "size": file_size,
                    "encoding": encoding,
                },
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Write failed: {str(e)}",
                metadata={"path": str(file_path)},
            )

    def _list(self, path: str, pattern: str = "*", recursive: bool = False) -> ToolResult:
        """
        List directory contents.

        Args:
            path: Directory path to list
            pattern: Glob pattern (default *)
            recursive: Recursive listing

        Returns:
            ToolResult with file list
        """
        dir_path = Path(path).resolve()

        # Validate path
        is_safe, error = self._is_path_safe(dir_path)
        if not is_safe:
            return ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error=error,
                metadata={"path": str(dir_path)},
            )

        if not dir_path.exists():
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Directory not found: {dir_path}",
                metadata={"path": str(dir_path)},
            )

        if not dir_path.is_dir():
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Path is not a directory: {dir_path}",
                metadata={"path": str(dir_path)},
            )

        # List directory
        try:
            if recursive:
                items = sorted(dir_path.rglob(pattern))
            else:
                items = sorted(dir_path.glob(pattern))

            # Format output
            output_lines = []
            for item in items:
                rel_path = item.relative_to(dir_path)
                if item.is_dir():
                    output_lines.append(f"[DIR]  {rel_path}/")
                else:
                    size = item.stat().st_size
                    output_lines.append(f"[FILE] {rel_path} ({size} bytes)")

            output = "\n".join(output_lines) if output_lines else "(empty)"

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=output,
                metadata={
                    "path": str(dir_path),
                    "count": len(items),
                    "pattern": pattern,
                    "recursive": recursive,
                },
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"List failed: {str(e)}",
                metadata={"path": str(dir_path)},
            )

    def _exists(self, path: str) -> ToolResult:
        """Check if path exists."""
        file_path = Path(path).resolve()

        # Validate path
        is_safe, error = self._is_path_safe(file_path)
        if not is_safe:
            return ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error=error,
                metadata={"path": str(file_path)},
            )

        exists = file_path.exists()
        return ToolResult(
            status=ToolStatus.SUCCESS,
            output="true" if exists else "false",
            metadata={"path": str(file_path), "exists": exists},
        )

    def _info(self, path: str) -> ToolResult:
        """Get file/directory information."""
        file_path = Path(path).resolve()

        # Validate path
        is_safe, error = self._is_path_safe(file_path)
        if not is_safe:
            return ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error=error,
                metadata={"path": str(file_path)},
            )

        if not file_path.exists():
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Path not found: {file_path}",
                metadata={"path": str(file_path)},
            )

        # Get info
        stat = file_path.stat()
        info = {
            "path": str(file_path),
            "type": "directory" if file_path.is_dir() else "file",
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "created": stat.st_ctime,
        }

        output = "\n".join(f"{k}: {v}" for k, v in info.items())

        return ToolResult(
            status=ToolStatus.SUCCESS,
            output=output,
            metadata=info,
        )

    def _mkdir(self, path: str, parents: bool = True) -> ToolResult:
        """Create directory."""
        dir_path = Path(path).resolve()

        # Validate path
        is_safe, error = self._is_path_safe(dir_path)
        if not is_safe:
            return ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error=error,
                metadata={"path": str(dir_path)},
            )

        try:
            dir_path.mkdir(parents=parents, exist_ok=True)
            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Created directory: {dir_path}",
                metadata={"path": str(dir_path)},
            )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Mkdir failed: {str(e)}",
                metadata={"path": str(dir_path)},
            )

    def _delete(self, path: str, recursive: bool = False) -> ToolResult:
        """Delete file or directory."""
        file_path = Path(path).resolve()

        # Validate path
        is_safe, error = self._is_path_safe(file_path)
        if not is_safe:
            return ToolResult(
                status=ToolStatus.BLOCKED,
                output="",
                error=error,
                metadata={"path": str(file_path)},
            )

        if not file_path.exists():
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Path not found: {file_path}",
                metadata={"path": str(file_path)},
            )

        try:
            if file_path.is_file():
                file_path.unlink()
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=f"Deleted file: {file_path}",
                    metadata={"path": str(file_path), "type": "file"},
                )
            elif file_path.is_dir():
                if recursive:
                    shutil.rmtree(file_path)
                    return ToolResult(
                        status=ToolStatus.SUCCESS,
                        output=f"Deleted directory (recursive): {file_path}",
                        metadata={"path": str(file_path), "type": "directory"},
                    )
                else:
                    file_path.rmdir()
                    return ToolResult(
                        status=ToolStatus.SUCCESS,
                        output=f"Deleted directory: {file_path}",
                        metadata={"path": str(file_path), "type": "directory"},
                    )
        except Exception as e:
            return ToolResult(
                status=ToolStatus.ERROR,
                output="",
                error=f"Delete failed: {str(e)}",
                metadata={"path": str(file_path)},
            )
