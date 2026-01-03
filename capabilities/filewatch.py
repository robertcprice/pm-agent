"""
File Watch Capability for PM Agent

Provides file watching and hot reload functionality including:
- File system monitoring
- Development server process management
- Automatic restart on crash
- Log tailing and error detection
- Pattern-based filtering

Uses watchdog for efficient file system monitoring.
"""

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from pm_agent.logger import ThoughtLogger


class EventType(Enum):
    """File system event types"""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"


class ProcessStatus(Enum):
    """Process states"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    CRASHED = "crashed"
    RESTARTING = "restarting"


@dataclass
class FileEvent:
    """File system event"""
    event_type: EventType
    path: Path
    timestamp: datetime
    is_directory: bool = False


@dataclass
class ProcessInfo:
    """Information about a managed process"""
    name: str
    command: List[str]
    pid: Optional[int] = None
    status: ProcessStatus = ProcessStatus.STOPPED
    started_at: Optional[datetime] = None
    last_restart: Optional[datetime] = None
    restart_count: int = 0
    exit_code: Optional[int] = None


@dataclass
class WatchConfig:
    """Configuration for file watching"""
    paths: List[Path]
    patterns: List[str] = field(default_factory=lambda: ["*.py", "*.js", "*.ts", "*.tsx"])
    ignore_patterns: List[str] = field(default_factory=lambda: ["node_modules", ".git", "__pycache__", "*.pyc", ".venv"])
    recursive: bool = True
    debounce_seconds: float = 0.5


@dataclass
class ServerConfig:
    """Configuration for development server management"""
    name: str
    command: str  # Command string to execute
    working_dir: Path
    env_vars: Dict[str, str] = field(default_factory=dict)
    auto_restart: bool = True
    max_restarts: int = 10
    restart_delay_seconds: float = 2.0
    health_check_url: Optional[str] = None
    health_check_interval: float = 5.0


class FileWatchError(Exception):
    """Base exception for file watch operations"""
    pass


class ProcessError(FileWatchError):
    """Process operation failed"""
    pass


class FileWatchCapability:
    """
    File watching and process management capability.

    Monitors files for changes and manages development servers
    with auto-restart and health monitoring.
    """

    def __init__(self, logger: Optional[ThoughtLogger] = None):
        """
        Initialize file watch capability.

        Args:
            logger: Optional thought logger for tracking operations
        """
        self.logger = logger or ThoughtLogger("filewatch_capability")
        self._watchers: Dict[str, Any] = {}
        self._processes: Dict[str, ProcessInfo] = {}
        self._event_callbacks: List[Callable[[FileEvent], None]] = []
        self._running = False

    async def watch_files(
        self,
        config: WatchConfig,
        callback: Optional[Callable[[FileEvent], None]] = None,
    ) -> None:
        """
        Watch files for changes.

        Args:
            config: Watch configuration
            callback: Optional callback for events

        Raises:
            FileWatchError: If watching fails
        """
        self.logger.log_thought(
            "watch_start",
            f"Watching {len(config.paths)} paths",
            {
                "paths": [str(p) for p in config.paths],
                "patterns": config.patterns
            }
        )

        if callback:
            self._event_callbacks.append(callback)

        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler, FileSystemEvent

            class EventHandler(FileSystemEventHandler):
                def __init__(self, capability: FileWatchCapability, config: WatchConfig):
                    self.capability = capability
                    self.config = config
                    self._pending_events: Dict[str, FileEvent] = {}
                    self._debounce_task: Optional[asyncio.Task] = None

                def _should_ignore(self, path: str) -> bool:
                    """Check if path should be ignored"""
                    for pattern in self.config.ignore_patterns:
                        if pattern in path or re.search(pattern, path):
                            return True
                    return False

                def _matches_patterns(self, path: str) -> bool:
                    """Check if path matches watch patterns"""
                    import fnmatch
                    for pattern in self.config.patterns:
                        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern):
                            return True
                    return False

                def _create_event(self, event: FileSystemEvent) -> Optional[FileEvent]:
                    """Create FileEvent from watchdog event"""
                    if self._should_ignore(event.src_path):
                        return None

                    if not self._matches_patterns(event.src_path):
                        return None

                    event_type_map = {
                        "created": EventType.CREATED,
                        "modified": EventType.MODIFIED,
                        "deleted": EventType.DELETED,
                        "moved": EventType.MOVED,
                    }

                    event_type = event_type_map.get(event.event_type, EventType.MODIFIED)

                    return FileEvent(
                        event_type=event_type,
                        path=Path(event.src_path),
                        timestamp=datetime.now(),
                        is_directory=event.is_directory
                    )

                def on_any_event(self, event: FileSystemEvent):
                    """Handle file system event"""
                    if event.is_directory:
                        return

                    file_event = self._create_event(event)
                    if file_event:
                        # Store event
                        key = str(file_event.path)
                        self._pending_events[key] = file_event

                        # Schedule debounced callback
                        if self._debounce_task and not self._debounce_task.done():
                            self._debounce_task.cancel()

                        async def debounced_callback():
                            await asyncio.sleep(self.config.debounce_seconds)
                            for ev in self._pending_events.values():
                                for cb in self.capability._event_callbacks:
                                    try:
                                        cb(ev)
                                    except Exception as e:
                                        self.capability.logger.log_thought(
                                            "callback_error",
                                            f"Event callback error: {e}",
                                            {"error": str(e)},
                                            level="warning"
                                        )
                            self._pending_events.clear()

                        self._debounce_task = asyncio.create_task(debounced_callback())

            # Create and start observer
            observer = Observer()

            for path in config.paths:
                handler = EventHandler(self, config)
                observer.schedule(handler, str(path), recursive=config.recursive)

            observer.start()
            self._watchers[ str(config.paths[0]) ] = observer

            self.logger.log_thought(
                "watch_active",
                "File watcher started",
                {"observer_id": id(observer)}
            )

        except ImportError:
            raise FileWatchError(
                "watchdog package not installed. "
                "Install with: pip install watchdog"
            )

    async def stop_watching(self, watcher_id: Optional[str] = None) -> None:
        """
        Stop file watching.

        Args:
            watcher_id: Specific watcher to stop, or all if None
        """
        if watcher_id:
            observer = self._watchers.get(watcher_id)
            if observer:
                observer.stop()
                observer.join()
                del self._watchers[watcher_id]
        else:
            for observer in self._watchers.values():
                observer.stop()
                observer.join()
            self._watchers.clear()

    async def start_server(
        self,
        config: ServerConfig,
    ) -> ProcessInfo:
        """
        Start and manage a development server.

        Args:
            config: Server configuration

        Returns:
            ProcessInfo for the started server

        Raises:
            ProcessError: If server fails to start
        """
        self.logger.log_thought(
            "server_start",
            f"Starting server: {config.name}",
            {"command": config.command, "cwd": str(config.working_dir)}
        )

        if config.name in self._processes:
            # Stop existing process
            await self.stop_server(config.name)

        process_info = ProcessInfo(
            name=config.name,
            command=config.command.split(),
            status=ProcessStatus.STARTING
        )

        self._processes[config.name] = process_info

        try:
            # Create environment
            import os
            env = os.environ.copy()
            env.update(config.env_vars)

            # Start process
            process = await asyncio.create_subprocess_shell(
                config.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=config.working_dir,
                env=env
            )

            process_info.pid = process.pid
            process_info.status = ProcessStatus.RUNNING
            process_info.started_at = datetime.now()

            self.logger.log_thought(
                "server_running",
                f"Server {config.name} started with PID {process.pid}",
                {"pid": process.pid}
            )

            # Start monitoring task
            asyncio.create_task(
                self._monitor_server(config, process, process_info)
            )

            return process_info

        except Exception as e:
            process_info.status = ProcessStatus.CRASHED
            raise ProcessError(f"Failed to start server: {e}")

    async def _monitor_server(
        self,
        config: ServerConfig,
        process: asyncio.subprocess.Process,
        process_info: ProcessInfo,
    ) -> None:
        """Monitor server process and handle crashes"""
        restart_count = 0

        while restart_count < config.max_restarts:
            # Wait for process to exit
            returncode = await process.wait()

            process_info.exit_code = returncode

            if returncode == 0:
                # Clean exit
                process_info.status = ProcessStatus.STOPPED
                self.logger.log_thought(
                    "server_exited",
                    f"Server {config.name} exited cleanly",
                    {"returncode": returncode}
                )
                break

            # Process crashed
            process_info.status = ProcessStatus.CRASHED
            process_info.restart_count += 1

            self.logger.log_thought(
                "server_crashed",
                f"Server {config.name} crashed (exit {returncode})",
                {
                    "returncode": returncode,
                    "restart_count": process_info.restart_count
                },
                level="warning"
            )

            if not config.auto_restart:
                break

            if process_info.restart_count >= config.max_restarts:
                self.logger.log_thought(
                    "server_max_restarts",
                    f"Server {config.name} reached max restarts",
                    {"max_restarts": config.max_restarts},
                    level="error"
                )
                break

            # Restart after delay
            await asyncio.sleep(config.restart_delay_seconds)

            process_info.status = ProcessStatus.RESTARTING
            process_info.last_restart = datetime.now()

            # Restart
            import os
            env = os.environ.copy()
            env.update(config.env_vars)

            process = await asyncio.create_subprocess_shell(
                config.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=config.working_dir,
                env=env
            )

            process_info.pid = process.pid
            process_info.status = ProcessStatus.RUNNING

            restart_count += 1

    async def stop_server(self, name: str) -> bool:
        """
        Stop a managed server.

        Args:
            name: Server name

        Returns:
            True if stopped successfully
        """
        process_info = self._processes.get(name)
        if not process_info:
            return False

        self.logger.log_thought(
            "server_stop",
            f"Stopping server: {name}",
            {"pid": process_info.pid}
        )

        # Send SIGTERM
        if process_info.pid:
            import signal
            try:
                import os
                os.kill(process_info.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        process_info.status = ProcessStatus.STOPPED
        return True

    async def tail_logs(
        self,
        name: str,
        lines: int = 100,
        follow: bool = True,
        callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Tail logs from a managed server.

        Args:
            name: Server name
            lines: Number of lines to fetch initially
            follow: Continue following log output
            callback: Optional callback for each log line
        """
        process_info = self._processes.get(name)
        if not process_info or not process_info.pid:
            raise ProcessError(f"Server {name} not running")

        # This would require the process to have been started with captured output
        # For now, it's a placeholder
        self.logger.log_thought(
            "tail_logs",
            f"Tailing logs for {name}",
            {"lines": lines, "follow": follow}
        )

    async def check_health(
        self,
        name: str,
    ) -> bool:
        """
        Check health of a managed server.

        Args:
            name: Server name

        Returns:
            True if server is healthy
        """
        process_info = self._processes.get(name)
        if not process_info:
            return False

        if process_info.status != ProcessStatus.RUNNING:
            return False

        # Check if process is alive
        if process_info.pid:
            import os
            import signal
            try:
                os.kill(process_info.pid, 0)  # Check if process exists
            except ProcessLookupError:
                return False

        # TODO: Add HTTP health check if configured
        return True

    async def get_process_status(self, name: str) -> Optional[ProcessInfo]:
        """
        Get status of a managed process.

        Args:
            name: Process name

        Returns:
            ProcessInfo or None if not found
        """
        return self._processes.get(name)

    async def list_processes(self) -> List[ProcessInfo]:
        """
        List all managed processes.

        Returns:
            List of process information
        """
        return list(self._processes.values())

    async def stop_all(self) -> None:
        """Stop all managed processes and watchers"""
        for name in list(self._processes.keys()):
            await self.stop_server(name)

        await self.stop_watching()


# Convenience functions
async def watch_and_reload(
    project_path: Path,
    command: str,
    patterns: Optional[List[str]] = None,
) -> FileWatchCapability:
    """
    Quick helper to watch files and reload server on changes.

    Example:
        watcher = await watch_and_reload(
            project_path=Path("/path/to/project"),
            command="npm run dev",
            patterns=["*.ts", "*.tsx"]
        )
    """
    watch_cap = FileWatchCapability()

    watch_config = WatchConfig(
        paths=[project_path],
        patterns=patterns or ["*.ts", "*.tsx", "*.js", "*.jsx", "*.py"]
    )

    server_config = ServerConfig(
        name="dev-server",
        command=command,
        working_dir=project_path,
        auto_restart=True
    )

    # Define reload callback
    async def on_change(event: FileEvent):
        if event.event_type == EventType.MODIFIED:
            print(f"File changed: {event.path}")
            await watch_cap.stop_server("dev-server")
            await watch_cap.start_server(server_config)

    # Start watching
    await watch_cap.watch_files(watch_config, lambda e: asyncio.create_task(on_change(e)))

    # Start server
    await watch_cap.start_server(server_config)

    return watch_cap


async def with_dev_server(
    command: str,
    working_dir: Path,
    func: callable,
    env_vars: Optional[Dict[str, str]] = None,
):
    """
    Context manager for running a function with a dev server.

    Automatically starts server before function and stops after.

    Example:
        result = await with_dev_server(
            command="npm run dev",
            working_dir=Path("/path/to/project"),
            func=my_test_function
        )
    """
    watch_cap = FileWatchCapability()

    config = ServerConfig(
        name="temp-dev-server",
        command=command,
        working_dir=working_dir,
        env_vars=env_vars or {},
        auto_restart=False
    )

    try:
        await watch_cap.start_server(config)
        # Give server time to start
        await asyncio.sleep(2)
        return await func()
    finally:
        await watch_cap.stop_server("temp-dev-server")
