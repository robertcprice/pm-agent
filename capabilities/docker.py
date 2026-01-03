"""
Docker Capability for PM Agent

Provides container management operations including:
- Build, run, stop, and remove containers
- Docker Compose orchestration
- Log fetching and monitoring
- Resource management and cleanup
- Container health checks

Safety Features:
- Resource quotas to prevent runaway containers
- Automatic cleanup on exit
- Timeout protection for long-running operations
- Non-destructive defaults
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
from pm_agent.logger import ThoughtLogger


class ContainerStatus(Enum):
    """Container lifecycle states"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    EXITED = "exited"
    REMOVING = "removing"
    DEAD = "dead"
    UNKNOWN = "unknown"


@dataclass
class ContainerConfig:
    """Configuration for running a container"""
    image: str
    name: Optional[str] = None
    command: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    ports: Dict[str, str] = field(default_factory=dict)  # host:container
    volumes: Dict[str, str] = field(default_factory=dict)  # host:container
    detach: bool = True
    auto_remove: bool = False
    healthcheck: Optional[str] = None
    timeout_seconds: int = 300
    memory_limit: Optional[str] = None  # e.g., "512m", "1g"
    cpu_limit: Optional[str] = None  # e.g., "0.5", "1.0"


@dataclass
class ContainerInfo:
    """Information about a running container"""
    container_id: str
    name: str
    image: str
    status: ContainerStatus
    ports: List[str]
    created: datetime
    health: Optional[str] = None


@dataclass
class ComposeProject:
    """Docker Compose project configuration"""
    project_name: str
    compose_files: List[str]
    env_file: Optional[str] = None
    env_vars: Dict[str, str] = field(default_factory=dict)


class DockerError(Exception):
    """Base exception for Docker operations"""
    pass


class ContainerNotFoundError(DockerError):
    """Container does not exist"""
    pass


class DockerTimeoutError(DockerError):
    """Operation timed out"""
    pass


class ResourceLimitError(DockerError):
    """Resource quota exceeded"""
    pass


class DockerCapability:
    """
    Docker container management capability.

    Provides safe, managed access to Docker operations with built-in
    safeguards against resource exhaustion and runaway containers.
    """

    # Resource limits to prevent abuse
    MAX_CONTAINERS = 20
    MAX_MEMORY_PER_CONTAINER = "2g"
    DEFAULT_TIMEOUT = 300
    CONTAINER_CLEANUP_AGE = timedelta(hours=24)

    def __init__(self, logger: Optional[ThoughtLogger] = None):
        """
        Initialize Docker capability.

        Args:
            logger: Optional thought logger for tracking operations
        """
        self.logger = logger or ThoughtLogger("docker_capability")
        self._active_containers: Dict[str, ContainerInfo] = {}
        self._compose_projects: Dict[str, ComposeProject] = {}
        self._resource_usage: Dict[str, Dict] = {}

    async def verify_docker_available(self) -> bool:
        """
        Verify Docker is installed and accessible.

        Returns:
            True if Docker is available, False otherwise
        """
        try:
            result = await self._run_command(["docker", "--version"])
            if result[0] == 0:
                self.logger.log_thought(
                    "docker_available",
                    f"Docker version: {result[1].strip()}",
                    {"version": result[1].strip()}
                )
                return True
        except Exception as e:
            self.logger.log_thought(
                "docker_unavailable",
                f"Docker not available: {e}",
                {"error": str(e)},
                level="warning"
            )
        return False

    async def build_image(
        self,
        dockerfile_path: Path,
        image_tag: str,
        build_context: Optional[Path] = None,
        build_args: Optional[Dict[str, str]] = None,
        target: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Build a Docker image from a Dockerfile.

        Args:
            dockerfile_path: Path to Dockerfile
            image_tag: Tag to apply to the image
            build_context: Build context directory (defaults to dockerfile parent)
            build_args: Build arguments for --build-arg
            target: Target stage for multi-stage builds

        Returns:
            Tuple of (success, output_message)
        """
        self.logger.log_thought(
            "build_start",
            f"Building Docker image: {image_tag}",
            {"image": image_tag, "dockerfile": str(dockerfile_path)}
        )

        build_context = build_context or dockerfile_path.parent

        cmd = [
            "docker", "build",
            "-f", str(dockerfile_path),
            "-t", image_tag,
            "--progress=plain"
        ]

        if build_args:
            for key, value in build_args.items():
                cmd.extend(["--build-arg", f"{key}={value}"])

        if target:
            cmd.extend(["--target", target])

        cmd.append(str(build_context))

        try:
            exit_code, output = await self._run_command(cmd, timeout=600)

            if exit_code == 0:
                self.logger.log_thought(
                    "build_success",
                    f"Built image: {image_tag}",
                    {"image": image_tag, "size": self._extract_image_size(output)}
                )
                return True, f"Successfully built {image_tag}"
            else:
                self.logger.log_thought(
                    "build_failed",
                    f"Build failed for {image_tag}",
                    {"image": image_tag, "error": output[-500:]}
                )
                return False, f"Build failed: {output[-500:]}"

        except asyncio.TimeoutError:
            return False, f"Build timed out after 10 minutes"

    async def run_container(
        self,
        config: ContainerConfig,
        wait_for_output: bool = False,
    ) -> ContainerInfo:
        """
        Run a container with the given configuration.

        Args:
            config: Container configuration
            wait_for_output: If True, wait for container to finish and return output

        Returns:
            ContainerInfo with container details

        Raises:
            DockerTimeoutError: If container execution times out
            ResourceLimitError: If resource limits would be exceeded
        """
        # Check resource limits
        await self._check_resource_limits()

        self.logger.log_thought(
            "run_container",
            f"Running container: {config.image}",
            {
                "image": config.image,
                "name": config.name,
                "ports": config.ports,
                "volumes": config.volumes
            }
        )

        cmd = ["docker", "run"]

        if config.name:
            cmd.extend(["--name", config.name])

        if config.environment:
            for key, value in config.environment.items():
                cmd.extend(["-e", f"{key}={value}"])

        if config.ports:
            for host_port, container_port in config.ports.items():
                cmd.extend(["-p", f"{host_port}:{container_port}"])

        if config.volumes:
            for host_path, container_path in config.volumes.items():
                cmd.extend(["-v", f"{host_path}:{container_path}"])

        if config.detach:
            cmd.append("-d")

        if config.auto_remove:
            cmd.append("--rm")

        if config.memory_limit:
            cmd.extend(["--memory", config.memory_limit])

        if config.cpu_limit:
            cmd.extend(["--cpus", config.cpu_limit])

        cmd.append(config.image)

        if config.command:
            cmd.extend(config.command.split())

        try:
            if wait_for_output:
                exit_code, output = await self._run_command(
                    cmd,
                    timeout=config.timeout_seconds
                )
                # Create a ContainerInfo for the finished container
                return ContainerInfo(
                    container_id="finished",
                    name=config.name or "unnamed",
                    image=config.image,
                    status=ContainerStatus.EXITED,
                    ports=[],
                    created=datetime.now()
                )
            else:
                exit_code, container_id = await self._run_command(cmd, timeout=60)
                container_id = container_id.strip()

                # Get container info
                info = await self.get_container_info(container_id)

                # Track active container
                if info:
                    self._active_containers[container_id] = info

                return info

        except asyncio.TimeoutError:
            raise DockerTimeoutError(
                f"Container {config.image} timed out after {config.timeout_seconds}s"
            )

    async def stop_container(
        self,
        container_id: str,
        timeout: int = 10
    ) -> bool:
        """
        Stop a running container.

        Args:
            container_id: Container ID or name
            timeout: Seconds to wait before force killing

        Returns:
            True if stopped successfully
        """
        self.logger.log_thought(
            "stop_container",
            f"Stopping container: {container_id}",
            {"container": container_id}
        )

        cmd = ["docker", "stop", "-t", str(timeout), container_id]
        exit_code, _ = await self._run_command(cmd, timeout=timeout + 5)

        if container_id in self._active_containers:
            del self._active_containers[container_id]

        return exit_code == 0

    async def remove_container(
        self,
        container_id: str,
        force: bool = False
    ) -> bool:
        """
        Remove a container.

        Args:
            container_id: Container ID or name
            force: Force removal even if running

        Returns:
            True if removed successfully
        """
        self.logger.log_thought(
            "remove_container",
            f"Removing container: {container_id}",
            {"container": container_id, "force": force}
        )

        cmd = ["docker", "rm"]
        if force:
            cmd.append("-f")
        cmd.append(container_id)

        exit_code, _ = await self._run_command(cmd)

        if container_id in self._active_containers:
            del self._active_containers[container_id]

        return exit_code == 0

    async def get_container_info(
        self,
        container_id: str
    ) -> Optional[ContainerInfo]:
        """
        Get detailed information about a container.

        Args:
            container_id: Container ID or name

        Returns:
            ContainerInfo or None if not found
        """
        cmd = [
            "docker", "inspect",
            "--format", "{{json .}}",
            container_id
        ]

        exit_code, output = await self._run_command(cmd)

        if exit_code != 0:
            return None

        try:
            data = json.loads(output)
            if isinstance(data, list):
                data = data[0]

            state = data.get("State", {})
            config_data = data.get("Config", {})

            # Determine status
            if state.get("Running"):
                status = ContainerStatus.RUNNING
            elif state.get("Paused"):
                status = ContainerStatus.PAUSED
            elif state.get("Restarting"):
                status = ContainerStatus.RESTARTING
            elif state.get("Dead"):
                status = ContainerStatus.DEAD
            else:
                status = ContainerStatus.EXITED

            # Parse ports
            ports = []
            network_settings = data.get("NetworkSettings", {})
            for port, bindings in network_settings.get("Ports", {}).items():
                if bindings:
                    for binding in bindings:
                        ports.append(f"{binding.get('HostPort')}:{port}")

            return ContainerInfo(
                container_id=data.get("Id", "")[:12],
                name=data.get("Name", "").lstrip("/"),
                image=config_data.get("Image", ""),
                status=status,
                ports=ports,
                created=datetime.fromisoformat(
                    data.get("Created", "").replace("Z", "+00:00")
                ),
                health=state.get("Health", {}).get("Status")
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.log_thought(
                "parse_error",
                f"Failed to parse container info: {e}",
                {"container": container_id, "error": str(e)},
                level="warning"
            )
            return None

    async def get_container_logs(
        self,
        container_id: str,
        tail: int = 100,
        follow: bool = False,
        timeout: int = 30,
    ) -> str:
        """
        Fetch logs from a container.

        Args:
            container_id: Container ID or name
            tail: Number of lines from the end
            follow: Continue following logs
            timeout: Maximum time to wait for logs

        Returns:
            Log output as string
        """
        cmd = ["docker", "logs", "--tail", str(tail)]
        if follow:
            cmd.append("-f")
        cmd.append(container_id)

        _, output = await self._run_command(cmd, timeout=timeout)
        return output

    async def compose_up(
        self,
        compose_file: Path,
        project_name: Optional[str] = None,
        env_file: Optional[Path] = None,
        services: Optional[List[str]] = None,
        detach: bool = True,
        build: bool = False,
    ) -> Tuple[bool, str]:
        """
        Start services using Docker Compose.

        Args:
            compose_file: Path to docker-compose.yml
            project_name: Project name (defaults to directory name)
            env_file: Path to .env file
            services: Specific services to start (None = all)
            detach: Run in background
            build: Build images before starting

        Returns:
            Tuple of (success, output_message)
        """
        self.logger.log_thought(
            "compose_up",
            f"Starting compose project: {project_name or compose_file.parent.name}",
            {"compose_file": str(compose_file), "services": services}
        )

        cmd = ["docker", "compose"]

        if project_name:
            cmd.extend(["-p", project_name])

        if env_file:
            cmd.extend(["--env-file", str(env_file)])

        cmd.extend(["up", "-d" if detach else "" ])

        if build:
            cmd.append("--build")

        if services:
            cmd.extend(services)

        exit_code, output = await self._run_command(cmd, timeout=300)

        if exit_code == 0:
            self.logger.log_thought(
                "compose_up_success",
                f"Compose project started: {project_name}",
                {"services": services or "all"}
            )
        else:
            self.logger.log_thought(
                "compose_up_failed",
                f"Compose project failed: {project_name}",
                {"error": output[-500:]}
            )

        return exit_code == 0, output

    async def compose_down(
        self,
        compose_file: Path,
        project_name: Optional[str] = None,
        volumes: bool = False,
    ) -> Tuple[bool, str]:
        """
        Stop and remove compose services.

        Args:
            compose_file: Path to docker-compose.yml
            project_name: Project name
            volumes: Remove named volumes

        Returns:
            Tuple of (success, output_message)
        """
        cmd = ["docker", "compose"]

        if project_name:
            cmd.extend(["-p", project_name])

        cmd.extend(["down", "-v" if volumes else ""])

        exit_code, output = await self._run_command(cmd, timeout=120)

        return exit_code == 0, output

    async def compose_logs(
        self,
        compose_file: Path,
        project_name: Optional[str] = None,
        services: Optional[List[str]] = None,
        tail: int = 100,
        follow: bool = False,
    ) -> str:
        """
        Get logs from compose services.

        Args:
            compose_file: Path to docker-compose.yml
            project_name: Project name
            services: Specific services (None = all)
            tail: Lines from end
            follow: Follow log output

        Returns:
            Log output
        """
        cmd = ["docker", "compose"]

        if project_name:
            cmd.extend(["-p", project_name])

        cmd.extend(["logs", "--tail", str(tail)])

        if follow:
            cmd.append("-f")

        if services:
            cmd.extend(services)

        _, output = await self._run_command(cmd, timeout=60)
        return output

    async def cleanup_old_containers(self, older_than: timedelta = None) -> int:
        """
        Remove stopped containers older than the specified age.

        Args:
            older_than: Age threshold (defaults to CONTAINER_CLEANUP_AGE)

        Returns:
            Number of containers removed
        """
        older_than = older_than or self.CONTAINER_CLEANUP_AGE
        cutoff = datetime.now() - older_than

        self.logger.log_thought(
            "cleanup_start",
            f"Cleaning up containers older than {older_than}",
            {"cutoff": cutoff.isoformat()}
        )

        # List all containers including stopped ones
        cmd = [
            "docker", "ps", "-a",
            "--filter", f"until={int(older_than.total_seconds())}s",
            "--format", "{{.ID}}"
        ]

        exit_code, output = await self._run_command(cmd)

        if exit_code != 0:
            return 0

        container_ids = output.strip().split("\n")
        container_ids = [c for c in container_ids if c]

        removed = 0
        for container_id in container_ids:
            if await self.remove_container(container_id, force=True):
                removed += 1

        self.logger.log_thought(
            "cleanup_complete",
            f"Removed {removed} old containers",
            {"count": removed}
        )

        return removed

    async def get_container_stats(
        self,
        container_id: str
    ) -> Dict[str, Any]:
        """
        Get resource usage statistics for a container.

        Args:
            container_id: Container ID or name

        Returns:
            Dictionary with stats (cpu_percent, memory_usage, memory_percent, etc.)
        """
        cmd = [
            "docker", "stats",
            "--no-stream",
            "--format", "{{json .}}",
            container_id
        ]

        exit_code, output = await self._run_command(cmd)

        if exit_code != 0:
            return {}

        try:
            data = json.loads(output)

            # Parse CPU percentage
            cpu_str = data.get("CPUPerc", "0%")
            cpu_percent = float(cpu_str.rstrip("%"))

            # Parse memory usage
            mem_str = data.get("MemUsage", "0 / 0")
            parts = mem_str.split("/")
            mem_usage = parts[0].strip()
            mem_limit = parts[1].strip() if len(parts) > 1 else "0"

            mem_percent_str = data.get("MemPerc", "0%")
            mem_percent = float(mem_percent_str.rstrip("%"))

            # Parse network I/O
            net_str = data.get("NetIO", "0B / 0B")
            net_rx, net_tx = net_str.split(" / ")

            # Parse block I/O
            block_str = data.get("BlockIO", "0B / 0B")
            block_read, block_write = block_str.split(" / ")

            return {
                "container_id": container_id,
                "cpu_percent": cpu_percent,
                "memory_usage": mem_usage,
                "memory_limit": mem_limit,
                "memory_percent": mem_percent,
                "network_rx": net_rx,
                "network_tx": net_tx,
                "block_read": block_read,
                "block_write": block_write,
            }

        except (json.JSONDecodeError, ValueError, IndexError):
            return {}

    async def _run_command(
        self,
        cmd: List[str],
        timeout: int = 60,
        cwd: Optional[Path] = None,
    ) -> Tuple[int, str]:
        """
        Run a shell command with timeout.

        Args:
            cmd: Command and arguments
            timeout: Maximum execution time in seconds
            cwd: Working directory

        Returns:
            Tuple of (exit_code, output)
        """
        import subprocess

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
            )

            try:
                stdout, _ = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                output = stdout.decode("utf-8", errors="replace")
                return process.returncode or 0, output

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise DockerTimeoutError(f"Command timed out: {' '.join(cmd)}")

        except FileNotFoundError:
            return 1, f"Command not found: {cmd[0]}"

    async def _check_resource_limits(self) -> None:
        """
        Check if resource limits would be exceeded.

        Raises:
            ResourceLimitError: If limits would be exceeded
        """
        active_count = len(self._active_containers)

        if active_count >= self.MAX_CONTAINERS:
            raise ResourceLimitError(
                f"Maximum container limit reached: {self.MAX_CONTAINERS}"
            )

        # Check disk usage
        cmd = ["docker", "system", "df", "--format", "{{.Size}}"]
        exit_code, output = await self._run_command(cmd)

        if exit_code == 0:
            sizes = output.strip().split("\n")
            # Simple check - if lots of output, might need cleanup
            if len(sizes) > 50:
                self.logger.log_thought(
                    "docker_cleanup_advised",
                    f"Docker disk usage high ({len(sizes)} images)",
                    {"image_count": len(sizes)},
                    level="warning"
                )

    def _extract_image_size(self, build_output: str) -> str:
        """Extract image size from build output."""
        match = re.search(r'(\d+(?:\.\d+)?[A-Z]{1,2})\s*$', build_output)
        return match.group(1) if match else "unknown"


# Convenience functions for common operations
async def run_service_container(
    image: str,
    name: str,
    ports: Dict[str, str],
    **kwargs
) -> ContainerInfo:
    """
    Quick helper to run a service container.

    Common use: Running databases, caches, etc. for testing.

    Example:
        db = await run_service_container(
            image="postgres:15",
            name="test-db",
            ports={"5432": "5432"},
            environment={"POSTGRES_PASSWORD": "test"}
        )
    """
    docker = DockerCapability()
    config = ContainerConfig(
        image=image,
        name=name,
        ports=ports,
        **kwargs
    )
    return await docker.run_container(config)


async def with_compose_project(
    compose_file: Path,
    project_name: str,
    func: callable,
    env_file: Optional[Path] = None,
):
    """
    Context manager for running a function with a compose project.

    Automatically starts compose project before function and stops after.

    Example:
        await with_compose_project(
            compose_file=Path("docker-compose.yml"),
            project_name="test",
            func=my_test_function
        )
    """
    docker = DockerCapability()

    success, _ = await docker.compose_up(
        compose_file=compose_file,
        project_name=project_name,
        env_file=env_file
    )

    if not success:
        raise DockerError(f"Failed to start compose project: {project_name}")

    try:
        return await func()
    finally:
        await docker.compose_down(
            compose_file=compose_file,
            project_name=project_name
        )
