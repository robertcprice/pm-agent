"""
Deployment Capability for PM Agent

Provides deployment automation including:
- Vercel deployments
- AWS deployments (S3, Lambda, ECS)
- Environment variable management
- Rollback capability
- Deployment status monitoring

Supports:
- Vercel (primary)
- AWS S3 (static sites)
- AWS Lambda (functions)
- Custom webhooks
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from pm_agent.logger import ThoughtLogger


class DeploymentPlatform(Enum):
    """Supported deployment platforms"""
    VERCEL = "vercel"
    NETLIFY = "netlify"
    AWS_S3 = "aws_s3"
    AWS_LAMBDA = "aws_lambda"
    AWS_ECS = "aws_ecs"
    RAILWAY = "railway"
    CUSTOM_WEBHOOK = "webhook"


class DeploymentStatus(Enum):
    """Deployment states"""
    BUILDING = "building"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ERROR = "error"
    QUEUED = "queued"


@dataclass
class DeploymentConfig:
    """Configuration for a deployment target"""
    platform: DeploymentPlatform
    project_name: str
    token: Optional[str] = None
    api_url: Optional[str] = None
    environment: str = "production"
    region: str = "us-east-1"
    build_command: Optional[str] = None
    output_directory: str = "dist"


@dataclass
class Deployment:
    """Information about a deployment"""
    deployment_id: str
    platform: DeploymentPlatform
    status: DeploymentStatus
    url: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    project: str
    environment: str
    commit_sha: Optional[str] = None
    commit_message: Optional[str] = None
    branch: Optional[str] = None
    logs_url: Optional[str] = None


@dataclass
class EnvironmentVariable:
    """Environment variable configuration"""
    key: str
    value: str
    environment: List[str] = field(default_factory=lambda: ["production"])
    sensitive: bool = True


class DeploymentError(Exception):
    """Base exception for deployment operations"""
    pass


class AuthenticationError(DeploymentError):
    """Deployment authentication failed"""
    pass


class DeploymentFailedError(DeploymentError):
    """Deployment operation failed"""
    pass


class DeploymentCapability:
    """
    Deployment automation capability.

    Supports multiple deployment platforms with unified
    interface for deployments, rollbacks, and monitoring.
    """

    def __init__(self, config: DeploymentConfig, logger: Optional[ThoughtLogger] = None):
        """
        Initialize deployment capability.

        Args:
            config: Deployment configuration
            logger: Optional thought logger for tracking operations
        """
        self.config = config
        self.logger = logger or ThoughtLogger("deployment_capability")
        self._session: Optional[aiohttp.ClientSession] = None

        # Set default API URLs
        if config.api_url is None:
            if config.platform == DeploymentPlatform.VERCEL:
                self.api_url = "https://api.vercel.com"
            elif config.platform == DeploymentPlatform.NETLIFY:
                self.api_url = "https://api.netlify.com"
            else:
                self.api_url = ""
        else:
            self.api_url = config.api_url

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            headers = {}
            if self.config.token:
                if self.config.platform == DeploymentPlatform.VERCEL:
                    headers["Authorization"] = f"Bearer {self.config.token}"
                elif self.config.platform == DeploymentPlatform.NETLIFY:
                    headers["Authorization"] = f"Bearer {self.config.token}"
                else:
                    headers["Authorization"] = f"Bearer {self.config.token}"

            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def deploy(
        self,
        project_path: Path,
        branch: Optional[str] = None,
        environment: Optional[str] = None,
        force: bool = False,
    ) -> Deployment:
        """
        Deploy a project.

        Args:
            project_path: Path to project directory
            branch: Git branch to deploy
            environment: Target environment
            force: Force new deployment even if no changes

        Returns:
            Deployment information

        Raises:
            DeploymentFailedError: If deployment fails
        """
        self.logger.log_thought(
            "deploy_start",
            f"Deploying {self.config.project_name} to {self.config.platform.value}",
            {
                "project": self.config.project_name,
                "environment": environment or self.config.environment,
                "branch": branch
            }
        )

        if self.config.platform == DeploymentPlatform.VERCEL:
            return await self._deploy_vercel(project_path, branch, environment, force)
        elif self.config.platform == DeploymentPlatform.AWS_S3:
            return await self._deploy_s3(project_path, branch, environment, force)
        elif self.config.platform == DeploymentPlatform.AWS_LAMBDA:
            return await self._deploy_lambda(project_path, branch, environment, force)
        else:
            raise DeploymentError(f"Platform {self.config.platform} not yet supported")

    async def _deploy_vercel(
        self,
        project_path: Path,
        branch: Optional[str],
        environment: Optional[str],
        force: bool,
    ) -> Deployment:
        """Deploy to Vercel"""
        session = await self._get_session()

        # Trigger deployment via CLI
        cmd = ["vercel", "--prod", "--yes", "--token", self.config.token]

        if force:
            cmd.append("--force")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=project_path,
        )

        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)

        output = stdout.decode()
        error = stderr.decode()

        if process.returncode != 0:
            raise DeploymentFailedError(f"Vercel deployment failed: {error}")

        # Parse output for deployment URL
        import re
        url_match = re.search(r'https://[\w.-]+\.vercel\.app', output)

        if not url_match:
            raise DeploymentFailedError("Could not find deployment URL")

        deployment = Deployment(
            deployment_id="vercel-" + datetime.now().strftime("%Y%m%d%H%M%S"),
            platform=DeploymentPlatform.VERCEL,
            status=DeploymentStatus.READY,
            url=url_match.group(0),
            created_at=datetime.now(),
            project=self.config.project_name,
            environment=environment or self.config.environment,
        )

        self.logger.log_thought(
            "deploy_complete",
            f"Vercel deployment complete: {deployment.url}",
            {"url": deployment.url, "deployment_id": deployment.deployment_id}
        )

        return deployment

    async def _deploy_s3(
        self,
        project_path: Path,
        branch: Optional[str],
        environment: Optional[str],
        force: bool,
    ) -> Deployment:
        """Deploy static site to S3"""
        import shutil

        # Build project first
        if self.config.build_command:
            build_cmd = self.config.build_command.split()
            process = await asyncio.create_subprocess_exec(
                *build_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_path,
            )

            await asyncio.wait_for(process.communicate(), timeout=300)

            if process.returncode != 0:
                raise DeploymentFailedError("Build command failed")

        # Sync to S3
        output_dir = project_path / self.config.output_directory
        bucket_name = self.config.project_name

        cmd = [
            "aws", "s3", "sync",
            str(output_dir),
            f"s3://{bucket_name}",
            "--delete"
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

        if process.returncode != 0:
            raise DeploymentFailedError(f"S3 sync failed: {stderr.decode()}")

        # Invalidate CloudFront cache if configured
        # (would need distribution ID in config)

        return Deployment(
            deployment_id="s3-" + datetime.now().strftime("%Y%m%d%H%M%S"),
            platform=DeploymentPlatform.AWS_S3,
            status=DeploymentStatus.READY,
            url=f"https://{bucket_name}.s3.amazonaws.com",
            created_at=datetime.now(),
            project=self.config.project_name,
            environment=environment or self.config.environment,
        )

    async def _deploy_lambda(
        self,
        project_path: Path,
        branch: Optional[str],
        environment: Optional[str],
        force: bool,
    ) -> Deployment:
        """Deploy to AWS Lambda"""
        # This would use boto3 or AWS CLI
        # Placeholder for now
        raise DeploymentError("Lambda deployment not yet implemented")

    async def rollback(
        self,
        deployment_id: Optional[str] = None,
        target_commit: Optional[str] = None,
    ) -> Deployment:
        """
        Rollback to a previous deployment.

        Args:
            deployment_id: Specific deployment to rollback to
            target_commit: Commit SHA to deploy

        Returns:
            New deployment information
        """
        self.logger.log_thought(
            "rollback_start",
            f"Rolling back deployment",
            {"deployment_id": deployment_id, "target_commit": target_commit}
        )

        if self.config.platform == DeploymentPlatform.VERCEL:
            return await self._rollback_vercel(deployment_id, target_commit)
        else:
            raise DeploymentError(f"Rollback not supported for {self.config.platform}")

    async def _rollback_vercel(
        self,
        deployment_id: Optional[str],
        target_commit: Optional[str],
    ) -> Deployment:
        """Rollback Vercel deployment"""
        session = await self._get_session()

        # Get deployment list to find previous one
        list_url = (
            f"{self.api_url}/v6/now/deployments"
        )

        params = {
            "projectId": self.config.project_name,
            "limit": 10
        }

        async with session.get(list_url, params=params) as response:
            if response.status == 401:
                raise AuthenticationError("Invalid Vercel token")
            response.raise_for_status()
            data = await response.json()

        deployments = data.get("deployments", [])
        if len(deployments) < 2:
            raise DeploymentError("No previous deployment to rollback to")

        # Get second-to-last deployment
        previous = deployments[1]

        # Promote previous deployment
        promote_url = (
            f"{self.api_url}/v2/now/deployments/{previous['uid']}/promote"
        )

        async with session.post(promote_url) as response:
            response.raise_for_status()
            result = await response.json()

        return Deployment(
            deployment_id=result.get("uid", ""),
            platform=DeploymentPlatform.VERCEL,
            status=DeploymentStatus.READY,
            url=result.get("url"),
            created_at=datetime.now(),
            project=self.config.project_name,
            environment=self.config.environment,
        )

    async def get_deployment_status(
        self,
        deployment_id: str,
    ) -> DeploymentStatus:
        """
        Get current status of a deployment.

        Args:
            deployment_id: Deployment identifier

        Returns:
            Current deployment status
        """
        if self.config.platform == DeploymentPlatform.VERCEL:
            return await self._get_vercel_status(deployment_id)
        else:
            return DeploymentStatus.ERROR

    async def _get_vercel_status(self, deployment_id: str) -> DeploymentStatus:
        """Get Vercel deployment status"""
        session = await self._get_session()

        url = f"{self.api_url}/v13/deployments/{deployment_id}"

        async with session.get(url) as response:
            if response.status == 404:
                return DeploymentStatus.ERROR
            response.raise_for_status()
            data = await response.json()

        state = data.get("state", "ERROR")
        status_map = {
            "BUILDING": DeploymentStatus.BUILDING,
            "READY": DeploymentStatus.READY,
            "FAILED": DeploymentStatus.FAILED,
            "CANCELED": DeploymentStatus.CANCELLED,
            "QUEUED": DeploymentStatus.QUEUED,
            "ERROR": DeploymentStatus.ERROR,
        }

        return status_map.get(state.upper(), DeploymentStatus.ERROR)

    async def set_environment_variables(
        self,
        variables: List[EnvironmentVariable],
    ) -> bool:
        """
        Set environment variables for the project.

        Args:
            variables: List of environment variables

        Returns:
            True if successful
        """
        self.logger.log_thought(
            "set_env_vars",
            f"Setting {len(variables)} environment variables",
            {"count": len(variables)}
        )

        if self.config.platform == DeploymentPlatform.VERCEL:
            return await self._set_vercel_env_vars(variables)
        else:
            self.logger.log_thought(
                "env_vars_not_supported",
                f"Environment variables not supported for {self.config.platform}",
                {},
                level="warning"
            )
            return False

    async def _set_vercel_env_vars(
        self,
        variables: List[EnvironmentVariable],
    ) -> bool:
        """Set Vercel environment variables"""
        session = await self._get_session()

        for var in variables:
            url = (
                f"{self.api_url}/v9/projects/"
                f"{self.config.project_name}/env/{var.key}"
            )

            payload = {
                "key": var.key,
                "value": var.value,
                "type": "encrypted" if var.sensitive else "plain",
                "target": var.environment
            }

            async with session.post(url, json=payload) as response:
                if response.status not in (200, 409):  # 409 = already exists
                    response.raise_for_status()

        return True

    async def get_deployment_logs(
        self,
        deployment_id: str,
        limit: int = 100,
    ) -> str:
        """
        Get logs from a deployment.

        Args:
            deployment_id: Deployment identifier
            limit: Maximum number of log lines

        Returns:
            Log output
        """
        if self.config.platform == DeploymentPlatform.VERCEL:
            return await self._get_vercel_logs(deployment_id, limit)
        else:
            return "Logs not available for this platform"

    async def _get_vercel_logs(self, deployment_id: str, limit: int) -> str:
        """Get Vercel deployment logs"""
        # Use vercel CLI to get logs
        cmd = ["vercel", "logs", deployment_id, "-n", str(limit)]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)

        return stdout.decode()

    async def list_deployments(
        self,
        limit: int = 20,
    ) -> List[Deployment]:
        """
        List recent deployments.

        Args:
            limit: Maximum number of deployments

        Returns:
            List of deployments
        """
        if self.config.platform == DeploymentPlatform.VERCEL:
            return await self._list_vercel_deployments(limit)
        else:
            return []

    async def _list_vercel_deployments(self, limit: int) -> List[Deployment]:
        """List Vercel deployments"""
        session = await self._get_session()

        url = f"{self.api_url}/v6/now/deployments"

        params = {
            "projectId": self.config.project_name,
            "limit": limit
        }

        async with session.get(url, params=params) as response:
            if response.status == 401:
                raise AuthenticationError("Invalid Vercel token")
            response.raise_for_status()
            data = await response.json()

        deployments = []
        for dep in data.get("deployments", []):
            state = dep.get("state", "UNKNOWN")
            status_map = {
                "BUILDING": DeploymentStatus.BUILDING,
                "READY": DeploymentStatus.READY,
                "FAILED": DeploymentStatus.FAILED,
                "CANCELED": DeploymentStatus.CANCELLED,
                "QUEUED": DeploymentStatus.QUEUED,
            }

            deployments.append(Deployment(
                deployment_id=dep.get("uid", ""),
                platform=DeploymentPlatform.VERCEL,
                status=status_map.get(state, DeploymentStatus.ERROR),
                url=dep.get("url"),
                created_at=datetime.fromisoformat(
                    dep.get("createdAt", "").replace("Z", "+00:00")
                ) if dep.get("createdAt") else datetime.now(),
                project=self.config.project_name,
                environment=self.config.environment,
                commit_sha=dep.get("gitCommitSha", "")[:7] if dep.get("gitCommitSha") else None,
            ))

        return deployments


# Convenience functions
async def deploy_to_vercel(
    token: str,
    project_name: str,
    project_path: Path,
    environment: str = "production",
) -> Deployment:
    """
    Quick helper to deploy to Vercel.

    Example:
        deployment = await deploy_to_vercel(
            token="vercel_token",
            project_name="my-app",
            project_path=Path("/path/to/project")
        )
        print(f"Deployed to: {deployment.url}")
    """
    config = DeploymentConfig(
        platform=DeploymentPlatform.VERCEL,
        project_name=project_name,
        token=token,
        environment=environment
    )

    deploy_cap = DeploymentCapability(config)
    try:
        return await deploy_cap.deploy(project_path)
    finally:
        await deploy_cap.close()


async def deploy_to_s3(
    project_name: str,
    project_path: Path,
    output_dir: str = "dist",
    build_command: Optional[str] = None,
) -> Deployment:
    """
    Quick helper to deploy static site to S3.

    Example:
        deployment = await deploy_to_s3(
            project_name="my-bucket",
            project_path=Path("/path/to/project"),
            build_command="npm run build"
        )
    """
    config = DeploymentConfig(
        platform=DeploymentPlatform.AWS_S3,
        project_name=project_name,
        output_directory=output_dir,
        build_command=build_command
    )

    deploy_cap = DeploymentCapability(config)
    return await deploy_cap.deploy(project_path)
