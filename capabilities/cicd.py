"""
CI/CD Capability for PM Agent

Provides integration with CI/CD platforms including:
- GitHub Actions workflow management
- GitLab CI pipeline monitoring
- Workflow status checking
- Manual workflow triggering
- Log fetching and analysis
- Deployment approval gating

Supports:
- GitHub Actions (primary)
- GitLab CI
- Jenkins (planned)
- CircleCI (planned)
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from pm_agent.logger import ThoughtLogger


class CIPlatform(Enum):
    """Supported CI/CD platforms"""
    GITHUB_ACTIONS = "github"
    GITLAB_CI = "gitlab"
    JENKINS = "jenkins"
    CIRCLECI = "circleci"


class WorkflowStatus(Enum):
    """Workflow execution states"""
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    PENDING = "pending"


class ConclusionStatus(Enum):
    """Workflow conclusion states"""
    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    ACTION_REQUIRED = "action_required"


@dataclass
class WorkflowRun:
    """Information about a workflow run"""
    run_id: str
    name: str
    status: WorkflowStatus
    conclusion: Optional[ConclusionStatus]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    url: str
    event: str  # push, pull_request, manual, etc.
    branch: str
    commit_sha: str
    actor: str


@dataclass
class JobStep:
    """Information about a job step"""
    name: str
    status: ConclusionStatus
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    log_url: str


@dataclass
class CIConfig:
    """Configuration for CI/CD integration"""
    platform: CIPlatform
    token: str
    repo_owner: str
    repo_name: str
    api_url: Optional[str] = None  # For self-hosted instances


@dataclass
class DeploymentGate:
    """Pre-deployment check requirements"""
    require_tests_pass: bool = True
    require_security_scan: bool = True
    require_manual_approval: bool = False
    min_coverage_percent: Optional[float] = None
    forbidden_branches: List[str] = field(default_factory=lambda: ["main", "master"])


class CICDError(Exception):
    """Base exception for CI/CD operations"""
    pass


class WorkflowNotFoundError(CICDError):
    """Workflow not found"""
    pass


class AuthenticationError(CICDError):
    """Authentication failed"""
    pass


class CICDCapability:
    """
    CI/CD integration capability.

    Provides workflow monitoring, triggering, and management
    for GitHub Actions, GitLab CI, and other platforms.
    """

    def __init__(self, config: CIConfig, logger: Optional[ThoughtLogger] = None):
        """
        Initialize CI/CD capability.

        Args:
            config: CI/CD configuration
            logger: Optional thought logger for tracking operations
        """
        self.config = config
        self.logger = logger or ThoughtLogger("cicd_capability")
        self._session: Optional[aiohttp.ClientSession] = None

        # Set default API URLs
        if config.api_url is None:
            if config.platform == CIPlatform.GITHUB_ACTIONS:
                self.api_url = "https://api.github.com"
            elif config.platform == CIPlatform.GITLAB_CI:
                self.api_url = "https://gitlab.com/api/v4"
            else:
                self.api_url = ""
        else:
            self.api_url = config.api_url

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            headers = {"Authorization": f"token {self.config.token}"}
            if self.config.platform == CIPlatform.GITLAB_CI:
                headers = {"PRIVATE-TOKEN": self.config.token}

            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_workflow_runs(
        self,
        branch: Optional[str] = None,
        status: Optional[WorkflowStatus] = None,
        limit: int = 20,
    ) -> List[WorkflowRun]:
        """
        Get workflow runs for the repository.

        Args:
            branch: Filter by branch
            status: Filter by status
            limit: Maximum number of runs to return

        Returns:
            List of workflow runs
        """
        self.logger.log_thought(
            "get_workflows",
            f"Fetching workflow runs for {self.config.repo_owner}/{self.config.repo_name}",
            {"branch": branch, "status": status.value if status else None}
        )

        if self.config.platform == CIPlatform.GITHUB_ACTIONS:
            return await self._get_github_workflows(branch, status, limit)
        elif self.config.platform == CIPlatform.GITLAB_CI:
            return await self._get_gitlab_pipelines(branch, status, limit)
        else:
            raise CICDError(f"Platform {self.config.platform} not yet supported")

    async def _get_github_workflows(
        self,
        branch: Optional[str],
        status: Optional[WorkflowStatus],
        limit: int,
    ) -> List[WorkflowRun]:
        """Get GitHub Actions workflow runs"""
        session = await self._get_session()

        url = (
            f"{self.api_url}/repos/{self.config.repo_owner}"
            f"/{self.config.repo_name}/actions/runs"
        )

        params = {"per_page": limit}
        if branch:
            params["branch"] = branch
        if status:
            params["status"] = status.value

        async with session.get(url, params=params) as response:
            if response.status == 401:
                raise AuthenticationError("Invalid GitHub token")
            response.raise_for_status()
            data = await response.json()

        workflows = []
        for run in data.get("workflow_runs", []):
            workflows.append(WorkflowRun(
                run_id=str(run.get("id")),
                name=run.get("name", ""),
                status=self._parse_github_status(run.get("status")),
                conclusion=self._parse_github_conclusion(run.get("conclusion")),
                created_at=datetime.fromisoformat(
                    run.get("created_at", "").replace("Z", "+00:00")
                ),
                started_at=datetime.fromisoformat(
                    run.get("run_started_at", "").replace("Z", "+00:00")
                ) if run.get("run_started_at") else None,
                completed_at=datetime.fromisoformat(
                    run.get("updated_at", "").replace("Z", "+00:00")
                ) if run.get("updated_at") else None,
                url=run.get("html_url", ""),
                event=run.get("event", ""),
                branch=run.get("head_branch", ""),
                commit_sha=run.get("head_sha", "")[:7],
                actor=run.get("triggering_actor", {}).get("login", ""),
            ))

        return workflows

    async def _get_gitlab_pipelines(
        self,
        branch: Optional[str],
        status: Optional[WorkflowStatus],
        limit: int,
    ) -> List[WorkflowRun]:
        """Get GitLab CI pipelines"""
        session = await self._get_session()

        project_id = await self._get_gitlab_project_id()
        url = f"{self.api_url}/projects/{project_id}/pipelines"

        params = {"per_page": limit}
        if branch:
            params["ref"] = branch
        if status:
            params["status"] = status.value

        async with session.get(url, params=params) as response:
            if response.status == 401:
                raise AuthenticationError("Invalid GitLab token")
            response.raise_for_status()
            data = await response.json()

        pipelines = []
        for pipeline in data:
            pipelines.append(WorkflowRun(
                run_id=str(pipeline.get("id")),
                name=f"Pipeline {pipeline.get('id')}",
                status=self._parse_gitlab_status(pipeline.get("status")),
                conclusion=self._parse_gitlab_status(pipeline.get("status")),
                created_at=datetime.fromisoformat(
                    pipeline.get("created_at", "").replace("Z", "+00:00")
                ),
                started_at=None,
                completed_at=datetime.fromisoformat(
                    pipeline.get("updated_at", "").replace("Z", "+00:00")
                ) if pipeline.get("updated_at") else None,
                url=pipeline.get("web_url", ""),
                event="push",
                branch=pipeline.get("ref", ""),
                commit_sha=pipeline.get("sha", "")[:7],
                actor=pipeline.get("user", {}).get("name", ""),
            ))

        return pipelines

    async def trigger_workflow(
        self,
        workflow_file: str,
        branch: str = "main",
        inputs: Optional[Dict[str, Any]] = None,
    ) -> WorkflowRun:
        """
        Manually trigger a workflow run.

        Args:
            workflow_file: Workflow filename (e.g., "ci.yml")
            branch: Branch to run against
            inputs: Workflow inputs

        Returns:
            The triggered workflow run

        Raises:
            WorkflowNotFoundError: If workflow doesn't exist
        """
        self.logger.log_thought(
            "trigger_workflow",
            f"Triggering workflow: {workflow_file} on {branch}",
            {"workflow": workflow_file, "branch": branch, "inputs": inputs}
        )

        if self.config.platform == CIPlatform.GITHUB_ACTIONS:
            return await self._trigger_github_workflow(workflow_file, branch, inputs)
        else:
            raise CICDError(f"Trigger not supported for {self.config.platform}")

    async def _trigger_github_workflow(
        self,
        workflow_file: str,
        branch: str,
        inputs: Optional[Dict[str, Any]],
    ) -> WorkflowRun:
        """Trigger GitHub Actions workflow"""
        session = await self._get_session()

        # Get workflow ID from filename
        workflows_url = (
            f"{self.api_url}/repos/{self.config.repo_owner}"
            f"/{self.config.repo_name}/actions/workflows"
        )

        async with session.get(workflows_url) as response:
            response.raise_for_status()
            data = await response.json()

        workflow_id = None
        for wf in data.get("workflows", []):
            if wf.get("path", "").endswith(workflow_file):
                workflow_id = wf.get("id")
                break

        if not workflow_id:
            raise WorkflowNotFoundError(f"Workflow not found: {workflow_file}")

        # Trigger the workflow
        dispatch_url = (
            f"{self.api_url}/repos/{self.config.repo_owner}"
            f"/{self.config.repo_name}/actions/workflows/{workflow_id}/dispatches"
        )

        payload = {"ref": branch}
        if inputs:
            payload["inputs"] = inputs

        async with session.post(dispatch_url, json=payload) as response:
            if response.status == 404:
                raise WorkflowNotFoundError(f"Workflow not found: {workflow_file}")
            response.raise_for_status()

        # Get the latest run to return
        runs = await self.get_workflow_runs(branch=branch, limit=1)
        return runs[0] if runs else None

    async def get_workflow_logs(
        self,
        run_id: str,
        job_name: Optional[str] = None,
    ) -> str:
        """
        Get logs from a workflow run.

        Args:
            run_id: Workflow run ID
            job_name: Specific job name (optional)

        Returns:
            Log output as string
        """
        self.logger.log_thought(
            "get_logs",
            f"Fetching logs for run: {run_id}",
            {"run_id": run_id, "job": job_name}
        )

        if self.config.platform == CIPlatform.GITHUB_ACTIONS:
            return await self._get_github_logs(run_id, job_name)
        else:
            raise CICDError(f"Logs not supported for {self.config.platform}")

    async def _get_github_logs(
        self,
        run_id: str,
        job_name: Optional[str],
    ) -> str:
        """Get GitHub Actions workflow logs"""
        session = await self._get_session()

        # First, get jobs to find the job ID
        jobs_url = (
            f"{self.api_url}/repos/{self.config.repo_owner}"
            f"/{self.config.repo_name}/actions/runs/{run_id}/jobs"
        )

        async with session.get(jobs_url) as response:
            response.raise_for_status()
            data = await response.json()

        jobs = data.get("jobs", [])
        if job_name:
            jobs = [j for j in jobs if j.get("name") == job_name]

        if not jobs:
            return "No jobs found"

        # Get logs for the first matching job
        job_id = jobs[0].get("id")
        logs_url = (
            f"{self.api_url}/repos/{self.config.repo_owner}"
            f"/{self.config.repo_name}/actions/jobs/{job_id}/logs"
        )

        async with session.get(logs_url) as response:
            if response.status == 404:
                return "Logs not found or expired"
            response.raise_for_status()

            # Logs are returned as plain text
            return await response.text()

    async def get_deployment_status(
        self,
        branch: str = "main",
    ) -> Dict[str, Any]:
        """
        Get deployment status for a branch.

        Args:
            branch: Branch to check

        Returns:
            Dictionary with deployment status info
        """
        self.logger.log_thought(
            "deployment_status",
            f"Checking deployment status for {branch}",
            {"branch": branch}
        )

        # Get recent workflows
        workflows = await self.get_workflow_runs(branch=branch, limit=10)

        # Filter for deployment workflows
        deployment_workflows = [
            w for w in workflows
            if "deploy" in w.name.lower() or "release" in w.name.lower()
        ]

        if not deployment_workflows:
            return {
                "status": "unknown",
                "last_deployment": None,
                "message": "No deployment workflows found"
            }

        latest = deployment_workflows[0]

        return {
            "status": latest.conclusion.value if latest.conclusion else "unknown",
            "last_deployment": latest.completed_at,
            "workflow": latest.name,
            "url": latest.url,
            "commit": latest.commit_sha,
        }

    async def check_deployment_gates(
        self,
        run: WorkflowRun,
        gates: DeploymentGate,
    ) -> Tuple[bool, List[str]]:
        """
        Check if deployment meets all gate requirements.

        Args:
            run: Workflow run to check
            gates: Deployment gate requirements

        Returns:
            Tuple of (passes_gate, failure_reasons)
        """
        failures = []

        # Check test requirement
        if gates.require_tests_pass:
            if run.conclusion != ConclusionStatus.SUCCESS:
                failures.append("Tests did not pass")

        # Check branch restriction
        if run.branch in gates.forbidden_branches:
            failures.append(f"Cannot deploy directly from {run.branch}")

        # Check manual approval
        if gates.require_manual_approval:
            # This would need to be implemented separately
            failures.append("Manual approval required (not implemented)")

        # Check coverage (would need to fetch from workflow artifacts)
        if gates.min_coverage_percent:
            # This requires parsing coverage reports
            failures.append("Coverage check not implemented")

        passes = len(failures) == 0

        self.logger.log_thought(
            "gate_check",
            f"Deployment gate check: {passes}",
            {"passes": passes, "failures": failures}
        )

        return passes, failures

    async def _get_gitlab_project_id(self) -> str:
        """Get GitLab project ID from owner/repo"""
        session = await self._get_session()

        # Encode project path
        project_path = f"{self.config.repo_owner}%2F{self.config.repo_name}"
        url = f"{self.api_url}/projects/{project_path}"

        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.json()

        return str(data.get("id"))

    def _parse_github_status(self, status: str) -> WorkflowStatus:
        """Parse GitHub status to WorkflowStatus"""
        mapping = {
            "queued": WorkflowStatus.QUEUED,
            "in_progress": WorkflowStatus.IN_PROGRESS,
            "completed": WorkflowStatus.COMPLETED,
            "pending": WorkflowStatus.PENDING,
        }
        return mapping.get(status, WorkflowStatus.PENDING)

    def _parse_github_conclusion(self, conclusion: Optional[str]) -> Optional[ConclusionStatus]:
        """Parse GitHub conclusion to ConclusionStatus"""
        if not conclusion:
            return None

        mapping = {
            "success": ConclusionStatus.SUCCESS,
            "failure": ConclusionStatus.FAILURE,
            "neutral": ConclusionStatus.NEUTRAL,
            "cancelled": ConclusionStatus.CANCELLED,
            "timed_out": ConclusionStatus.TIMED_OUT,
            "action_required": ConclusionStatus.ACTION_REQUIRED,
        }
        return mapping.get(conclusion)

    def _parse_gitlab_status(self, status: str) -> ConclusionStatus:
        """Parse GitLab status to ConclusionStatus"""
        mapping = {
            "success": ConclusionStatus.SUCCESS,
            "failed": ConclusionStatus.FAILURE,
            "canceled": ConclusionStatus.CANCELLED,
            "skipped": ConclusionStatus.NEUTRAL,
            "manual": ConclusionStatus.ACTION_REQUIRED,
        }
        return mapping.get(status, ConclusionStatus.NEUTRAL)


# Convenience functions
async def check_ci_status(
    token: str,
    repo_owner: str,
    repo_name: str,
    branch: str = "main",
) -> Dict[str, Any]:
    """
    Quick helper to check CI/CD status for a repository.

    Example:
        status = await check_ci_status(
            token="ghp_xxx",
            repo_owner="owner",
            repo_name="repo",
            branch="main"
        )
    """
    config = CIConfig(
        platform=CIPlatform.GITHUB_ACTIONS,
        token=token,
        repo_owner=repo_owner,
        repo_name=repo_name
    )

    cicd = CICDCapability(config)
    try:
        workflows = await cicd.get_workflow_runs(branch=branch, limit=5)

        if not workflows:
            return {"status": "no_workflows", "latest": None}

        latest = workflows[0]
        return {
            "status": latest.conclusion.value if latest.conclusion else latest.status.value,
            "latest": {
                "name": latest.name,
                "created_at": latest.created_at.isoformat(),
                "url": latest.url,
                "commit": latest.commit_sha,
            }
        }
    finally:
        await cicd.close()
