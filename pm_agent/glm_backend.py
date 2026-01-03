"""
GLM 4.7 Backend for PM Agent - Use GLM instead of Claude Code.

This module provides a GLM-powered coding agent that can be used as an
alternative to Claude Code CLI. It supports both:

1. Direct GLM API calls (for simple generation)
2. Claude Code CLI with GLM backend (via Z.ai proxy)

GLM 4.7 offers excellent code generation at lower cost than Claude.
"""

import os
import json
import logging
import subprocess
import shutil
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
from enum import Enum
import httpx

logger = logging.getLogger(__name__)


# GLM API configuration
# Direct GLM API (for direct calls without Claude Code)
GLM_API_BASE = "https://open.bigmodel.cn/api/paas/v4"

# Z.AI Proxy - Makes Claude Code use GLM as backend
# This is the recommended way: Claude Code CLI → Z.AI Proxy → GLM
ZAI_PROXY_BASE = "https://api.z.ai/api/anthropic"


class GLMModel(Enum):
    """Available GLM models."""
    GLM_4_AIR = "glm-4-air"           # Balanced
    GLM_4 = "glm-4"                   # Standard
    GLM_4_PLUS = "glm-4-plus"         # Advanced
    GLM_4_7 = "glm-4.7"               # Latest and best


@dataclass
class GLMResult:
    """Result from GLM execution."""
    success: bool
    output: str
    files_modified: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    tokens_used: int = 0
    model: str = "glm-4.7"


@dataclass
class GLMConfig:
    """Configuration for GLM backend."""
    api_key: str
    model: GLMModel = GLMModel.GLM_4_7
    base_url: str = GLM_API_BASE
    timeout_seconds: int = 300
    max_tokens: int = 4096
    temperature: float = 0.7
    use_zai_proxy: bool = False  # Use Z.ai proxy for Claude Code compatibility
    zai_proxy_url: str = ZAI_PROXY_BASE


class GLMDirectClient:
    """Direct client for GLM API calls."""

    def __init__(self, config: GLMConfig):
        self.config = config
        self.client = httpx.Client(timeout=config.timeout_seconds)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GLMResult:
        """Generate a response from GLM.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            temperature: Override temperature
            max_tokens: Override max tokens

        Returns:
            GLMResult with the response
        """
        start_time = time.time()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.post(
                f"{self.config.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model.value,
                    "messages": messages,
                    "temperature": temperature or self.config.temperature,
                    "max_tokens": max_tokens or self.config.max_tokens,
                }
            )

            response.raise_for_status()
            data = response.json()

            output = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)

            return GLMResult(
                success=True,
                output=output,
                duration_seconds=time.time() - start_time,
                tokens_used=tokens,
                model=self.config.model.value,
            )

        except httpx.HTTPStatusError as e:
            return GLMResult(
                success=False,
                output="",
                error_message=f"HTTP error: {e.response.status_code} - {e.response.text}",
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            return GLMResult(
                success=False,
                output="",
                error_message=str(e),
                duration_seconds=time.time() - start_time,
            )


class GLMCodeAgent:
    """GLM-powered coding agent using Claude Code CLI with GLM backend.

    This spawns Claude Code CLI but routes requests through Z.ai proxy
    to use GLM as the underlying model. This gives you Claude Code's
    powerful tool use (file editing, search, etc.) with GLM's model.
    """

    # Model mapping for Claude Code - all use GLM 4.7
    MODEL_MAP = {
        "haiku": "glm-4.7",
        "sonnet": "glm-4.7",
        "opus": "glm-4.7",
    }

    def __init__(self, config: GLMConfig):
        """Initialize the GLM code agent.

        Args:
            config: GLM configuration
        """
        self.config = config
        self.direct_client = GLMDirectClient(config)

    def is_available(self) -> bool:
        """Check if the agent can be used."""
        if not self.config.api_key:
            logger.warning("GLM: No API key configured")
            return False

        if self.config.use_zai_proxy and not shutil.which("claude"):
            logger.warning("GLM: 'claude' CLI not found for proxy mode")
            return False

        return True

    def execute_task(
        self,
        description: str,
        working_directory: Optional[Path] = None,
        context_files: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        acceptance_criteria: Optional[List[str]] = None,
        model_tier: str = "sonnet",
        timeout_seconds: int = 600,
    ) -> GLMResult:
        """Execute a coding task using GLM.

        Args:
            description: Task description
            working_directory: Directory to work in
            context_files: Files to include as context
            constraints: Task constraints
            acceptance_criteria: Success criteria
            model_tier: Model tier (haiku, sonnet, opus)
            timeout_seconds: Timeout for the task

        Returns:
            GLMResult with task outcome
        """
        start_time = time.time()
        cwd = working_directory or Path.cwd()

        # Build full prompt with context
        full_prompt = self._build_task_prompt(
            description, context_files, constraints, acceptance_criteria, cwd
        )

        if self.config.use_zai_proxy:
            # Use Claude Code CLI with GLM backend via Z.ai
            return self._execute_via_proxy(
                full_prompt, cwd, model_tier, timeout_seconds, start_time
            )
        else:
            # Use direct GLM API (simpler, no tool use)
            return self._execute_direct(
                full_prompt, cwd, context_files, start_time
            )

    def _build_task_prompt(
        self,
        description: str,
        context_files: Optional[List[str]],
        constraints: Optional[List[str]],
        acceptance_criteria: Optional[List[str]],
        cwd: Path,
    ) -> str:
        """Build the full task prompt with context."""
        parts = [f"# Task\n{description}"]

        # Add file context
        if context_files:
            context_parts = []
            for file_path in context_files[:10]:  # Limit files
                full_path = Path(file_path) if Path(file_path).is_absolute() else cwd / file_path
                if full_path.exists():
                    try:
                        content = full_path.read_text()[:20000]
                        context_parts.append(f"## {file_path}\n```\n{content}\n```")
                    except Exception as e:
                        context_parts.append(f"## {file_path}\n[Error reading: {e}]")

            if context_parts:
                parts.append("# Context Files\n" + "\n\n".join(context_parts))

        # Add constraints
        if constraints:
            parts.append("# Constraints\n" + "\n".join(f"- {c}" for c in constraints))

        # Add acceptance criteria
        if acceptance_criteria:
            parts.append("# Acceptance Criteria\n" + "\n".join(f"- {c}" for c in acceptance_criteria))

        return "\n\n".join(parts)

    def _execute_via_proxy(
        self,
        prompt: str,
        cwd: Path,
        model_tier: str,
        timeout_seconds: int,
        start_time: float,
    ) -> GLMResult:
        """Execute task via Claude Code CLI with GLM backend."""
        # Build environment for GLM backend
        env = os.environ.copy()
        env["ANTHROPIC_AUTH_TOKEN"] = self.config.api_key
        env["ANTHROPIC_BASE_URL"] = self.config.zai_proxy_url
        env["API_TIMEOUT_MS"] = str(timeout_seconds * 1000)

        # Model mappings - all use GLM 4.7
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = "glm-4.7"
        env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = "glm-4.7"
        env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = "glm-4.7"

        # Build command
        cmd = [
            "claude",
            "--print",
            "--dangerously-skip-permissions",
            "--model", model_tier,
            "--allowedTools", "Read,Glob,Grep,Write,Edit,Bash",
            "-p", prompt,
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )

            output = result.stdout.strip()
            if result.stderr:
                output += f"\n\n[stderr]: {result.stderr.strip()}"

            # Parse modified/created files from output
            files_modified = self._extract_files_from_output(output, "modified")
            files_created = self._extract_files_from_output(output, "created")

            return GLMResult(
                success=result.returncode == 0,
                output=output if output else "Task completed with no output",
                files_modified=files_modified,
                files_created=files_created,
                duration_seconds=time.time() - start_time,
                model=self.MODEL_MAP.get(model_tier, "glm-4.7"),
            )

        except subprocess.TimeoutExpired:
            return GLMResult(
                success=False,
                output="",
                error_message=f"Task timed out after {timeout_seconds} seconds",
                duration_seconds=timeout_seconds,
            )
        except FileNotFoundError:
            return GLMResult(
                success=False,
                output="",
                error_message="'claude' command not found. Is Claude Code installed?",
                duration_seconds=time.time() - start_time,
            )
        except Exception as e:
            return GLMResult(
                success=False,
                output="",
                error_message=str(e),
                duration_seconds=time.time() - start_time,
            )

    def _execute_direct(
        self,
        prompt: str,
        cwd: Path,
        context_files: Optional[List[str]],
        start_time: float,
    ) -> GLMResult:
        """Execute task using direct GLM API (without Claude Code tools)."""
        system_prompt = """You are an expert software engineer. You are given coding tasks to complete.

For each task:
1. Analyze the requirements carefully
2. Write clean, well-documented code
3. Follow best practices for the language/framework
4. Include error handling where appropriate

Output your code in markdown code blocks with the filename as a comment at the top.
Example:
```python
# filename: src/utils.py
def helper():
    pass
```
"""

        result = self.direct_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        if result.success:
            # Extract and write files from the response
            files_created = self._extract_and_write_files(result.output, cwd)
            result.files_created = files_created

        result.duration_seconds = time.time() - start_time
        return result

    def _extract_files_from_output(self, output: str, file_type: str) -> List[str]:
        """Extract file paths from output text."""
        files = []
        import re

        # Look for patterns like "Modified: file.py" or "Created: file.py"
        pattern = rf'{file_type}[:\s]+([^\s\n]+)'
        matches = re.findall(pattern, output, re.IGNORECASE)
        files.extend(matches)

        return files

    def _extract_and_write_files(self, output: str, cwd: Path) -> List[str]:
        """Extract code blocks from output and write to files."""
        import re
        files_created = []

        # Pattern to match code blocks with filename
        pattern = r'```(\w+)?\n#\s*filename:\s*([^\n]+)\n(.*?)```'
        matches = re.findall(pattern, output, re.DOTALL)

        for lang, filename, content in matches:
            filename = filename.strip()
            file_path = cwd / filename

            try:
                # Create parent directories
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content.strip())
                files_created.append(filename)
                logger.info(f"Created file: {filename}")
            except Exception as e:
                logger.error(f"Failed to write {filename}: {e}")

        return files_created


class GLMPMBackend:
    """PM Agent backend using GLM instead of Claude Code.

    This class provides the same interface as ClaudeCodeTool but uses
    GLM 4.7 as the underlying model.
    """

    def __init__(
        self,
        api_key: str,
        model: GLMModel = GLMModel.GLM_4_7,
        use_proxy: bool = True,
        data_dir: Optional[Path] = None,
    ):
        """Initialize the GLM PM backend.

        Args:
            api_key: GLM API key
            model: GLM model to use
            use_proxy: Whether to use Z.ai proxy (for Claude Code tools)
            data_dir: Directory for data storage
        """
        self.config = GLMConfig(
            api_key=api_key,
            model=model,
            use_zai_proxy=use_proxy,
        )
        self.agent = GLMCodeAgent(self.config)
        self.data_dir = data_dir or Path.cwd() / "data" / "glm_pm"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Track task history
        self.task_history: List[Dict[str, Any]] = []

    def execute_task(
        self,
        description: str,
        working_directory: Optional[Path] = None,
        context_files: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        acceptance_criteria: Optional[List[str]] = None,
        task_id: Optional[str] = None,
        timeout_seconds: int = 600,
    ) -> GLMResult:
        """Execute a coding task.

        This method provides Claude Code-compatible interface for the PM Agent.

        Args:
            description: Task description
            working_directory: Directory to work in
            context_files: Files to include as context
            constraints: Task constraints
            acceptance_criteria: Success criteria
            task_id: Optional task ID for tracking
            timeout_seconds: Timeout

        Returns:
            GLMResult with task outcome
        """
        result = self.agent.execute_task(
            description=description,
            working_directory=working_directory,
            context_files=context_files,
            constraints=constraints,
            acceptance_criteria=acceptance_criteria,
            timeout_seconds=timeout_seconds,
        )

        # Track in history
        self.task_history.append({
            "task_id": task_id,
            "description": description[:200],
            "success": result.success,
            "duration": result.duration_seconds,
            "files_modified": result.files_modified,
            "files_created": result.files_created,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Save history periodically
        if len(self.task_history) % 10 == 0:
            self._save_history()

        return result

    def quick_generate(self, prompt: str, system_prompt: Optional[str] = None) -> GLMResult:
        """Quick generation without code execution.

        Args:
            prompt: The prompt
            system_prompt: Optional system prompt

        Returns:
            GLMResult with generated text
        """
        return self.agent.direct_client.generate(prompt, system_prompt)

    def is_available(self) -> bool:
        """Check if the backend is available."""
        return self.agent.is_available()

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        if not self.task_history:
            return {"tasks": 0}

        success_count = sum(1 for t in self.task_history if t["success"])
        total_duration = sum(t["duration"] for t in self.task_history)

        return {
            "tasks": len(self.task_history),
            "success_rate": success_count / len(self.task_history),
            "total_duration_seconds": total_duration,
            "avg_duration_seconds": total_duration / len(self.task_history),
        }

    def _save_history(self):
        """Save task history to disk."""
        history_file = self.data_dir / "task_history.json"
        try:
            with open(history_file, "w") as f:
                json.dump(self.task_history[-1000:], f, indent=2)  # Keep last 1000
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")


# ============================================================================
# Factory functions
# ============================================================================

def create_glm_backend(
    api_key: str,
    use_proxy: bool = True,
    data_dir: Optional[Path] = None,
) -> GLMPMBackend:
    """Create a GLM PM backend.

    Args:
        api_key: GLM API key
        use_proxy: Whether to use Z.ai proxy for Claude Code tools
        data_dir: Directory for data storage

    Returns:
        Configured GLMPMBackend
    """
    return GLMPMBackend(
        api_key=api_key,
        model=GLMModel.GLM_4_7,
        use_proxy=use_proxy,
        data_dir=data_dir,
    )


def create_glm_agent(api_key: str, use_proxy: bool = True) -> GLMCodeAgent:
    """Create a standalone GLM code agent.

    Args:
        api_key: GLM API key
        use_proxy: Whether to use Z.ai proxy

    Returns:
        Configured GLMCodeAgent
    """
    config = GLMConfig(
        api_key=api_key,
        model=GLMModel.GLM_4_7,
        use_zai_proxy=use_proxy,
    )
    return GLMCodeAgent(config)
