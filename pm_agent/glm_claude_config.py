"""
PM Agent with GLM 4.7 as default backend via Claude Code.

This configures the PM Agent to use GLM 4.7 for all coding tasks through
the Claude Code CLI, providing significant cost savings while maintaining
full tool capabilities.

Configuration is stored in ~/.claude/settings.json:
- ANTHROPIC_BASE_URL → https://api.z.ai/api/anthropic (Z.AI proxy)
- ANTHROPIC_AUTH_TOKEN → Your Z.AI API key
- ANTHROPIC_DEFAULT_*_MODEL → GLM 4.7 (all models)
"""

import os
import json
from pathlib import Path

# Z.AI API configuration for GLM
ZAI_API_KEY = "ff2be656926745dd864a9f09d6e306d5.DWaYrM15oGRZu1eK"
ZAI_BASE_URL = "https://api.z.ai/api/anthropic"

# Model mappings - all use GLM 4.7
MODEL_MAPPINGS = {
    "opus": "glm-4.7",
    "sonnet": "glm-4.7",
    "haiku": "glm-4.7",
}


def setup_claude_for_glm(api_key: str = ZAI_API_KEY, base_url: str = ZAI_BASE_URL) -> bool:
    """
    Configure Claude Code to use GLM via Z.AI proxy.

    Args:
        api_key: Z.AI API key
        base_url: Z.AI proxy base URL

    Returns:
        True if successful
    """
    claude_dir = Path.home() / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "env": {
            "ANTHROPIC_AUTH_TOKEN": api_key,
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-4.7",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-4.7",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-4.7",
            "API_TIMEOUT_MS": "300000",
        }
    }

    settings_file = claude_dir / "settings.json"

    # Backup existing settings
    if settings_file.exists():
        backup_file = claude_dir / "settings.json.backup"
        import shutil
        shutil.copy(settings_file, backup_file)
        print(f"Backed up existing settings to {backup_file}")

    # Write new settings
    with open(settings_file, "w") as f:
        json.dump(settings, f, indent=2)

    print(f"✅ Claude Code configured to use GLM 4.7")
    print(f"   Settings: {settings_file}")
    return True


def get_glm_env() -> dict:
    """Get environment variables for GLM via Claude Code."""
    return {
        "ANTHROPIC_AUTH_TOKEN": ZAI_API_KEY,
        "ANTHROPIC_BASE_URL": ZAI_BASE_URL,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-4.7",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-4.7",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-4.7",
        "API_TIMEOUT_MS": "300000",
    }


# Auto-setup on import
if __name__ != "__main__":
    setup_claude_for_glm()
