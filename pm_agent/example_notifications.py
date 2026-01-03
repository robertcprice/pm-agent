#!/usr/bin/env python3
"""
Simple example demonstrating PM Agent notification system.

This script shows how to:
1. Configure notifications
2. Send different types of notifications
3. Monitor channel status
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from conch_dna.pm import (
    NotificationManager,
    NotificationConfig,
    ConsoleNotifier,
)


def example_1_basic_console():
    """Example 1: Basic console notifications."""
    print("\n" + "="*80)
    print("Example 1: Basic Console Notifications")
    print("="*80 + "\n")

    # Create manager with console only
    manager = NotificationManager()

    # Send notifications of different severities
    manager.notify("Information", "This is an informational message", "info")
    manager.notify("Success", "Operation completed successfully", "success")
    manager.notify("Warning", "This might need attention", "warning")
    manager.notify("Error", "Something went wrong", "error")
    manager.notify("Critical", "Immediate action required", "critical")


def example_2_milestone_notifications():
    """Example 2: Milestone notifications."""
    print("\n" + "="*80)
    print("Example 2: Milestone Notifications")
    print("="*80 + "\n")

    manager = NotificationManager()

    # Notify about project milestones
    manager.notify_milestone(
        "Initial setup completed",
        {
            "files_created": 15,
            "tests_written": 8,
            "coverage": "95%",
            "time_taken": "12 minutes",
        }
    )

    manager.notify_milestone(
        "First user authentication successful",
        {
            "username": "test_user",
            "authentication_method": "JWT",
            "timestamp": "2026-01-02 15:30:00",
        }
    )


def example_3_error_notifications():
    """Example 3: Error notifications."""
    print("\n" + "="*80)
    print("Example 3: Error Notifications")
    print("="*80 + "\n")

    manager = NotificationManager()

    # Notify about errors with context
    manager.notify_error(
        "Database connection failed",
        {
            "host": "localhost",
            "port": 5432,
            "database": "project_db",
            "error": "Connection refused",
            "retry_count": 3,
        }
    )

    manager.notify_error(
        "Task execution timeout",
        {
            "task_id": "task-123",
            "timeout_seconds": 600,
            "elapsed_seconds": 605,
            "status": "killed",
        }
    )


def example_4_custom_notification():
    """Example 4: Custom notification with specific severity."""
    print("\n" + "="*80)
    print("Example 4: Custom Notifications")
    print("="*80 + "\n")

    manager = NotificationManager()

    # Send custom formatted notification
    title = "🔍 Code Review Required"
    message = """
**Pull Request:** PR #42 - Add authentication system
**Author:** Claude Code
**Files Changed:** 12
**Lines Added:** 450
**Lines Removed:** 23

**Changes:**
- Implemented JWT authentication
- Added user registration endpoint
- Created password hashing utility
- Added comprehensive tests

**Review Checklist:**
- [ ] Security review of authentication logic
- [ ] Test coverage verification
- [ ] Performance impact assessment
- [ ] Documentation completeness
"""

    manager.notify(title, message, "warning")


def example_5_channel_configuration():
    """Example 5: Configure multiple channels."""
    print("\n" + "="*80)
    print("Example 5: Multi-Channel Configuration")
    print("="*80 + "\n")

    # Configure with environment-like settings
    config = NotificationConfig(
        console_enabled=True,
        console_use_color=True,
        # Uncomment to add real channels:
        # slack_webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK",
        # discord_webhook_url="https://discord.com/api/webhooks/YOUR/WEBHOOK",
        # email_smtp_host="smtp.gmail.com",
        # email_from="pm-agent@example.com",
        # email_to=["team@example.com"],
    )

    manager = NotificationManager(config)

    # Send a test notification
    manager.notify(
        "Multi-Channel Test",
        "This notification would be sent to all configured channels",
        "info"
    )

    # Check channel status
    print("\nChannel Status:")
    for channel in manager.get_channel_status():
        enabled_str = "✓ Enabled" if channel['enabled'] else "✗ Disabled"
        print(f"  {channel['name']:<15} {enabled_str} ({channel['failure_count']} failures)")


def example_6_manual_channel_setup():
    """Example 6: Manually add channels."""
    print("\n" + "="*80)
    print("Example 6: Manual Channel Setup")
    print("="*80 + "\n")

    # Create manager without auto-initialization
    manager = NotificationManager(NotificationConfig(console_enabled=False))

    # Add channels manually
    manager.add_channel(ConsoleNotifier(use_color=True))

    # Uncomment to add real channels:
    # from conch_dna.pm import SlackNotifier, DiscordNotifier
    # manager.add_channel(SlackNotifier(webhook_url="..."))
    # manager.add_channel(DiscordNotifier(webhook_url="..."))

    manager.notify(
        "Manually Configured Channels",
        "This demonstrates adding channels programmatically",
        "success"
    )


def example_7_environment_configuration():
    """Example 7: Load configuration from environment."""
    print("\n" + "="*80)
    print("Example 7: Environment Configuration")
    print("="*80 + "\n")

    # Load from environment variables
    # Set these before running:
    # export PM_SLACK_WEBHOOK="..."
    # export PM_DISCORD_WEBHOOK="..."
    # export PM_EMAIL_SMTP_HOST="..."

    config = NotificationConfig.from_env()
    manager = NotificationManager(config)

    manager.notify(
        "Environment-Based Configuration",
        "Configuration loaded from environment variables",
        "info"
    )

    print("\nLoaded Configuration:")
    print(f"  Console Enabled: {config.console_enabled}")
    print(f"  Slack Configured: {'Yes' if config.slack_webhook_url else 'No'}")
    print(f"  Discord Configured: {'Yes' if config.discord_webhook_url else 'No'}")
    print(f"  Email Configured: {'Yes' if config.email_smtp_host else 'No'}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("PM Agent Notification System Examples")
    print("="*80)

    example_1_basic_console()
    example_2_milestone_notifications()
    example_3_error_notifications()
    example_4_custom_notification()
    example_5_channel_configuration()
    example_6_manual_channel_setup()
    example_7_environment_configuration()

    print("\n" + "="*80)
    print("All Examples Completed")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
