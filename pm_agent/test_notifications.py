#!/usr/bin/env python3
"""
Test script for PM Agent notification system.

Tests all notification channels and templates to ensure proper functionality.

Usage:
    python test_notifications.py
    python test_notifications.py --test-slack
    python test_notifications.py --test-discord
    python test_notifications.py --test-email
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import uuid

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from conch_dna.pm.notifications import (
    NotificationManager,
    NotificationConfig,
    ConsoleNotifier,
    SlackNotifier,
    DiscordNotifier,
    EmailNotifier,
)
from conch_dna.pm.task_queue import (
    Escalation,
    Task,
    Goal,
    TaskStatus,
    GoalStatus,
    TaskPriority,
)


def create_test_escalation() -> Escalation:
    """Create a test escalation."""
    return Escalation(
        id=str(uuid.uuid4()),
        task_id=str(uuid.uuid4()),
        reason="Task failed after 3 attempts. Error: Connection timeout when accessing external API.",
        status="pending",
        created_at=datetime.now(),
    )


def create_test_task() -> Task:
    """Create a test task."""
    return Task(
        id=str(uuid.uuid4()),
        goal_id=str(uuid.uuid4()),
        description="Implement user authentication API endpoints with JWT tokens",
        status=TaskStatus.ESCALATED,
        priority=TaskPriority.HIGH,
        attempt_count=3,
        max_attempts=3,
        context_files=["api/auth.py", "api/middleware.py", "tests/test_auth.py"],
        acceptance_criteria=[
            "POST /api/auth/login returns JWT token",
            "POST /api/auth/register creates new user",
            "Middleware validates JWT on protected routes",
        ],
        error_message="Connection timeout when accessing external API",
        created_at=datetime.now(),
    )


def create_test_goal() -> Goal:
    """Create a test goal."""
    return Goal(
        id=str(uuid.uuid4()),
        description="Build complete authentication system with JWT and refresh tokens",
        project_id=str(uuid.uuid4()),
        status=GoalStatus.IN_PROGRESS,
        priority=TaskPriority.HIGH,
        created_at=datetime.now(),
    )


def test_console_notifications():
    """Test console notifications."""
    print("\n" + "="*80)
    print("Testing Console Notifications")
    print("="*80 + "\n")

    notifier = ConsoleNotifier(use_color=True)

    # Test all severity levels
    notifier.notify("Info Test", "This is an informational message", "info")
    notifier.notify("Success Test", "This is a success message", "success")
    notifier.notify("Warning Test", "This is a warning message", "warning")
    notifier.notify("Error Test", "This is an error message", "error")
    notifier.notify("Critical Test", "This is a critical message", "critical")


def test_notification_manager():
    """Test notification manager with console only."""
    print("\n" + "="*80)
    print("Testing Notification Manager")
    print("="*80 + "\n")

    config = NotificationConfig(
        console_enabled=True,
        console_use_color=True,
    )

    manager = NotificationManager(config)

    # Test escalation notification
    escalation = create_test_escalation()
    task = create_test_task()
    goal = create_test_goal()

    print("\n--- Testing Escalation Notification ---\n")
    manager.notify_escalation(escalation, task, goal)

    # Test milestone notification
    print("\n--- Testing Milestone Notification ---\n")
    manager.notify_milestone(
        "Project setup completed successfully",
        {
            "tasks_completed": 5,
            "time_taken": "15 minutes",
            "files_created": 12,
        }
    )

    # Test error notification
    print("\n--- Testing Error Notification ---\n")
    manager.notify_error(
        "Failed to connect to database",
        {
            "host": "localhost:5432",
            "database": "project_db",
            "error": "Connection refused",
        }
    )

    # Test goal completion notification
    print("\n--- Testing Goal Completion Notification ---\n")
    completed_goal = create_test_goal()
    completed_goal.status = GoalStatus.COMPLETED
    completed_goal.completed_at = datetime.now()
    manager.notify_goal_completed(completed_goal, task_count=5, duration_minutes=45.5)


def test_slack_notifications(webhook_url: str):
    """Test Slack notifications."""
    print("\n" + "="*80)
    print("Testing Slack Notifications")
    print("="*80 + "\n")

    notifier = SlackNotifier(webhook_url=webhook_url)

    if not notifier.enabled:
        print("ERROR: Slack notifier not enabled (check webhook URL)")
        return

    escalation = create_test_escalation()
    task = create_test_task()
    goal = create_test_goal()

    config = NotificationConfig(
        slack_webhook_url=webhook_url,
        console_enabled=False,
    )

    manager = NotificationManager(config)
    success = manager.notify_escalation(escalation, task, goal)

    if success:
        print("✓ Slack notification sent successfully")
    else:
        print("✗ Failed to send Slack notification")


def test_discord_notifications(webhook_url: str):
    """Test Discord notifications."""
    print("\n" + "="*80)
    print("Testing Discord Notifications")
    print("="*80 + "\n")

    notifier = DiscordNotifier(webhook_url=webhook_url)

    if not notifier.enabled:
        print("ERROR: Discord notifier not enabled (check webhook URL)")
        return

    escalation = create_test_escalation()
    task = create_test_task()
    goal = create_test_goal()

    config = NotificationConfig(
        discord_webhook_url=webhook_url,
        console_enabled=False,
    )

    manager = NotificationManager(config)
    success = manager.notify_escalation(escalation, task, goal)

    if success:
        print("✓ Discord notification sent successfully")
    else:
        print("✗ Failed to send Discord notification")


def test_email_notifications(
    smtp_host: str,
    smtp_port: int,
    from_addr: str,
    to_addr: str,
    username: str = None,
    password: str = None,
):
    """Test email notifications."""
    print("\n" + "="*80)
    print("Testing Email Notifications")
    print("="*80 + "\n")

    notifier = EmailNotifier(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        from_address=from_addr,
        to_addresses=[to_addr],
        username=username,
        password=password,
    )

    if not notifier.enabled:
        print("ERROR: Email notifier not enabled (check configuration)")
        return

    escalation = create_test_escalation()
    task = create_test_task()
    goal = create_test_goal()

    config = NotificationConfig(
        email_smtp_host=smtp_host,
        email_smtp_port=smtp_port,
        email_from=from_addr,
        email_to=[to_addr],
        email_smtp_username=username,
        email_smtp_password=password,
        console_enabled=False,
    )

    manager = NotificationManager(config)
    success = manager.notify_escalation(escalation, task, goal)

    if success:
        print(f"✓ Email notification sent successfully to {to_addr}")
    else:
        print("✗ Failed to send email notification")


def test_channel_status():
    """Test channel status reporting."""
    print("\n" + "="*80)
    print("Testing Channel Status")
    print("="*80 + "\n")

    config = NotificationConfig(
        console_enabled=True,
        slack_webhook_url="https://hooks.slack.com/test",  # Will fail
    )

    manager = NotificationManager(config)

    # Try to send a notification (Slack will fail)
    manager.notify("Test", "This is a test", "info")

    # Get status
    status = manager.get_channel_status()

    print("\nChannel Status:")
    for channel in status:
        print(f"  {channel['name']}:")
        print(f"    Enabled: {channel['enabled']}")
        print(f"    Failures: {channel['failure_count']}/{channel['max_failures']}")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test PM Agent notification system")
    parser.add_argument("--test-slack", metavar="WEBHOOK_URL", help="Test Slack notifications")
    parser.add_argument("--test-discord", metavar="WEBHOOK_URL", help="Test Discord notifications")
    parser.add_argument("--test-email", action="store_true", help="Test email notifications")
    parser.add_argument("--smtp-host", default="localhost", help="SMTP host")
    parser.add_argument("--smtp-port", type=int, default=587, help="SMTP port")
    parser.add_argument("--from-addr", help="From email address")
    parser.add_argument("--to-addr", help="To email address")
    parser.add_argument("--smtp-user", help="SMTP username")
    parser.add_argument("--smtp-pass", help="SMTP password")

    args = parser.parse_args()

    # Always run console tests
    test_console_notifications()
    test_notification_manager()
    test_channel_status()

    # Run optional tests
    if args.test_slack:
        test_slack_notifications(args.test_slack)

    if args.test_discord:
        test_discord_notifications(args.test_discord)

    if args.test_email:
        if not args.from_addr or not args.to_addr:
            print("\nERROR: --from-addr and --to-addr required for email testing")
            sys.exit(1)

        test_email_notifications(
            smtp_host=args.smtp_host,
            smtp_port=args.smtp_port,
            from_addr=args.from_addr,
            to_addr=args.to_addr,
            username=args.smtp_user,
            password=args.smtp_pass,
        )

    print("\n" + "="*80)
    print("All Tests Completed")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
