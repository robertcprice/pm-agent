#!/usr/bin/env python3
"""
Example: Integrating Notifications with PM Agent

This demonstrates how the notification system integrates with the PM Agent
to provide real-time alerts about task escalations, goal completions, and errors.
"""

import asyncio
from pathlib import Path
from datetime import datetime

from conch_dna.pm import (
    PMAgent,
    PMConfig,
    TaskQueue,
    NotificationManager,
    NotificationConfig,
    Task,
    Goal,
    Project,
    TaskPriority,
    GoalStatus,
)


async def example_pm_with_notifications():
    """Example: Create PM Agent with notification system."""
    print("\n" + "="*80)
    print("PM Agent with Notification Integration")
    print("="*80 + "\n")

    # 1. Setup paths
    project_root = Path.cwd()
    data_dir = project_root / "pm_data"
    data_dir.mkdir(exist_ok=True)

    # 2. Create task queue
    queue = TaskQueue(data_dir / "tasks.db")

    # 3. Configure notifications
    notification_config = NotificationConfig(
        console_enabled=True,
        console_use_color=True,
        # Add your production channels here:
        # slack_webhook_url=os.getenv("PM_SLACK_WEBHOOK"),
        # discord_webhook_url=os.getenv("PM_DISCORD_WEBHOOK"),
        # email_smtp_host=os.getenv("PM_EMAIL_SMTP_HOST"),
        # email_from=os.getenv("PM_EMAIL_FROM"),
        # email_to=os.getenv("PM_EMAIL_TO", "").split(","),
    )
    notification_manager = NotificationManager(notification_config)

    print("✓ Notification manager configured with channels:")
    for channel in notification_manager.get_channel_status():
        print(f"  - {channel['name']}: {'Enabled' if channel['enabled'] else 'Disabled'}")

    # 4. Create PM configuration
    pm_config = PMConfig(
        project_root=project_root,
        data_dir=data_dir,
        max_concurrent_tasks=1,
        task_timeout_seconds=600,
        max_task_attempts=3,
        auto_approve_threshold=0.9,
    )

    # 5. Create mock Claude Code tool (replace with real implementation)
    from unittest.mock import MagicMock
    claude_code = MagicMock()

    # 6. Create PM Agent with notification integration
    agent = PMAgent(
        config=pm_config,
        task_queue=queue,
        claude_code=claude_code,
        notification_manager=notification_manager,
    )

    print("\n✓ PM Agent initialized with notification system")

    # 7. Demonstrate notification scenarios

    print("\n" + "-"*80)
    print("Scenario 1: Task Escalation")
    print("-"*80 + "\n")

    # Create a test project and goal
    project = Project(
        id="test-project",
        name="Test Project",
        root_path=str(project_root),
        description="Example project for notification testing",
    )
    queue.add_project(project)

    goal = Goal(
        id="test-goal",
        project_id=project.id,
        description="Implement authentication system",
        status=GoalStatus.IN_PROGRESS,
        priority=TaskPriority.HIGH,
    )
    queue.add_goal(goal)

    # Create a failing task
    task = Task(
        id="test-task",
        goal_id=goal.id,
        description="Implement JWT authentication endpoints",
        priority=TaskPriority.HIGH,
        attempt_count=3,
        max_attempts=3,
        context_files=["api/auth.py"],
        error_message="Connection timeout after 3 attempts",
    )
    queue.add_task(task)

    # Simulate escalation
    print("Escalating task due to repeated failures...\n")
    escalation_id = agent._escalate_task(task, "Task failed after 3 attempts")
    print(f"\n✓ Escalation created: {escalation_id}")
    print("  Notifications sent to all configured channels")

    print("\n" + "-"*80)
    print("Scenario 2: Goal Completion")
    print("-"*80 + "\n")

    # Create a completed goal
    completed_goal = Goal(
        id="completed-goal",
        project_id=project.id,
        description="Setup development environment",
        status=GoalStatus.COMPLETED,
        priority=TaskPriority.MEDIUM,
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    queue.add_goal(completed_goal)

    # Simulate goal completion notification
    print("Notifying about goal completion...\n")
    notification_manager.notify_goal_completed(
        goal=completed_goal,
        task_count=5,
        duration_minutes=23.5,
    )
    print("\n✓ Goal completion notification sent")

    print("\n" + "-"*80)
    print("Scenario 3: System Error")
    print("-"*80 + "\n")

    # Simulate a system error
    print("Simulating PM Agent error...\n")
    notification_manager.notify_error(
        "PM Agent cycle error: Database connection lost",
        {
            "cycle": 42,
            "state": "reviewing",
            "current_task": "task-789",
            "timestamp": datetime.now().isoformat(),
        }
    )
    print("\n✓ Error notification sent")

    print("\n" + "-"*80)
    print("Scenario 4: Project Milestone")
    print("-"*80 + "\n")

    # Simulate milestone achievement
    print("Celebrating project milestone...\n")
    notification_manager.notify_milestone(
        "50% of project tasks completed",
        {
            "total_tasks": 100,
            "completed_tasks": 50,
            "failed_tasks": 3,
            "pending_tasks": 47,
            "completion_rate": "94%",
            "average_task_time": "8.5 minutes",
        }
    )
    print("\n✓ Milestone notification sent")

    print("\n" + "="*80)
    print("Integration Example Complete")
    print("="*80)
    print("\nThe PM Agent is now configured to send notifications for:")
    print("  - Task escalations requiring human intervention")
    print("  - Goal completions and achievements")
    print("  - Critical system errors")
    print("  - Project milestones")
    print("\nAll notifications fail gracefully and won't interrupt PM Agent operation.")
    print("="*80 + "\n")


def show_configuration_options():
    """Show different ways to configure notifications."""
    print("\n" + "="*80)
    print("Notification Configuration Options")
    print("="*80 + "\n")

    print("Option 1: Environment Variables (Recommended for Production)")
    print("-" * 80)
    print("""
export PM_SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK"
export PM_DISCORD_WEBHOOK="https://discord.com/api/webhooks/YOUR/WEBHOOK"
export PM_EMAIL_SMTP_HOST="smtp.gmail.com"
export PM_EMAIL_FROM="pm-agent@company.com"
export PM_EMAIL_TO="oncall@company.com,manager@company.com"

# Then in code:
config = NotificationConfig.from_env()
manager = NotificationManager(config)
""")

    print("\nOption 2: Programmatic Configuration")
    print("-" * 80)
    print("""
config = NotificationConfig(
    slack_webhook_url="https://hooks.slack.com/...",
    discord_webhook_url="https://discord.com/api/webhooks/...",
    email_smtp_host="smtp.sendgrid.net",
    email_smtp_port=587,
    email_from="pm-agent@company.com",
    email_to=["team@company.com"],
    console_enabled=True,
)
manager = NotificationManager(config)
""")

    print("\nOption 3: JSON Configuration File")
    print("-" * 80)
    print("""
# notifications.json
{
  "slack_webhook_url": "https://hooks.slack.com/...",
  "email_smtp_host": "smtp.gmail.com",
  "email_from": "pm-agent@company.com",
  "email_to": ["team@company.com"]
}

# Load in code:
config = NotificationConfig.from_file(Path("notifications.json"))
manager = NotificationManager(config)
""")

    print("\nOption 4: Manual Channel Setup")
    print("-" * 80)
    print("""
from conch_dna.pm import (
    NotificationManager,
    SlackNotifier,
    DiscordNotifier,
    EmailNotifier,
)

manager = NotificationManager()
manager.add_channel(SlackNotifier(webhook_url="..."))
manager.add_channel(DiscordNotifier(webhook_url="..."))
manager.add_channel(EmailNotifier(
    smtp_host="smtp.gmail.com",
    smtp_port=587,
    from_address="pm@company.com",
    to_addresses=["team@company.com"],
    password="app-password",
))
""")

    print("\n" + "="*80 + "\n")


def show_real_world_example():
    """Show a real-world production setup."""
    print("\n" + "="*80)
    print("Real-World Production Setup")
    print("="*80 + "\n")

    print("Dockerfile Configuration:")
    print("-" * 80)
    print("""
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

# Configure notifications via environment
ENV PM_SLACK_WEBHOOK=""
ENV PM_EMAIL_SMTP_HOST="smtp.sendgrid.net"
ENV PM_EMAIL_FROM="pm-agent@company.com"
ENV PM_EMAIL_TO="oncall@company.com"

CMD ["python", "-m", "conch_dna.pm.run_agent"]
""")

    print("\ndocker-compose.yml:")
    print("-" * 80)
    print("""
version: '3.8'
services:
  pm-agent:
    build: .
    environment:
      PM_SLACK_WEBHOOK: ${SLACK_WEBHOOK}
      PM_DISCORD_WEBHOOK: ${DISCORD_WEBHOOK}
      PM_EMAIL_SMTP_HOST: smtp.sendgrid.net
      PM_EMAIL_SMTP_PORT: 587
      PM_EMAIL_FROM: pm-agent@company.com
      PM_EMAIL_TO: oncall@company.com,manager@company.com
      PM_EMAIL_SMTP_USERNAME: apikey
      PM_EMAIL_SMTP_PASSWORD: ${SENDGRID_API_KEY}
    volumes:
      - ./data:/app/data
      - ./project:/app/project
    restart: unless-stopped
""")

    print("\n.env file (not committed):")
    print("-" * 80)
    print("""
SLACK_WEBHOOK=https://hooks.slack.com/services/T00/B00/XXX
DISCORD_WEBHOOK=https://discord.com/api/webhooks/000/XXX
SENDGRID_API_KEY=SG.xxxxxxxxxx
""")

    print("\n" + "="*80 + "\n")


async def main():
    """Run all examples."""
    await example_pm_with_notifications()
    show_configuration_options()
    show_real_world_example()


if __name__ == "__main__":
    asyncio.run(main())
