"""
Notification System for PM Agent.

Provides flexible notification channels for human escalations, milestones,
and error alerts. All channels fail gracefully to ensure the PM Agent
continues operation even if notifications fail.

Supported Channels:
    - Slack (webhook)
    - Discord (webhook)
    - Email (SMTP)
    - Console (local development)

Usage:
    from conch_dna.pm.notifications import NotificationManager, SlackNotifier

    manager = NotificationManager()
    manager.add_channel(SlackNotifier(webhook_url))
    manager.notify_escalation(escalation)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import json
import logging
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .task_queue import Escalation, Task, Goal


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class NotificationConfig:
    """Configuration for notification channels."""

    # Slack settings
    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None
    slack_username: str = "PM Agent"

    # Discord settings
    discord_webhook_url: Optional[str] = None
    discord_username: str = "PM Agent"

    # Email settings
    email_smtp_host: Optional[str] = None
    email_smtp_port: int = 587
    email_smtp_username: Optional[str] = None
    email_smtp_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: Optional[List[str]] = None
    email_use_tls: bool = True

    # Console settings
    console_enabled: bool = True
    console_use_color: bool = True

    # General settings
    max_retry_attempts: int = 3
    notification_timeout_seconds: int = 10
    include_stack_traces: bool = False

    @classmethod
    def from_env(cls) -> "NotificationConfig":
        """Load configuration from environment variables."""
        email_to_str = os.getenv("PM_EMAIL_TO", "")
        email_to = [e.strip() for e in email_to_str.split(",")] if email_to_str else None

        return cls(
            slack_webhook_url=os.getenv("PM_SLACK_WEBHOOK"),
            slack_channel=os.getenv("PM_SLACK_CHANNEL"),
            slack_username=os.getenv("PM_SLACK_USERNAME", "PM Agent"),
            discord_webhook_url=os.getenv("PM_DISCORD_WEBHOOK"),
            discord_username=os.getenv("PM_DISCORD_USERNAME", "PM Agent"),
            email_smtp_host=os.getenv("PM_EMAIL_SMTP_HOST"),
            email_smtp_port=int(os.getenv("PM_EMAIL_SMTP_PORT", "587")),
            email_smtp_username=os.getenv("PM_EMAIL_SMTP_USERNAME"),
            email_smtp_password=os.getenv("PM_EMAIL_SMTP_PASSWORD"),
            email_from=os.getenv("PM_EMAIL_FROM"),
            email_to=email_to,
            email_use_tls=os.getenv("PM_EMAIL_USE_TLS", "true").lower() == "true",
            console_enabled=os.getenv("PM_CONSOLE_NOTIFICATIONS", "true").lower() == "true",
            console_use_color=os.getenv("PM_CONSOLE_COLOR", "true").lower() == "true",
            include_stack_traces=os.getenv("PM_INCLUDE_STACK_TRACES", "false").lower() == "true",
        )

    @classmethod
    def from_file(cls, config_path: Path) -> "NotificationConfig":
        """Load configuration from JSON file."""
        with open(config_path) as f:
            data = json.load(f)
        return cls(**data)


# =============================================================================
# Notification Templates
# =============================================================================

class NotificationTemplate:
    """Templates for formatting notification messages."""

    @staticmethod
    def escalation(escalation: Escalation, task: Optional[Task] = None, goal: Optional[Goal] = None) -> Dict[str, str]:
        """Format escalation notification."""
        title = "🚨 Task Escalated - Human Attention Required"

        details = [
            f"**Escalation ID:** `{escalation.id}`",
            f"**Reason:** {escalation.reason}",
            f"**Created:** {escalation.created_at.strftime('%Y-%m-%d %H:%M:%S') if escalation.created_at else 'Unknown'}",
        ]

        if task:
            details.extend([
                "",
                "**Task Details:**",
                f"- ID: `{task.id}`",
                f"- Description: {task.description}",
                f"- Status: {task.status.value}",
                f"- Priority: {task.priority.name}",
                f"- Attempts: {task.attempt_count}/{task.max_attempts}",
            ])

            if task.error_message:
                details.append(f"- Error: {task.error_message}")

            if task.context_files:
                details.append(f"- Context Files: {', '.join(task.context_files[:3])}")

        if goal:
            details.extend([
                "",
                "**Related Goal:**",
                f"- Description: {goal.description}",
                f"- Status: {goal.status.value}",
            ])

        details.extend([
            "",
            "**Action Required:**",
            "Please review and provide guidance to resolve this escalation.",
        ])

        return {
            "title": title,
            "body": "\n".join(details),
            "severity": "critical",
        }

    @staticmethod
    def milestone(message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Format milestone notification."""
        title = "🎯 Project Milestone"

        body_parts = [message]

        if details:
            body_parts.append("")
            body_parts.append("**Details:**")
            for key, value in details.items():
                body_parts.append(f"- {key}: {value}")

        return {
            "title": title,
            "body": "\n".join(body_parts),
            "severity": "info",
        }

    @staticmethod
    def error(error_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Format error notification."""
        title = "❌ PM Agent Error"

        body_parts = [
            f"**Error:** {error_message}",
        ]

        if context:
            body_parts.append("")
            body_parts.append("**Context:**")
            for key, value in context.items():
                body_parts.append(f"- {key}: {value}")

        return {
            "title": title,
            "body": "\n".join(body_parts),
            "severity": "error",
        }

    @staticmethod
    def goal_completed(goal: Goal, task_count: int, duration_minutes: Optional[float] = None) -> Dict[str, str]:
        """Format goal completion notification."""
        title = "✅ Goal Completed Successfully"

        details = [
            f"**Goal:** {goal.description}",
            f"**Priority:** {goal.priority.name}",
            f"**Tasks Completed:** {task_count}",
        ]

        if duration_minutes:
            details.append(f"**Duration:** {duration_minutes:.1f} minutes")

        if goal.metadata:
            details.append("")
            details.append("**Metadata:**")
            for key, value in goal.metadata.items():
                details.append(f"- {key}: {value}")

        return {
            "title": title,
            "body": "\n".join(details),
            "severity": "success",
        }


# =============================================================================
# Abstract Base Channel
# =============================================================================

class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    def __init__(self, enabled: bool = True):
        """
        Initialize notification channel.

        Args:
            enabled: Whether this channel is active
        """
        self.enabled = enabled
        self.failure_count = 0
        self.max_failures = 3

    @abstractmethod
    def send(self, title: str, message: str, severity: str = "info") -> bool:
        """
        Send a notification.

        Args:
            title: Notification title
            message: Notification body
            severity: Severity level (info, warning, error, critical, success)

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get channel name for logging."""
        pass

    def notify(self, title: str, message: str, severity: str = "info") -> bool:
        """
        Send notification with error handling and circuit breaking.

        Args:
            title: Notification title
            message: Notification body
            severity: Severity level

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            logger.debug(f"Channel {self.get_name()} is disabled, skipping notification")
            return False

        # Circuit breaker: disable channel after max failures
        if self.failure_count >= self.max_failures:
            logger.warning(
                f"Channel {self.get_name()} has exceeded max failures ({self.max_failures}), "
                f"disabling for this session"
            )
            self.enabled = False
            return False

        try:
            success = self.send(title, message, severity)

            if success:
                # Reset failure count on success
                self.failure_count = 0
                logger.debug(f"Notification sent successfully via {self.get_name()}")
            else:
                self.failure_count += 1
                logger.warning(f"Failed to send notification via {self.get_name()}")

            return success

        except Exception as e:
            self.failure_count += 1
            logger.error(
                f"Exception in {self.get_name()} notification: {e}",
                exc_info=True
            )
            return False


# =============================================================================
# Concrete Channel Implementations
# =============================================================================

class ConsoleNotifier(NotificationChannel):
    """Console/terminal notification for local development."""

    # ANSI color codes
    COLORS = {
        "info": "\033[94m",      # Blue
        "success": "\033[92m",   # Green
        "warning": "\033[93m",   # Yellow
        "error": "\033[91m",     # Red
        "critical": "\033[95m",  # Magenta
        "reset": "\033[0m",
        "bold": "\033[1m",
    }

    def __init__(self, use_color: bool = True):
        """
        Initialize console notifier.

        Args:
            use_color: Whether to use ANSI colors
        """
        super().__init__(enabled=True)
        self.use_color = use_color

    def get_name(self) -> str:
        return "Console"

    def send(self, title: str, message: str, severity: str = "info") -> bool:
        """Print notification to console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if self.use_color:
            color = self.COLORS.get(severity, self.COLORS["info"])
            reset = self.COLORS["reset"]
            bold = self.COLORS["bold"]

            print(f"\n{color}{bold}{'='*80}{reset}")
            print(f"{color}{bold}[{timestamp}] {title}{reset}")
            print(f"{color}{bold}{'='*80}{reset}")
            print(f"{message}")
            print(f"{color}{bold}{'='*80}{reset}\n")
        else:
            print(f"\n{'='*80}")
            print(f"[{timestamp}] {title}")
            print(f"{'='*80}")
            print(f"{message}")
            print(f"{'='*80}\n")

        return True


class SlackNotifier(NotificationChannel):
    """Slack notification via webhook."""

    def __init__(self, webhook_url: str, channel: Optional[str] = None, username: str = "PM Agent"):
        """
        Initialize Slack notifier.

        Args:
            webhook_url: Slack webhook URL
            channel: Optional channel override (e.g., "#alerts")
            username: Bot username
        """
        super().__init__(enabled=bool(webhook_url))
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username

        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available, Slack notifications disabled")
            self.enabled = False

    def get_name(self) -> str:
        return "Slack"

    def send(self, title: str, message: str, severity: str = "info") -> bool:
        """Send notification to Slack."""
        if not REQUESTS_AVAILABLE:
            return False

        # Map severity to emoji
        emoji_map = {
            "info": ":information_source:",
            "success": ":white_check_mark:",
            "warning": ":warning:",
            "error": ":x:",
            "critical": ":rotating_light:",
        }
        emoji = emoji_map.get(severity, ":speech_balloon:")

        # Map severity to color
        color_map = {
            "info": "#36a64f",      # Green
            "success": "#2eb886",   # Teal
            "warning": "#ff9900",   # Orange
            "error": "#d00000",     # Red
            "critical": "#9f00d0",  # Purple
        }
        color = color_map.get(severity, "#808080")

        payload = {
            "username": self.username,
            "icon_emoji": emoji,
            "attachments": [
                {
                    "color": color,
                    "title": title,
                    "text": message,
                    "footer": "PM Agent",
                    "ts": int(datetime.now().timestamp()),
                }
            ],
        }

        if self.channel:
            payload["channel"] = self.channel

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False


class DiscordNotifier(NotificationChannel):
    """Discord notification via webhook."""

    def __init__(self, webhook_url: str, username: str = "PM Agent"):
        """
        Initialize Discord notifier.

        Args:
            webhook_url: Discord webhook URL
            username: Bot username
        """
        super().__init__(enabled=bool(webhook_url))
        self.webhook_url = webhook_url
        self.username = username

        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available, Discord notifications disabled")
            self.enabled = False

    def get_name(self) -> str:
        return "Discord"

    def send(self, title: str, message: str, severity: str = "info") -> bool:
        """Send notification to Discord."""
        if not REQUESTS_AVAILABLE:
            return False

        # Map severity to color (decimal)
        color_map = {
            "info": 3447003,      # Blue
            "success": 3066993,   # Green
            "warning": 16776960,  # Yellow
            "error": 15158332,    # Red
            "critical": 10181046, # Purple
        }
        color = color_map.get(severity, 8421504)  # Gray

        payload = {
            "username": self.username,
            "embeds": [
                {
                    "title": title,
                    "description": message,
                    "color": color,
                    "timestamp": datetime.now().isoformat(),
                    "footer": {
                        "text": "PM Agent"
                    }
                }
            ]
        }

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            return response.status_code in (200, 204)
        except Exception as e:
            logger.error(f"Discord notification failed: {e}")
            return False


class EmailNotifier(NotificationChannel):
    """Email notification via SMTP."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        from_address: str,
        to_addresses: List[str],
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
    ):
        """
        Initialize email notifier.

        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            from_address: Sender email address
            to_addresses: List of recipient email addresses
            username: SMTP username (if authentication required)
            password: SMTP password (if authentication required)
            use_tls: Whether to use TLS encryption
        """
        super().__init__(enabled=bool(smtp_host and from_address and to_addresses))
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.from_address = from_address
        self.to_addresses = to_addresses
        self.username = username or from_address
        self.password = password
        self.use_tls = use_tls

    def get_name(self) -> str:
        return "Email"

    def send(self, title: str, message: str, severity: str = "info") -> bool:
        """Send notification via email."""
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{severity.upper()}] {title}"
            msg["From"] = self.from_address
            msg["To"] = ", ".join(self.to_addresses)

            # Plain text version
            text_body = f"{title}\n\n{message}"

            # HTML version with basic styling
            severity_colors = {
                "info": "#3498db",
                "success": "#2ecc71",
                "warning": "#f39c12",
                "error": "#e74c3c",
                "critical": "#9b59b6",
            }
            color = severity_colors.get(severity, "#95a5a6")

            html_body = f"""
            <html>
                <head>
                    <style>
                        body {{ font-family: Arial, sans-serif; }}
                        .header {{ background-color: {color}; color: white; padding: 20px; }}
                        .content {{ padding: 20px; background-color: #f5f5f5; }}
                        .footer {{ padding: 10px; text-align: center; color: #888; font-size: 12px; }}
                        pre {{ background-color: #fff; padding: 10px; border-left: 3px solid {color}; }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h2>{title}</h2>
                    </div>
                    <div class="content">
                        <pre>{message}</pre>
                    </div>
                    <div class="footer">
                        PM Agent Notification System
                    </div>
                </body>
            </html>
            """

            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            # Send email
            if self.use_tls:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
                server.starttls()
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)

            if self.password:
                server.login(self.username, self.password)

            server.sendmail(self.from_address, self.to_addresses, msg.as_string())
            server.quit()

            return True

        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False


# =============================================================================
# Notification Manager
# =============================================================================

class NotificationManager:
    """
    Manages multiple notification channels and provides high-level APIs.

    Handles:
    - Multiple notification channels
    - Message formatting with templates
    - Graceful failure handling
    - Channel prioritization
    """

    def __init__(self, config: Optional[NotificationConfig] = None):
        """
        Initialize notification manager.

        Args:
            config: Notification configuration (defaults to environment)
        """
        self.config = config or NotificationConfig.from_env()
        self.channels: List[NotificationChannel] = []
        self._initialize_channels()

    def _initialize_channels(self) -> None:
        """Initialize notification channels from configuration."""
        # Always add console notifier for development
        if self.config.console_enabled:
            self.add_channel(ConsoleNotifier(use_color=self.config.console_use_color))

        # Add Slack if configured
        if self.config.slack_webhook_url:
            self.add_channel(SlackNotifier(
                webhook_url=self.config.slack_webhook_url,
                channel=self.config.slack_channel,
                username=self.config.slack_username,
            ))

        # Add Discord if configured
        if self.config.discord_webhook_url:
            self.add_channel(DiscordNotifier(
                webhook_url=self.config.discord_webhook_url,
                username=self.config.discord_username,
            ))

        # Add Email if configured
        if (self.config.email_smtp_host and
            self.config.email_from and
            self.config.email_to):
            self.add_channel(EmailNotifier(
                smtp_host=self.config.email_smtp_host,
                smtp_port=self.config.email_smtp_port,
                from_address=self.config.email_from,
                to_addresses=self.config.email_to,
                username=self.config.email_smtp_username,
                password=self.config.email_smtp_password,
                use_tls=self.config.email_use_tls,
            ))

        logger.info(f"Initialized {len(self.channels)} notification channels")

    def add_channel(self, channel: NotificationChannel) -> None:
        """
        Add a notification channel.

        Args:
            channel: Notification channel instance
        """
        self.channels.append(channel)
        logger.debug(f"Added notification channel: {channel.get_name()}")

    def notify(self, title: str, message: str, severity: str = "info") -> bool:
        """
        Send notification to all enabled channels.

        Args:
            title: Notification title
            message: Notification body
            severity: Severity level

        Returns:
            True if at least one channel succeeded
        """
        if not self.channels:
            logger.warning("No notification channels configured")
            return False

        success_count = 0
        for channel in self.channels:
            if channel.notify(title, message, severity):
                success_count += 1

        return success_count > 0

    def notify_escalation(
        self,
        escalation: Escalation,
        task: Optional[Task] = None,
        goal: Optional[Goal] = None,
    ) -> bool:
        """
        Notify about a task escalation.

        Args:
            escalation: Escalation instance
            task: Associated task (optional)
            goal: Associated goal (optional)

        Returns:
            True if notification sent successfully
        """
        template = NotificationTemplate.escalation(escalation, task, goal)
        return self.notify(
            title=template["title"],
            message=template["body"],
            severity=template["severity"],
        )

    def notify_milestone(self, message: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Notify about a project milestone.

        Args:
            message: Milestone message
            details: Additional details

        Returns:
            True if notification sent successfully
        """
        template = NotificationTemplate.milestone(message, details)
        return self.notify(
            title=template["title"],
            message=template["body"],
            severity=template["severity"],
        )

    def notify_error(self, error_message: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Notify about an error.

        Args:
            error_message: Error message
            context: Additional context

        Returns:
            True if notification sent successfully
        """
        template = NotificationTemplate.error(error_message, context)
        return self.notify(
            title=template["title"],
            message=template["body"],
            severity=template["severity"],
        )

    def notify_goal_completed(
        self,
        goal: Goal,
        task_count: int,
        duration_minutes: Optional[float] = None,
    ) -> bool:
        """
        Notify about goal completion.

        Args:
            goal: Completed goal
            task_count: Number of tasks completed
            duration_minutes: Time taken in minutes

        Returns:
            True if notification sent successfully
        """
        template = NotificationTemplate.goal_completed(goal, task_count, duration_minutes)
        return self.notify(
            title=template["title"],
            message=template["body"],
            severity=template["severity"],
        )

    def get_channel_status(self) -> List[Dict[str, Any]]:
        """
        Get status of all notification channels.

        Returns:
            List of channel status dictionaries
        """
        return [
            {
                "name": channel.get_name(),
                "enabled": channel.enabled,
                "failure_count": channel.failure_count,
                "max_failures": channel.max_failures,
            }
            for channel in self.channels
        ]
