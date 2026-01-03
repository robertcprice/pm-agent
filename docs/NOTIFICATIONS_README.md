# PM Agent Notification System

The notification system enables the PM Agent to alert humans when it needs help, encounters errors, or achieves milestones.

## Quick Start

### Basic Usage (Console Only)

```python
from conch_dna.pm import PMAgent, NotificationManager

# Console notifications work out of the box
notification_manager = NotificationManager()

agent = PMAgent(
    config=pm_config,
    task_queue=task_queue,
    claude_code=claude_code,
    notification_manager=notification_manager,
)

await agent.run()
```

### Multi-Channel Setup

```bash
# Configure via environment variables
export PM_SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
export PM_DISCORD_WEBHOOK="https://discord.com/api/webhooks/YOUR/WEBHOOK/URL"
export PM_EMAIL_SMTP_HOST="smtp.gmail.com"
export PM_EMAIL_FROM="pm-agent@yourdomain.com"
export PM_EMAIL_TO="manager@yourdomain.com"
```

```python
from conch_dna.pm import NotificationManager, NotificationConfig

# Load configuration from environment
config = NotificationConfig.from_env()
manager = NotificationManager(config)

# All channels configured automatically
agent = PMAgent(..., notification_manager=manager)
```

## Features

### Notification Types

1. **Escalations** - Tasks requiring human intervention
2. **Milestones** - Project achievements
3. **Errors** - Critical system errors
4. **Goal Completions** - Successful goal achievements

### Supported Channels

- **Console** - Terminal output (always available)
- **Slack** - Webhook integration
- **Discord** - Webhook integration
- **Email** - SMTP delivery

### Graceful Failure

- Channels fail independently
- Circuit breaker prevents spam
- PM Agent continues operation even if notifications fail
- All failures are logged

## Architecture

```
NotificationManager
├── ConsoleNotifier (always available)
├── SlackNotifier (optional)
├── DiscordNotifier (optional)
└── EmailNotifier (optional)
```

### Notification Flow

```
PM Agent Event
    ↓
NotificationManager.notify_escalation()
    ↓
NotificationTemplate.escalation()
    ↓
Send to all enabled channels in parallel
    ↓
    ├→ Console: Print to terminal
    ├→ Slack: POST to webhook
    ├→ Discord: POST to webhook
    └→ Email: Send via SMTP
```

## Configuration

See [NOTIFICATIONS_CONFIG.md](./NOTIFICATIONS_CONFIG.md) for detailed configuration documentation.

## Testing

Run the test suite:

```bash
cd conch_dna/pm
python test_notifications.py
```

Test specific channels:

```bash
# Test Slack
python test_notifications.py --test-slack "https://hooks.slack.com/..."

# Test Discord
python test_notifications.py --test-discord "https://discord.com/api/webhooks/..."

# Test Email
python test_notifications.py --test-email \
    --smtp-host smtp.gmail.com \
    --from-addr your@email.com \
    --to-addr recipient@email.com
```

## Examples

### Simple Console Notifications

```python
from conch_dna.pm import NotificationManager

manager = NotificationManager()

# Send different types of notifications
manager.notify_milestone("Database migration completed", {"migrations": 5})
manager.notify_error("Connection failed", {"host": "localhost:5432"})
```

### Multi-Channel Production Setup

```python
from conch_dna.pm import NotificationManager, NotificationConfig

config = NotificationConfig(
    slack_webhook_url=os.getenv("SLACK_WEBHOOK"),
    discord_webhook_url=os.getenv("DISCORD_WEBHOOK"),
    email_smtp_host="smtp.sendgrid.net",
    email_from="pm-agent@company.com",
    email_to=["oncall@company.com"],
    console_enabled=True,
)

manager = NotificationManager(config)

# Notifications will be sent to all configured channels
agent = PMAgent(..., notification_manager=manager)
```

### Custom Notification Channel

```python
from conch_dna.pm.notifications import NotificationChannel

class PagerDutyNotifier(NotificationChannel):
    def __init__(self, api_key: str, service_id: str):
        super().__init__()
        self.api_key = api_key
        self.service_id = service_id

    def get_name(self) -> str:
        return "PagerDuty"

    def send(self, title: str, message: str, severity: str = "info") -> bool:
        # Only send critical notifications to PagerDuty
        if severity != "critical":
            return True

        # Implement PagerDuty API call
        try:
            # Create incident via PagerDuty API
            return True
        except Exception as e:
            logger.error(f"PagerDuty notification failed: {e}")
            return False

# Add custom channel
manager = NotificationManager()
manager.add_channel(PagerDutyNotifier(api_key="...", service_id="..."))
```

## Integration with PM Agent

The PM Agent automatically sends notifications for:

1. **Task Escalations** - When a task fails after max attempts or encounters safety blocks
2. **Goal Completions** - When all tasks in a goal are completed
3. **Cycle Errors** - When the PM Agent encounters critical errors
4. **Partial Completions** - When goals complete with some failed tasks

### Escalation Example

```
🚨 Task Escalated - Human Attention Required

Escalation ID: abc-123
Reason: Task failed after 3 attempts. Error: Connection timeout

Task Details:
- ID: task-456
- Description: Implement authentication API
- Status: escalated
- Priority: HIGH
- Attempts: 3/3
- Error: Connection timeout when accessing external API
- Context Files: api/auth.py, api/middleware.py

Related Goal:
- Description: Build authentication system
- Status: in_progress

Action Required:
Please review and provide guidance to resolve this escalation.
```

### Goal Completion Example

```
✅ Goal Completed Successfully

Goal: Build complete authentication system with JWT and refresh tokens
Priority: HIGH
Tasks Completed: 5
Duration: 45.5 minutes
```

## Security Considerations

1. **Never commit secrets** - Use environment variables or secret management
2. **Rotate credentials** - Regularly update API keys and passwords
3. **Use app passwords** - For Gmail and similar providers
4. **Monitor webhook usage** - Check for unauthorized access
5. **Limit email recipients** - Only send to authorized team members

## Troubleshooting

### Notifications not being sent

1. Check if channels are enabled:
   ```python
   status = manager.get_channel_status()
   print(status)
   ```

2. Check logs for errors:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. Verify configuration:
   ```python
   print(config.__dict__)
   ```

### Circuit breaker triggering

If a channel fails 3 times in a row, it's automatically disabled. Check:

```python
status = manager.get_channel_status()
for channel in status:
    if not channel['enabled']:
        print(f"{channel['name']} disabled after {channel['failure_count']} failures")
```

To re-enable, restart the PM Agent.

### Slack webhook not working

Test webhook directly:
```bash
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Test message"}' \
  YOUR_WEBHOOK_URL
```

### Email authentication failing

For Gmail, use an [app password](https://support.google.com/accounts/answer/185833) instead of your regular password.

## API Reference

### NotificationManager

```python
class NotificationManager:
    def __init__(self, config: Optional[NotificationConfig] = None)
    def add_channel(self, channel: NotificationChannel) -> None
    def notify(self, title: str, message: str, severity: str = "info") -> bool
    def notify_escalation(self, escalation: Escalation, task: Task, goal: Goal) -> bool
    def notify_milestone(self, message: str, details: Dict[str, Any]) -> bool
    def notify_error(self, error_message: str, context: Dict[str, Any]) -> bool
    def notify_goal_completed(self, goal: Goal, task_count: int, duration_minutes: float) -> bool
    def get_channel_status(self) -> List[Dict[str, Any]]
```

### NotificationConfig

```python
@dataclass
class NotificationConfig:
    # Slack
    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None
    slack_username: str = "PM Agent"

    # Discord
    discord_webhook_url: Optional[str] = None
    discord_username: str = "PM Agent"

    # Email
    email_smtp_host: Optional[str] = None
    email_smtp_port: int = 587
    email_smtp_username: Optional[str] = None
    email_smtp_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: Optional[List[str]] = None
    email_use_tls: bool = True

    # Console
    console_enabled: bool = True
    console_use_color: bool = True

    @classmethod
    def from_env(cls) -> "NotificationConfig"
    @classmethod
    def from_file(cls, config_path: Path) -> "NotificationConfig"
```

### NotificationChannel (Abstract)

```python
class NotificationChannel(ABC):
    def __init__(self, enabled: bool = True)
    @abstractmethod
    def send(self, title: str, message: str, severity: str) -> bool
    @abstractmethod
    def get_name(self) -> str
    def notify(self, title: str, message: str, severity: str) -> bool
```

## Files

- `notifications.py` - Core notification system implementation
- `test_notifications.py` - Comprehensive test suite
- `NOTIFICATIONS_CONFIG.md` - Detailed configuration guide
- `NOTIFICATIONS_README.md` - This file

## Dependencies

Required:
- `smtplib` (Python standard library)
- `email` (Python standard library)

Optional:
- `requests` - Required for Slack and Discord webhooks
  ```bash
  pip install requests
  ```

## License

Part of the KVRM Conscious project.
