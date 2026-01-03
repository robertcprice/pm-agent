# PM Agent Notification System Configuration

This document explains how to configure the notification system for the PM Agent.

## Overview

The PM Agent notification system supports multiple channels for alerting humans about:
- **Escalations**: Tasks that need human intervention
- **Milestones**: Important project achievements
- **Errors**: Critical system errors
- **Goal Completions**: Successful goal achievements

## Supported Notification Channels

### 1. Console Notifications (Default)

Always enabled by default for local development. Prints formatted notifications to the terminal.

**Configuration:**
```python
from conch_dna.pm import NotificationConfig

config = NotificationConfig(
    console_enabled=True,
    console_use_color=True,  # Use ANSI colors
)
```

**Environment Variables:**
```bash
export PM_CONSOLE_NOTIFICATIONS=true
export PM_CONSOLE_COLOR=true
```

### 2. Slack Notifications

Send notifications to Slack via webhook.

**Setup:**
1. Create a Slack webhook:
   - Go to https://api.slack.com/apps
   - Create a new app or use existing
   - Enable "Incoming Webhooks"
   - Create a webhook URL for your channel
   - Copy the webhook URL (e.g., `https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX`)

2. Configure in code:
```python
config = NotificationConfig(
    slack_webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    slack_channel="#pm-alerts",  # Optional: override webhook's default channel
    slack_username="PM Agent",   # Optional: custom bot name
)
```

3. Or use environment variables:
```bash
export PM_SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
export PM_SLACK_CHANNEL="#pm-alerts"
export PM_SLACK_USERNAME="PM Agent"
```

**Features:**
- Color-coded messages based on severity
- Rich formatting with attachments
- Emoji indicators for different event types
- Automatic timestamping

### 3. Discord Notifications

Send notifications to Discord via webhook.

**Setup:**
1. Create a Discord webhook:
   - Go to your Discord server settings
   - Select "Integrations" → "Webhooks"
   - Click "New Webhook"
   - Choose the channel for notifications
   - Copy the webhook URL

2. Configure in code:
```python
config = NotificationConfig(
    discord_webhook_url="https://discord.com/api/webhooks/YOUR/WEBHOOK/URL",
    discord_username="PM Agent",  # Optional: custom bot name
)
```

3. Or use environment variables:
```bash
export PM_DISCORD_WEBHOOK="https://discord.com/api/webhooks/YOUR/WEBHOOK/URL"
export PM_DISCORD_USERNAME="PM Agent"
```

**Features:**
- Rich embeds with color coding
- Formatted markdown messages
- Automatic timestamping
- Footer branding

### 4. Email Notifications

Send notifications via SMTP email.

**Setup:**

1. Configure SMTP settings:
```python
config = NotificationConfig(
    email_smtp_host="smtp.gmail.com",
    email_smtp_port=587,
    email_smtp_username="your-email@gmail.com",
    email_smtp_password="your-app-password",  # Use app password for Gmail
    email_from="pm-agent@yourdomain.com",
    email_to=["manager@yourdomain.com", "team@yourdomain.com"],
    email_use_tls=True,
)
```

2. Or use environment variables:
```bash
export PM_EMAIL_SMTP_HOST="smtp.gmail.com"
export PM_EMAIL_SMTP_PORT=587
export PM_EMAIL_SMTP_USERNAME="your-email@gmail.com"
export PM_EMAIL_SMTP_PASSWORD="your-app-password"
export PM_EMAIL_FROM="pm-agent@yourdomain.com"
export PM_EMAIL_TO="manager@yourdomain.com,team@yourdomain.com"
export PM_EMAIL_USE_TLS=true
```

**Common SMTP Providers:**

- **Gmail:**
  - Host: `smtp.gmail.com`
  - Port: `587` (TLS) or `465` (SSL)
  - Note: Use [app passwords](https://support.google.com/accounts/answer/185833)

- **Outlook/Office365:**
  - Host: `smtp.office365.com`
  - Port: `587`

- **SendGrid:**
  - Host: `smtp.sendgrid.net`
  - Port: `587`
  - Username: `apikey`
  - Password: Your SendGrid API key

- **AWS SES:**
  - Host: `email-smtp.us-east-1.amazonaws.com` (region-specific)
  - Port: `587`
  - Credentials from SES SMTP settings

**Features:**
- Plain text and HTML versions
- Color-coded subject lines
- Professional HTML formatting
- Multiple recipients

## Loading Configuration

### From Environment Variables (Recommended)

```python
from conch_dna.pm import NotificationManager, NotificationConfig

# Load from environment
config = NotificationConfig.from_env()
manager = NotificationManager(config)
```

### From JSON File

Create a `notifications.json` file:
```json
{
  "slack_webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
  "slack_channel": "#pm-alerts",
  "discord_webhook_url": "https://discord.com/api/webhooks/YOUR/WEBHOOK/URL",
  "email_smtp_host": "smtp.gmail.com",
  "email_smtp_port": 587,
  "email_from": "pm-agent@yourdomain.com",
  "email_to": ["manager@yourdomain.com"],
  "console_enabled": true
}
```

Load in code:
```python
from pathlib import Path

config = NotificationConfig.from_file(Path("notifications.json"))
manager = NotificationManager(config)
```

### Programmatic Configuration

```python
config = NotificationConfig(
    slack_webhook_url="...",
    discord_webhook_url="...",
    email_smtp_host="smtp.gmail.com",
    email_smtp_port=587,
    email_from="pm@example.com",
    email_to=["team@example.com"],
    console_enabled=True,
    console_use_color=True,
)
manager = NotificationManager(config)
```

## Integration with PM Agent

### Basic Integration

```python
from conch_dna.pm import PMAgent, PMConfig, TaskQueue, NotificationManager

# Create notification manager
notification_manager = NotificationManager()  # Uses env vars by default

# Create PM Agent with notifications
agent = PMAgent(
    config=pm_config,
    task_queue=task_queue,
    claude_code=claude_code,
    notification_manager=notification_manager,
)

# Run the agent
await agent.run()
```

### Custom Notification Channels

You can add custom notification channels:

```python
from conch_dna.pm.notifications import NotificationChannel

class CustomNotifier(NotificationChannel):
    def get_name(self) -> str:
        return "Custom"

    def send(self, title: str, message: str, severity: str = "info") -> bool:
        # Your custom notification logic
        print(f"Custom: {title} - {message}")
        return True

# Add to manager
manager = NotificationManager()
manager.add_channel(CustomNotifier())
```

## Notification Types

### 1. Escalation Notifications

Sent when a task needs human intervention:

```python
# Automatic (handled by PM Agent)
manager.notify_escalation(escalation, task, goal)
```

**Contains:**
- Escalation reason
- Task details (description, priority, attempts)
- Error messages
- Context files
- Related goal information

### 2. Milestone Notifications

Sent for important achievements:

```python
manager.notify_milestone(
    "Database migration completed",
    {
        "migrations_applied": 5,
        "time_taken": "2 minutes",
        "database": "production",
    }
)
```

### 3. Error Notifications

Sent for critical errors:

```python
manager.notify_error(
    "PM Agent cycle error",
    {
        "cycle": 42,
        "state": "delegating",
        "error": "Connection timeout",
    }
)
```

### 4. Goal Completion Notifications

Sent when goals are completed:

```python
# Automatic (handled by PM Agent)
manager.notify_goal_completed(
    goal=completed_goal,
    task_count=5,
    duration_minutes=45.5,
)
```

## Error Handling

The notification system is designed to fail gracefully:

1. **Circuit Breaker**: Channels are disabled after 3 consecutive failures
2. **Graceful Degradation**: PM Agent continues even if notifications fail
3. **Logging**: All notification failures are logged for debugging
4. **Status Monitoring**: Check channel health with `get_channel_status()`

```python
# Check notification channel status
status = manager.get_channel_status()
for channel in status:
    print(f"{channel['name']}: {channel['enabled']} ({channel['failure_count']} failures)")
```

## Testing Notifications

Test all notification channels:

```bash
cd conch_dna/pm
python test_notifications.py
```

Test specific channels:

```bash
# Test Slack
python test_notifications.py --test-slack "https://hooks.slack.com/services/YOUR/WEBHOOK"

# Test Discord
python test_notifications.py --test-discord "https://discord.com/api/webhooks/YOUR/WEBHOOK"

# Test Email
python test_notifications.py --test-email \
    --smtp-host smtp.gmail.com \
    --smtp-port 587 \
    --from-addr your-email@gmail.com \
    --to-addr recipient@example.com \
    --smtp-user your-email@gmail.com \
    --smtp-pass your-app-password
```

## Security Considerations

1. **Never commit credentials**: Use environment variables or secure secret management
2. **Use app passwords**: For Gmail and other providers that support them
3. **Limit webhook access**: Webhooks should only be accessible by your PM Agent
4. **Rotate credentials**: Regularly rotate API keys and passwords
5. **Monitor usage**: Check for unauthorized notification activity

## Environment Variables Reference

Complete list of environment variables:

```bash
# Console
export PM_CONSOLE_NOTIFICATIONS=true
export PM_CONSOLE_COLOR=true

# Slack
export PM_SLACK_WEBHOOK="https://hooks.slack.com/services/..."
export PM_SLACK_CHANNEL="#pm-alerts"
export PM_SLACK_USERNAME="PM Agent"

# Discord
export PM_DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."
export PM_DISCORD_USERNAME="PM Agent"

# Email
export PM_EMAIL_SMTP_HOST="smtp.gmail.com"
export PM_EMAIL_SMTP_PORT=587
export PM_EMAIL_SMTP_USERNAME="your-email@gmail.com"
export PM_EMAIL_SMTP_PASSWORD="your-app-password"
export PM_EMAIL_FROM="pm-agent@yourdomain.com"
export PM_EMAIL_TO="manager@yourdomain.com,team@yourdomain.com"
export PM_EMAIL_USE_TLS=true

# Advanced
export PM_INCLUDE_STACK_TRACES=false
```

## Troubleshooting

### Slack notifications not working

1. Verify webhook URL is correct
2. Check that the Slack app has permission to post
3. Test webhook with curl:
   ```bash
   curl -X POST -H 'Content-type: application/json' \
     --data '{"text":"Test message"}' \
     YOUR_WEBHOOK_URL
   ```

### Discord notifications not working

1. Verify webhook URL is correct
2. Check channel permissions
3. Test webhook with curl:
   ```bash
   curl -X POST -H 'Content-type: application/json' \
     --data '{"content":"Test message"}' \
     YOUR_WEBHOOK_URL
   ```

### Email notifications not working

1. Verify SMTP credentials
2. Check firewall/network settings (port 587 must be open)
3. For Gmail: Ensure "Less secure app access" is enabled or use app password
4. Check spam folder
5. Test SMTP connection:
   ```python
   import smtplib
   server = smtplib.SMTP('smtp.gmail.com', 587)
   server.starttls()
   server.login('your-email@gmail.com', 'your-password')
   server.quit()
   ```

### requests library not available

Install required dependencies:
```bash
pip install requests
```

## Examples

### Minimal Setup (Console Only)

```python
from conch_dna.pm import PMAgent, NotificationManager

# Console notifications enabled by default
manager = NotificationManager()
agent = PMAgent(..., notification_manager=manager)
```

### Production Setup (Multi-Channel)

```bash
# Set environment variables
export PM_SLACK_WEBHOOK="..."
export PM_DISCORD_WEBHOOK="..."
export PM_EMAIL_SMTP_HOST="smtp.sendgrid.net"
export PM_EMAIL_FROM="pm-agent@company.com"
export PM_EMAIL_TO="oncall@company.com,manager@company.com"
```

```python
from conch_dna.pm import NotificationManager, NotificationConfig

# Load from environment
config = NotificationConfig.from_env()
manager = NotificationManager(config)

# All channels configured automatically
agent = PMAgent(..., notification_manager=manager)
```

### Custom Setup

```python
from conch_dna.pm import NotificationManager, SlackNotifier, EmailNotifier

# Manual channel setup
manager = NotificationManager()
manager.add_channel(SlackNotifier(webhook_url="..."))
manager.add_channel(EmailNotifier(
    smtp_host="smtp.sendgrid.net",
    smtp_port=587,
    from_address="pm@company.com",
    to_addresses=["team@company.com"],
    username="apikey",
    password="SG.xxx",
))

agent = PMAgent(..., notification_manager=manager)
```
