# Quick Start: PM Agent Notifications

Get notifications up and running in 5 minutes.

## Step 1: Basic Setup (30 seconds)

Console notifications work out of the box with zero configuration.

```python
from conch_dna.pm import PMAgent, NotificationManager

# Create notification manager
notification_manager = NotificationManager()

# Add to PM Agent
agent = PMAgent(
    config=pm_config,
    task_queue=task_queue,
    claude_code=claude_code,
    notification_manager=notification_manager,  # <-- Add this line
)

# That's it! Console notifications enabled.
```

## Step 2: Test It (1 minute)

```bash
cd conch_dna/pm
python test_notifications.py
```

You should see colorful console output showing different notification types.

## Step 3: Add Slack (2 minutes)

1. Create a Slack webhook:
   - Go to https://api.slack.com/apps
   - Create app → "Incoming Webhooks" → Create webhook
   - Copy the URL

2. Set environment variable:
   ```bash
   export PM_SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
   ```

3. Restart your PM Agent:
   ```python
   from conch_dna.pm import NotificationConfig, NotificationManager

   config = NotificationConfig.from_env()
   manager = NotificationManager(config)
   ```

4. Test it:
   ```bash
   python test_notifications.py --test-slack "$PM_SLACK_WEBHOOK"
   ```

## Step 4: Add Discord (Optional, 2 minutes)

1. Create Discord webhook:
   - Server Settings → Integrations → Webhooks
   - Create webhook → Copy URL

2. Set environment variable:
   ```bash
   export PM_DISCORD_WEBHOOK="https://discord.com/api/webhooks/YOUR/WEBHOOK"
   ```

3. Test it:
   ```bash
   python test_notifications.py --test-discord "$PM_DISCORD_WEBHOOK"
   ```

## Step 5: Add Email (Optional, 3 minutes)

For Gmail:

1. Create app password:
   - Go to https://myaccount.google.com/apppasswords
   - Generate app password
   - Copy the password

2. Set environment variables:
   ```bash
   export PM_EMAIL_SMTP_HOST="smtp.gmail.com"
   export PM_EMAIL_FROM="your-email@gmail.com"
   export PM_EMAIL_TO="recipient@example.com"
   export PM_EMAIL_SMTP_USERNAME="your-email@gmail.com"
   export PM_EMAIL_SMTP_PASSWORD="your-app-password"
   ```

3. Test it:
   ```bash
   python test_notifications.py --test-email \
     --smtp-host smtp.gmail.com \
     --from-addr your-email@gmail.com \
     --to-addr recipient@example.com \
     --smtp-user your-email@gmail.com \
     --smtp-pass your-app-password
   ```

## Complete Example

```python
from conch_dna.pm import (
    PMAgent,
    PMConfig,
    TaskQueue,
    NotificationManager,
    NotificationConfig,
)
from pathlib import Path

# Configure notifications from environment
notification_config = NotificationConfig.from_env()
notification_manager = NotificationManager(notification_config)

# Create PM Agent
pm_config = PMConfig(
    project_root=Path.cwd(),
    data_dir=Path("pm_data"),
)

task_queue = TaskQueue(Path("pm_data/tasks.db"))

agent = PMAgent(
    config=pm_config,
    task_queue=task_queue,
    claude_code=claude_code,
    notification_manager=notification_manager,
)

# Run it
await agent.run()
```

## Environment Variables Reference

```bash
# Console (enabled by default)
export PM_CONSOLE_NOTIFICATIONS=true
export PM_CONSOLE_COLOR=true

# Slack (optional)
export PM_SLACK_WEBHOOK="https://hooks.slack.com/services/..."
export PM_SLACK_CHANNEL="#pm-alerts"

# Discord (optional)
export PM_DISCORD_WEBHOOK="https://discord.com/api/webhooks/..."

# Email (optional)
export PM_EMAIL_SMTP_HOST="smtp.gmail.com"
export PM_EMAIL_SMTP_PORT=587
export PM_EMAIL_FROM="pm-agent@yourdomain.com"
export PM_EMAIL_TO="manager@yourdomain.com,team@yourdomain.com"
export PM_EMAIL_SMTP_USERNAME="your-email@gmail.com"
export PM_EMAIL_SMTP_PASSWORD="your-app-password"
```

## Common Issues

### Slack notifications not working

1. Verify webhook URL is correct
2. Test with curl:
   ```bash
   curl -X POST -H 'Content-type: application/json' \
     --data '{"text":"Test"}' \
     "$PM_SLACK_WEBHOOK"
   ```

### Email "Authentication failed"

For Gmail:
- Use an [app password](https://support.google.com/accounts/answer/185833), not your regular password
- Enable "Less secure app access" (not recommended, use app passwords instead)

### requests library not found

Install it:
```bash
pip install requests
```

This is only needed for Slack/Discord webhooks.

## What Gets Notified?

The PM Agent automatically sends notifications for:

1. **Escalations** - When tasks need human help (always notify)
2. **Goal Completions** - When goals finish successfully (always notify)
3. **Errors** - When PM Agent encounters errors (always notify)
4. **Milestones** - Custom milestone events (optional)

## Next Steps

- Read [NOTIFICATIONS_README.md](./NOTIFICATIONS_README.md) for detailed documentation
- See [NOTIFICATIONS_CONFIG.md](./NOTIFICATIONS_CONFIG.md) for advanced configuration
- Run [example_notifications.py](./example_notifications.py) for usage examples
- Check [example_pm_integration.py](./example_pm_integration.py) for integration patterns

## Need Help?

1. Run the test suite: `python test_notifications.py`
2. Check logs for errors
3. Verify environment variables are set
4. Review [NOTIFICATIONS_CONFIG.md](./NOTIFICATIONS_CONFIG.md) troubleshooting section

---

**You're all set!** The PM Agent will now notify you when it needs help or achieves milestones.
