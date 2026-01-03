# PM Agent

An autonomous Project Manager agent that orchestrates Claude Code CLI and GLM 4.7 to manage software development projects from goal to deployment.

## Features

### Core Capabilities
- **Intelligent Task Breakdown**: Automatically decomposes goals into actionable tasks
- **Multi-Backend Support**: Claude Code (primary), GLM 4.7 (cost-effective), or Hybrid routing
- **Self-Improvement**: Learns from task outcomes to improve planning and execution
- **Project Memory**: Remembers decisions, patterns, and context across sessions
- **Real-time Dashboards**: CLI and Web-based monitoring interfaces

### Production Features
- **Docker Orchestration**: Build, run, and manage containers
- **Database Management**: Migrations, backups, and seeding
- **CI/CD Integration**: GitHub Actions, GitLab CI monitoring
- **Test Automation**: pytest, Jest, Playwright support
- **Error Monitoring**: Sentry integration for error tracking
- **Deployment Automation**: Vercel, AWS S3/Lambda support
- **File Watching**: Hot reload and development server management

## Installation

```bash
# Clone the repository
git clone https://github.com/robertcprice/pm-agent.git
cd pm-agent

# Install dependencies
pip install -e .

# Install extra dependencies for full capabilities
pip install -e ".[docker,database,testing,monitoring,deployment]"
```

## Quick Start

### Basic Usage

```python
from pm_agent import PMAgent, PMConfig

# Configure the agent
config = PMConfig(
    project_path="/path/to/project",
    claude_code_path="/usr/local/bin/claude",
)

# Initialize agent
agent = PMAgent(config)

# Give it a goal
await agent.add_goal(
    name="Build REST API",
    description="Create a FastAPI REST API with user authentication"
)

# Let it work
await agent.run_cycle()
```

### With GLM Backend (Cost-Effective)

```python
from pm_agent import PMAgent, PMConfig
from pm_agent.glm_backend import GLMBackend

config = PMConfig(
    project_path="/path/to/project",
    backend=GLMBackend(
        api_key="your-glm-key",
        base_url="https://api.z.ai/api/anthropic"
    )
)

agent = PMAgent(config)
```

### Using Capabilities

```python
from pm_agent.capabilities import (
    DockerCapability,
    DatabaseCapability,
    TestCapability,
)

# Manage containers
docker = DockerCapability()
await docker.compose_up(
    compose_file=Path("docker-compose.yml"),
    project_name="myapp"
)

# Run database migrations
db = DatabaseCapability()
await db.run_migrations(
    config=DatabaseConfig(
        db_type=DatabaseType.POSTGRESQL,
        host="localhost",
        port=5432,
        database="myapp",
        username="user",
        password="pass"
    ),
    framework=MigrationFramework.ALEMBIC,
    project_path=Path("/path/to/project")
)

# Run tests
testing = TestCapability()
result = await testing.run_tests(
    project_path=Path("/path/to/project"),
    framework=TestFramework.PYTEST,
    coverage=True
)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        PM AGENT                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │   PLANNER     │  │  DELEGATOR    │  │   REVIEWER    │  │
│  │               │  │               │  │               │  │
│  │ Goal Analysis │  │ Task Routing  │  │ Validation    │  │
│  │ Breakdown     │  │ Claude Code   │  │ Quality Check │  │
│  │ Prioritization│  │ GLM 4.7       │  │ Auto-Fix      │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  │
│         │                    │                    │         │
│         └────────────────────┴────────────────────┘         │
│                            │                                │
│  ┌─────────────────────────▼───────────────────────────┐  │
│  │                  TASK QUEUE (SQLite)                 │  │
│  │  • Projects  • Goals  • Tasks  • Logs  • Escalations │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                    CAPABILITIES                      │  │
│  ├─────────────┬─────────────┬─────────────┬───────────┤  │
│  │   Docker    │  Database   │  CI/CD      │  Testing  │  │
│  ├─────────────┼─────────────┼─────────────┼───────────┤  │
│  │ Monitoring  │ Deployment  │ File Watch  │   EGO     │  │
│  └─────────────┴─────────────┴─────────────┴───────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Capabilities Reference

### Docker Capability
- Build and run containers
- Docker Compose orchestration
- Log fetching and monitoring
- Resource management
- Automatic cleanup

### Database Capability
- Migration execution (Alembic, Prisma, Django)
- Backup and restore
- Database seeding
- Query execution
- Safety checks and dry-run mode

### CI/CD Capability
- GitHub Actions integration
- Workflow status monitoring
- Manual workflow triggering
- Log fetching
- Deployment gates

### Testing Capability
- Test discovery (pytest, Jest, Go test)
- Automated test execution
- Coverage analysis
- Flaky test detection
- Test generation

### Monitoring Capability
- Sentry error tracking
- Prometheus metrics
- Event aggregation
- Alert routing

### Deployment Capability
- Vercel deployments
- AWS S3/Lambda deployments
- Rollback support
- Environment variable management
- Deployment status monitoring

### File Watch Capability
- File system monitoring
- Development server management
- Auto-restart on crash
- Log tailing
- Pattern-based filtering

## Configuration

### Environment Variables

```bash
# Claude Code Configuration
ANTHROPIC_AUTH_TOKEN=your-token
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_DEFAULT_MODEL=claude-opus-4-20250514

# GLM Configuration (optional)
GLM_API_KEY=your-glm-key
GLM_BASE_URL=https://api.z.ai/api/anthropic

# Sentry (optional)
SENTRY_DSN=your-sentry-dsn
SENTRY_ENVIRONMENT=development

# Notifications (optional)
SLACK_WEBHOOK_URL=your-webhook-url
DISCORD_WEBHOOK_URL=your-discord-webhook
```

### Project Settings

```python
from pm_agent import PMConfig

config = PMConfig(
    project_path="/path/to/project",
    work_hours="9-18",  # Working hours
    max_parallel_tasks=3,
    auto_escalate=True,
    notification_channels=["slack", "discord"],
)
```

## CLI Usage

```bash
# Start the agent with Claude Code backend
pm-agent run --project /path/to/project

# Start with GLM backend (cost-effective)
pm-agent run --project /path/to/project --backend glm

# Start with web dashboard
pm-agent run --project /path/to/project --dashboard web

# Add a goal
pm-agent goal add "Add user authentication"

# List tasks
pm-agent task list

# Show status
pm-agent status
```

## Development

### Running Tests

```bash
# Unit tests
pytest tests/unit/

# Integration tests (requires API keys)
pytest tests/integration/

# With coverage
pytest --cov=pm_agent
```

### Project Structure

```
pm-agent/
├── pm_agent/              # Main package
│   ├── agent.py          # Core agent logic
│   ├── task_queue.py     # SQLite-backed task management
│   ├── logger.py         # Thought tracking
│   ├── cli_dashboard.py  # Terminal UI
│   ├── web_dashboard.py  # Web interface
│   ├── goal_analyzer.py  # Task breakdown
│   ├── notifications.py  # Multi-channel alerts
│   ├── ego_integration.py# EGO model integration
│   ├── project_memory.py # Learning system
│   ├── adaptive_learner.py    # Self-improvement
│   ├── intelligent_retry.py   # Smart retries
│   ├── model_selector.py      # Cost optimization
│   ├── claude_mentor.py       # Conversational learning
│   ├── glm_backend.py         # GLM integration
│   ├── hybrid_backend.py      # Hybrid routing
│   └── production.py          # Production features
├── capabilities/          # Modular capabilities
│   ├── docker.py
│   ├── database.py
│   ├── cicd.py
│   ├── testing.py
│   ├── monitoring.py
│   ├── deployment.py
│   └── filewatch.py
├── tests/               # Test suites
├── docs/                # Documentation
├── config/              # Configuration files
└── scripts/             # Utility scripts
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.

## Acknowledgments

Built with Claude Code and powered by Anthropic's Claude models.
