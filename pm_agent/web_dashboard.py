#!/usr/bin/env python3
"""
PM Agent Web Dashboard - FastAPI + WebSocket real-time interface.

Features:
- Real-time status updates via WebSocket
- Live log streaming
- Interactive terminal console
- Task queue management
- Goal submission
- Escalation handling
"""

import asyncio
import json
import uvicorn
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Set
from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .task_queue import TaskQueue, Task, Goal, TaskStatus, GoalStatus, TaskPriority
from .logger import PMLogger, PMLogEntry, LogLevel


# ============================================================================
# Pydantic Models
# ============================================================================

class GoalCreate(BaseModel):
    description: str
    project_id: str
    priority: str = "medium"


class EscalationResponse(BaseModel):
    escalation_id: str
    response: str
    new_status: str = "queued"


class CommandInput(BaseModel):
    command: str


# ============================================================================
# WebSocket Manager
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connections."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected
        for conn in disconnected:
            self.active_connections.discard(conn)

    async def send_personal(self, websocket: WebSocket, message: dict):
        await websocket.send_json(message)


# ============================================================================
# Dashboard HTML Template
# ============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PM Agent Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
            background: #0d1117;
            color: #c9d1d9;
            min-height: 100vh;
        }
        .container {
            display: grid;
            grid-template-columns: 2fr 1fr;
            grid-template-rows: auto 1fr auto;
            gap: 12px;
            padding: 12px;
            height: 100vh;
        }
        .header {
            grid-column: 1 / -1;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            font-size: 1.4em;
            color: #58a6ff;
        }
        .header .status {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .header .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .header .status-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }
        .status-idle { background: #21262d; color: #8b949e; }
        .status-planning { background: #3d2d00; color: #f0c000; }
        .status-delegating { background: #0d2d3d; color: #58a6ff; }
        .status-reviewing { background: #2d1d3d; color: #bc8cff; }
        .status-escalating { background: #3d1d1d; color: #f85149; }

        .main-content {
            display: flex;
            flex-direction: column;
            gap: 12px;
            overflow: hidden;
        }
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 12px;
            overflow: hidden;
        }
        .panel {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        .panel-header {
            padding: 12px 16px;
            background: #21262d;
            border-bottom: 1px solid #30363d;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .panel-content {
            padding: 12px;
            overflow-y: auto;
            flex: 1;
        }

        /* Terminal/Console */
        .terminal {
            flex: 1;
            min-height: 300px;
        }
        .terminal-output {
            font-family: inherit;
            font-size: 0.85em;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .terminal-input-container {
            display: flex;
            gap: 8px;
            padding: 12px;
            background: #21262d;
            border-top: 1px solid #30363d;
        }
        .terminal-prompt {
            color: #58a6ff;
            font-weight: bold;
        }
        .terminal-input {
            flex: 1;
            background: transparent;
            border: none;
            color: #c9d1d9;
            font-family: inherit;
            font-size: 0.9em;
            outline: none;
        }

        /* Log styling */
        .log-entry {
            padding: 4px 0;
            border-bottom: 1px solid #21262d;
        }
        .log-time { color: #6e7681; font-size: 0.8em; }
        .log-level { font-weight: 600; margin: 0 8px; }
        .log-level.debug { color: #6e7681; }
        .log-level.info { color: #58a6ff; }
        .log-level.thought { color: #bc8cff; }
        .log-level.action { color: #3fb950; }
        .log-level.result { color: #3fb950; }
        .log-level.warning { color: #d29922; }
        .log-level.error { color: #f85149; }
        .log-level.milestone { color: #3fb950; font-weight: bold; }

        /* Thoughts */
        .thought-entry {
            padding: 8px;
            margin-bottom: 8px;
            background: #21262d;
            border-radius: 6px;
            border-left: 3px solid #bc8cff;
        }
        .thought-type {
            font-size: 0.75em;
            color: #bc8cff;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .thought-content {
            font-size: 0.9em;
            color: #c9d1d9;
        }

        /* Tasks */
        .task-stats {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }
        .stat-card {
            background: #21262d;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }
        .stat-value {
            font-size: 1.8em;
            font-weight: bold;
        }
        .stat-label {
            font-size: 0.75em;
            color: #8b949e;
            text-transform: uppercase;
        }
        .stat-pending .stat-value { color: #d29922; }
        .stat-completed .stat-value { color: #3fb950; }
        .stat-failed .stat-value { color: #f85149; }
        .stat-escalated .stat-value { color: #bc8cff; }

        /* Goal input */
        .goal-form {
            padding: 12px;
            background: #21262d;
            border-top: 1px solid #30363d;
        }
        .goal-input {
            width: 100%;
            padding: 10px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-family: inherit;
            resize: vertical;
            min-height: 60px;
        }
        .goal-submit {
            width: 100%;
            margin-top: 8px;
            padding: 10px;
            background: #238636;
            border: none;
            border-radius: 6px;
            color: white;
            font-weight: 600;
            cursor: pointer;
        }
        .goal-submit:hover {
            background: #2ea043;
        }

        /* Connection status */
        .connection-status {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85em;
        }
        .connection-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }
        .connected .connection-dot { background: #3fb950; }
        .disconnected .connection-dot { background: #f85149; }

        /* Footer */
        .footer {
            grid-column: 1 / -1;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 12px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85em;
            color: #8b949e;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 PM Agent Dashboard</h1>
            <div class="status">
                <div class="status-item">
                    <span>State:</span>
                    <span id="agent-state" class="status-badge status-idle">IDLE</span>
                </div>
                <div class="status-item">
                    <span>Cycle:</span>
                    <span id="cycle-count">0</span>
                </div>
                <div class="status-item connection-status" id="connection-status">
                    <span class="connection-dot"></span>
                    <span>Connecting...</span>
                </div>
            </div>
        </div>

        <div class="main-content">
            <div class="panel terminal">
                <div class="panel-header">
                    <span>📺 Live Console</span>
                    <span style="font-size: 0.8em; color: #8b949e;" id="log-count">0 entries</span>
                </div>
                <div class="panel-content" id="terminal-output">
                    <div class="terminal-output"></div>
                </div>
                <div class="terminal-input-container">
                    <span class="terminal-prompt">pm&gt;</span>
                    <input type="text" class="terminal-input" id="terminal-input"
                           placeholder="Type command (help for list)..."
                           autocomplete="off">
                </div>
            </div>
        </div>

        <div class="sidebar">
            <div class="panel" style="flex: 0 0 auto;">
                <div class="panel-header">📊 Task Stats</div>
                <div class="panel-content">
                    <div class="task-stats">
                        <div class="stat-card stat-pending">
                            <div class="stat-value" id="stat-pending">0</div>
                            <div class="stat-label">Pending</div>
                        </div>
                        <div class="stat-card stat-completed">
                            <div class="stat-value" id="stat-completed">0</div>
                            <div class="stat-label">Completed</div>
                        </div>
                        <div class="stat-card stat-failed">
                            <div class="stat-value" id="stat-failed">0</div>
                            <div class="stat-label">Failed</div>
                        </div>
                        <div class="stat-card stat-escalated">
                            <div class="stat-value" id="stat-escalated">0</div>
                            <div class="stat-label">Escalated</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="panel" style="flex: 1;">
                <div class="panel-header">🧠 Recent Thoughts</div>
                <div class="panel-content" id="thoughts-container">
                    <div style="color: #6e7681; font-style: italic;">
                        Waiting for thoughts...
                    </div>
                </div>
            </div>

            <div class="panel" style="flex: 0 0 auto;">
                <div class="panel-header">🎯 Add Goal</div>
                <div class="goal-form">
                    <textarea class="goal-input" id="goal-input"
                              placeholder="Describe what you want to accomplish..."></textarea>
                    <button class="goal-submit" onclick="submitGoal()">Submit Goal</button>
                </div>
            </div>
        </div>

        <div class="footer">
            <div>
                <span id="current-task">No active task</span>
            </div>
            <div>
                <span>Uptime: </span><span id="uptime">00:00:00</span>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let logCount = 0;
        let startTime = Date.now();

        function connect() {
            ws = new WebSocket(`ws://${window.location.host}/ws`);

            ws.onopen = () => {
                updateConnectionStatus(true);
                appendLog('system', 'Connected to PM Agent', 'info');
            };

            ws.onclose = () => {
                updateConnectionStatus(false);
                appendLog('system', 'Disconnected. Reconnecting...', 'warning');
                setTimeout(connect, 3000);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
        }

        function updateConnectionStatus(connected) {
            const el = document.getElementById('connection-status');
            el.className = 'status-item connection-status ' + (connected ? 'connected' : 'disconnected');
            el.querySelector('span:last-child').textContent = connected ? 'Connected' : 'Disconnected';
        }

        function handleMessage(data) {
            switch(data.type) {
                case 'status':
                    updateStatus(data.data);
                    break;
                case 'log':
                    appendLog(data.data.category, data.data.message, data.data.level);
                    break;
                case 'thought':
                    addThought(data.data);
                    break;
                case 'stats':
                    updateStats(data.data);
                    break;
                case 'command_result':
                    appendLog('cmd', data.data.result, 'info');
                    break;
            }
        }

        function updateStatus(status) {
            const stateEl = document.getElementById('agent-state');
            stateEl.textContent = status.state.toUpperCase();
            stateEl.className = 'status-badge status-' + status.state;

            document.getElementById('cycle-count').textContent = status.cycle_count || 0;

            const taskEl = document.getElementById('current-task');
            taskEl.textContent = status.current_task
                ? `Working on: ${status.current_task.substring(0, 50)}...`
                : 'No active task';
        }

        function updateStats(stats) {
            document.getElementById('stat-pending').textContent = stats.pending || 0;
            document.getElementById('stat-completed').textContent = stats.completed || 0;
            document.getElementById('stat-failed').textContent = stats.failed || 0;
            document.getElementById('stat-escalated').textContent = stats.escalated || 0;
        }

        function appendLog(category, message, level) {
            const output = document.querySelector('#terminal-output .terminal-output');
            const time = new Date().toLocaleTimeString();

            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `
                <span class="log-time">${time}</span>
                <span class="log-level ${level}">[${level.toUpperCase()}]</span>
                <span class="log-message">[${category}] ${message}</span>
            `;
            output.appendChild(entry);

            // Auto-scroll
            const container = document.getElementById('terminal-output');
            container.scrollTop = container.scrollHeight;

            logCount++;
            document.getElementById('log-count').textContent = `${logCount} entries`;
        }

        function addThought(thought) {
            const container = document.getElementById('thoughts-container');

            // Clear placeholder
            if (container.querySelector('div[style]')) {
                container.innerHTML = '';
            }

            const entry = document.createElement('div');
            entry.className = 'thought-entry';
            entry.innerHTML = `
                <div class="thought-type">${thought.thought_type}</div>
                <div class="thought-content">${thought.content.substring(0, 100)}${thought.content.length > 100 ? '...' : ''}</div>
            `;

            container.insertBefore(entry, container.firstChild);

            // Keep only 10 thoughts
            while (container.children.length > 10) {
                container.removeChild(container.lastChild);
            }
        }

        function submitGoal() {
            const input = document.getElementById('goal-input');
            const goal = input.value.trim();
            if (!goal) return;

            fetch('/api/goals', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    description: goal,
                    project_id: 'default',
                    priority: 'medium'
                })
            })
            .then(r => r.json())
            .then(data => {
                appendLog('goal', `Goal submitted: ${goal.substring(0, 40)}...`, 'action');
                input.value = '';
            })
            .catch(err => {
                appendLog('error', `Failed to submit goal: ${err}`, 'error');
            });
        }

        function sendCommand(cmd) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'command', command: cmd}));
                appendLog('cmd', `> ${cmd}`, 'info');
            }
        }

        // Terminal input handler
        document.getElementById('terminal-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const cmd = e.target.value.trim();
                if (cmd) {
                    sendCommand(cmd);
                    e.target.value = '';
                }
            }
        });

        // Goal input handler
        document.getElementById('goal-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                submitGoal();
            }
        });

        // Uptime counter
        setInterval(() => {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const hours = Math.floor(elapsed / 3600).toString().padStart(2, '0');
            const mins = Math.floor((elapsed % 3600) / 60).toString().padStart(2, '0');
            const secs = (elapsed % 60).toString().padStart(2, '0');
            document.getElementById('uptime').textContent = `${hours}:${mins}:${secs}`;
        }, 1000);

        // Start connection
        connect();
    </script>
</body>
</html>
"""


# ============================================================================
# Web Dashboard Application
# ============================================================================

class WebDashboard:
    """
    FastAPI-based web dashboard for PM Agent.

    Provides:
    - Real-time status updates via WebSocket
    - REST API for task/goal management
    - Interactive terminal console
    """

    def __init__(
        self,
        task_queue: TaskQueue,
        logger: PMLogger,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        self.queue = task_queue
        self.logger = logger
        self.host = host
        self.port = port

        self.app = FastAPI(title="PM Agent Dashboard")
        self.manager = ConnectionManager()

        self._agent_state = "idle"
        self._current_task = None
        self._cycle_count = 0

        # Subscribe to logger
        self.logger.subscribe(self._on_log)
        self.logger.subscribe_thoughts(self._on_thought)

        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return DASHBOARD_HTML

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.manager.connect(websocket)
            try:
                # Send initial state
                await self.manager.send_personal(websocket, {
                    "type": "status",
                    "data": {
                        "state": self._agent_state,
                        "current_task": self._current_task,
                        "cycle_count": self._cycle_count,
                    }
                })
                await self.manager.send_personal(websocket, {
                    "type": "stats",
                    "data": self.queue.get_stats()
                })

                while True:
                    data = await websocket.receive_json()
                    await self._handle_ws_message(websocket, data)

            except WebSocketDisconnect:
                self.manager.disconnect(websocket)

        @self.app.get("/api/status")
        async def get_status():
            return {
                "state": self._agent_state,
                "current_task": self._current_task,
                "cycle_count": self._cycle_count,
                "stats": self.queue.get_stats(),
            }

        @self.app.get("/api/logs")
        async def get_logs(limit: int = 50):
            logs = self.logger.get_recent_logs(limit)
            return [l.to_dict() for l in logs]

        @self.app.get("/api/thoughts")
        async def get_thoughts(limit: int = 20):
            thoughts = self.logger.get_recent_thoughts(limit)
            return [t.to_dict() for t in thoughts]

        @self.app.post("/api/goals")
        async def create_goal(goal: GoalCreate):
            new_goal = Goal(
                id="",
                description=goal.description,
                project_id=goal.project_id,
                priority=TaskPriority[goal.priority.upper()],
            )
            goal_id = self.queue.add_goal(new_goal)
            return {"goal_id": goal_id, "status": "created"}

        @self.app.get("/api/escalations")
        async def get_escalations():
            return [
                {
                    "id": e.id,
                    "task_id": e.task_id,
                    "reason": e.reason,
                    "status": e.status,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in self.queue.get_pending_escalations()
            ]

        @self.app.post("/api/escalations/{escalation_id}/resolve")
        async def resolve_escalation(escalation_id: str, response: EscalationResponse):
            try:
                self.queue.resolve_escalation(
                    escalation_id,
                    response.response,
                    TaskStatus[response.new_status.upper()]
                )
                return {"status": "resolved"}
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

    async def _handle_ws_message(self, websocket: WebSocket, data: dict):
        """Handle incoming WebSocket message."""
        msg_type = data.get("type")

        if msg_type == "command":
            result = await self._execute_command(data.get("command", ""))
            await self.manager.send_personal(websocket, {
                "type": "command_result",
                "data": {"result": result}
            })

    async def _execute_command(self, command: str) -> str:
        """Execute a terminal command."""
        parts = command.strip().split()
        if not parts:
            return ""

        cmd = parts[0].lower()

        if cmd == "help":
            return """Available commands:
  status    - Show current agent status
  stats     - Show task statistics
  goals     - List active goals
  tasks     - List pending tasks
  escalations - List pending escalations
  pause     - Pause the agent
  resume    - Resume the agent
  help      - Show this help"""

        elif cmd == "status":
            return f"State: {self._agent_state}, Cycle: {self._cycle_count}, Task: {self._current_task or 'None'}"

        elif cmd == "stats":
            stats = self.queue.get_stats()
            return f"Pending: {stats['pending']}, Completed: {stats['completed']}, Failed: {stats['failed']}, Escalated: {stats['escalated']}"

        elif cmd == "goals":
            goals = self.queue.get_active_goals()
            if not goals:
                return "No active goals"
            return "\n".join([f"- {g.description[:50]}..." for g in goals[:5]])

        elif cmd == "escalations":
            escalations = self.queue.get_pending_escalations()
            if not escalations:
                return "No pending escalations"
            return "\n".join([f"- [{e.id[:8]}] {e.reason[:40]}..." for e in escalations])

        else:
            return f"Unknown command: {cmd}. Type 'help' for available commands."

    def _on_log(self, entry: PMLogEntry):
        """Handle new log entry."""
        asyncio.create_task(self.manager.broadcast({
            "type": "log",
            "data": entry.to_dict()
        }))

    def _on_thought(self, thought: ThoughtEntry):
        """Handle new thought."""
        asyncio.create_task(self.manager.broadcast({
            "type": "thought",
            "data": thought.to_dict()
        }))

    def update_state(
        self,
        state: str = None,
        current_task: str = None,
        cycle_count: int = None,
    ):
        """Update agent state and broadcast to clients."""
        if state is not None:
            self._agent_state = state
        if current_task is not None:
            self._current_task = current_task
        if cycle_count is not None:
            self._cycle_count = cycle_count

        asyncio.create_task(self.manager.broadcast({
            "type": "status",
            "data": {
                "state": self._agent_state,
                "current_task": self._current_task,
                "cycle_count": self._cycle_count,
            }
        }))

        # Also broadcast stats
        asyncio.create_task(self.manager.broadcast({
            "type": "stats",
            "data": self.queue.get_stats()
        }))

    def run(self):
        """Run the web server."""
        uvicorn.run(self.app, host=self.host, port=self.port)

    async def run_async(self):
        """Run the web server asynchronously."""
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()


def run_standalone_web_dashboard(db_path: Path, log_dir: Path, port: int = 8080):
    """Run web dashboard as standalone."""
    queue = TaskQueue(db_path)
    logger = PMLogger(log_dir, console_output=False)

    dashboard = WebDashboard(queue, logger, port=port)
    print(f"Starting web dashboard at http://localhost:{port}")
    dashboard.run()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PM Agent Web Dashboard")
    parser.add_argument("--db", required=True, help="Path to tasks database")
    parser.add_argument("--logs", required=True, help="Path to logs directory")
    parser.add_argument("--port", type=int, default=8080, help="Port to run on")
    args = parser.parse_args()

    run_standalone_web_dashboard(Path(args.db), Path(args.logs), args.port)
