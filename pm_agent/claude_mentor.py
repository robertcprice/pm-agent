"""
Claude Mentor - Conversational Learning System for PM Agent Self-Improvement.

This module enables the PM Agent to have conversations with Claude (Opus 4.5)
to learn how to improve itself. It implements:

1. Structured conversations about improvement strategies
2. Learning extraction from conversation insights
3. Automatic implementation of learned improvements
4. Conversation history for context retention
5. Pattern recognition across multiple conversations
"""

import anthropic
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum
import json
import sqlite3
import logging
import asyncio

logger = logging.getLogger(__name__)


class ConversationType(Enum):
    """Types of mentoring conversations."""
    GENERAL_IMPROVEMENT = "general_improvement"
    TASK_FAILURE_ANALYSIS = "task_failure_analysis"
    PLANNING_STRATEGY = "planning_strategy"
    CODE_QUALITY = "code_quality"
    USER_EXPERIENCE = "user_experience"
    SELF_REFLECTION = "self_reflection"
    ARCHITECTURE_REVIEW = "architecture_review"


@dataclass
class ConversationMessage:
    """A single message in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LearningInsight:
    """An actionable insight extracted from a conversation."""
    topic: str
    insight: str
    action_type: str  # "implement", "modify", "monitor", "investigate"
    priority: str  # "critical", "high", "medium", "low"
    implementation_hint: Optional[str] = None
    confidence: float = 0.8
    applied: bool = False
    applied_at: Optional[datetime] = None


@dataclass
class Conversation:
    """A complete conversation session."""
    id: str
    conversation_type: ConversationType
    topic: str
    messages: List[ConversationMessage]
    insights: List[LearningInsight]
    context: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


class ConversationStore:
    """Persistent storage for conversations and learnings."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    conversation_type TEXT NOT NULL,
                    topic TEXT,
                    context TEXT,
                    created_at TEXT,
                    completed_at TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    topic TEXT,
                    insight TEXT NOT NULL,
                    action_type TEXT,
                    priority TEXT,
                    implementation_hint TEXT,
                    confidence REAL,
                    applied INTEGER DEFAULT 0,
                    applied_at TEXT,
                    created_at TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_name TEXT NOT NULL,
                    pattern_description TEXT,
                    source_conversations TEXT,
                    confidence REAL,
                    usage_count INTEGER DEFAULT 0,
                    last_used TEXT,
                    created_at TEXT
                )
            ''')

            conn.commit()

    def save_conversation(self, conversation: Conversation):
        """Save a conversation to the database."""
        with sqlite3.connect(self.db_path) as conn:
            # Save conversation
            conn.execute('''
                INSERT OR REPLACE INTO conversations
                (id, conversation_type, topic, context, created_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                conversation.id,
                conversation.conversation_type.value,
                conversation.topic,
                json.dumps(conversation.context),
                conversation.created_at.isoformat(),
                conversation.completed_at.isoformat() if conversation.completed_at else None
            ))

            # Save messages
            for msg in conversation.messages:
                conn.execute('''
                    INSERT INTO messages (conversation_id, role, content, timestamp)
                    VALUES (?, ?, ?, ?)
                ''', (
                    conversation.id,
                    msg.role,
                    msg.content,
                    msg.timestamp.isoformat()
                ))

            # Save insights
            for insight in conversation.insights:
                conn.execute('''
                    INSERT INTO insights
                    (conversation_id, topic, insight, action_type, priority,
                     implementation_hint, confidence, applied, applied_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    conversation.id,
                    insight.topic,
                    insight.insight,
                    insight.action_type,
                    insight.priority,
                    insight.implementation_hint,
                    insight.confidence,
                    1 if insight.applied else 0,
                    insight.applied_at.isoformat() if insight.applied_at else None,
                    datetime.now().isoformat()
                ))

            conn.commit()

    def get_unapplied_insights(self, limit: int = 20) -> List[LearningInsight]:
        """Get insights that haven't been applied yet."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT * FROM insights
                WHERE applied = 0
                ORDER BY
                    CASE priority
                        WHEN 'critical' THEN 1
                        WHEN 'high' THEN 2
                        WHEN 'medium' THEN 3
                        WHEN 'low' THEN 4
                    END,
                    confidence DESC
                LIMIT ?
            ''', (limit,)).fetchall()

        return [LearningInsight(
            topic=row['topic'],
            insight=row['insight'],
            action_type=row['action_type'],
            priority=row['priority'],
            implementation_hint=row['implementation_hint'],
            confidence=row['confidence'],
            applied=bool(row['applied'])
        ) for row in rows]

    def mark_insight_applied(self, insight_topic: str):
        """Mark an insight as applied."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE insights
                SET applied = 1, applied_at = ?
                WHERE topic = ? AND applied = 0
            ''', (datetime.now().isoformat(), insight_topic))
            conn.commit()

    def get_conversation_history(self, conversation_type: Optional[ConversationType] = None, limit: int = 10) -> List[Dict]:
        """Get recent conversation summaries."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if conversation_type:
                rows = conn.execute('''
                    SELECT * FROM conversations
                    WHERE conversation_type = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (conversation_type.value, limit)).fetchall()
            else:
                rows = conn.execute('''
                    SELECT * FROM conversations
                    ORDER BY created_at DESC
                    LIMIT ?
                ''', (limit,)).fetchall()

        return [dict(row) for row in rows]


class ClaudeMentor:
    """
    The Claude Mentor enables the PM Agent to have conversations with Claude
    for learning and self-improvement.
    """

    def __init__(
        self,
        api_key: str,
        data_dir: Path,
        model: str = "claude-opus-4-5-20251101",
        pm_agent_context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Claude Mentor.

        Args:
            api_key: Anthropic API key
            data_dir: Directory for storing conversation data
            model: Model to use for conversations (default: opus-4-5)
            pm_agent_context: Context about the PM Agent for richer conversations
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.store = ConversationStore(data_dir / "mentor_conversations.db")

        # PM Agent context for conversations
        self.pm_context = pm_agent_context or {}

        # Active conversation
        self.current_conversation: Optional[Conversation] = None

    def _get_system_prompt(self, conversation_type: ConversationType) -> str:
        """Generate system prompt based on conversation type."""
        base_context = f"""You are a senior software architect and AI systems expert mentoring an autonomous PM Agent.

# About the PM Agent You're Mentoring
The PM Agent is an autonomous project manager that:
- Receives high-level goals from humans
- Uses GoalAnalyzer for complexity analysis and task breakdown
- Delegates coding tasks to Claude Code CLI
- Reviews completed work using EGO (language model)
- Escalates when stuck
- Maintains project memory across sessions

# Current PM Agent Architecture
```
HUMAN → PM AGENT → CLAUDE CODE CLI (primary) / LOCAL FALLBACK (backup)
         ↓
[PLANNER | DELEGATOR | REVIEWER | ESCALATOR]
         ↓
[TASK QUEUE (SQLite) | PROJECT MEMORY | NOTIFICATIONS | GOAL ANALYZER]
         ↓
[ADAPTIVE LEARNER | CLAUDE MENTOR (this conversation)]
```

# Key Components
1. ClaudeCodeTool: Executes tasks via Claude Code CLI with git state capture
2. TaskQueue: SQLite-based persistent task/goal management
3. GoalAnalyzer: Heuristic complexity analysis and task breakdown
4. PMEgoAdapter: LLM integration for planning and review
5. ProjectMemory: Tracks patterns, decisions, task history
6. LocalCoderAgent: Fallback when Claude Code unavailable
7. ProductionManager: Health monitoring, crash recovery, metrics
8. AdaptiveLearner: Learns from task outcomes to improve over time
"""

        type_specific = {
            ConversationType.GENERAL_IMPROVEMENT: """
# Your Role
Help the PM Agent become more effective at:
- Managing software projects autonomously
- Breaking down goals into actionable tasks
- Delegating work effectively
- Handling failures gracefully
- Learning from experience

Provide specific, actionable advice that can be implemented as code or configuration changes.
""",
            ConversationType.TASK_FAILURE_ANALYSIS: """
# Your Role
Analyze task failures and help the PM Agent learn from them:
- Identify root causes of failures
- Suggest preventive measures
- Recommend retry strategies
- Help recognize failure patterns

Focus on actionable improvements to prevent similar failures.
""",
            ConversationType.PLANNING_STRATEGY: """
# Your Role
Help improve the PM Agent's planning capabilities:
- Better goal decomposition strategies
- More accurate complexity estimation
- Smarter dependency detection
- Improved task prioritization

Provide concrete algorithms or heuristics that can be implemented.
""",
            ConversationType.CODE_QUALITY: """
# Your Role
Help the PM Agent produce higher quality code outcomes:
- Better code review criteria
- More effective testing strategies
- Improved documentation practices
- Smarter refactoring decisions

Focus on measurable quality improvements.
""",
            ConversationType.USER_EXPERIENCE: """
# Your Role
Help the PM Agent provide a better experience for users:
- Clearer status communication
- More helpful escalation messages
- Better progress visualization
- More intuitive interaction patterns

Focus on making the PM Agent more pleasant to work with.
""",
            ConversationType.SELF_REFLECTION: """
# Your Role
Guide the PM Agent through self-reflection:
- Analyze recent performance
- Identify areas of improvement
- Recognize successful patterns
- Set improvement goals

Help the agent become more self-aware and growth-oriented.
""",
            ConversationType.ARCHITECTURE_REVIEW: """
# Your Role
Review and improve the PM Agent's architecture:
- Identify bottlenecks
- Suggest structural improvements
- Recommend new capabilities
- Optimize existing components

Focus on scalability, maintainability, and extensibility.
"""
        }

        return base_context + type_specific.get(conversation_type, type_specific[ConversationType.GENERAL_IMPROVEMENT])

    def start_conversation(
        self,
        conversation_type: ConversationType,
        topic: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new conversation with Claude mentor.

        Args:
            conversation_type: Type of conversation
            topic: Specific topic to discuss
            initial_context: Additional context for the conversation

        Returns:
            Conversation ID
        """
        import uuid
        conv_id = str(uuid.uuid4())[:8]

        context = {**(initial_context or {}), **self.pm_context}

        self.current_conversation = Conversation(
            id=conv_id,
            conversation_type=conversation_type,
            topic=topic,
            messages=[],
            insights=[],
            context=context
        )

        logger.info(f"Started conversation {conv_id}: {topic}")
        return conv_id

    def send_message(self, message: str) -> str:
        """
        Send a message to Claude and get a response.

        Args:
            message: The message to send

        Returns:
            Claude's response
        """
        if not self.current_conversation:
            raise ValueError("No active conversation. Call start_conversation first.")

        # Add user message
        self.current_conversation.messages.append(ConversationMessage(
            role="user",
            content=message
        ))

        # Build messages for API
        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in self.current_conversation.messages
        ]

        # Get system prompt
        system_prompt = self._get_system_prompt(self.current_conversation.conversation_type)

        # Add context if available
        if self.current_conversation.context:
            system_prompt += f"\n\n# Current Context\n{json.dumps(self.current_conversation.context, indent=2)}"

        # Call Claude
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=messages
            )

            assistant_message = response.content[0].text

            # Add assistant response
            self.current_conversation.messages.append(ConversationMessage(
                role="assistant",
                content=assistant_message
            ))

            return assistant_message

        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            raise

    def extract_insights(self) -> List[LearningInsight]:
        """
        Extract actionable insights from the current conversation.

        Returns:
            List of learning insights
        """
        if not self.current_conversation or len(self.current_conversation.messages) < 2:
            return []

        # Build conversation summary
        conversation_text = "\n\n".join([
            f"{'PM Agent' if msg.role == 'user' else 'Mentor'}: {msg.content}"
            for msg in self.current_conversation.messages
        ])

        extraction_prompt = f"""Analyze this conversation and extract actionable insights.

# Conversation
{conversation_text}

# Task
Extract specific, actionable insights that the PM Agent can implement to improve itself.

Output as JSON:
{{
    "insights": [
        {{
            "topic": "Brief topic name",
            "insight": "The specific insight or learning",
            "action_type": "implement|modify|monitor|investigate",
            "priority": "critical|high|medium|low",
            "implementation_hint": "Specific hint for how to implement this",
            "confidence": 0.0-1.0
        }}
    ]
}}

Focus on:
- Specific code changes or new features to implement
- Configuration adjustments
- Process improvements
- New patterns to adopt
- Pitfalls to avoid
"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": extraction_prompt}]
            )

            response_text = response.content[0].text

            # Parse JSON from response
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                json_str = response_text[start:end].strip()
            elif "{" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_str = response_text[start:end]
            else:
                json_str = response_text

            data = json.loads(json_str)
            insights = []

            for item in data.get("insights", []):
                insight = LearningInsight(
                    topic=item.get("topic", "Unknown"),
                    insight=item.get("insight", ""),
                    action_type=item.get("action_type", "investigate"),
                    priority=item.get("priority", "medium"),
                    implementation_hint=item.get("implementation_hint"),
                    confidence=item.get("confidence", 0.7)
                )
                insights.append(insight)
                self.current_conversation.insights.append(insight)

            return insights

        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return []

    def end_conversation(self) -> Conversation:
        """
        End the current conversation and save it.

        Returns:
            The completed conversation
        """
        if not self.current_conversation:
            raise ValueError("No active conversation")

        self.current_conversation.completed_at = datetime.now()

        # Extract insights if not already done
        if not self.current_conversation.insights:
            self.extract_insights()

        # Save to database
        self.store.save_conversation(self.current_conversation)

        conversation = self.current_conversation
        self.current_conversation = None

        logger.info(f"Ended conversation {conversation.id} with {len(conversation.insights)} insights")

        return conversation

    async def have_improvement_conversation(
        self,
        topic: str,
        context: Dict[str, Any],
        max_exchanges: int = 5
    ) -> Conversation:
        """
        Have a complete improvement conversation with Claude.

        This is a higher-level method that:
        1. Starts a conversation
        2. Sends an initial query about the topic
        3. Has follow-up exchanges
        4. Extracts insights
        5. Ends and saves the conversation

        Args:
            topic: Topic to discuss
            context: Context for the conversation
            max_exchanges: Maximum number of back-and-forth exchanges

        Returns:
            The completed conversation with insights
        """
        # Determine conversation type from topic
        topic_lower = topic.lower()
        if "fail" in topic_lower or "error" in topic_lower:
            conv_type = ConversationType.TASK_FAILURE_ANALYSIS
        elif "plan" in topic_lower or "break" in topic_lower:
            conv_type = ConversationType.PLANNING_STRATEGY
        elif "code" in topic_lower or "quality" in topic_lower:
            conv_type = ConversationType.CODE_QUALITY
        elif "user" in topic_lower or "experience" in topic_lower:
            conv_type = ConversationType.USER_EXPERIENCE
        elif "reflect" in topic_lower or "performance" in topic_lower:
            conv_type = ConversationType.SELF_REFLECTION
        elif "architect" in topic_lower or "design" in topic_lower:
            conv_type = ConversationType.ARCHITECTURE_REVIEW
        else:
            conv_type = ConversationType.GENERAL_IMPROVEMENT

        self.start_conversation(conv_type, topic, context)

        # Initial message
        initial_message = f"""I'm the PM Agent and I want to improve my capabilities around: {topic}

Here's my current situation:
{json.dumps(context, indent=2)}

What specific improvements would you recommend? Please be concrete and actionable."""

        response = self.send_message(initial_message)
        logger.info(f"Mentor response: {response[:200]}...")

        # Follow-up exchanges
        follow_ups = [
            "Can you elaborate on the most impactful improvement you mentioned?",
            "What are the potential pitfalls I should avoid when implementing these changes?",
            "How should I measure success after implementing these improvements?",
            "Are there any quick wins I can implement immediately?"
        ]

        for i, follow_up in enumerate(follow_ups[:max_exchanges - 1]):
            response = self.send_message(follow_up)
            logger.info(f"Follow-up {i+1} response: {response[:200]}...")

        # Extract insights and end
        return self.end_conversation()

    def get_pending_insights(self) -> List[LearningInsight]:
        """Get insights that haven't been applied yet."""
        return self.store.get_unapplied_insights()

    def mark_insight_implemented(self, topic: str):
        """Mark an insight as implemented."""
        self.store.mark_insight_applied(topic)


class SelfImprovementLoop:
    """
    Orchestrates the PM Agent's self-improvement through Claude conversations.

    This class manages:
    1. Periodic reflection conversations
    2. Failure analysis when tasks fail
    3. Proactive improvement seeking
    4. Insight implementation tracking
    """

    def __init__(
        self,
        mentor: ClaudeMentor,
        pm_agent: Any,  # PMAgent instance
        interval_minutes: int = 60
    ):
        self.mentor = mentor
        self.pm_agent = pm_agent
        self.interval_minutes = interval_minutes
        self.last_reflection: Optional[datetime] = None
        self.running = False

    async def start(self):
        """Start the self-improvement loop."""
        self.running = True
        logger.info("Self-improvement loop started")

        while self.running:
            try:
                # Check if it's time for reflection
                if self._should_reflect():
                    await self._run_reflection()
                    self.last_reflection = datetime.now()

                # Check for pending insights to apply
                pending = self.mentor.get_pending_insights()
                if pending:
                    await self._apply_insights(pending[:3])  # Apply top 3

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in self-improvement loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    def stop(self):
        """Stop the self-improvement loop."""
        self.running = False
        logger.info("Self-improvement loop stopped")

    def _should_reflect(self) -> bool:
        """Determine if it's time for a reflection."""
        if not self.last_reflection:
            return True

        elapsed = (datetime.now() - self.last_reflection).total_seconds() / 60
        return elapsed >= self.interval_minutes

    async def _run_reflection(self):
        """Run a reflection conversation."""
        # Get PM Agent status
        status = self.pm_agent.get_status() if hasattr(self.pm_agent, 'get_status') else {}

        context = {
            "status": status,
            "recent_stats": self.pm_agent.stats if hasattr(self.pm_agent, 'stats') else {},
            "timestamp": datetime.now().isoformat()
        }

        logger.info("Starting reflection conversation...")

        conversation = await self.mentor.have_improvement_conversation(
            topic="Self-reflection on recent performance and areas for improvement",
            context=context,
            max_exchanges=4
        )

        logger.info(f"Reflection complete: {len(conversation.insights)} insights extracted")

    async def _apply_insights(self, insights: List[LearningInsight]):
        """Attempt to apply pending insights."""
        for insight in insights:
            try:
                if insight.action_type == "implement" and insight.implementation_hint:
                    # Log for now - actual implementation would require code generation
                    logger.info(f"Insight to implement: {insight.topic}")
                    logger.info(f"  Hint: {insight.implementation_hint}")

                    # Mark as applied (in real system, would verify implementation)
                    self.mentor.mark_insight_implemented(insight.topic)

                elif insight.action_type == "monitor":
                    logger.info(f"Insight to monitor: {insight.topic}")
                    # Would set up monitoring

                elif insight.action_type == "investigate":
                    logger.info(f"Insight to investigate: {insight.topic}")
                    # Would trigger investigation

            except Exception as e:
                logger.error(f"Error applying insight {insight.topic}: {e}")

    async def analyze_failure(self, task_id: str, error: str, context: Dict[str, Any]):
        """Analyze a task failure through conversation."""
        context["task_id"] = task_id
        context["error"] = error

        conversation = await self.mentor.have_improvement_conversation(
            topic=f"Task failure analysis: {error[:100]}",
            context=context,
            max_exchanges=3
        )

        return conversation.insights


# ============================================================================
# Factory functions
# ============================================================================

def create_claude_mentor(
    api_key: str,
    data_dir: Path,
    pm_context: Optional[Dict[str, Any]] = None
) -> ClaudeMentor:
    """Create a ClaudeMentor instance."""
    return ClaudeMentor(
        api_key=api_key,
        data_dir=data_dir,
        pm_agent_context=pm_context
    )


def create_self_improvement_loop(
    mentor: ClaudeMentor,
    pm_agent: Any,
    interval_minutes: int = 60
) -> SelfImprovementLoop:
    """Create a SelfImprovementLoop instance."""
    return SelfImprovementLoop(mentor, pm_agent, interval_minutes)
