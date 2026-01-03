"""
Adaptive Learning System - Learn from task outcomes to improve future planning.

This module provides the AdaptiveLearner class which:
1. Tracks task success/failure patterns
2. Correlates outcomes with task characteristics
3. Adjusts complexity estimates based on historical data
4. Improves breakdown strategies over time
5. Identifies high-risk task patterns
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
import logging
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class OutcomeType(Enum):
    """Task outcome types for learning."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ESCALATED = "escalated"
    TIMEOUT = "timeout"


@dataclass
class TaskOutcome:
    """Record of a task's outcome for learning."""
    task_id: str
    goal_id: str
    project_id: str
    description: str
    outcome: OutcomeType

    # Characteristics
    estimated_complexity: str  # low, medium, high, complex
    actual_duration_minutes: float
    estimated_duration_minutes: float
    attempt_count: int

    # Task features
    keywords: List[str]
    file_types: List[str]
    task_type: str  # implementation, refactoring, bug_fix, etc.

    # Results
    files_modified: List[str]
    files_created: List[str]
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Metadata
    model_used: str = "sonnet"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PatternInsight:
    """An insight derived from outcome patterns."""
    pattern_type: str  # "high_failure_rate", "underestimated_complexity", etc.
    description: str
    confidence: float
    supporting_evidence: List[str]
    recommendation: str
    task_filters: Dict[str, Any]  # Filters to identify matching tasks


@dataclass
class StrategyAdjustment:
    """Adjustment to planning strategy based on learning."""
    adjustment_type: str
    description: str
    parameters: Dict[str, Any]
    applied_to: str  # "complexity_estimation", "breakdown", "model_selection", etc.
    confidence: float


class TaskOutcomeTracker:
    """Tracks and stores task outcomes for learning."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the outcomes database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS task_outcomes (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    goal_id TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    estimated_complexity TEXT,
                    actual_duration_minutes REAL,
                    estimated_duration_minutes REAL,
                    attempt_count INTEGER,
                    keywords TEXT,
                    file_types TEXT,
                    task_type TEXT,
                    files_modified TEXT,
                    files_created TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    model_used TEXT,
                    timestamp TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS pattern_insights (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    description TEXT,
                    confidence REAL,
                    supporting_evidence TEXT,
                    recommendation TEXT,
                    task_filters TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS strategy_adjustments (
                    id TEXT PRIMARY KEY,
                    adjustment_type TEXT NOT NULL,
                    description TEXT,
                    parameters TEXT,
                    applied_to TEXT,
                    confidence REAL,
                    active INTEGER DEFAULT 1,
                    created_at TEXT
                )
            ''')

            # Indexes for efficient queries
            conn.execute('CREATE INDEX IF NOT EXISTS idx_outcomes_project ON task_outcomes(project_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_outcomes_type ON task_outcomes(task_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_outcomes_outcome ON task_outcomes(outcome)')

            conn.commit()

    def record_outcome(self, outcome: TaskOutcome) -> str:
        """Record a task outcome."""
        import uuid
        outcome_id = str(uuid.uuid4())[:8]

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO task_outcomes (
                    id, task_id, goal_id, project_id, description, outcome,
                    estimated_complexity, actual_duration_minutes, estimated_duration_minutes,
                    attempt_count, keywords, file_types, task_type,
                    files_modified, files_created, error_type, error_message,
                    model_used, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                outcome_id, outcome.task_id, outcome.goal_id, outcome.project_id,
                outcome.description, outcome.outcome.value, outcome.estimated_complexity,
                outcome.actual_duration_minutes, outcome.estimated_duration_minutes,
                outcome.attempt_count, json.dumps(outcome.keywords),
                json.dumps(outcome.file_types), outcome.task_type,
                json.dumps(outcome.files_modified), json.dumps(outcome.files_created),
                outcome.error_type, outcome.error_message, outcome.model_used,
                outcome.timestamp.isoformat()
            ))
            conn.commit()

        return outcome_id

    def get_outcomes(
        self,
        project_id: Optional[str] = None,
        task_type: Optional[str] = None,
        outcome: Optional[OutcomeType] = None,
        since: Optional[datetime] = None,
        limit: int = 100
    ) -> List[TaskOutcome]:
        """Query task outcomes with filters."""
        query = 'SELECT * FROM task_outcomes WHERE 1=1'
        params = []

        if project_id:
            query += ' AND project_id = ?'
            params.append(project_id)
        if task_type:
            query += ' AND task_type = ?'
            params.append(task_type)
        if outcome:
            query += ' AND outcome = ?'
            params.append(outcome.value)
        if since:
            query += ' AND timestamp >= ?'
            params.append(since.isoformat())

        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_outcome(row) for row in rows]

    def _row_to_outcome(self, row) -> TaskOutcome:
        """Convert database row to TaskOutcome."""
        return TaskOutcome(
            task_id=row['task_id'],
            goal_id=row['goal_id'],
            project_id=row['project_id'],
            description=row['description'],
            outcome=OutcomeType(row['outcome']),
            estimated_complexity=row['estimated_complexity'],
            actual_duration_minutes=row['actual_duration_minutes'],
            estimated_duration_minutes=row['estimated_duration_minutes'],
            attempt_count=row['attempt_count'],
            keywords=json.loads(row['keywords']) if row['keywords'] else [],
            file_types=json.loads(row['file_types']) if row['file_types'] else [],
            task_type=row['task_type'],
            files_modified=json.loads(row['files_modified']) if row['files_modified'] else [],
            files_created=json.loads(row['files_created']) if row['files_created'] else [],
            error_type=row['error_type'],
            error_message=row['error_message'],
            model_used=row['model_used'],
            timestamp=datetime.fromisoformat(row['timestamp']) if row['timestamp'] else datetime.now()
        )


class PatternMiner:
    """Mines patterns from task outcomes to generate insights."""

    def __init__(self, tracker: TaskOutcomeTracker):
        self.tracker = tracker

    def analyze_failure_patterns(self, project_id: Optional[str] = None) -> List[PatternInsight]:
        """Identify patterns in task failures."""
        insights = []

        # Get recent failures
        failures = self.tracker.get_outcomes(
            project_id=project_id,
            outcome=OutcomeType.FAILURE,
            limit=500
        )

        if len(failures) < 5:
            return insights  # Not enough data

        # Analyze by task type
        type_failures = defaultdict(list)
        for f in failures:
            type_failures[f.task_type].append(f)

        # Find task types with high failure rates
        all_outcomes = self.tracker.get_outcomes(project_id=project_id, limit=1000)
        type_counts = defaultdict(lambda: {'success': 0, 'failure': 0})

        for o in all_outcomes:
            if o.outcome == OutcomeType.SUCCESS:
                type_counts[o.task_type]['success'] += 1
            elif o.outcome == OutcomeType.FAILURE:
                type_counts[o.task_type]['failure'] += 1

        for task_type, counts in type_counts.items():
            total = counts['success'] + counts['failure']
            if total >= 5:  # Need meaningful sample
                failure_rate = counts['failure'] / total
                if failure_rate > 0.3:  # High failure rate
                    insights.append(PatternInsight(
                        pattern_type="high_failure_rate",
                        description=f"Task type '{task_type}' has {failure_rate:.0%} failure rate",
                        confidence=min(0.9, total / 20),  # More data = higher confidence
                        supporting_evidence=[
                            f"Observed {counts['failure']} failures out of {total} tasks"
                        ],
                        recommendation=f"Consider breaking down '{task_type}' tasks into smaller steps or using more context",
                        task_filters={"task_type": task_type}
                    ))

        # Analyze error patterns
        error_patterns = defaultdict(list)
        for f in failures:
            if f.error_type:
                error_patterns[f.error_type].append(f)

        for error_type, fails in error_patterns.items():
            if len(fails) >= 3:
                # Extract common keywords
                common_keywords = self._find_common_keywords([f.keywords for f in fails])

                insights.append(PatternInsight(
                    pattern_type="recurring_error",
                    description=f"Error type '{error_type}' occurs frequently ({len(fails)} times)",
                    confidence=min(0.85, len(fails) / 10),
                    supporting_evidence=[f.description[:100] for f in fails[:3]],
                    recommendation=f"Add error handling or validation for '{error_type}' scenarios",
                    task_filters={"error_type": error_type, "keywords": common_keywords}
                ))

        return insights

    def analyze_complexity_accuracy(self, project_id: Optional[str] = None) -> List[PatternInsight]:
        """Analyze how accurate complexity estimates are."""
        insights = []

        outcomes = self.tracker.get_outcomes(project_id=project_id, limit=500)

        if len(outcomes) < 10:
            return insights

        # Compare estimated vs actual duration by complexity
        complexity_accuracy = defaultdict(lambda: {'over': 0, 'under': 0, 'accurate': 0, 'ratio_sum': 0, 'count': 0})

        for o in outcomes:
            if o.estimated_duration_minutes and o.actual_duration_minutes:
                ratio = o.actual_duration_minutes / o.estimated_duration_minutes
                complexity_accuracy[o.estimated_complexity]['ratio_sum'] += ratio
                complexity_accuracy[o.estimated_complexity]['count'] += 1

                if ratio > 1.5:
                    complexity_accuracy[o.estimated_complexity]['under'] += 1
                elif ratio < 0.67:
                    complexity_accuracy[o.estimated_complexity]['over'] += 1
                else:
                    complexity_accuracy[o.estimated_complexity]['accurate'] += 1

        for complexity, stats in complexity_accuracy.items():
            if stats['count'] >= 5:
                avg_ratio = stats['ratio_sum'] / stats['count']

                if avg_ratio > 1.3:
                    insights.append(PatternInsight(
                        pattern_type="underestimated_complexity",
                        description=f"'{complexity}' complexity tasks take {avg_ratio:.1f}x longer than estimated",
                        confidence=min(0.9, stats['count'] / 15),
                        supporting_evidence=[
                            f"Based on {stats['count']} tasks",
                            f"{stats['under']} were underestimated, {stats['accurate']} accurate"
                        ],
                        recommendation=f"Increase time estimates for '{complexity}' complexity by {(avg_ratio - 1) * 100:.0f}%",
                        task_filters={"estimated_complexity": complexity}
                    ))
                elif avg_ratio < 0.7:
                    insights.append(PatternInsight(
                        pattern_type="overestimated_complexity",
                        description=f"'{complexity}' complexity tasks complete {1/avg_ratio:.1f}x faster than estimated",
                        confidence=min(0.9, stats['count'] / 15),
                        supporting_evidence=[
                            f"Based on {stats['count']} tasks",
                            f"{stats['over']} were overestimated"
                        ],
                        recommendation=f"Reduce time estimates for '{complexity}' complexity by {(1 - avg_ratio) * 100:.0f}%",
                        task_filters={"estimated_complexity": complexity}
                    ))

        return insights

    def analyze_model_performance(self, project_id: Optional[str] = None) -> List[PatternInsight]:
        """Analyze which model performs best for which tasks."""
        insights = []

        outcomes = self.tracker.get_outcomes(project_id=project_id, limit=500)

        # Group by model and task type
        model_type_stats = defaultdict(lambda: defaultdict(lambda: {'success': 0, 'failure': 0}))

        for o in outcomes:
            if o.outcome == OutcomeType.SUCCESS:
                model_type_stats[o.model_used][o.task_type]['success'] += 1
            elif o.outcome == OutcomeType.FAILURE:
                model_type_stats[o.model_used][o.task_type]['failure'] += 1

        # Find significant differences
        for task_type in set(tt for m in model_type_stats.values() for tt in m.keys()):
            model_performance = {}

            for model, type_stats in model_type_stats.items():
                if task_type in type_stats:
                    stats = type_stats[task_type]
                    total = stats['success'] + stats['failure']
                    if total >= 3:
                        model_performance[model] = {
                            'success_rate': stats['success'] / total,
                            'total': total
                        }

            if len(model_performance) >= 2:
                best_model = max(model_performance.items(), key=lambda x: x[1]['success_rate'])
                worst_model = min(model_performance.items(), key=lambda x: x[1]['success_rate'])

                if best_model[1]['success_rate'] - worst_model[1]['success_rate'] > 0.2:
                    insights.append(PatternInsight(
                        pattern_type="model_preference",
                        description=f"'{best_model[0]}' performs better than '{worst_model[0]}' for '{task_type}' tasks",
                        confidence=min(0.85, (best_model[1]['total'] + worst_model[1]['total']) / 20),
                        supporting_evidence=[
                            f"{best_model[0]}: {best_model[1]['success_rate']:.0%} success ({best_model[1]['total']} tasks)",
                            f"{worst_model[0]}: {worst_model[1]['success_rate']:.0%} success ({worst_model[1]['total']} tasks)"
                        ],
                        recommendation=f"Prefer '{best_model[0]}' model for '{task_type}' tasks",
                        task_filters={"task_type": task_type, "recommended_model": best_model[0]}
                    ))

        return insights

    def _find_common_keywords(self, keyword_lists: List[List[str]]) -> List[str]:
        """Find keywords common across multiple tasks."""
        if not keyword_lists:
            return []

        keyword_counts = defaultdict(int)
        for keywords in keyword_lists:
            for kw in keywords:
                keyword_counts[kw] += 1

        threshold = len(keyword_lists) * 0.5
        return [kw for kw, count in keyword_counts.items() if count >= threshold]


class StrategyAdapter:
    """Adapts planning strategies based on learned patterns."""

    def __init__(self, tracker: TaskOutcomeTracker, miner: PatternMiner):
        self.tracker = tracker
        self.miner = miner
        self._adjustments: List[StrategyAdjustment] = []

    def refresh_adjustments(self, project_id: Optional[str] = None):
        """Generate new strategy adjustments from patterns."""
        self._adjustments = []

        # Get all pattern insights
        failure_insights = self.miner.analyze_failure_patterns(project_id)
        complexity_insights = self.miner.analyze_complexity_accuracy(project_id)
        model_insights = self.miner.analyze_model_performance(project_id)

        # Convert insights to adjustments
        for insight in failure_insights:
            if insight.pattern_type == "high_failure_rate":
                self._adjustments.append(StrategyAdjustment(
                    adjustment_type="increase_breakdown",
                    description=f"Break down {insight.task_filters.get('task_type', 'unknown')} tasks more granularly",
                    parameters={
                        "task_type": insight.task_filters.get('task_type'),
                        "breakdown_multiplier": 1.5
                    },
                    applied_to="breakdown",
                    confidence=insight.confidence
                ))
            elif insight.pattern_type == "recurring_error":
                self._adjustments.append(StrategyAdjustment(
                    adjustment_type="add_validation",
                    description=f"Add validation for {insight.task_filters.get('error_type', 'unknown')} errors",
                    parameters={
                        "error_type": insight.task_filters.get('error_type'),
                        "add_constraint": f"Validate against {insight.task_filters.get('error_type')} errors"
                    },
                    applied_to="task_constraints",
                    confidence=insight.confidence
                ))

        for insight in complexity_insights:
            if insight.pattern_type == "underestimated_complexity":
                # Extract multiplier from recommendation
                multiplier = 1.3  # default
                match = re.search(r'(\d+)%', insight.recommendation)
                if match:
                    multiplier = 1 + int(match.group(1)) / 100

                self._adjustments.append(StrategyAdjustment(
                    adjustment_type="adjust_time_estimate",
                    description=insight.description,
                    parameters={
                        "complexity": insight.task_filters.get('estimated_complexity'),
                        "multiplier": multiplier
                    },
                    applied_to="complexity_estimation",
                    confidence=insight.confidence
                ))

        for insight in model_insights:
            if insight.pattern_type == "model_preference":
                self._adjustments.append(StrategyAdjustment(
                    adjustment_type="model_selection",
                    description=insight.description,
                    parameters={
                        "task_type": insight.task_filters.get('task_type'),
                        "preferred_model": insight.task_filters.get('recommended_model')
                    },
                    applied_to="model_selection",
                    confidence=insight.confidence
                ))

        logger.info(f"Generated {len(self._adjustments)} strategy adjustments")

    def get_time_multiplier(self, complexity: str, task_type: Optional[str] = None) -> float:
        """Get adjusted time multiplier for a complexity level."""
        multiplier = 1.0

        for adj in self._adjustments:
            if adj.applied_to == "complexity_estimation":
                if adj.parameters.get('complexity') == complexity:
                    multiplier = max(multiplier, adj.parameters.get('multiplier', 1.0))

        return multiplier

    def get_breakdown_multiplier(self, task_type: str) -> float:
        """Get breakdown granularity multiplier for a task type."""
        multiplier = 1.0

        for adj in self._adjustments:
            if adj.applied_to == "breakdown":
                if adj.parameters.get('task_type') == task_type:
                    multiplier = max(multiplier, adj.parameters.get('breakdown_multiplier', 1.0))

        return multiplier

    def get_preferred_model(self, task_type: str) -> Optional[str]:
        """Get preferred model for a task type if one is learned."""
        for adj in self._adjustments:
            if adj.applied_to == "model_selection" and adj.confidence > 0.7:
                if adj.parameters.get('task_type') == task_type:
                    return adj.parameters.get('preferred_model')
        return None

    def get_additional_constraints(self, task_type: str, keywords: List[str]) -> List[str]:
        """Get additional constraints based on learned patterns."""
        constraints = []

        for adj in self._adjustments:
            if adj.applied_to == "task_constraints":
                constraint = adj.parameters.get('add_constraint')
                if constraint:
                    constraints.append(constraint)

        return constraints


class AdaptiveLearner:
    """
    Main interface for the adaptive learning system.

    Combines outcome tracking, pattern mining, and strategy adaptation
    to enable continuous self-improvement of the PM Agent.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        db_path = data_dir / "adaptive_learning.db"
        self.tracker = TaskOutcomeTracker(db_path)
        self.miner = PatternMiner(self.tracker)
        self.adapter = StrategyAdapter(self.tracker, self.miner)

        # Refresh adjustments on initialization
        self.adapter.refresh_adjustments()

    def record_task_completion(
        self,
        task_id: str,
        goal_id: str,
        project_id: str,
        description: str,
        success: bool,
        estimated_complexity: str,
        actual_duration: float,
        estimated_duration: float,
        attempt_count: int,
        files_modified: List[str],
        files_created: List[str],
        model_used: str = "sonnet",
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        """Record a completed task for learning."""
        # Extract features from description
        keywords = self._extract_keywords(description)
        file_types = self._extract_file_types(files_modified + files_created)
        task_type = self._classify_task_type(description, keywords)

        outcome = TaskOutcome(
            task_id=task_id,
            goal_id=goal_id,
            project_id=project_id,
            description=description,
            outcome=OutcomeType.SUCCESS if success else OutcomeType.FAILURE,
            estimated_complexity=estimated_complexity,
            actual_duration_minutes=actual_duration,
            estimated_duration_minutes=estimated_duration,
            attempt_count=attempt_count,
            keywords=keywords,
            file_types=file_types,
            task_type=task_type,
            files_modified=files_modified,
            files_created=files_created,
            error_type=error_type,
            error_message=error_message,
            model_used=model_used,
        )

        self.tracker.record_outcome(outcome)

        # Periodically refresh strategy adjustments
        # In practice, do this less frequently (e.g., every 10 tasks)
        self._maybe_refresh_strategies()

    def get_adjusted_time_estimate(self, complexity: str, base_minutes: int, task_type: Optional[str] = None) -> int:
        """Get time estimate adjusted by learned patterns."""
        multiplier = self.adapter.get_time_multiplier(complexity, task_type)
        return int(base_minutes * multiplier)

    def get_breakdown_recommendation(self, task_type: str) -> Dict[str, Any]:
        """Get recommendations for breaking down a task type."""
        multiplier = self.adapter.get_breakdown_multiplier(task_type)
        return {
            "granularity_multiplier": multiplier,
            "should_subdivide": multiplier > 1.2,
            "recommended_subtask_count": int(3 * multiplier)
        }

    def get_model_recommendation(self, task_type: str, default: str = "sonnet") -> str:
        """Get recommended model for a task type."""
        preferred = self.adapter.get_preferred_model(task_type)
        return preferred or default

    def get_risk_assessment(self, description: str, task_type: str) -> Dict[str, Any]:
        """Assess risk level for a task based on learned patterns."""
        keywords = self._extract_keywords(description)

        # Get failure patterns
        failure_insights = self.miner.analyze_failure_patterns()

        risk_level = "low"
        risk_factors = []

        for insight in failure_insights:
            if insight.task_filters.get('task_type') == task_type:
                risk_level = "high"
                risk_factors.append(insight.description)
            elif insight.task_filters.get('keywords'):
                common = set(keywords) & set(insight.task_filters['keywords'])
                if len(common) >= 2:
                    risk_level = "medium" if risk_level == "low" else risk_level
                    risk_factors.append(f"Keywords overlap with failure pattern: {common}")

        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendations": self.adapter.get_additional_constraints(task_type, keywords)
        }

    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about what has been learned."""
        outcomes = self.tracker.get_outcomes(limit=1000)

        if not outcomes:
            return {"total_tasks": 0, "message": "No data yet"}

        success_count = len([o for o in outcomes if o.outcome == OutcomeType.SUCCESS])
        failure_count = len([o for o in outcomes if o.outcome == OutcomeType.FAILURE])

        return {
            "total_tasks": len(outcomes),
            "success_rate": success_count / len(outcomes) if outcomes else 0,
            "failure_count": failure_count,
            "patterns_identified": len(self.miner.analyze_failure_patterns()) +
                                   len(self.miner.analyze_complexity_accuracy()) +
                                   len(self.miner.analyze_model_performance()),
            "active_adjustments": len(self.adapter._adjustments),
            "task_types_seen": len(set(o.task_type for o in outcomes)),
        }

    def _extract_keywords(self, description: str) -> List[str]:
        """Extract relevant keywords from description."""
        keywords = []
        desc_lower = description.lower()

        # Technical keywords
        tech_keywords = [
            'api', 'database', 'auth', 'test', 'refactor', 'bug', 'fix',
            'implement', 'add', 'create', 'update', 'delete', 'remove',
            'optimize', 'performance', 'security', 'validation', 'error',
            'ui', 'frontend', 'backend', 'server', 'client', 'async',
            'cache', 'config', 'deploy', 'migrate', 'schema'
        ]

        for kw in tech_keywords:
            if kw in desc_lower:
                keywords.append(kw)

        return keywords

    def _extract_file_types(self, files: List[str]) -> List[str]:
        """Extract file extensions from file list."""
        extensions = set()
        for f in files:
            if '.' in f:
                ext = f.rsplit('.', 1)[-1].lower()
                extensions.add(ext)
        return list(extensions)

    def _classify_task_type(self, description: str, keywords: List[str]) -> str:
        """Classify task into a type category."""
        desc_lower = description.lower()

        if any(kw in desc_lower for kw in ['fix', 'bug', 'error', 'issue']):
            return 'bug_fix'
        elif any(kw in desc_lower for kw in ['test', 'spec', 'unittest']):
            return 'testing'
        elif any(kw in desc_lower for kw in ['refactor', 'clean', 'reorganize']):
            return 'refactoring'
        elif any(kw in desc_lower for kw in ['implement', 'add', 'create', 'feature']):
            return 'implementation'
        elif any(kw in desc_lower for kw in ['optimize', 'performance', 'speed']):
            return 'optimization'
        elif any(kw in desc_lower for kw in ['doc', 'comment', 'readme']):
            return 'documentation'
        elif any(kw in desc_lower for kw in ['config', 'setup', 'install']):
            return 'configuration'
        else:
            return 'general'

    def _maybe_refresh_strategies(self):
        """Periodically refresh strategy adjustments."""
        # Simple heuristic: refresh every time for now
        # In production, track last refresh and only do periodically
        self.adapter.refresh_adjustments()


# ============================================================================
# Factory function
# ============================================================================

def create_adaptive_learner(data_dir: Path) -> AdaptiveLearner:
    """Create and initialize an AdaptiveLearner instance."""
    return AdaptiveLearner(data_dir)
