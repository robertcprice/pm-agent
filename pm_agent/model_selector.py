"""
Smart Model Selection - Dynamically select optimal model based on task characteristics.

This module provides intelligent model selection that:
1. Estimates task complexity
2. Considers budget constraints
3. Tracks model performance by task type
4. Optimizes cost while maintaining quality
5. Learns from past model performance
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)


class Model(Enum):
    """Available models with their characteristics."""
    HAIKU = "haiku"      # Fast, cheap, good for simple tasks
    SONNET = "sonnet"    # Balanced, default choice
    OPUS = "opus"        # Most capable, expensive


@dataclass
class ModelCost:
    """Cost information for a model."""
    input_per_million: float   # $ per million input tokens
    output_per_million: float  # $ per million output tokens
    average_latency_ms: int    # Average response time


# Current pricing (approximate, update as needed)
MODEL_COSTS = {
    Model.HAIKU: ModelCost(0.25, 1.25, 500),
    Model.SONNET: ModelCost(3.0, 15.0, 2000),
    Model.OPUS: ModelCost(15.0, 75.0, 5000),
}


@dataclass
class TaskComplexityScore:
    """Complexity analysis of a task."""
    overall_score: float  # 0.0 (trivial) to 1.0 (extremely complex)
    factors: Dict[str, float]
    recommended_model: Model
    confidence: float
    reasoning: str


@dataclass
class ModelPerformance:
    """Performance record for a model on task types."""
    model: Model
    task_type: str
    success_rate: float
    average_duration_seconds: float
    sample_count: int
    last_updated: datetime


@dataclass
class ModelSelection:
    """Result of model selection."""
    selected_model: Model
    reasoning: str
    confidence: float
    alternatives: List[Tuple[Model, str]]  # (model, why not selected)
    estimated_cost: Optional[float] = None


class ComplexityEstimator:
    """Estimates task complexity from description and characteristics."""

    # Complexity indicators
    COMPLEXITY_KEYWORDS = {
        'low': [
            'simple', 'basic', 'trivial', 'minor', 'small', 'quick',
            'typo', 'comment', 'rename', 'format', 'style'
        ],
        'medium': [
            'add', 'update', 'modify', 'implement', 'create', 'extend',
            'refactor', 'improve', 'enhance', 'fix bug'
        ],
        'high': [
            'complex', 'integrate', 'architecture', 'system', 'api',
            'database', 'security', 'authentication', 'performance',
            'optimize', 'migrate'
        ],
        'very_high': [
            'redesign', 'rewrite', 'overhaul', 'distributed', 'scalability',
            'real-time', 'concurrent', 'multi-threaded', 'machine learning'
        ]
    }

    # Scope indicators
    SCOPE_MULTIPLIERS = {
        'single_file': 0.8,
        'few_files': 1.0,
        'multiple_files': 1.2,
        'project_wide': 1.5,
        'multi_project': 2.0,
    }

    def estimate(
        self,
        description: str,
        context_files: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        acceptance_criteria: Optional[List[str]] = None
    ) -> TaskComplexityScore:
        """
        Estimate task complexity.

        Args:
            description: Task description
            context_files: Files involved in the task
            constraints: Task constraints
            acceptance_criteria: Acceptance criteria

        Returns:
            TaskComplexityScore with analysis
        """
        desc_lower = description.lower()
        factors = {}

        # Keyword complexity
        keyword_scores = {
            'low': sum(1 for kw in self.COMPLEXITY_KEYWORDS['low'] if kw in desc_lower),
            'medium': sum(1 for kw in self.COMPLEXITY_KEYWORDS['medium'] if kw in desc_lower),
            'high': sum(1 for kw in self.COMPLEXITY_KEYWORDS['high'] if kw in desc_lower),
            'very_high': sum(1 for kw in self.COMPLEXITY_KEYWORDS['very_high'] if kw in desc_lower),
        }

        # Calculate weighted keyword score
        keyword_score = (
            keyword_scores['low'] * 0.1 +
            keyword_scores['medium'] * 0.3 +
            keyword_scores['high'] * 0.6 +
            keyword_scores['very_high'] * 0.9
        ) / max(1, sum(keyword_scores.values()))

        factors['keyword_complexity'] = keyword_score

        # Scope factor
        file_count = len(context_files) if context_files else 1
        if file_count == 1:
            scope_factor = self.SCOPE_MULTIPLIERS['single_file']
        elif file_count <= 3:
            scope_factor = self.SCOPE_MULTIPLIERS['few_files']
        elif file_count <= 10:
            scope_factor = self.SCOPE_MULTIPLIERS['multiple_files']
        else:
            scope_factor = self.SCOPE_MULTIPLIERS['project_wide']

        factors['scope'] = (scope_factor - 0.8) / 1.2  # Normalize to 0-1

        # Constraint factor
        constraint_count = len(constraints) if constraints else 0
        factors['constraints'] = min(1.0, constraint_count * 0.15)

        # Criteria factor
        criteria_count = len(acceptance_criteria) if acceptance_criteria else 0
        factors['acceptance_criteria'] = min(1.0, criteria_count * 0.1)

        # Description length factor (longer = more complex usually)
        word_count = len(description.split())
        factors['description_length'] = min(1.0, word_count / 100)

        # Calculate overall score
        weights = {
            'keyword_complexity': 0.35,
            'scope': 0.25,
            'constraints': 0.15,
            'acceptance_criteria': 0.15,
            'description_length': 0.10,
        }

        overall_score = sum(factors[k] * weights[k] for k in factors)

        # Map score to recommended model
        if overall_score < 0.25:
            recommended_model = Model.HAIKU
            reasoning = "Task is simple enough for the fastest model"
        elif overall_score < 0.6:
            recommended_model = Model.SONNET
            reasoning = "Task has moderate complexity, balanced model recommended"
        else:
            recommended_model = Model.OPUS
            reasoning = "Task is complex, using most capable model for best results"

        # Confidence based on how clear the indicators are
        confidence = 0.7 + (0.3 * abs(overall_score - 0.5) / 0.5)

        return TaskComplexityScore(
            overall_score=overall_score,
            factors=factors,
            recommended_model=recommended_model,
            confidence=confidence,
            reasoning=reasoning
        )


class CostOptimizer:
    """Optimizes model selection based on cost constraints."""

    def __init__(self, daily_budget: Optional[float] = None, monthly_budget: Optional[float] = None):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.daily_spend = 0.0
        self.monthly_spend = 0.0
        self.last_reset_day = datetime.now().date()
        self.last_reset_month = datetime.now().month

    def check_budget(self, model: Model, estimated_tokens: int = 2000) -> Tuple[bool, str]:
        """
        Check if using a model fits within budget.

        Args:
            model: Model to check
            estimated_tokens: Estimated tokens for the task

        Returns:
            Tuple of (within_budget, reason)
        """
        self._reset_if_needed()

        cost = MODEL_COSTS[model]
        estimated_cost = (
            (estimated_tokens / 1_000_000) * cost.input_per_million +
            (estimated_tokens / 1_000_000) * cost.output_per_million
        )

        # Check daily budget
        if self.daily_budget and (self.daily_spend + estimated_cost) > self.daily_budget:
            return False, f"Daily budget ({self.daily_budget}) would be exceeded"

        # Check monthly budget
        if self.monthly_budget and (self.monthly_spend + estimated_cost) > self.monthly_budget:
            return False, f"Monthly budget ({self.monthly_budget}) would be exceeded"

        return True, "Within budget"

    def record_usage(self, model: Model, input_tokens: int, output_tokens: int):
        """Record actual token usage."""
        cost = MODEL_COSTS[model]
        actual_cost = (
            (input_tokens / 1_000_000) * cost.input_per_million +
            (output_tokens / 1_000_000) * cost.output_per_million
        )

        self.daily_spend += actual_cost
        self.monthly_spend += actual_cost

    def get_affordable_models(self, estimated_tokens: int = 2000) -> List[Model]:
        """Get list of models that fit within budget."""
        affordable = []
        for model in Model:
            within_budget, _ = self.check_budget(model, estimated_tokens)
            if within_budget:
                affordable.append(model)
        return affordable

    def _reset_if_needed(self):
        """Reset counters if day/month changed."""
        now = datetime.now()
        if now.date() != self.last_reset_day:
            self.daily_spend = 0.0
            self.last_reset_day = now.date()
        if now.month != self.last_reset_month:
            self.monthly_spend = 0.0
            self.last_reset_month = now.month


class ModelPerformanceTracker:
    """Tracks model performance by task type for learning."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    duration_seconds REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    timestamp TEXT
                )
            ''')
            conn.commit()

    def record_task(
        self,
        model: Model,
        task_type: str,
        success: bool,
        duration_seconds: float,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None
    ):
        """Record a task execution for performance tracking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO model_performance
                (model, task_type, success, duration_seconds, input_tokens, output_tokens, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                model.value,
                task_type,
                1 if success else 0,
                duration_seconds,
                input_tokens,
                output_tokens,
                datetime.now().isoformat()
            ))
            conn.commit()

    def get_performance(self, task_type: str, min_samples: int = 5) -> Dict[Model, ModelPerformance]:
        """Get performance data for a task type."""
        results = {}

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            for model in Model:
                rows = conn.execute('''
                    SELECT
                        COUNT(*) as count,
                        AVG(success) as success_rate,
                        AVG(duration_seconds) as avg_duration,
                        MAX(timestamp) as last_updated
                    FROM model_performance
                    WHERE model = ? AND task_type = ?
                    GROUP BY model
                ''', (model.value, task_type)).fetchone()

                if rows and rows['count'] >= min_samples:
                    results[model] = ModelPerformance(
                        model=model,
                        task_type=task_type,
                        success_rate=rows['success_rate'] or 0.0,
                        average_duration_seconds=rows['avg_duration'] or 0.0,
                        sample_count=rows['count'],
                        last_updated=datetime.fromisoformat(rows['last_updated']) if rows['last_updated'] else datetime.now()
                    )

        return results

    def get_best_model_for_type(self, task_type: str) -> Optional[Model]:
        """Get the best performing model for a task type."""
        performance = self.get_performance(task_type)

        if not performance:
            return None

        # Rank by success rate, then by speed
        ranked = sorted(
            performance.items(),
            key=lambda x: (x[1].success_rate, -x[1].average_duration_seconds),
            reverse=True
        )

        return ranked[0][0] if ranked else None


class SmartModelSelector:
    """
    Main interface for smart model selection.

    Combines complexity estimation, cost optimization, and performance tracking
    to make intelligent model selection decisions.
    """

    def __init__(
        self,
        data_dir: Path,
        daily_budget: Optional[float] = None,
        monthly_budget: Optional[float] = None,
        default_model: Model = Model.SONNET
    ):
        self.estimator = ComplexityEstimator()
        self.cost_optimizer = CostOptimizer(daily_budget, monthly_budget)
        self.performance_tracker = ModelPerformanceTracker(data_dir / "model_performance.db")
        self.default_model = default_model

    def select_model(
        self,
        description: str,
        task_type: Optional[str] = None,
        context_files: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        acceptance_criteria: Optional[List[str]] = None,
        prefer_speed: bool = False,
        prefer_quality: bool = False,
        force_model: Optional[Model] = None
    ) -> ModelSelection:
        """
        Select the best model for a task.

        Args:
            description: Task description
            task_type: Type of task (for learning)
            context_files: Files involved
            constraints: Task constraints
            acceptance_criteria: Acceptance criteria
            prefer_speed: Prefer faster model
            prefer_quality: Prefer more capable model
            force_model: Override with specific model

        Returns:
            ModelSelection with selected model and reasoning
        """
        # Handle forced model
        if force_model:
            return ModelSelection(
                selected_model=force_model,
                reasoning=f"Model forced to {force_model.value}",
                confidence=1.0,
                alternatives=[]
            )

        # Get complexity estimation
        complexity = self.estimator.estimate(
            description, context_files, constraints, acceptance_criteria
        )

        # Get learned best model for task type if available
        learned_best = None
        if task_type:
            learned_best = self.performance_tracker.get_best_model_for_type(task_type)

        # Get affordable models
        affordable = self.cost_optimizer.get_affordable_models()

        if not affordable:
            # Fallback to cheapest if budget exhausted
            return ModelSelection(
                selected_model=Model.HAIKU,
                reasoning="Budget constraints - using most affordable model",
                confidence=0.5,
                alternatives=[(Model.SONNET, "Over budget"), (Model.OPUS, "Over budget")]
            )

        # Decision logic
        alternatives = []
        selected = None
        reasoning = ""

        # Priority 1: Learned best model (if available and affordable)
        if learned_best and learned_best in affordable:
            perf = self.performance_tracker.get_performance(task_type or "general").get(learned_best)
            if perf and perf.success_rate > 0.8:
                selected = learned_best
                reasoning = f"Historical data shows {learned_best.value} performs best for {task_type}"

        # Priority 2: User preference
        if not selected:
            if prefer_speed and Model.HAIKU in affordable:
                selected = Model.HAIKU
                reasoning = "Prioritizing speed - using fastest model"
            elif prefer_quality and Model.OPUS in affordable:
                selected = Model.OPUS
                reasoning = "Prioritizing quality - using most capable model"

        # Priority 3: Complexity-based selection
        if not selected:
            if complexity.recommended_model in affordable:
                selected = complexity.recommended_model
                reasoning = complexity.reasoning
            else:
                # Fall back to best affordable
                selected = max(affordable, key=lambda m: list(Model).index(m))
                reasoning = f"Complexity suggests {complexity.recommended_model.value} but using {selected.value} due to budget"

        # Build alternatives
        for model in Model:
            if model != selected:
                if model not in affordable:
                    alternatives.append((model, "Over budget"))
                elif model == Model.HAIKU and complexity.overall_score > 0.5:
                    alternatives.append((model, "Task too complex"))
                elif model == Model.OPUS and complexity.overall_score < 0.3:
                    alternatives.append((model, "Overkill for simple task"))
                else:
                    alternatives.append((model, "Not optimal for this task"))

        return ModelSelection(
            selected_model=selected,
            reasoning=reasoning,
            confidence=complexity.confidence,
            alternatives=alternatives,
            estimated_cost=self._estimate_cost(selected)
        )

    def record_task_result(
        self,
        model: Model,
        task_type: str,
        success: bool,
        duration_seconds: float,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None
    ):
        """Record task result for learning."""
        self.performance_tracker.record_task(
            model, task_type, success, duration_seconds, input_tokens, output_tokens
        )

        if input_tokens and output_tokens:
            self.cost_optimizer.record_usage(model, input_tokens, output_tokens)

    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about model usage and performance."""
        return {
            "daily_spend": self.cost_optimizer.daily_spend,
            "monthly_spend": self.cost_optimizer.monthly_spend,
            "daily_budget": self.cost_optimizer.daily_budget,
            "monthly_budget": self.cost_optimizer.monthly_budget,
            "default_model": self.default_model.value,
        }

    def _estimate_cost(self, model: Model, tokens: int = 2000) -> float:
        """Estimate cost for a model."""
        cost = MODEL_COSTS[model]
        return (tokens / 1_000_000) * (cost.input_per_million + cost.output_per_million)


# ============================================================================
# Factory function
# ============================================================================

def create_smart_model_selector(
    data_dir: Path,
    daily_budget: Optional[float] = None,
    monthly_budget: Optional[float] = None
) -> SmartModelSelector:
    """Create a SmartModelSelector instance."""
    return SmartModelSelector(data_dir, daily_budget, monthly_budget)
