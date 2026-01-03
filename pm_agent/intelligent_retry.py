"""
Intelligent Retry System - Adaptive retry strategies based on failure type.

This module provides intelligent retry capabilities that:
1. Classify errors by type and root cause
2. Select appropriate retry strategies
3. Mutate approach between retries
4. Know when to escalate vs retry
5. Learn from retry outcomes
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
from datetime import datetime
import re
import logging

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors for strategy selection."""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    LOGIC_ERROR = "logic_error"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    PERMISSION_ERROR = "permission_error"
    DEPENDENCY_ERROR = "dependency_error"
    CONTEXT_ERROR = "context_error"  # Missing context/information
    AMBIGUITY_ERROR = "ambiguity_error"  # Unclear requirements
    EXTERNAL_ERROR = "external_error"  # External service issues
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Available retry strategies."""
    SIMPLE_RETRY = "simple_retry"  # Just try again
    ADD_CONTEXT = "add_context"  # Add more context/examples
    SIMPLIFY_TASK = "simplify_task"  # Break into smaller parts
    UPGRADE_MODEL = "upgrade_model"  # Use more capable model
    ADD_CONSTRAINTS = "add_constraints"  # Add explicit constraints
    CHANGE_APPROACH = "change_approach"  # Different implementation strategy
    ESCALATE = "escalate"  # Give up and escalate
    WAIT_AND_RETRY = "wait_and_retry"  # Wait for external issues to resolve


@dataclass
class ErrorClassification:
    """Classification of an error."""
    category: ErrorCategory
    confidence: float
    root_cause: str
    suggested_strategies: List[RetryStrategy]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryPlan:
    """Plan for retrying a failed task."""
    strategy: RetryStrategy
    modifications: Dict[str, Any]
    reasoning: str
    estimated_success_probability: float
    max_additional_attempts: int = 2


@dataclass
class ApproachMutation:
    """A modification to the task approach."""
    mutation_type: str
    original_value: Any
    new_value: Any
    reasoning: str


class ErrorClassifier:
    """Classifies errors to enable intelligent retry strategies."""

    # Error patterns mapped to categories
    ERROR_PATTERNS = {
        ErrorCategory.SYNTAX_ERROR: [
            r'SyntaxError',
            r'IndentationError',
            r'invalid syntax',
            r'unexpected token',
            r'parsing error',
            r'malformed',
        ],
        ErrorCategory.RUNTIME_ERROR: [
            r'RuntimeError',
            r'TypeError',
            r'ValueError',
            r'AttributeError',
            r'KeyError',
            r'IndexError',
            r'ZeroDivisionError',
            r'NameError',
        ],
        ErrorCategory.LOGIC_ERROR: [
            r'assertion.*fail',
            r'test.*fail',
            r'expect.*but.*got',
            r'incorrect.*result',
            r'wrong.*output',
            r'logic error',
        ],
        ErrorCategory.TIMEOUT: [
            r'timeout',
            r'timed out',
            r'took too long',
            r'exceeded.*time',
            r'deadline exceeded',
        ],
        ErrorCategory.RESOURCE_LIMIT: [
            r'memory',
            r'out of memory',
            r'OOM',
            r'disk.*full',
            r'quota.*exceeded',
            r'rate.*limit',
            r'too many',
        ],
        ErrorCategory.PERMISSION_ERROR: [
            r'permission.*denied',
            r'access.*denied',
            r'forbidden',
            r'unauthorized',
            r'not allowed',
            r'EACCES',
        ],
        ErrorCategory.DEPENDENCY_ERROR: [
            r'import.*error',
            r'module.*not.*found',
            r'package.*not.*found',
            r'dependency',
            r'cannot find module',
            r'no such file',
            r'ENOENT',
        ],
        ErrorCategory.CONTEXT_ERROR: [
            r'undefined',
            r'not defined',
            r'missing.*context',
            r'unknown.*reference',
            r'cannot find',
            r'does not exist',
        ],
        ErrorCategory.AMBIGUITY_ERROR: [
            r'ambiguous',
            r'unclear',
            r'multiple.*possible',
            r'which.*one',
            r'not sure',
        ],
        ErrorCategory.EXTERNAL_ERROR: [
            r'connection.*refused',
            r'network.*error',
            r'ECONNREFUSED',
            r'service.*unavailable',
            r'502',
            r'503',
            r'504',
        ],
    }

    # Strategy recommendations by error category
    STRATEGY_MAP = {
        ErrorCategory.SYNTAX_ERROR: [
            RetryStrategy.ADD_CONTEXT,
            RetryStrategy.UPGRADE_MODEL,
            RetryStrategy.SIMPLE_RETRY,
        ],
        ErrorCategory.RUNTIME_ERROR: [
            RetryStrategy.ADD_CONSTRAINTS,
            RetryStrategy.ADD_CONTEXT,
            RetryStrategy.CHANGE_APPROACH,
        ],
        ErrorCategory.LOGIC_ERROR: [
            RetryStrategy.ADD_CONTEXT,
            RetryStrategy.CHANGE_APPROACH,
            RetryStrategy.UPGRADE_MODEL,
        ],
        ErrorCategory.TIMEOUT: [
            RetryStrategy.SIMPLIFY_TASK,
            RetryStrategy.WAIT_AND_RETRY,
            RetryStrategy.ESCALATE,
        ],
        ErrorCategory.RESOURCE_LIMIT: [
            RetryStrategy.SIMPLIFY_TASK,
            RetryStrategy.WAIT_AND_RETRY,
            RetryStrategy.ESCALATE,
        ],
        ErrorCategory.PERMISSION_ERROR: [
            RetryStrategy.ESCALATE,  # Usually needs human intervention
        ],
        ErrorCategory.DEPENDENCY_ERROR: [
            RetryStrategy.ADD_CONSTRAINTS,
            RetryStrategy.ESCALATE,
        ],
        ErrorCategory.CONTEXT_ERROR: [
            RetryStrategy.ADD_CONTEXT,
            RetryStrategy.ADD_CONSTRAINTS,
            RetryStrategy.UPGRADE_MODEL,
        ],
        ErrorCategory.AMBIGUITY_ERROR: [
            RetryStrategy.ADD_CONSTRAINTS,
            RetryStrategy.ESCALATE,
        ],
        ErrorCategory.EXTERNAL_ERROR: [
            RetryStrategy.WAIT_AND_RETRY,
            RetryStrategy.ESCALATE,
        ],
        ErrorCategory.UNKNOWN: [
            RetryStrategy.ADD_CONTEXT,
            RetryStrategy.UPGRADE_MODEL,
            RetryStrategy.ESCALATE,
        ],
    }

    def classify(self, error_message: str, error_type: Optional[str] = None, task_output: Optional[str] = None) -> ErrorClassification:
        """
        Classify an error to determine appropriate retry strategy.

        Args:
            error_message: The error message
            error_type: Optional error type from the system
            task_output: Optional full task output for more context

        Returns:
            ErrorClassification with category and suggested strategies
        """
        combined_text = f"{error_message} {error_type or ''} {task_output or ''}".lower()

        # Find matching category
        best_category = ErrorCategory.UNKNOWN
        best_confidence = 0.0
        matched_patterns = []

        for category, patterns in self.ERROR_PATTERNS.items():
            match_count = 0
            for pattern in patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    match_count += 1
                    matched_patterns.append(pattern)

            if match_count > 0:
                confidence = min(0.9, 0.5 + match_count * 0.15)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_category = category

        # Get suggested strategies
        strategies = self.STRATEGY_MAP.get(best_category, [RetryStrategy.ESCALATE])

        # Determine root cause
        root_cause = self._determine_root_cause(best_category, error_message, matched_patterns)

        return ErrorClassification(
            category=best_category,
            confidence=best_confidence,
            root_cause=root_cause,
            suggested_strategies=strategies,
            details={
                "matched_patterns": matched_patterns,
                "original_error": error_message,
                "error_type": error_type,
            }
        )

    def _determine_root_cause(self, category: ErrorCategory, error_message: str, patterns: List[str]) -> str:
        """Determine the likely root cause of the error."""
        causes = {
            ErrorCategory.SYNTAX_ERROR: "Code syntax is invalid or malformed",
            ErrorCategory.RUNTIME_ERROR: "Code runs but encounters an error during execution",
            ErrorCategory.LOGIC_ERROR: "Code runs but produces incorrect results",
            ErrorCategory.TIMEOUT: "Task took too long to complete",
            ErrorCategory.RESOURCE_LIMIT: "System resources were exhausted",
            ErrorCategory.PERMISSION_ERROR: "Insufficient permissions for the operation",
            ErrorCategory.DEPENDENCY_ERROR: "Required dependency is missing or incompatible",
            ErrorCategory.CONTEXT_ERROR: "Required information or context is missing",
            ErrorCategory.AMBIGUITY_ERROR: "Requirements are unclear or have multiple interpretations",
            ErrorCategory.EXTERNAL_ERROR: "External service or network issue",
            ErrorCategory.UNKNOWN: "Error cause could not be determined",
        }
        return causes.get(category, "Unknown cause")


class RetryStrategySelector:
    """Selects the best retry strategy based on error classification and history."""

    def __init__(self):
        # Track strategy effectiveness
        self.strategy_history: Dict[str, Dict[str, int]] = {}  # {category: {strategy: success_count}}

    def select_strategy(
        self,
        classification: ErrorClassification,
        attempt_count: int,
        max_attempts: int,
        previous_strategies: List[RetryStrategy] = None
    ) -> RetryPlan:
        """
        Select the best retry strategy.

        Args:
            classification: Error classification
            attempt_count: Current attempt number
            max_attempts: Maximum allowed attempts
            previous_strategies: Strategies already tried

        Returns:
            RetryPlan with selected strategy and modifications
        """
        previous_strategies = previous_strategies or []
        remaining_attempts = max_attempts - attempt_count

        # If we're out of attempts, escalate
        if remaining_attempts <= 0:
            return RetryPlan(
                strategy=RetryStrategy.ESCALATE,
                modifications={},
                reasoning=f"Maximum attempts ({max_attempts}) reached",
                estimated_success_probability=0.0,
                max_additional_attempts=0
            )

        # Filter out already-tried strategies
        available_strategies = [
            s for s in classification.suggested_strategies
            if s not in previous_strategies
        ]

        # If all suggested strategies tried, escalate or repeat best one
        if not available_strategies:
            if remaining_attempts > 0 and classification.suggested_strategies:
                # Try the first suggestion again with modifications
                strategy = classification.suggested_strategies[0]
                return self._create_plan_with_mutation(strategy, classification, attempt_count)
            else:
                return RetryPlan(
                    strategy=RetryStrategy.ESCALATE,
                    modifications={},
                    reasoning="All retry strategies exhausted",
                    estimated_success_probability=0.0,
                    max_additional_attempts=0
                )

        # Select the best available strategy
        strategy = available_strategies[0]

        # Create plan with appropriate modifications
        return self._create_plan(strategy, classification, attempt_count, remaining_attempts)

    def _create_plan(
        self,
        strategy: RetryStrategy,
        classification: ErrorClassification,
        attempt_count: int,
        remaining_attempts: int
    ) -> RetryPlan:
        """Create a retry plan for the selected strategy."""
        plans = {
            RetryStrategy.SIMPLE_RETRY: RetryPlan(
                strategy=strategy,
                modifications={},
                reasoning="Simple retry - error may be transient",
                estimated_success_probability=0.3,
                max_additional_attempts=1
            ),
            RetryStrategy.ADD_CONTEXT: RetryPlan(
                strategy=strategy,
                modifications={
                    "add_context": True,
                    "context_areas": ["error_details", "file_contents", "examples"],
                    "context_request": f"Previous error: {classification.details.get('original_error', 'unknown')}"
                },
                reasoning=f"Adding more context to address {classification.root_cause}",
                estimated_success_probability=0.5,
                max_additional_attempts=2
            ),
            RetryStrategy.SIMPLIFY_TASK: RetryPlan(
                strategy=strategy,
                modifications={
                    "simplify": True,
                    "break_into_subtasks": True,
                    "reduce_scope": True,
                },
                reasoning="Breaking task into smaller, more manageable parts",
                estimated_success_probability=0.6,
                max_additional_attempts=3
            ),
            RetryStrategy.UPGRADE_MODEL: RetryPlan(
                strategy=strategy,
                modifications={
                    "upgrade_model": True,
                    "from_model": "sonnet",
                    "to_model": "opus",
                },
                reasoning="Using more capable model for complex task",
                estimated_success_probability=0.7,
                max_additional_attempts=1
            ),
            RetryStrategy.ADD_CONSTRAINTS: RetryPlan(
                strategy=strategy,
                modifications={
                    "add_constraints": True,
                    "new_constraints": [
                        f"Avoid this error: {classification.root_cause}",
                        "Validate output before completing",
                        "Check for edge cases",
                    ],
                },
                reasoning="Adding explicit constraints to guide implementation",
                estimated_success_probability=0.5,
                max_additional_attempts=2
            ),
            RetryStrategy.CHANGE_APPROACH: RetryPlan(
                strategy=strategy,
                modifications={
                    "change_approach": True,
                    "approach_hint": "Try a different implementation strategy",
                    "avoid_previous": True,
                },
                reasoning="Previous approach failed, trying alternative",
                estimated_success_probability=0.5,
                max_additional_attempts=2
            ),
            RetryStrategy.WAIT_AND_RETRY: RetryPlan(
                strategy=strategy,
                modifications={
                    "wait_seconds": 30 * (attempt_count + 1),  # Exponential backoff
                },
                reasoning="Waiting for transient issue to resolve",
                estimated_success_probability=0.4,
                max_additional_attempts=2
            ),
            RetryStrategy.ESCALATE: RetryPlan(
                strategy=strategy,
                modifications={},
                reasoning="Task requires human intervention",
                estimated_success_probability=0.0,
                max_additional_attempts=0
            ),
        }

        return plans.get(strategy, plans[RetryStrategy.ESCALATE])

    def _create_plan_with_mutation(
        self,
        strategy: RetryStrategy,
        classification: ErrorClassification,
        attempt_count: int
    ) -> RetryPlan:
        """Create a plan with additional mutations for repeated strategies."""
        base_plan = self._create_plan(strategy, classification, attempt_count, 1)

        # Add mutations to differentiate from previous attempt
        base_plan.modifications["mutation_applied"] = True
        base_plan.modifications["mutation_reason"] = f"Retry #{attempt_count + 1} with variations"
        base_plan.estimated_success_probability *= 0.8  # Lower probability for repeated strategy

        return base_plan

    def record_outcome(self, category: str, strategy: RetryStrategy, success: bool):
        """Record the outcome of a retry strategy for learning."""
        if category not in self.strategy_history:
            self.strategy_history[category] = {}

        key = f"{strategy.value}_{success}"
        self.strategy_history[category][key] = self.strategy_history[category].get(key, 0) + 1


class ApproachMutator:
    """Mutates task approaches between retries to try different solutions."""

    def mutate_task(
        self,
        task_description: str,
        constraints: List[str],
        context_files: List[str],
        retry_plan: RetryPlan,
        previous_output: Optional[str] = None
    ) -> Tuple[str, List[str], List[str]]:
        """
        Mutate a task based on the retry plan.

        Args:
            task_description: Original task description
            constraints: Original constraints
            context_files: Original context files
            retry_plan: The retry plan to apply
            previous_output: Output from previous attempt

        Returns:
            Tuple of (new_description, new_constraints, new_context_files)
        """
        new_description = task_description
        new_constraints = list(constraints)
        new_context_files = list(context_files)

        mods = retry_plan.modifications

        if mods.get("add_context"):
            # Add context about the error
            context_request = mods.get("context_request", "")
            new_description = f"{task_description}\n\nIMPORTANT CONTEXT:\n{context_request}"

            # Add constraint about the error
            new_constraints.append(f"Avoid previous error: {context_request}")

        if mods.get("simplify"):
            new_description = f"SIMPLIFIED TASK: Focus only on the core requirement.\n\n{task_description}"
            new_constraints.append("Keep the solution simple and minimal")
            new_constraints.append("Avoid unnecessary complexity")

        if mods.get("add_constraints"):
            for constraint in mods.get("new_constraints", []):
                if constraint not in new_constraints:
                    new_constraints.append(constraint)

        if mods.get("change_approach"):
            new_description = f"{task_description}\n\nNOTE: Try a different approach than before."
            if previous_output:
                new_constraints.append(f"Previous approach failed. Try something different.")

        if mods.get("avoid_previous"):
            new_constraints.append("Use a different implementation strategy than the previous attempt")

        return new_description, new_constraints, new_context_files


class IntelligentRetrySystem:
    """
    Main interface for the intelligent retry system.

    Combines error classification, strategy selection, and approach mutation
    to enable smart retries that learn and adapt.
    """

    def __init__(self):
        self.classifier = ErrorClassifier()
        self.strategy_selector = RetryStrategySelector()
        self.mutator = ApproachMutator()

        # Track retry history per task
        self.retry_history: Dict[str, List[RetryPlan]] = {}

    def analyze_failure(
        self,
        task_id: str,
        error_message: str,
        error_type: Optional[str] = None,
        task_output: Optional[str] = None
    ) -> ErrorClassification:
        """Analyze a task failure."""
        return self.classifier.classify(error_message, error_type, task_output)

    def get_retry_plan(
        self,
        task_id: str,
        error_message: str,
        attempt_count: int,
        max_attempts: int = 3,
        error_type: Optional[str] = None,
        task_output: Optional[str] = None
    ) -> RetryPlan:
        """
        Get a retry plan for a failed task.

        Args:
            task_id: ID of the failed task
            error_message: The error message
            attempt_count: Current attempt number
            max_attempts: Maximum allowed attempts
            error_type: Optional error type
            task_output: Optional full task output

        Returns:
            RetryPlan with strategy and modifications
        """
        # Classify the error
        classification = self.classifier.classify(error_message, error_type, task_output)

        logger.info(
            f"Error classified as {classification.category.value} "
            f"with confidence {classification.confidence:.2f}"
        )

        # Get previous strategies for this task
        previous_strategies = [
            plan.strategy for plan in self.retry_history.get(task_id, [])
        ]

        # Select strategy
        plan = self.strategy_selector.select_strategy(
            classification,
            attempt_count,
            max_attempts,
            previous_strategies
        )

        # Record for history
        if task_id not in self.retry_history:
            self.retry_history[task_id] = []
        self.retry_history[task_id].append(plan)

        logger.info(
            f"Selected retry strategy: {plan.strategy.value} "
            f"(estimated success: {plan.estimated_success_probability:.0%})"
        )

        return plan

    def apply_retry_plan(
        self,
        task_description: str,
        constraints: List[str],
        context_files: List[str],
        retry_plan: RetryPlan,
        previous_output: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply a retry plan to a task.

        Args:
            task_description: Original task description
            constraints: Original constraints
            context_files: Original context files
            retry_plan: The retry plan to apply
            previous_output: Output from previous attempt

        Returns:
            Dictionary with modified task parameters
        """
        if retry_plan.strategy == RetryStrategy.ESCALATE:
            return {
                "should_retry": False,
                "escalate": True,
                "reason": retry_plan.reasoning
            }

        if retry_plan.strategy == RetryStrategy.WAIT_AND_RETRY:
            wait_seconds = retry_plan.modifications.get("wait_seconds", 30)
            return {
                "should_retry": True,
                "wait_seconds": wait_seconds,
                "description": task_description,
                "constraints": constraints,
                "context_files": context_files,
            }

        # Apply mutations
        new_desc, new_constraints, new_files = self.mutator.mutate_task(
            task_description,
            constraints,
            context_files,
            retry_plan,
            previous_output
        )

        # Handle model upgrade
        model_override = None
        if retry_plan.modifications.get("upgrade_model"):
            model_override = retry_plan.modifications.get("to_model", "opus")

        return {
            "should_retry": True,
            "description": new_desc,
            "constraints": new_constraints,
            "context_files": new_files,
            "model_override": model_override,
            "strategy_applied": retry_plan.strategy.value,
        }

    def record_retry_outcome(self, task_id: str, success: bool):
        """Record the outcome of a retry for learning."""
        if task_id in self.retry_history and self.retry_history[task_id]:
            last_plan = self.retry_history[task_id][-1]
            # Would record to adaptive learner for pattern learning
            logger.info(
                f"Retry outcome for {task_id}: {last_plan.strategy.value} "
                f"{'succeeded' if success else 'failed'}"
            )

    def should_retry(self, task_id: str, attempt_count: int, max_attempts: int) -> bool:
        """Determine if a task should be retried."""
        if attempt_count >= max_attempts:
            return False

        # Check if we've exhausted all strategies
        if task_id in self.retry_history:
            strategies_tried = len(self.retry_history[task_id])
            if strategies_tried >= len(RetryStrategy) - 1:  # -1 for ESCALATE
                return False

        return True

    def clear_history(self, task_id: str):
        """Clear retry history for a task."""
        if task_id in self.retry_history:
            del self.retry_history[task_id]


# ============================================================================
# Factory function
# ============================================================================

def create_intelligent_retry_system() -> IntelligentRetrySystem:
    """Create an IntelligentRetrySystem instance."""
    return IntelligentRetrySystem()
