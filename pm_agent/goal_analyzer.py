"""
Goal Analyzer - Intelligent goal complexity analysis and task breakdown.

This module provides the GoalAnalyzer class which uses heuristics and pattern
matching to analyze goal complexity and suggest intelligent task breakdowns.
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
import re


class ComplexityLevel(Enum):
    """Goal complexity levels."""
    LOW = "low"              # Simple, single-file changes
    MEDIUM = "medium"        # Multi-file or multi-step changes
    HIGH = "high"            # Complex features requiring coordination
    COMPLEX = "complex"      # Large-scale architecture changes


@dataclass
class SubtaskSuggestion:
    """A suggested subtask for a goal."""
    description: str
    acceptance_criteria: List[str]
    estimated_minutes: int
    dependencies: List[int]  # Indices of tasks this depends on
    priority: str = "medium"
    context_hint: Optional[str] = None  # Hint about what files/areas involved


@dataclass
class GoalAnalysis:
    """Complete analysis of a goal."""
    complexity: ComplexityLevel
    estimated_total_minutes: int
    subtasks: List[SubtaskSuggestion]
    reasoning: str
    risks: List[str]
    key_entities: List[str]  # Important files, classes, systems mentioned


class GoalAnalyzer:
    """
    Analyzes goal descriptions to determine complexity and suggest task breakdowns.

    This class uses heuristic analysis to:
    1. Identify complexity based on keywords, scope, and patterns
    2. Suggest logical subtask breakdowns
    3. Estimate time requirements
    4. Identify dependencies between subtasks
    """

    # Complexity indicators
    COMPLEXITY_KEYWORDS = {
        'low': {
            'fix', 'typo', 'rename', 'update comment', 'add comment',
            'format', 'lint', 'style', 'minor'
        },
        'medium': {
            'add', 'implement', 'create', 'update', 'modify', 'refactor',
            'improve', 'enhance', 'extend'
        },
        'high': {
            'integrate', 'migrate', 'redesign', 'architecture',
            'system', 'framework', 'api', 'database', 'authentication',
            'authorization', 'security', 'performance optimization'
        },
        'complex': {
            'rebuild', 'rewrite', 'overhaul', 'complete redesign',
            'multi-system', 'microservices', 'distributed',
            'real-time', 'scalability', 'infrastructure'
        }
    }

    # Scope multipliers
    SCOPE_KEYWORDS = {
        'single': ['one', 'single', 'specific', 'this file'],
        'multiple': ['multiple', 'several', 'various', 'across'],
        'project_wide': ['entire', 'whole', 'all', 'everywhere', 'project-wide', 'codebase']
    }

    # Common task breakdown patterns
    TASK_PATTERNS = {
        'feature_implementation': {
            'keywords': ['implement', 'add feature', 'create'],
            'subtasks': [
                'Design and plan implementation approach',
                'Implement core logic',
                'Add tests',
                'Update documentation'
            ]
        },
        'refactoring': {
            'keywords': ['refactor', 'restructure', 'reorganize'],
            'subtasks': [
                'Analyze current implementation',
                'Design new structure',
                'Implement refactoring',
                'Verify tests still pass'
            ]
        },
        'bug_fix': {
            'keywords': ['fix', 'bug', 'error', 'issue'],
            'subtasks': [
                'Reproduce and diagnose issue',
                'Implement fix',
                'Add regression test'
            ]
        },
        'integration': {
            'keywords': ['integrate', 'connect', 'link'],
            'subtasks': [
                'Research integration requirements',
                'Implement integration layer',
                'Add integration tests',
                'Update configuration and documentation'
            ]
        },
        'optimization': {
            'keywords': ['optimize', 'improve performance', 'speed up'],
            'subtasks': [
                'Profile and identify bottlenecks',
                'Implement optimizations',
                'Benchmark improvements',
                'Document performance changes'
            ]
        }
    }

    def __init__(self):
        """Initialize the goal analyzer."""
        pass

    def analyze_complexity(self, goal_description: str) -> ComplexityLevel:
        """
        Analyze the complexity of a goal based on its description.

        Args:
            goal_description: The goal description to analyze

        Returns:
            ComplexityLevel indicating the estimated complexity
        """
        desc_lower = goal_description.lower()

        # Count keyword matches at each level
        scores = {
            'low': 0,
            'medium': 0,
            'high': 0,
            'complex': 0
        }

        for level, keywords in self.COMPLEXITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    scores[level] += 1

        # Adjust for scope
        scope_multiplier = 1.0
        if any(word in desc_lower for word in self.SCOPE_KEYWORDS['project_wide']):
            scope_multiplier = 2.0
        elif any(word in desc_lower for word in self.SCOPE_KEYWORDS['multiple']):
            scope_multiplier = 1.5

        # Apply scope multiplier to higher complexity levels
        scores['high'] = int(scores['high'] * scope_multiplier)
        scores['complex'] = int(scores['complex'] * scope_multiplier)

        # Determine complexity based on scores
        if scores['complex'] >= 2 or (scores['complex'] >= 1 and scores['high'] >= 1):
            return ComplexityLevel.COMPLEX
        elif scores['high'] >= 2 or (scores['high'] >= 1 and scores['medium'] >= 2):
            return ComplexityLevel.HIGH
        elif scores['medium'] >= 1 or scores['low'] >= 2:
            return ComplexityLevel.MEDIUM if scores['medium'] >= 1 else ComplexityLevel.LOW
        elif scores['low'] >= 1:
            return ComplexityLevel.LOW
        else:
            # Default to MEDIUM for ambiguous cases
            return ComplexityLevel.MEDIUM

    def suggest_breakdown(self, goal_description: str) -> List[SubtaskSuggestion]:
        """
        Suggest a breakdown of subtasks for a goal.

        Args:
            goal_description: The goal description to analyze

        Returns:
            List of suggested subtasks
        """
        desc_lower = goal_description.lower()
        complexity = self.analyze_complexity(goal_description)

        # Find matching pattern
        pattern_match = None
        for pattern_name, pattern_info in self.TASK_PATTERNS.items():
            if any(keyword in desc_lower for keyword in pattern_info['keywords']):
                pattern_match = pattern_info
                break

        # If complexity is LOW, don't break down
        if complexity == ComplexityLevel.LOW:
            return [SubtaskSuggestion(
                description=goal_description,
                acceptance_criteria=[f"Goal '{goal_description}' is achieved"],
                estimated_minutes=15,
                dependencies=[],
                priority="medium"
            )]

        # For MEDIUM+, create structured breakdown
        subtasks = []

        if pattern_match:
            # Use pattern-based breakdown
            for i, task_template in enumerate(pattern_match['subtasks']):
                # Customize the template with goal context
                description = self._customize_task_description(task_template, goal_description)
                criteria = self._generate_acceptance_criteria(description, goal_description)

                subtasks.append(SubtaskSuggestion(
                    description=description,
                    acceptance_criteria=criteria,
                    estimated_minutes=self._estimate_subtask_time(description, complexity),
                    dependencies=list(range(i)),  # Sequential dependencies
                    priority="high" if i == len(pattern_match['subtasks']) - 1 else "medium"
                ))
        else:
            # Generic breakdown based on complexity
            subtasks = self._generic_breakdown(goal_description, complexity)

        return subtasks

    def estimate_time(self, goal_description: str) -> int:
        """
        Estimate time required for a goal in minutes.

        Args:
            goal_description: The goal description to analyze

        Returns:
            Estimated time in minutes
        """
        complexity = self.analyze_complexity(goal_description)
        subtasks = self.suggest_breakdown(goal_description)

        # Sum up subtask estimates
        total_time = sum(task.estimated_minutes for task in subtasks)

        # Add buffer based on complexity
        buffers = {
            ComplexityLevel.LOW: 1.1,
            ComplexityLevel.MEDIUM: 1.2,
            ComplexityLevel.HIGH: 1.3,
            ComplexityLevel.COMPLEX: 1.5
        }

        return int(total_time * buffers[complexity])

    def identify_dependencies(self, subtasks: List[SubtaskSuggestion]) -> Dict[int, List[int]]:
        """
        Identify and validate dependencies between subtasks.

        Args:
            subtasks: List of subtasks to analyze

        Returns:
            Dictionary mapping task index to list of dependency indices
        """
        dependencies = {}

        for i, task in enumerate(subtasks):
            # Validate dependencies are within bounds
            valid_deps = [dep for dep in task.dependencies if 0 <= dep < len(subtasks) and dep != i]
            dependencies[i] = valid_deps

        # Check for circular dependencies
        def has_cycle(node: int, visited: Set[int], rec_stack: Set[int]) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for dep in dependencies.get(node, []):
                if dep not in visited:
                    if has_cycle(dep, visited, rec_stack):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        # Remove circular dependencies if found
        visited = set()
        for i in range(len(subtasks)):
            if i not in visited:
                if has_cycle(i, visited, set()):
                    # Clear all dependencies to break cycles
                    for key in dependencies:
                        dependencies[key] = []
                    break

        return dependencies

    def analyze_goal(self, goal_description: str) -> GoalAnalysis:
        """
        Perform complete analysis of a goal.

        Args:
            goal_description: The goal description to analyze

        Returns:
            Complete GoalAnalysis with all insights
        """
        complexity = self.analyze_complexity(goal_description)
        subtasks = self.suggest_breakdown(goal_description)
        estimated_time = self.estimate_time(goal_description)

        # Extract key entities (file references, system names, etc.)
        entities = self._extract_entities(goal_description)

        # Identify risks based on complexity and content
        risks = self._identify_risks(goal_description, complexity)

        # Generate reasoning
        reasoning = self._generate_reasoning(goal_description, complexity, subtasks)

        return GoalAnalysis(
            complexity=complexity,
            estimated_total_minutes=estimated_time,
            subtasks=subtasks,
            reasoning=reasoning,
            risks=risks,
            key_entities=entities
        )

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _customize_task_description(self, template: str, goal_description: str) -> str:
        """Customize a task template with context from the goal."""
        # Extract key subject from goal (simplified approach)
        # E.g., "Implement user authentication" -> "user authentication"

        # Find main action and subject
        desc_lower = goal_description.lower()

        # Common patterns: "implement X", "add X", "create X", etc.
        for verb in ['implement', 'add', 'create', 'update', 'fix', 'refactor']:
            if verb in desc_lower:
                subject_start = desc_lower.find(verb) + len(verb)
                subject = goal_description[subject_start:].strip()
                return f"{template} for {subject}"

        return f"{template} for: {goal_description}"

    def _generate_acceptance_criteria(self, task_description: str, goal_description: str) -> List[str]:
        """Generate acceptance criteria for a subtask."""
        criteria = []

        desc_lower = task_description.lower()

        # Add criteria based on task type
        if 'test' in desc_lower:
            criteria.extend([
                "Tests are written and passing",
                "Test coverage is adequate"
            ])
        elif 'implement' in desc_lower or 'create' in desc_lower:
            criteria.extend([
                "Implementation is complete and functional",
                "Code follows project conventions"
            ])
        elif 'document' in desc_lower:
            criteria.extend([
                "Documentation is clear and comprehensive",
                "Examples are provided where appropriate"
            ])
        elif 'design' in desc_lower or 'plan' in desc_lower:
            criteria.extend([
                "Design/plan is documented",
                "Approach is reviewed and approved"
            ])
        else:
            criteria.append(f"Task '{task_description}' is successfully completed")

        return criteria

    def _estimate_subtask_time(self, task_description: str, complexity: ComplexityLevel) -> int:
        """Estimate time for a single subtask."""
        desc_lower = task_description.lower()

        # Base times for different task types
        if 'design' in desc_lower or 'plan' in desc_lower or 'research' in desc_lower:
            base_time = 20
        elif 'test' in desc_lower:
            base_time = 25
        elif 'document' in desc_lower:
            base_time = 15
        elif 'implement' in desc_lower or 'create' in desc_lower:
            base_time = 30
        else:
            base_time = 20

        # Adjust based on overall complexity
        multipliers = {
            ComplexityLevel.LOW: 0.8,
            ComplexityLevel.MEDIUM: 1.0,
            ComplexityLevel.HIGH: 1.3,
            ComplexityLevel.COMPLEX: 1.5
        }

        return int(base_time * multipliers[complexity])

    def _generic_breakdown(self, goal_description: str, complexity: ComplexityLevel) -> List[SubtaskSuggestion]:
        """Create a generic breakdown when no pattern matches."""
        subtasks = []

        if complexity in (ComplexityLevel.MEDIUM, ComplexityLevel.HIGH):
            # Standard 3-phase breakdown
            phases = [
                ("Analyze and design solution", "Design is documented and reviewed"),
                (f"Implement {goal_description}", "Implementation is complete and functional"),
                ("Add tests and validation", "Tests are passing and coverage is adequate")
            ]

            for i, (desc, criteria) in enumerate(phases):
                subtasks.append(SubtaskSuggestion(
                    description=desc,
                    acceptance_criteria=[criteria],
                    estimated_minutes=self._estimate_subtask_time(desc, complexity),
                    dependencies=list(range(i)),
                    priority="medium"
                ))

        elif complexity == ComplexityLevel.COMPLEX:
            # 5-phase breakdown for complex goals
            phases = [
                ("Research and architectural design", "Architecture is designed and approved"),
                ("Implement core functionality", "Core features are working"),
                ("Implement secondary features", "All features are implemented"),
                ("Add comprehensive tests", "Test coverage is >80%"),
                ("Documentation and polish", "Documentation is complete")
            ]

            for i, (desc, criteria) in enumerate(phases):
                subtasks.append(SubtaskSuggestion(
                    description=desc,
                    acceptance_criteria=[criteria],
                    estimated_minutes=self._estimate_subtask_time(desc, complexity),
                    dependencies=list(range(i)),
                    priority="high" if i < 3 else "medium"
                ))

        return subtasks

    def _extract_entities(self, goal_description: str) -> List[str]:
        """Extract key entities (files, systems, classes) from goal description."""
        entities = []

        # Find file paths (e.g., path/to/file.py)
        file_pattern = r'\b[\w/]+\.\w{2,4}\b'
        files = re.findall(file_pattern, goal_description)
        entities.extend(files)

        # Find class names (CamelCase words)
        class_pattern = r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b'
        classes = re.findall(class_pattern, goal_description)
        entities.extend(classes)

        # Find system/component names (quoted or capitalized)
        quoted_pattern = r'["\']([^"\']+)["\']'
        quoted = re.findall(quoted_pattern, goal_description)
        entities.extend(quoted)

        return list(set(entities))  # Remove duplicates

    def _identify_risks(self, goal_description: str, complexity: ComplexityLevel) -> List[str]:
        """Identify potential risks based on goal content."""
        risks = []
        desc_lower = goal_description.lower()

        # Risk indicators
        if any(word in desc_lower for word in ['database', 'migration', 'schema']):
            risks.append("Database changes may require migration and data backup")

        if any(word in desc_lower for word in ['api', 'breaking', 'interface']):
            risks.append("API changes may break existing integrations")

        if any(word in desc_lower for word in ['security', 'auth', 'permission']):
            risks.append("Security-sensitive changes require thorough review")

        if any(word in desc_lower for word in ['performance', 'optimize', 'scale']):
            risks.append("Performance changes need benchmarking and validation")

        if complexity in (ComplexityLevel.HIGH, ComplexityLevel.COMPLEX):
            risks.append("High complexity increases risk of scope creep")

        if 'entire' in desc_lower or 'whole' in desc_lower or 'all' in desc_lower:
            risks.append("Project-wide changes may have unexpected side effects")

        return risks

    def _generate_reasoning(
        self,
        goal_description: str,
        complexity: ComplexityLevel,
        subtasks: List[SubtaskSuggestion]
    ) -> str:
        """Generate reasoning explanation for the analysis."""
        reasoning_parts = [
            f"Analyzed goal: '{goal_description}'",
            f"Complexity level: {complexity.value}",
            f"Breakdown: {len(subtasks)} subtask(s)"
        ]

        if len(subtasks) > 1:
            reasoning_parts.append(
                f"Sequential breakdown with dependencies to ensure proper implementation order"
            )

        return ". ".join(reasoning_parts) + "."
