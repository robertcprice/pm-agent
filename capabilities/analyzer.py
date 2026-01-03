"""
Project Analyzer Capability for PM Agent

Analyzes existing codebases to avoid redundant work by:
- Scanning all source files
- Categorizing by feature/tech stack
- Identifying what's already implemented
- Finding TODOs, FIXMEs, and gaps
- Reading file contents to understand actual implementation status

This is CRITICAL for avoiding redundant work!
"""

import ast
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter

try:
    from pm_agent.logger import PMLogger
except ImportError:
    PMLogger = None  # Will use SimpleLogger fallback


class SimpleLogger:
    """Simple fallback logger"""
    def __init__(self, name):
        self.name = name
    def log_thought(self, thought_type, message, data=None, level="info"):
        pass  # Silent fallback


class FileType(Enum):
    """Types of source files"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JSX = "jsx"
    TSX = "tsx"
    CSS = "css"
    HTML = "html"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """High-level component categories"""
    ROUTER = "router"
    MODEL = "model"
    SERVICE = "service"
    SCHEMA = "schema"
    COMPONENT = "ui_component"
    PAGE = "page"
    TEST = "test"
    CONFIG = "config"
    UTIL = "utility"


@dataclass
class SourceFile:
    """Information about a source file"""
    path: Path
    file_type: FileType
    component_type: Optional[ComponentType]
    size_bytes: int
    lines_of_code: int
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    todos: List[str] = field(default_factory=list)
    fixmes: List[str] = field(default_factory=list)
    is_stub: bool = False  # File is a stub/incomplete


@dataclass
class FeatureAnalysis:
    """Analysis of a specific feature"""
    name: str
    status: str  # "complete", "partial", "missing", "stub"
    files: List[SourceFile] = field(default_factory=list)
    todos_count: int = 0
    description: str = ""


@dataclass
class ProjectAnalysis:
    """Complete analysis of a project"""
    project_name: str
    project_path: Path
    total_files: int
    total_loc: int
    languages: Counter = field(default_factory=Counter)
    features: Dict[str, FeatureAnalysis] = field(default_factory=dict)
    all_todos: List[str] = field(default_factory=list)
    all_fixmes: List[str] = field(default_factory=list)
    scanned_at: datetime = field(default_factory=datetime.now)

    def get_summary(self) -> str:
        """Get a text summary of the analysis"""
        lines = [
            f"Project: {self.project_name}",
            f"Path: {self.project_path}",
            f"Files: {self.total_files}",
            f"Lines of Code: {self.total_loc}",
            f"Languages: {', '.join(f'{lang} ({count})' for lang, count in self.languages.most_common())}",
            "",
            "Features:"
        ]

        for feature_name, analysis in self.features.items():
            status_icon = {
                "complete": "✅",
                "partial": "🟡",
                "missing": "❌",
                "stub": "⚠️"
            }.get(analysis.status, "?")

            lines.append(f"  {status_icon} {feature_name}: {analysis.status}")
            if analysis.files:
                lines.append(f"      Files: {len(analysis.files)}")
            if analysis.todos_count:
                lines.append(f"      TODOs: {analysis.todos_count}")

        if self.all_todos:
            lines.append(f"\nTODOs: {len(self.all_todos)} total")

        return "\n".join(lines)


class ProjectAnalyzer:
    """
    Analyzes projects to understand what exists before working on them.

    This is the FIRST capability any PM Agent should use to avoid
    doing redundant work!
    """

    def __init__(self, logger=None):
        if logger is None:
            if PMLogger:
                import tempfile
                self.logger = PMLogger(log_dir=Path(tempfile.gettempdir()) / "pm_analyzer_logs")
            else:
                self.logger = SimpleLogger("project_analyzer")
        else:
            self.logger = logger

        # File patterns to ignore
        self.ignore_patterns = [
            "__pycache__",
            ".next",
            "node_modules",
            ".git",
            "venv",
            ".venv",
            "dist",
            "build",
            ".pytest_cache",
            "*.pyc",
        ]

        # Feature detection patterns
        self.feature_patterns = {
            "auth": ["auth", "login", "register", "oauth", "jwt", "token"],
            "resume": ["resume", "cv", "parsing"],
            "jobs": ["job", "application", "scraping"],
            "matching": ["match", "similarity", "embedding", "vector"],
            "payments": ["payment", "stripe", "billing", "credit"],
            "email": ["email", "notification", "mail"],
            "user": ["user", "profile", "account"],
            "database": ["db", "database", "model", "schema"],
            "api": ["router", "endpoint", "api"],
            "frontend": ["page", "component", "ui"],
        }

    def analyze_project(
        self,
        project_path: Path,
        scan_content: bool = True,
    ) -> ProjectAnalysis:
        """
        Analyze a project comprehensively.

        Args:
            project_path: Path to project root
            scan_content: Whether to read file contents (slower but more accurate)

        Returns:
            ProjectAnalysis with complete status
        """
        self.logger.info(
            "analyze_project_start",
            f"Analyzing project: {project_path}",
            {"path": str(project_path), "scan_content": scan_content}
        )

        project_name = project_path.name
        all_files = []
        all_todos = []
        all_fixmes = []
        language_counts = Counter()

        # Find all source files
        source_files = self._find_source_files(project_path)

        for file_path in source_files:
            try:
                source_file = self._analyze_file(file_path, scan_content=scan_content)
                all_files.append(source_file)

                # Track languages
                language_counts[source_file.file_type.value] += 1

                # Collect TODOs/FIXMEs
                all_todos.extend(source_file.todos)
                all_fixmes.extend(source_file.fixmes)

            except Exception as e:
                self.logger.info(
                    "analyze_file_error",
                    f"Error analyzing {file_path}: {e}",
                    {"file": str(file_path), "error": str(e)},
                    level="warning"
                )

        # Categorize into features
        features = self._categorize_features(all_files)

        total_loc = sum(f.lines_of_code for f in all_files)

        analysis = ProjectAnalysis(
            project_name=project_name,
            project_path=project_path,
            total_files=len(all_files),
            total_loc=total_loc,
            languages=language_counts,
            features=features,
            all_todos=all_todos,
            all_fixmes=all_fixmes,
        )

        self.logger.info(
            "analyze_project_complete",
            f"Analysis complete: {len(all_files)} files, {total_loc} LOC",
            {
                "files": len(all_files),
                "loc": total_loc,
                "features": len(features),
                "todos": len(all_todos)
            }
        )

        return analysis

    def _find_source_files(self, project_path: Path) -> List[Path]:
        """Find all source files in project"""
        source_files = []

        extensions = {
            ".py": FileType.PYTHON,
            ".js": FileType.JAVASCRIPT,
            ".jsx": FileType.JSX,
            ".ts": FileType.TYPESCRIPT,
            ".tsx": FileType.TSX,
            ".css": FileType.CSS,
            ".html": FileType.HTML,
        }

        for ext, file_type in extensions.items():
            matches = project_path.rglob(f"*{ext}")
            for match in matches:
                if match.is_file() and not self._should_ignore(match):
                    source_files.append(match)

        return source_files

    def _should_ignore(self, path: Path) -> bool:
        """Check if path should be ignored"""
        path_str = str(path)
        for pattern in self.ignore_patterns:
            if pattern in path_str:
                return True
        return False

    def _analyze_file(
        self,
        file_path: Path,
        scan_content: bool = True,
    ) -> SourceFile:
        """Analyze a single source file"""
        # Get file type
        suffix = file_path.suffix.lower()
        file_type_map = {
            ".py": FileType.PYTHON,
            ".js": FileType.JAVASCRIPT,
            ".jsx": FileType.JSX,
            ".ts": FileType.TYPESCRIPT,
            ".tsx": FileType.TSX,
            ".css": FileType.CSS,
            ".html": FileType.HTML,
        }
        file_type = file_type_map.get(suffix, FileType.UNKNOWN)

        # Get basic stats
        size_bytes = file_path.stat().st_size

        # Read content for detailed analysis
        content = ""
        if scan_content and size_bytes < 500000:  # Skip files > 500KB
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            except:
                pass

        lines_of_code = len(content.splitlines()) if content else 0

        # Extract info based on file type
        imports, exports, classes, functions = [], [], [], []
        todos, fixmes = [], []
        is_stub = False

        if file_type == FileType.PYTHON and content:
            imports, exports, classes, functions, todos, fixmes, is_stub = self._analyze_python(content, file_path)
        elif file_type in (FileType.JAVASCRIPT, FileType.TYPESCRIPT, FileType.JSX, FileType.TSX) and content:
            imports, exports, classes, functions, todos, fixmes, is_stub = self._analyze_javascript(content, file_path)

        # Detect component type
        component_type = self._detect_component_type(file_path, content, classes, functions)

        return SourceFile(
            path=file_path,
            file_type=file_type,
            component_type=component_type,
            size_bytes=size_bytes,
            lines_of_code=lines_of_code,
            imports=imports,
            exports=exports,
            classes=classes,
            functions=functions,
            todos=todos,
            fixmes=fixmes,
            is_stub=is_stub,
        )

    def _analyze_python(
        self,
        content: str,
        file_path: Path
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], bool]:
        """Analyze Python file"""
        imports = []
        classes = []
        functions = []
        todos = []
        fixmes = []

        # Parse AST
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
        except:
            pass

        # Find TODOs/FIXMEs
        for line_num, line in enumerate(content.splitlines(), 1):
            if "TODO" in line or "todo" in line:
                todos.append(f"{file_path.name}:{line_num}: {line.strip()}")
            if "FIXME" in line or "fixme" in line:
                fixmes.append(f"{file_path.name}:{line_num}: {line.strip()}")

        # Check if it's a stub (minimal implementation, lots of ... or pass)
        is_stub = (
            content.count("pass") > 3 or
            content.count("...") > 3 or
            content.count("NotImplementedError") > 0
        )

        return imports, [], classes, functions, todos, fixmes, is_stub

    def _analyze_javascript(
        self,
        content: str,
        file_path: Path
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str], bool]:
        """Analyze JavaScript/TypeScript file"""
        imports = []
        exports = []
        classes = []
        functions = []
        todos = []
        fixmes = []

        # Find imports
        import_patterns = [
            r'import\s+{?([^}\n]+)}?\s+from\s+["\']([^"\']+)["\']',
            r'import\s+["\']([^"\']+)["\']',
            r'require\(["\']([^"\']+)["\']\)',
        ]

        for pattern in import_patterns:
            for match in re.finditer(pattern, content):
                imports.append(match.group(2) if match.lastindex >= 2 else match.group(1))

        # Find exports
        export_patterns = [
            r'export\s+(?:default\s+)?(?:class|function|const|let)\s+(\w+)',
            r'export\s+{\s*([^}\n]+)\s*}',
        ]

        for pattern in export_patterns:
            for match in re.finditer(pattern, content):
                exports.append(match.group(1))

        # Find classes and functions
        classes.extend(re.findall(r'class\s+(\w+)', content))
        functions.extend(re.findall(r'function\s+(\w+)', content))
        functions.extend(re.findall(r'const\s+(\w+)\s*=\s*(?:async\s+)?\(?=>', content))

        # Find TODOs
        for line_num, line in enumerate(content.splitlines(), 1):
            if "TODO" in line or "todo" in line:
                todos.append(f"{file_path.name}:{line_num}: {line.strip()}")
            if "FIXME" in line or "fixme" in line:
                fixmes.append(f"{file_path.name}:{line_num}: {line.strip()}")

        # Check for stub
        is_stub = (
            "TODO" in content * 5 > len(content.splitlines()) or
            content.count("throw new Error") > len(content.splitlines()) / 2
        )

        return imports, exports, classes, functions, todos, fixmes, is_stub

    def _detect_component_type(
        self,
        file_path: Path,
        content: str,
        classes: List[str],
        functions: List[str],
    ) -> Optional[ComponentType]:
        """Detect the type of component based on path and content"""
        path_str = str(file_path).lower()

        # Check path for hints
        if "router" in path_str or "routers" in path_str:
            return ComponentType.ROUTER
        if "model" in path_str or "models" in path_str:
            return ComponentType.MODEL
        if "service" in path_str or "services" in path_str:
            return ComponentType.SERVICE
        if "schema" in path_str or "schemas" in path_str:
            return ComponentType.SCHEMA
        if "component" in path_str or "components" in path_str:
            return ComponentType.COMPONENT
        if "test" in path_str:
            return ComponentType.TEST

        # Check content for hints
        if content:
            if "APIRouter" in content or "router" in content:
                return ComponentType.ROUTER
            if "BaseModel" in content or "Schema" in content:
                return ComponentType.SCHEMA
            if "React" in content or "Component" in content:
                return ComponentType.COMPONENT

        return None

    def _categorize_features(self, files: List[SourceFile]) -> Dict[str, FeatureAnalysis]:
        """Categorize files into features"""
        features = {}

        for file in files:
            path_str = str(file.path).lower()

            # Determine which feature this file belongs to
            assigned_features = set()
            for feature_name, patterns in self.feature_patterns.items():
                for pattern in patterns:
                    if pattern in path_str:
                        assigned_features.add(feature_name)
                        break

            # If no feature matched, try to guess from content
            if not assigned_features:
                if "page" in path_str or "app/" in path_str:
                    assigned_features.add("frontend")

            # Add file to each assigned feature
            for feature_name in assigned_features:
                if feature_name not in features:
                    features[feature_name] = FeatureAnalysis(
                        name=feature_name,
                        status="partial",
                        files=[],
                        todos_count=0,
                    )

                features[feature_name].files.append(file)
                features[feature_name].todos_count += len(file.todos)

        # Determine status of each feature
        for feature_name, analysis in features.items():
            if not analysis.files:
                analysis.status = "missing"
            elif all(f.is_stub for f in analysis.files):
                analysis.status = "stub"
            elif analysis.todos_count > len(analysis.files) * 2:
                analysis.status = "partial"
            else:
                analysis.status = "complete"

        return features

    def find_gaps(
        self,
        analysis: ProjectAnalysis,
        expected_features: List[str],
    ) -> List[str]:
        """
        Find missing features based on what's expected.

        Returns:
            List of missing feature names
        """
        missing = []

        for feature in expected_features:
            if feature not in analysis.features:
                missing.append(feature)
            else:
                status = analysis.features[feature].status
                if status in ("missing", "stub"):
                    missing.append(f"{feature} ({status})")

        return missing

    def get_file_summary(self, file_path: Path) -> str:
        """Get a detailed summary of a specific file"""
        source_file = self._analyze_file(file_path, scan_content=True)

        lines = [
            f"File: {source_file.path.relative_to(source_file.path.parents[2] if len(source_file.path.parents) > 2 else source_file.path)}",
            f"Type: {source_file.file_type.value}",
            f"Lines: {source_file.lines_of_code}",
            f"Size: {source_file.size_bytes} bytes",
            "",
        ]

        if source_file.classes:
            lines.append(f"Classes: {', '.join(source_file.classes)}")
        if source_file.functions:
            lines.append(f"Functions: {', '.join(source.functions[:10])}")
            if len(source_file.functions) > 10:
                lines.append(f"  ... and {len(source_file.functions) - 10} more")
        if source_file.todos:
            lines.append(f"TODOs: {len(source_file.todos)}")
            for todo in source_file.todos[:3]:
                lines.append(f"  - {todo}")
        if source_file.is_stub:
            lines.append("⚠️ Appears to be a stub/incomplete")

        return "\n".join(lines)


# Convenience function
def analyze_project(project_path: str) -> ProjectAnalysis:
    """
    Quick convenience function to analyze a project.

    Example:
        analysis = analyze_project("/path/to/project")
        print(analysis.get_summary())
    """
    analyzer = ProjectAnalyzer()
    return analyzer.analyze_project(Path(project_path))
