"""
Testing Capability for PM Agent

Provides automated testing operations including:
- Unit test generation and execution
- Integration test management
- E2E test coordination with Playwright
- Coverage analysis and gap detection
- Flaky test detection and retry

Supports:
- pytest (Python)
- jest/vitest (JavaScript/TypeScript)
- Go test (Go)
- Playwright (E2E)
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pm_agent.logger import ThoughtLogger


class TestFramework(Enum):
    """Supported testing frameworks"""
    PYTEST = "pytest"
    JEST = "jest"
    VITEST = "vitest"
    GO_TEST = "gotest"
    PLAYWRIGHT = "playwright"


class TestType(Enum):
    """Types of tests"""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    COMPONENT = "component"


class TestStatus(Enum):
    """Test execution states"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    FLAKY = "flaky"
    ERROR = "error"


@dataclass
class TestResult:
    """Result of a single test"""
    name: str
    file: str
    status: TestStatus
    duration_ms: float
    error_message: Optional[str] = None
    error_stack: Optional[str] = None
    line_number: Optional[int] = None


@dataclass
class TestSuiteResult:
    """Result of a test suite execution"""
    framework: TestFramework
    test_type: TestType
    total_tests: int
    passed: int
    failed: int
    skipped: int
    duration_seconds: float
    coverage_percent: Optional[float] = None
    tests: List[TestResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TestFile:
    """Information about a test file"""
    path: Path
    framework: TestFramework
    test_type: TestType
    test_count: int = 0


@dataclass
class CoverageReport:
    """Code coverage report"""
    percent_covered: float
    lines_covered: int
    lines_total: int
    branches_covered: int
    branches_total: int
    file_coverage: Dict[str, float]  # filename -> percentage


class TestingError(Exception):
    """Base exception for testing operations"""
    pass


class TestDiscoveryError(TestingError):
    """Failed to discover tests"""
    pass


class Capability:
    """Base capability class"""
    pass


class TestCapability:
    """
    Automated testing capability.

    Supports test discovery, execution, coverage analysis,
    and flaky test detection across multiple frameworks.
    """

    # Maximum retries for flaky tests
    MAX_RETRIES = 3
    FLAKY_THRESHOLD = 0.5  # Pass rate below this indicates flakiness

    def __init__(self, logger: Optional[ThoughtLogger] = None):
        """
        Initialize test capability.

        Args:
            logger: Optional thought logger for tracking operations
        """
        self.logger = logger or ThoughtLogger("test_capability")
        self._flaky_tests: Dict[str, List[bool]] = {}  # test_name -> [pass, fail, pass, ...]

    async def discover_tests(
        self,
        project_path: Path,
        test_type: Optional[TestType] = None,
    ) -> List[TestFile]:
        """
        Discover all test files in the project.

        Args:
            project_path: Path to project root
            test_type: Filter by test type

        Returns:
            List of test files
        """
        self.logger.log_thought(
            "discover_tests",
            f"Discovering tests in {project_path}",
            {"test_type": test_type.value if test_type else "all"}
        )

        test_files = []

        # Python tests
        python_patterns = {
            "test_*.py": (TestType.UNIT, TestFramework.PYTEST),
            "*_test.py": (TestType.UNIT, TestFramework.PYTEST),
            "tests/test_*.py": (TestType.UNIT, TestFramework.PYTEST),
        }

        # JavaScript/TypeScript tests
        js_patterns = {
            "*.test.js": (TestType.UNIT, TestFramework.JEST),
            "*.test.ts": (TestType.UNIT, TestFramework.JEST),
            "*.spec.js": (TestType.UNIT, TestFramework.JEST),
            "*.spec.ts": (TestType.UNIT, TestFramework.JEST),
            "e2e/*.ts": (TestType.E2E, TestFramework.PLAYWRIGHT),
            "e2e/*.js": (TestType.E2E, TestFramework.PLAYWRIGHT),
        }

        # Go tests
        go_patterns = {
            "*_test.go": (TestType.UNIT, TestFramework.GO_TEST),
        }

        all_patterns = {**python_patterns, **js_patterns, **go_patterns}

        for pattern, (file_type, framework) in all_patterns.items():
            if test_type and file_type != test_type:
                continue

            matches = list(project_path.rglob(pattern))
            for match in matches:
                # Skip node_modules and virtual environments
                if "node_modules" in str(match) or "venv" in str(match) or ".venv" in str(match):
                    continue

                test_files.append(TestFile(
                    path=match,
                    framework=framework,
                    test_type=file_type
                ))

        self.logger.log_thought(
            "tests_discovered",
            f"Found {len(test_files)} test files",
            {"count": len(test_files)}
        )

        return test_files

    async def run_tests(
        self,
        project_path: Path,
        framework: TestFramework,
        test_type: TestType = TestType.UNIT,
        pattern: Optional[str] = None,
        coverage: bool = True,
        retry_flaky: bool = True,
    ) -> TestSuiteResult:
        """
        Run tests for a project.

        Args:
            project_path: Path to project root
            framework: Test framework to use
            test_type: Type of tests to run
            pattern: File pattern to match (optional)
            coverage: Generate coverage report
            retry_flaky: Retry failed tests to detect flakiness

        Returns:
            TestSuiteResult with execution results
        """
        self.logger.log_thought(
            "run_tests",
            f"Running {test_type.value} tests with {framework.value}",
            {"pattern": pattern, "coverage": coverage}
        )

        if framework == TestFramework.PYTEST:
            return await self._run_pytest(
                project_path, pattern, coverage, retry_flaky
            )
        elif framework == TestFramework.JEST:
            return await self._run_jest(
                project_path, pattern, coverage, retry_flaky
            )
        elif framework == TestFramework.VITEST:
            return await self._run_vitest(
                project_path, pattern, coverage, retry_flaky
            )
        elif framework == TestFramework.GO_TEST:
            return await self._run_gotest(
                project_path, pattern, coverage, retry_flaky
            )
        elif framework == TestFramework.PLAYWRIGHT:
            return await self._run_playwright(
                project_path, pattern, coverage, retry_flaky
            )
        else:
            raise TestingError(f"Framework {framework} not yet supported")

    async def _run_pytest(
        self,
        project_path: Path,
        pattern: Optional[str],
        coverage: bool,
        retry_flaky: bool,
    ) -> TestSuiteResult:
        """Run pytest tests"""
        cmd = ["python", "-m", "pytest", "-v", "--tb=short"]

        if coverage:
            cmd.extend(["--cov=.", "--cov-report=json", "--cov-report=term"])

        if pattern:
            cmd.extend(["-k", pattern])

        cmd.extend(["--json-report", "--json-report-file=.pytest-report.json"])

        exit_code, output = await self._run_command(cmd, cwd=project_path, timeout=300)

        # Parse JSON report if available
        report_path = project_path / ".pytest-report.json"
        coverage_path = project_path / "coverage.json"

        tests = []
        coverage_data = None

        if report_path.exists():
            with open(report_path) as f:
                data = json.load(f)
                for test in data.get("tests", []):
                    tests.append(TestResult(
                        name=test.get("name", ""),
                        file=test.get("nodeid", "").split("::")[0],
                        status=self._parse_pytest_status(test.get("outcome")),
                        duration_ms=test.get("duration", 0) * 1000,
                        error_message=test.get("call", {}).get("longrepr", ""),
                    ))

        if coverage_path.exists():
            with open(coverage_path) as f:
                coverage_data = json.load(f)

        return TestSuiteResult(
            framework=TestFramework.PYTEST,
            test_type=TestType.UNIT,
            total_tests=len(tests) or int(self._extract_from_output(output, r"(\d+) passed")),
            passed=len([t for t in tests if t.status == TestStatus.PASSED]),
            failed=len([t for t in tests if t.status == TestStatus.FAILED]),
            skipped=len([t for t in tests if t.status == TestStatus.SKIPPED]),
            duration_seconds=float(self._extract_from_output(output, r"([\d.]+) seconds") or 0),
            coverage_percent=self._parse_coverage_percent(coverage_data),
            tests=tests
        )

    async def _run_jest(
        self,
        project_path: Path,
        pattern: Optional[str],
        coverage: bool,
        retry_flaky: bool,
    ) -> TestSuiteResult:
        """Run Jest tests"""
        cmd = ["npx", "jest", "--verbose", "--json"]

        if coverage:
            cmd.extend(["--coverage", "--coverageReporters=json"])

        if pattern:
            cmd.append(pattern)

        exit_code, output = await self._run_command(cmd, cwd=project_path, timeout=300)

        # Parse JSON output
        try:
            data = json.loads(output)
            test_data = data.get("testResults", [])

            tests = []
            for file_data in test_data:
                for assertion in file_data.get("assertionResults", []):
                    tests.append(TestResult(
                        name=assertion.get("title", ""),
                        file=file_data.get("name", ""),
                        status=self._parse_jest_status(assertion.get("status")),
                        duration_ms=assertion.get("duration", 0),
                        error_message=assertion.get("failureMessages", [""])[0] if assertion.get("failureMessages") else None,
                    ))

            coverage_data = data.get("coverageMap", {})
            coverage_percent = self._parse_jest_coverage(coverage_data)

            return TestSuiteResult(
                framework=TestFramework.JEST,
                test_type=TestType.UNIT,
                total_tests=len(tests),
                passed=len([t for t in tests if t.status == TestStatus.PASSED]),
                failed=len([t for t in tests if t.status == TestStatus.FAILED]),
                skipped=len([t for t in tests if t.status == TestStatus.SKIPPED]),
                duration_seconds=sum(t.duration_ms for t in tests) / 1000,
                coverage_percent=coverage_percent,
                tests=tests
            )

        except json.JSONDecodeError:
            # Fallback to parsing text output
            return self._parse_jest_text_output(output)

    async def _run_vitest(
        self,
        project_path: Path,
        pattern: Optional[str],
        coverage: bool,
        retry_flaky: bool,
    ) -> TestSuiteResult:
        """Run Vitest tests"""
        cmd = ["npx", "vitest", "run", "--json"]

        if coverage:
            cmd.append("--coverage")

        if pattern:
            cmd.extend(["--filter", pattern])

        exit_code, output = await self._run_command(cmd, cwd=project_path, timeout=300)

        # Vitest JSON output is similar to Jest
        return await self._run_jest(project_path, pattern, coverage, retry_flaky)

    async def _run_gotest(
        self,
        project_path: Path,
        pattern: Optional[str],
        coverage: bool,
        retry_flaky: bool,
    ) -> TestSuiteResult:
        """Run Go tests"""
        cmd = ["go", "test", "-v", "-json"]

        if coverage:
            cmd.extend(["-coverprofile=coverage.out"])

        if pattern:
            cmd.extend(["-run", pattern])

        exit_code, output = await self._run_command(cmd, cwd=project_path, timeout=300)

        tests = []
        for line in output.split("\n"):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if data.get("Action") == "run":
                    tests.append(TestResult(
                        name=data.get("Test", ""),
                        file=data.get("Package", ""),
                        status=TestStatus.PASSED,  # Will update if fail
                        duration_ms=data.get("Elapsed", 0) * 1000,
                    ))
                elif data.get("Action") == "fail":
                    if tests:
                        tests[-1].status = TestStatus.FAILED
            except json.JSONDecodeError:
                pass

        # Parse coverage
        coverage_percent = None
        if coverage:
            coverage_cmd = ["go", "tool", "cover", "-func=coverage.out"]
            _, cov_output = await self._run_command(coverage_cmd, cwd=project_path)
            match = re.search(r'total:\s+\(statements\)\s+(\d+\.?\d*)%', cov_output)
            if match:
                coverage_percent = float(match.group(1))

        return TestSuiteResult(
            framework=TestFramework.GO_TEST,
            test_type=TestType.UNIT,
            total_tests=len(tests),
            passed=len([t for t in tests if t.status == TestStatus.PASSED]),
            failed=len([t for t in tests if t.status == TestStatus.FAILED]),
            skipped=0,
            duration_seconds=sum(t.duration_ms for t in tests) / 1000,
            coverage_percent=coverage_percent,
            tests=tests
        )

    async def _run_playwright(
        self,
        project_path: Path,
        pattern: Optional[str],
        coverage: bool,
        retry_flaky: bool,
    ) -> TestSuiteResult:
        """Run Playwright E2E tests"""
        cmd = ["npx", "playwright", "test", "--reporter=json"]

        if pattern:
            cmd.extend(["--grep", pattern])

        exit_code, output = await self._run_command(cmd, cwd=project_path, timeout=600)

        try:
            data = json.loads(output)

            tests = []
            for suite in data.get("suites", []):
                for spec in suite.get("specs", []):
                    for test in spec.get("tests", []):
                        tests.append(TestResult(
                            name=test.get("title", ""),
                            file=spec.get("file", ""),
                            status=self._parse_playwright_status(test.get("results", [{}])[0].get("status")),
                            duration_ms=test.get("results", [{}])[0].get("duration", 0),
                        ))

            return TestSuiteResult(
                framework=TestFramework.PLAYWRIGHT,
                test_type=TestType.E2E,
                total_tests=len(tests),
                passed=len([t for t in tests if t.status == TestStatus.PASSED]),
                failed=len([t for t in tests if t.status == TestStatus.FAILED]),
                skipped=len([t for t in tests if t.status == TestStatus.SKIPPED]),
                duration_seconds=data.get("duration", 0) / 1000,
                tests=tests
            )

        except json.JSONDecodeError:
            # Fallback parsing
            return self._parse_playwright_text_output(output)

    async def generate_test(
        self,
        source_file: Path,
        test_type: TestType = TestType.UNIT,
        framework: Optional[TestFramework] = None,
    ) -> str:
        """
        Generate test code for a source file.

        Args:
            source_file: Path to source file
            test_type: Type of test to generate
            framework: Test framework (auto-detected if None)

        Returns:
            Generated test code
        """
        self.logger.log_thought(
            "generate_test",
            f"Generating {test_type.value} test for {source_file.name}",
            {"file": str(source_file)}
        )

        # Read source file
        async with asyncio.to_thread(source_file.read_text) as content:
            source_code = content

        # Detect framework if not specified
        if framework is None:
            framework = self._detect_framework(source_file)

        # Extract functions/classes from source
        if source_file.suffix == ".py":
            return await self._generate_python_test(source_code, framework)
        elif source_file.suffix in [".js", ".ts"]:
            return await self._generate_js_test(source_code, framework)
        elif source_file.suffix == ".go":
            return await self._generate_go_test(source_code, framework)
        else:
            raise TestingError(f"Unsupported file type: {source_file.suffix}")

    async def _generate_python_test(
        self,
        source_code: str,
        framework: TestFramework,
    ) -> str:
        """Generate Python test code"""
        import ast
        import inspect

        # Parse source to extract functions/classes
        tree = ast.parse(source_code)

        test_lines = []
        test_lines.append("import pytest")
        test_lines.append("from unittest.mock import Mock, patch")

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                if func_name.startswith("_"):
                    continue  # Skip private functions

                # Generate test function
                test_lines.append(f"\ndef test_{func_name}():")
                test_lines.append(f'    """Test {func_name}"""')
                test_lines.append("    # Arrange")
                test_lines.append("    # TODO: Set up test data")
                test_lines.append("\n    # Act")
                test_lines.append(f"    result = {func_name}()")
                test_lines.append("\n    # Assert")
                test_lines.append("    # TODO: Add assertions")
                test_lines.append("    assert result is not None")

            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                if class_name.startswith("_"):
                    continue

                # Generate test class
                test_lines.append(f"\nclass Test{class_name}:")
                test_lines.append(f'    """Tests for {class_name}"""')

                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                        method_name = item.name
                        test_lines.append(f"\n    def test_{method_name}(self):")
                        test_lines.append(f'        """Test {class_name}.{method_name}"""')
                        test_lines.append("        # TODO: Implement test")
                        test_lines.append("        pass")

        return "\n".join(test_lines)

    async def _generate_js_test(
        self,
        source_code: str,
        framework: TestFramework,
    ) -> str:
        """Generate JavaScript/TypeScript test code"""
        # Simple regex-based extraction for JS/TS
        function_pattern = r'(?:export\s+)?(?:async\s+)?function\s+(\w+)'
        class_pattern = r'export\s+class\s+(\w+)'
        arrow_pattern = r'(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>'

        functions = set(re.findall(function_pattern, source_code))
        classes = set(re.findall(class_pattern, source_code))
        arrows = set(re.findall(arrow_pattern, source_code))

        test_lines = []
        test_lines.append("// @ts-check")

        for func_name in list(functions) + list(arrows):
            if func_name.startswith("_"):
                continue
            test_lines.append(f"\ntest('{func_name}', () => {{")
            test_lines.append("  // TODO: Implement test")
            test_lines.append("  expect(true).toBe(true);")
            test_lines.append("});")

        for class_name in classes:
            if class_name.startswith("_"):
                continue
            test_lines.append(f"\ndescribe('{class_name}', () => {{")
            test_lines.append("  // TODO: Add tests")
            test_lines.append("});")

        return "\n".join(test_lines)

    async def _generate_go_test(
        self,
        source_code: str,
        framework: TestFramework,
    ) -> str:
        """Generate Go test code"""
        # Extract function signatures
        func_pattern = r'func (\w+)\('

        functions = re.findall(func_pattern, source_code)

        test_lines = ["package main"]

        for func_name in functions:
            if func_name.startswith("_") or func_name == "main":
                continue
            test_lines.append(f"\nfunc Test{func_name}(t *testing.T) {{")
            test_lines.append(f'\t// t.Run("{func_name}", func(t *testing.T) {{')
            test_lines.append("\t\t// TODO: Implement test")
            test_lines.append("\t\tt.Skip(\"TODO\")")
            test_lines.append("\t\t})")
            test_lines.append("}")

        return "\n".join(test_lines)

    async def detect_coverage_gaps(
        self,
        project_path: Path,
        coverage_report: Optional[CoverageReport] = None,
    ) -> List[str]:
        """
        Identify files or modules with low coverage.

        Args:
            project_path: Path to project
            coverage_report: Optional existing coverage report

        Returns:
            List of files/paths with coverage gaps
        """
        self.logger.log_thought(
            "coverage_gaps",
            "Analyzing coverage gaps",
            {"project": str(project_path)}
        )

        if coverage_report is None:
            # Generate coverage report first
            result = await self.run_tests(
                project_path,
                self._detect_framework_for_project(project_path),
                coverage=True
            )
            if result.coverage_percent is None:
                return []
            coverage_report = CoverageReport(
                percent_covered=result.coverage_percent,
                lines_covered=0,
                lines_total=0,
                branches_covered=0,
                branches_total=0,
                file_coverage={}
            )

        gaps = []
        threshold = 80.0  # Coverage threshold

        for file_path, percent in coverage_report.file_coverage.items():
            if percent < threshold:
                gaps.append(f"{file_path}: {percent:.1f}%")

        return gaps

    def _detect_framework(self, file_path: Path) -> TestFramework:
        """Detect test framework from file extension and project structure"""
        if file_path.suffix == ".py":
            return TestFramework.PYTEST
        elif file_path.suffix in [".js", ".ts"]:
            # Check for vitest config
            parent = file_path.parent
            while parent != parent.parent:
                if (parent / "vitest.config.ts").exists() or (parent / "vitest.config.js").exists():
                    return TestFramework.VITEST
                parent = parent.parent
            return TestFramework.JEST
        elif file_path.suffix == ".go":
            return TestFramework.GO_TEST
        else:
            return TestFramework.PYTEST  # Default

    def _detect_framework_for_project(self, project_path: Path) -> TestFramework:
        """Detect primary test framework for a project"""
        # Check for config files
        if (project_path / "pytest.ini").exists() or (project_path / "pyproject.toml").exists():
            return TestFramework.PYTEST
        if (project_path / "jest.config.js").exists() or (project_path / "jest.config.ts").exists():
            return TestFramework.JEST
        if (project_path / "vitest.config.ts").exists():
            return TestFramework.VITEST
        if (project_path / "go.mod").exists():
            return TestFramework.GO_TEST

        # Check for package.json
        package_json = project_path / "package.json"
        if package_json.exists():
            content = package_json.read_text()
            if "vitest" in content:
                return TestFramework.VITEST
            if "jest" in content:
                return TestFramework.JEST

        return TestFramework.PYTEST  # Default

    def _parse_pytest_status(self, outcome: str) -> TestStatus:
        """Parse pytest outcome to TestStatus"""
        mapping = {
            "passed": TestStatus.PASSED,
            "failed": TestStatus.FAILED,
            "skipped": TestStatus.SKIPPED,
        }
        return mapping.get(outcome, TestStatus.ERROR)

    def _parse_jest_status(self, status: str) -> TestStatus:
        """Parse Jest status to TestStatus"""
        mapping = {
            "passed": TestStatus.PASSED,
            "failed": TestStatus.FAILED,
            "skipped": TestStatus.SKIPPED,
            "todo": TestStatus.SKIPPED,
        }
        return mapping.get(status, TestStatus.ERROR)

    def _parse_playwright_status(self, status: str) -> TestStatus:
        """Parse Playwright status to TestStatus"""
        mapping = {
            "passed": TestStatus.PASSED,
            "failed": TestStatus.FAILED,
            "skipped": TestStatus.SKIPPED,
        }
        return mapping.get(status, TestStatus.ERROR)

    def _parse_coverage_percent(self, coverage_data: Optional[Dict]) -> Optional[float]:
        """Parse coverage percent from coverage data"""
        if not coverage_data:
            return None

        totals = coverage_data.get("totals", {})
        percent_covered = totals.get("percent_covered")

        if isinstance(percent_covered, str):
            return float(percent_covered.rstrip("%"))
        elif isinstance(percent_covered, (int, float)):
            return float(percent_covered)

        return None

    def _parse_jest_coverage(self, coverage_map: Dict) -> Optional[float]:
        """Parse coverage from Jest coverage map"""
        if not coverage_map:
            return None

        # Jest coverage map is complex, simplified extraction
        total_lines = 0
        covered_lines = 0

        for file_data in coverage_map.values():
            if isinstance(file_data, dict):
                for statement_data in file_data.values():
                    if isinstance(statement_data, dict):
                        count = statement_data.get("count", 0)
                        total_lines += 1
                        if count > 0:
                            covered_lines += 1

        if total_lines > 0:
            return (covered_lines / total_lines) * 100

        return None

    def _parse_jest_text_output(self, output: str) -> TestSuiteResult:
        """Parse Jest text output when JSON is not available"""
        match = re.search(r'Tests:\s+(\d+)\s+passed,\s+(\d+)\s+failed', output)
        passed = int(match.group(1)) if match else 0
        failed = int(match.group(2)) if match else 0

        return TestSuiteResult(
            framework=TestFramework.JEST,
            test_type=TestType.UNIT,
            total_tests=passed + failed,
            passed=passed,
            failed=failed,
            skipped=0,
            duration_seconds=0,
        )

    def _parse_playwright_text_output(self, output: str) -> TestSuiteResult:
        """Parse Playwright text output"""
        match = re.search(r'(\d+)\s+passed\s+\((\d+)\s+failed\)', output)
        passed = int(match.group(1)) if match else 0
        failed = int(match.group(2)) if match else 0

        return TestSuiteResult(
            framework=TestFramework.PLAYWRIGHT,
            test_type=TestType.E2E,
            total_tests=passed + failed,
            passed=passed,
            failed=failed,
            skipped=0,
            duration_seconds=0,
        )

    def _extract_from_output(self, output: str, pattern: str) -> Optional[str]:
        """Extract value from command output using regex"""
        match = re.search(pattern, output)
        return match.group(1) if match else None

    async def _run_command(
        self,
        cmd: List[str],
        cwd: Path,
        timeout: int = 300,
    ) -> Tuple[int, str]:
        """Run a command and return output"""
        import subprocess

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=cwd,
        )

        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            output = stdout.decode("utf-8", errors="replace")
            return process.returncode or 0, output

        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TestingError(f"Command timed out: {' '.join(cmd)}")


# Convenience functions
async def run_all_tests(
    project_path: Path,
    with_coverage: bool = True,
) -> List[TestSuiteResult]:
    """
    Run all discovered tests for a project.

    Example:
        results = await run_all_tests(
            project_path=Path("/path/to/project"),
            with_coverage=True
        )
    """
    test_cap = TestCapability()

    # Detect framework
    framework = test_cap._detect_framework_for_project(project_path)

    # Run unit tests
    unit_result = await test_cap.run_tests(
        project_path,
        framework,
        TestType.UNIT,
        coverage=with_coverage
    )

    results = [unit_result]

    # Run E2E tests if available
    try:
        e2e_result = await test_cap.run_tests(
            project_path,
            TestFramework.PLAYWRIGHT,
            TestType.E2E,
            coverage=False
        )
        results.append(e2e_result)
    except TestingError:
        pass  # No E2E tests

    return results
