"""
Monitoring Capability for PM Agent

Provides error monitoring and metrics integration including:
- Sentry error tracking
- Prometheus metrics scraping
- Log aggregation
- Performance monitoring
- Alert routing to notifications

Supports:
- Sentry (error tracking)
- Prometheus (metrics)
- Custom log forwarding
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp

from pm_agent.logger import ThoughtLogger


class MonitorType(Enum):
    """Types of monitoring"""
    ERRORS = "errors"
    METRICS = "metrics"
    LOGS = "logs"
    PERFORMANCE = "performance"
    UPTIME = "uptime"


class SeverityLevel(Enum):
    """Severity levels for events"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


class AlertStatus(Enum):
    """Alert states"""
    RESOLVED = "resolved"
    FIRING = "firing"
    PENDING = "pending"
    SILENCED = "silenced"


@dataclass
class ErrorEvent:
    """Error event data"""
    message: str
    level: SeverityLevel
    exception_type: Optional[str] = None
    stacktrace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    fingerprint: Optional[str] = None


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metric_type: str = "gauge"  # gauge, counter, histogram


@dataclass
class Alert:
    """Alert information"""
    alert_name: str
    status: AlertStatus
    severity: SeverityLevel
    message: str
    labels: Dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    url: Optional[str] = None


@dataclass
class SentryConfig:
    """Sentry configuration"""
    dsn: str
    environment: str = "development"
    release: Optional[str] = None
    sample_rate: float = 1.0
    traces_sample_rate: float = 0.1
    attach_stacktrace: bool = True


@dataclass
class PrometheusConfig:
    """Prometheus configuration"""
    pushgateway_url: Optional[str] = None
    scrape_url: Optional[str] = None
    job_name: str = "pm_agent"


class MonitoringError(Exception):
    """Base exception for monitoring operations"""
    pass


class SentryCapability:
    """
    Sentry error tracking integration.

    Provides error capture, event querying,
    and issue management for Sentry.
    """

    def __init__(self, config: SentryConfig, logger: Optional[ThoughtLogger] = None):
        """
        Initialize Sentry capability.

        Args:
            config: Sentry configuration
            logger: Optional thought logger for tracking operations
        """
        self.config = config
        self.logger = logger or ThoughtLogger("sentry_capability")
        self._session: Optional[aiohttp.ClientSession] = None

        # Extract organization and project from DSN
        self.dsn_parts = self._parse_dsn(config.dsn)

    def _parse_dsn(self, dsn: str) -> Dict[str, str]:
        """Parse Sentry DSN to extract components"""
        import re
        pattern = r'https://(\w+)@(\w+)/(\d+)'
        match = re.match(pattern, dsn)
        if match:
            return {
                "key": match.group(1),
                "host": match.group(2),
                "project_id": match.group(3)
            }
        return {}

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            auth = aiohttp.BasicAuth(self.dsn_parts.get("key", ""))
            self._session = aiohttp.ClientSession(auth=auth)
        return self._session

    async def capture_exception(
        self,
        error: ErrorEvent,
    ) -> Optional[str]:
        """
        Send an error to Sentry.

        Args:
            error: Error event to capture

        Returns:
            Event ID if successful, None otherwise
        """
        self.logger.log_thought(
            "capture_exception",
            f"Sending error to Sentry: {error.message[:100]}",
            {"level": error.level.value, "type": error.exception_type}
        )

        session = await self._get_session()

        url = f"https://{self.dsn_parts.get('host')}/api/{self.dsn_parts.get('project_id')}/store/"

        payload = {
            "message": error.message,
            "level": error.level.value,
            "platform": "python",
            "environment": self.config.environment,
            "tags": error.tags,
            "extra": error.context,
            "timestamp": error.timestamp.isoformat(),
        }

        if error.release:
            payload["release"] = error.release

        if error.stacktrace:
            payload["stacktrace"] = {
                "frames": self._parse_stacktrace(error.stacktrace)
            }

        if error.exception_type:
            payload["exception"] = {
                "values": [{
                    "type": error.exception_type,
                    "value": error.message,
                    "stacktrace": payload.get("stacktrace")
                }]
            }

        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    event_id = data.get("id")
                    self.logger.log_thought(
                        "exception_captured",
                        f"Error captured with ID: {event_id}",
                        {"event_id": event_id}
                    )
                    return event_id
        except Exception as e:
            self.logger.log_thought(
                "capture_failed",
                f"Failed to capture error: {e}",
                {"error": str(e)},
                level="warning"
            )

        return None

    async def capture_message(
        self,
        message: str,
        level: SeverityLevel = SeverityLevel.INFO,
        tags: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Send a message to Sentry.

        Args:
            message: Message to send
            level: Severity level
            tags: Optional tags

        Returns:
            Event ID if successful
        """
        error = ErrorEvent(
            message=message,
            level=level,
            tags=tags or {}
        )
        return await self.capture_exception(error)

    async def get_recent_errors(
        self,
        limit: int = 50,
        level: Optional[SeverityLevel] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get recent errors from Sentry.

        Args:
            limit: Maximum number of errors
            level: Filter by severity level
            since: Only errors after this time

        Returns:
            List of error events
        """
        # This requires Sentry API with auth token
        # For now, return placeholder
        return []

    def _parse_stacktrace(self, stacktrace: str) -> List[Dict]:
        """Parse stacktrace string into Sentry format"""
        frames = []
        for line in stacktrace.split("\n"):
            if "File " in line and "line " in line:
                # Parse frame
                import re
                match = re.search(r'File "([^"]+)", line (\d+)', line)
                if match:
                    frames.append({
                        "filename": match.group(1),
                        "lineno": int(match.group(2)),
                        "context_line": line,
                    })
        return list(reversed(frames))


class PrometheusCapability:
    """
    Prometheus metrics integration.

    Provides metric pushing and scraping for Prometheus.
    """

    def __init__(self, config: PrometheusConfig, logger: Optional[ThoughtLogger] = None):
        """
        Initialize Prometheus capability.

        Args:
            config: Prometheus configuration
            logger: Optional thought logger for tracking operations
        """
        self.config = config
        self.logger = logger or ThoughtLogger("prometheus_capability")
        self._metrics: Dict[str, List[MetricPoint]] = {}

    async def push_metric(
        self,
        metric: MetricPoint,
    ) -> bool:
        """
        Push a metric to Prometheus Pushgateway.

        Args:
            metric: Metric data point

        Returns:
            True if successful
        """
        if not self.config.pushgateway_url:
            self.logger.log_thought(
                "pushgateway_not_configured",
                "Pushgateway URL not configured, metric not pushed",
                {"metric": metric.name},
                level="warning"
            )
            return False

        self.logger.log_thought(
            "push_metric",
            f"Pushing metric: {metric.name}",
            {"value": metric.value, "labels": metric.labels}
        )

        # Store metric locally
        if metric.name not in self._metrics:
            self._metrics[metric.name] = []
        self._metrics[metric.name].append(metric)

        # Push to Pushgateway
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.config.pushgateway_url}/metrics/job/{self.config.job_name}"

                # Format metric for Prometheus
                labels_str = ",".join([f'{k}="{v}"' for k, v in metric.labels.items()])
                metric_line = f"{metric.name}{{{labels_str}}} {metric.value} {int(metric.timestamp.timestamp())}\n"

                async with session.post(url, data=metric_line) as response:
                    return response.status == 200

        except Exception as e:
            self.logger.log_thought(
                "push_failed",
                f"Failed to push metric: {e}",
                {"error": str(e)},
                level="warning"
            )
            return False

    async def push_metrics_batch(
        self,
        metrics: List[MetricPoint],
    ) -> int:
        """
        Push multiple metrics at once.

        Args:
            metrics: List of metric data points

        Returns:
            Number of metrics successfully pushed
        """
        if not self.config.pushgateway_url:
            return 0

        # Format all metrics
        lines = []
        for metric in metrics:
            labels_str = ",".join([f'{k}="{v}"' for k, v in metric.labels.items()])
            lines.append(
                f"{metric.name}{{{labels_str}}} {metric.value} "
                f"{int(metric.timestamp.timestamp())}"
            )

        data = "\n".join(lines) + "\n"

        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.config.pushgateway_url}/metrics/job/{self.config.job_name}"

                async with session.post(url, data=data) as response:
                    if response.status == 200:
                        return len(metrics)
        except Exception as e:
            self.logger.log_thought(
                "batch_push_failed",
                f"Failed to push metrics batch: {e}",
                {"error": str(e)},
                level="warning"
            )

        return 0

    def get_metric(
        self,
        name: str,
        since: Optional[datetime] = None,
    ) -> List[MetricPoint]:
        """
        Get stored metric points.

        Args:
            name: Metric name
            since: Only points after this time

        Returns:
            List of metric points
        """
        points = self._metrics.get(name, [])

        if since:
            points = [p for p in points if p.timestamp >= since]

        return points

    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Increment a counter metric.

        Args:
            name: Metric name
            value: Value to add
            labels: Metric labels
        """
        metric = MetricPoint(
            name=name,
            value=value,
            labels=labels or {},
            metric_type="counter"
        )
        self._metrics.setdefault(name, []).append(metric)

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Set a gauge metric value.

        Args:
            name: Metric name
            value: Current value
            labels: Metric labels
        """
        metric = MetricPoint(
            name=name,
            value=value,
            labels=labels or {},
            metric_type="gauge"
        )
        self._metrics[name] = [metric]  # Gauges replace previous value

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        Observe a histogram value.

        Args:
            name: Metric name
            value: Observed value
            labels: Metric labels
        """
        metric = MetricPoint(
            name=f"{name}_bucket",
            value=value,
            labels=labels or {},
            metric_type="histogram"
        )
        self._metrics.setdefault(name, []).append(metric)


class MonitoringCapability:
    """
    Unified monitoring capability combining error tracking
    and metrics collection.
    """

    def __init__(
        self,
        sentry_config: Optional[SentryConfig] = None,
        prometheus_config: Optional[PrometheusConfig] = None,
        logger: Optional[ThoughtLogger] = None,
    ):
        """
        Initialize unified monitoring capability.

        Args:
            sentry_config: Optional Sentry configuration
            prometheus_config: Optional Prometheus configuration
            logger: Optional thought logger for tracking operations
        """
        self.logger = logger or ThoughtLogger("monitoring_capability")

        self.sentry = SentryCapability(sentry_config) if sentry_config else None
        self.prometheus = PrometheusCapability(prometheus_config) if prometheus_config else None

    async def track_event(
        self,
        event: ErrorEvent,
    ) -> Optional[str]:
        """
        Track an error event.

        Args:
            event: Error event to track

        Returns:
            Event ID if sent to Sentry
        """
        if self.sentry:
            return await self.sentry.capture_exception(event)
        return None

    async def track_metric(
        self,
        metric: MetricPoint,
    ) -> bool:
        """
        Track a metric.

        Args:
            metric: Metric data point

        Returns:
            True if successfully tracked
        """
        if self.prometheus:
            return await self.prometheus.push_metric(metric)
        return False

    async def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all tracked metrics.

        Returns:
            Dictionary with metric summaries
        """
        if not self.prometheus:
            return {}

        summary = {}
        for name, points in self.prometheus._metrics.items():
            if points:
                values = [p.value for p in points]
                summary[name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "latest": values[-1] if values else None,
                }

        return summary

    def set_logger(self, logger: ThoughtLogger):
        """Set or update the logger"""
        self.logger = logger
        if self.sentry:
            self.sentry.logger = logger
        if self.prometheus:
            self.prometheus.logger = logger


# Convenience functions
async def track_error(
    dsn: str,
    message: str,
    exception_type: Optional[str] = None,
    stacktrace: Optional[str] = None,
    level: SeverityLevel = SeverityLevel.ERROR,
) -> Optional[str]:
    """
    Quick helper to send an error to Sentry.

    Example:
        event_id = await track_error(
            dsn="https://key@host/project_id",
            message="Database connection failed",
            exception_type="ConnectionError",
            stacktrace=traceback.format_exc()
        )
    """
    config = SentryConfig(dsn=dsn)
    monitoring = MonitoringCapability(sentry_config=config)

    error = ErrorEvent(
        message=message,
        exception_type=exception_type,
        stacktrace=stacktrace,
        level=level
    )

    return await monitoring.track_event(error)


def create_metrics(
    prometheus_url: str,
    job_name: str = "pm_agent",
) -> PrometheusCapability:
    """
    Quick helper to create a metrics capability.

    Example:
        metrics = create_metrics(
            prometheus_url="http://localhost:9091",
            job_name="my_app"
        )
        metrics.increment_counter("requests_total", labels={"endpoint": "/api"})
    """
    config = PrometheusConfig(pushgateway_url=prometheus_url, job_name=job_name)
    return PrometheusCapability(config)
