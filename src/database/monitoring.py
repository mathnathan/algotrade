# src/database/monitoring.py
"""
Comprehensive database monitoring for trading systems.
This provides the observability needed to maintain optimal performance.
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Metrics for individual database queries."""

    query_type: str
    execution_time: float
    rows_affected: int
    timestamp: datetime
    success: bool
    error_message: str | None = None


@dataclass
class ConnectionMetrics:
    """Metrics for database connection usage."""

    active_connections: int = 0
    total_connections: int = 0
    connection_errors: int = 0
    average_session_duration: float = 0.0
    last_health_check: datetime | None = None


class DatabaseMetrics:
    """
    Comprehensive metrics collection for database operations.

    This class tracks all the performance indicators that matter for
    trading systems where database latency directly impacts profitability.
    """

    def __init__(self, retention_minutes: int = 60):
        self.retention_period = timedelta(minutes=retention_minutes)

        # Query performance tracking
        self.query_history: deque[QueryMetrics] = deque(maxlen=10000)
        self.slow_queries: deque[QueryMetrics] = deque(maxlen=1000)
        self.query_types: defaultdict[str, list[float]] = defaultdict(list)

        # Connection tracking
        self.connection_metrics = ConnectionMetrics()
        self.session_durations: deque[float] = deque(maxlen=1000)

        # Health monitoring
        self.health_checks: deque[dict[str, Any]] = deque(maxlen=100)
        self.error_rates: defaultdict[str, int] = defaultdict(int)

        # Performance baselines
        self.baseline_query_times: dict[str, float] = {}
        self.performance_degradation_threshold = 2.0  # 2x slower than baseline

    def record_query(
        self,
        query_type: str,
        execution_time: float,
        rows_affected: int = 0,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        """Record metrics for a database query."""

        metrics = QueryMetrics(
            query_type=query_type,
            execution_time=execution_time,
            rows_affected=rows_affected,
            timestamp=datetime.now(),
            success=success,
            error_message=error,
        )

        self.query_history.append(metrics)
        self.query_types[query_type].append(execution_time)

        # Track slow queries for investigation
        if execution_time > 0.1:  # Configurable slow query threshold
            self.slow_queries.append(metrics)
            logger.warning(f"Slow query detected: {query_type} took {execution_time:.3f}s")

        # Update performance baselines
        self._update_baseline(query_type, execution_time)

        # Check for performance degradation
        self._check_performance_degradation(query_type, execution_time)

    def record_session_created(self) -> None:
        """Record that a new database session was created."""
        self.connection_metrics.active_connections += 1
        self.connection_metrics.total_connections += 1

    def record_session_closed(self, duration: float) -> None:
        """Record that a database session was closed."""
        self.connection_metrics.active_connections -= 1
        self.session_durations.append(duration)

        # Update average session duration
        if self.session_durations:
            self.connection_metrics.average_session_duration = sum(self.session_durations) / len(
                self.session_durations
            )

    def record_session_error(self) -> None:
        """Record that a database session encountered an error."""
        self.connection_metrics.connection_errors += 1
        self.error_rates["session_errors"] += 1

    def record_health_check_success(self, duration: float) -> None:
        """Record a successful health check."""
        self.health_checks.append(
            {"timestamp": datetime.now(), "success": True, "duration": duration}
        )
        self.connection_metrics.last_health_check = datetime.now()

    def record_health_check_failure(self) -> None:
        """Record a failed health check."""
        self.health_checks.append({"timestamp": datetime.now(), "success": False, "duration": None})
        self.error_rates["health_check_failures"] += 1

    def _update_baseline(self, query_type: str, execution_time: float) -> None:
        """Update performance baseline for query type."""
        if query_type not in self.baseline_query_times:
            self.baseline_query_times[query_type] = execution_time
        else:
            # Use exponential moving average for baseline
            alpha = 0.1  # Smoothing factor
            current_baseline = self.baseline_query_times[query_type]
            self.baseline_query_times[query_type] = (
                alpha * execution_time + (1 - alpha) * current_baseline
            )

    def _check_performance_degradation(self, query_type: str, execution_time: float) -> None:
        """Check if query performance has degraded significantly."""
        if query_type in self.baseline_query_times:
            baseline = self.baseline_query_times[query_type]
            if execution_time > baseline * self.performance_degradation_threshold:
                logger.warning(
                    f"Performance degradation detected: {query_type} took {execution_time:.3f}s "
                    f"(baseline: {baseline:.3f}s, {execution_time / baseline:.1f}x slower)"
                )

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary for monitoring dashboards."""

        # Calculate query statistics
        recent_queries = [
            q for q in self.query_history if datetime.now() - q.timestamp < timedelta(minutes=5)
        ]

        if recent_queries:
            avg_query_time = sum(q.execution_time for q in recent_queries) / len(recent_queries)
            max_query_time = max(q.execution_time for q in recent_queries)
            success_rate = sum(1 for q in recent_queries if q.success) / len(recent_queries)
        else:
            avg_query_time = max_query_time = success_rate = 0

        # Calculate health statistics
        recent_health_checks = [
            h for h in self.health_checks if datetime.now() - h["timestamp"] < timedelta(minutes=5)
        ]
        health_success_rate = sum(1 for h in recent_health_checks if h["success"]) / max(
            1, len(recent_health_checks)
        )

        return {
            "query_performance": {
                "average_query_time": avg_query_time,
                "max_query_time": max_query_time,
                "queries_per_minute": len(recent_queries),
                "success_rate": success_rate,
                "slow_queries_count": len([q for q in recent_queries if q.execution_time > 0.1]),
            },
            "connection_health": {
                "active_connections": self.connection_metrics.active_connections,
                "total_connections": self.connection_metrics.total_connections,
                "connection_errors": self.connection_metrics.connection_errors,
                "average_session_duration": self.connection_metrics.average_session_duration,
                "health_success_rate": health_success_rate,
            },
            "performance_baselines": dict(self.baseline_query_times),
            "error_rates": dict(self.error_rates),
        }

    def get_slow_queries_report(self) -> list[dict[str, Any]]:
        """Get detailed report of slow queries for optimization."""

        def _create_empty_stats() -> dict[str, Any]:
            return {
                "count": 0,
                "total_time": 0,
                "max_time": 0,
                "avg_time": 0,
                "recent_examples": [],
            }

        slow_query_summary: defaultdict[str, dict[str, Any]] = defaultdict(_create_empty_stats)

        for query in self.slow_queries:
            query_type = query.query_type
            slow_query_summary[query_type]["count"] += 1
            slow_query_summary[query_type]["total_time"] += query.execution_time
            slow_query_summary[query_type]["max_time"] = max(
                slow_query_summary[query_type]["max_time"], query.execution_time
            )

            # Keep recent examples for investigation
            if len(slow_query_summary[query_type]["recent_examples"]) < 3:
                slow_query_summary[query_type]["recent_examples"].append(
                    {
                        "timestamp": query.timestamp.isoformat(),
                        "execution_time": query.execution_time,
                        "rows_affected": query.rows_affected,
                    }
                )

        # Calculate averages
        for query_type, stats in slow_query_summary.items():
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["query_type"] = query_type

        return list(slow_query_summary.values())


class QueryProfiler:
    """
    Context manager for profiling individual database operations.

    This makes it easy to track performance of specific trading operations
    and identify bottlenecks in your algorithmic trading pipeline.
    """

    def __init__(self, metrics: DatabaseMetrics, operation_name: str):
        self.metrics = metrics
        self.operation_name = operation_name
        self.start_time = None
        self.rows_affected = 0

    async def __aenter__(self):
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        success = exc_type is None
        error_message = str(exc_val) if exc_val else None

        self.metrics.record_query(
            query_type=self.operation_name,
            execution_time=execution_time,
            rows_affected=self.rows_affected,
            success=success,
            error=error_message,
        )

        return False  # Don't suppress exceptions

    def set_rows_affected(self, count: int) -> None:
        """Set the number of rows affected by this operation."""
        self.rows_affected = count
