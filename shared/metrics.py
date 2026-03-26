"""
Prometheus metrics utilities shared across services.
Each service should mount /metrics using the prometheus_fastapi_instrumentator.
"""
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
import time

# --- Custom metrics ---

REQUEST_COUNT = Counter(
    "ai_platform_requests_total",
    "Total number of requests by service and endpoint",
    ["service", "endpoint", "status_code"],
)

INFERENCE_LATENCY = Histogram(
    "ai_platform_inference_latency_seconds",
    "Inference duration per service endpoint",
    ["service", "endpoint"],
)


def instrument_app(app, service_name: str):
    """Attach Prometheus FastAPI instrumentator and expose /metrics."""
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")


class LatencyTracker:
    """Context manager for measuring and recording inference latency."""

    def __init__(self, service: str, endpoint: str):
        self.service = service
        self.endpoint = endpoint
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        INFERENCE_LATENCY.labels(service=self.service, endpoint=self.endpoint).observe(elapsed)
