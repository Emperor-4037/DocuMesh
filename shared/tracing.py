"""
OpenTelemetry tracing setup for all microservices.
Call setup_tracing(service_name) on application startup.
"""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter


def setup_tracing(service_name: str) -> trace.Tracer:
    """Configure OTLP trace exporter and return a tracer instance."""
    provider = TracerProvider()

    # OTLP exporter — by default sends to http://localhost:4318/v1/traces
    # Override OTEL_EXPORTER_OTLP_ENDPOINT env var in production.
    exporter = OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    return trace.get_tracer(service_name)
