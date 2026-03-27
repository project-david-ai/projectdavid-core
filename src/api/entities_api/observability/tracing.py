import os

from opentelemetry import trace

# Ensure you are using the GRPC exporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON


def setup_tracing(app):
    service_name = os.getenv("OTEL_SERVICE_NAME", "entities-api")
    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource, sampler=ALWAYS_ON)
    trace.set_tracer_provider(provider)

    # REMOVED hardcoded arguments.
    # This will now automatically read OTEL_EXPORTER_OTLP_ENDPOINT from Docker Env.
    otlp_exporter = OTLPSpanExporter()

    span_processor = BatchSpanProcessor(otlp_exporter)
    provider.add_span_processor(span_processor)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)

    # Instrument Requests
    RequestsInstrumentor().instrument(tracer_provider=provider)
