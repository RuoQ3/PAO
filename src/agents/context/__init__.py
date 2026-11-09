"""Read-only ProcessContext assembly for IMA and agent callers."""

from .builder import ProcessContextBuilder, ProcessContextBuildError
from .loader import ContextLoader, LoadedContext
from .adapter import CaseInputConfig, ProcessCaseAdapter


# Functional read-only API for integrations that do not want to bind helper
# methods to a concrete builder instance.
def get_equipment(context, equipment_id): return context.get_equipment(equipment_id)
def get_stream(context, stream_id): return context.get_stream(stream_id)
def downstream(context, equipment_id): return context.downstream(equipment_id)
def upstream(context, equipment_id): return context.upstream(equipment_id)
def trace_stream(context, stream_id): return context.trace_stream(stream_id)
def trace_path(context, start_id, end_id): return context.trace_path(start_id, end_id)
def terminal_streams(context): return context.terminal_streams()
def boundary_nodes(context): return context.boundary_nodes()
def summarize_process(context): return context.summarize_process()
def compare_compositions(context, input_stream, output_stream): return context.compare_compositions(input_stream, output_stream)

__all__ = ["ProcessContextBuilder", "ProcessContextBuildError", "CaseInputConfig", "ProcessCaseAdapter", "ContextLoader", "LoadedContext",
           "get_equipment", "get_stream", "downstream", "upstream", "trace_stream",
           "trace_path", "terminal_streams", "boundary_nodes", "summarize_process",
           "compare_compositions"]
