from __future__ import annotations

import argparse

from .._cli.common import (
    CliHelpFormatter,
    add_checkpoint_arg,
    add_decode_mode_arg,
    add_device_arg,
    add_discard_overlapping_predicted_spans_arg,
    add_n_ctx_arg,
    add_output_mode_arg,
    add_trim_whitespace_args,
    add_viterbi_args,
    resolve_prog,
)

_SERVE_DESCRIPTION = (
    "Run an HTTP redaction service with async request queueing and micro-batching."
)


def build_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    """Build the parser for the service CLI mode."""
    parser = argparse.ArgumentParser(
        description=_SERVE_DESCRIPTION,
        formatter_class=CliHelpFormatter,
        prog=prog or resolve_prog("opf serve"),
    )
    network_group = parser.add_argument_group("Network")
    runtime_group = parser.add_argument_group("Model / Runtime")
    decode_group = parser.add_argument_group("Decode")
    batching_group = parser.add_argument_group("Batching / Queueing")

    network_group.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface to bind the HTTP service to.",
    )
    network_group.add_argument(
        "--port",
        type=int,
        default=8000,
        help="HTTP port to listen on.",
    )
    network_group.add_argument(
        "--log-level",
        choices=("critical", "error", "warning", "info", "debug", "trace"),
        default="info",
        help="Uvicorn log level.",
    )

    add_checkpoint_arg(runtime_group)
    add_device_arg(runtime_group)
    add_n_ctx_arg(runtime_group)
    add_output_mode_arg(runtime_group)

    add_decode_mode_arg(decode_group)
    add_discard_overlapping_predicted_spans_arg(decode_group)
    add_trim_whitespace_args(parser, decode_group)
    add_viterbi_args(decode_group)

    batching_group.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum number of queued HTTP requests to combine into one inference batch.",
    )
    batching_group.add_argument(
        "--batch-timeout-ms",
        type=float,
        default=10.0,
        help="How long to wait for additional requests before flushing one micro-batch.",
    )
    batching_group.add_argument(
        "--max-queue-size",
        type=int,
        default=512,
        help="Maximum number of pending requests allowed in the async queue.",
    )
    batching_group.add_argument(
        "--window-batch-size",
        type=int,
        default=None,
        help=(
            "Maximum number of model windows to combine in one batched forward pass. "
            "Defaults to the runtime policy (cuda=8, cpu=1, or OPF_WINDOW_BATCH_SIZE if set)."
        ),
    )
    batching_group.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="Skip loading the runtime and decoder during startup.",
    )
    parser.set_defaults(warmup=True)
    return parser


def parse_args(
    argv: list[str] | None = None, *, prog: str | None = None
) -> argparse.Namespace:
    """Parse service-mode arguments and validate them."""
    args = build_parser(prog=prog).parse_args(argv)
    if args.port <= 0 or args.port > 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.max_batch_size <= 0:
        raise ValueError("max_batch_size must be > 0")
    if args.batch_timeout_ms < 0:
        raise ValueError("batch_timeout_ms must be >= 0")
    if args.max_queue_size <= 0:
        raise ValueError("max_queue_size must be > 0")
    if args.window_batch_size is not None and args.window_batch_size <= 0:
        raise ValueError("window_batch_size must be > 0")
    return args
