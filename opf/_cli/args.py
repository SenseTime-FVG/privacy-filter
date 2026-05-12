from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

from .common import (
    add_decode_mode_arg,
    add_device_arg,
    add_discard_overlapping_predicted_spans_arg,
    add_n_ctx_arg,
    add_output_mode_arg,
    add_trim_whitespace_args,
    add_viterbi_args,
)


_line_editing_warning_emitted = False


def _warn_line_editing_unavailable(reason: str) -> None:
    """Emit a one-time warning that readline support is unavailable."""
    global _line_editing_warning_emitted
    if _line_editing_warning_emitted:
        return
    print(
        "WARNING: interactive line editing unavailable; "
        f"falling back to plain input ({reason}).",
        file=sys.stderr,
    )
    _line_editing_warning_emitted = True


def _enable_interactive_line_editing() -> None:
    """Enable readline-backed interactive editing when available."""
    try:
        import readline  # noqa: F401
    except ImportError:
        _warn_line_editing_unavailable("readline module not available")


def add_common_redaction_args(
    parser: argparse.ArgumentParser,
    *,
    runtime_group: object,
    decode_group: object,
    output_group: object,
) -> None:
    """Add the shared redaction/runtime/decode/output arguments."""
    parser.set_defaults(
        text=None,
        interactive_banner="OPF inference. Type '/exit' (or 'quit') to stop.",
        interactive_prompt="text> ",
    )
    add_device_arg(runtime_group)
    add_n_ctx_arg(runtime_group)
    runtime_group.add_argument(
        "--input-batch-size",
        type=int,
        default=32,
        help=(
            "Maximum number of input texts to accumulate per non-interactive "
            "batch before calling batched inference."
        ),
    )
    add_decode_mode_arg(decode_group)
    add_discard_overlapping_predicted_spans_arg(decode_group)
    add_trim_whitespace_args(parser, decode_group)
    add_viterbi_args(decode_group)
    add_output_mode_arg(output_group)
    output_group.add_argument(
        "--json-indent",
        type=int,
        default=2,
        help="Indentation level for printed JSON",
    )
    output_group.add_argument(
        "--no-print-color-coded-text",
        dest="print_color_coded_text",
        action="store_false",
        help="Do not print the ANSI color-coded text section after JSON output.",
    )
    parser.set_defaults(print_color_coded_text=True)


def _read_text_file(path: str) -> str:
    """Read one input text file as UTF-8."""
    file_path = Path(path).expanduser()
    return file_path.read_text(encoding="utf-8")


def iter_inputs(args: argparse.Namespace) -> Iterator[str]:
    """Yield input texts from CLI arguments, files, stdin, or the prompt."""
    text_items = args.text
    if text_items:
        for item in text_items:
            yield item
        return

    text_files = args.text_file
    if text_files:
        for file_path in text_files:
            text = _read_text_file(file_path)
            if not text:
                continue
            yield text
        return

    if not sys.stdin.isatty():
        for raw in sys.stdin:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            yield line
        return

    print(args.interactive_banner)
    _enable_interactive_line_editing()
    while True:
        try:
            line = input(args.interactive_prompt)
        except EOFError:
            print()
            return
        command = line.strip().lower()
        if command in {"/exit", "exit", "quit", ":q"}:
            return
        if not line:
            continue
        yield line


def using_interactive_prompt(args: argparse.Namespace) -> bool:
    """Return whether the CLI should enter interactive prompt mode."""
    return not args.text and not args.text_file and sys.stdin.isatty()
