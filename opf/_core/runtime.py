"""Core runtime loading and one-shot text prediction helpers."""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
import tiktoken

from .._common.constants import (
    OUTPUT_MODES,
    REDACTED_OUTPUT_LABEL,
    REDACTED_OUTPUT_PLACEHOLDER,
)
from .._common.env import get_env_bool
from .decoding import ViterbiCRFDecoder
from .._common.label_space import resolve_label_space_from_config
from .spans import (
    decode_text_with_offsets,
    discard_overlapping_spans_by_label,
    labels_to_spans,
    token_spans_to_char_spans,
    trim_char_spans_whitespace,
)
from .sequence_labeling import (
    ExampleAggregation,
    LabelInfo,
    TokenizedExample,
    build_label_info,
    example_to_windows,
)
from .._model.model import Transformer


@dataclass(frozen=True)
class InferenceRuntime:
    """Loaded model runtime and decode-time metadata for one OPF instance."""

    checkpoint: str
    model: Transformer
    encoding: tiktoken.Encoding
    label_info: LabelInfo
    device: torch.device
    n_ctx: int
    trim_span_whitespace: bool
    discard_overlapping_predicted_spans: bool
    output_mode: str
    active_encoding_name: str
    pad_token_id: int
    bidirectional_context: bool
    category_version: str


@dataclass(frozen=True)
class DetectedSpan:
    """One detected character span ready for rendering or serialization."""

    label: str
    start: int
    end: int
    text: str
    placeholder: str


@dataclass(frozen=True)
class PredictionResult:
    """Raw inference output before higher-level API serialization."""

    text: str
    spans: tuple[DetectedSpan, ...]
    decoded_mismatch: bool


def build_detection_summary(
    *,
    output_mode: str,
    labels: Sequence[str],
    decoded_mismatch: bool,
) -> dict[str, object]:
    """Build a compact summary for structured prediction output."""
    by_label: dict[str, int] = {}
    for label in labels:
        by_label[label] = by_label.get(label, 0) + 1
    return {
        "output_mode": output_mode,
        "span_count": len(labels),
        "by_label": dict(sorted(by_label.items(), key=lambda item: item[0])),
        "decoded_mismatch": decoded_mismatch,
    }


def _apply_output_mode_to_detected_spans(
    spans: Sequence[DetectedSpan],
    *,
    output_mode: str,
) -> list[DetectedSpan]:
    """Apply typed vs redacted output rendering to detected spans."""
    if output_mode == "typed":
        return list(spans)
    if output_mode != "redacted":
        raise ValueError(f"Unsupported output_mode: {output_mode!r}")
    return [
        DetectedSpan(
            label=REDACTED_OUTPUT_LABEL,
            start=span.start,
            end=span.end,
            text=span.text,
            placeholder=REDACTED_OUTPUT_PLACEHOLDER,
        )
        for span in spans
    ]


def _load_checkpoint_config(checkpoint_dir: str) -> dict[str, object]:
    """Load and validate the checkpoint JSON config object."""
    config_path = Path(checkpoint_dir) / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Checkpoint config at {config_path} must contain a JSON object"
        )
    return payload


def _resolve_n_ctx(
    checkpoint_config: dict[str, object],
    override_n_ctx: int | None,
    device: torch.device,
) -> int:
    """Resolve the effective context length for the current runtime."""
    if override_n_ctx is not None:
        if override_n_ctx <= 0:
            raise ValueError("n_ctx must be positive")
        return override_n_ctx
    if device.type == "cpu":
        # CPU full-eval/demo should default to a safer context size.
        return 4096

    for field_name in (
        "default_n_ctx",
        "initial_context_length",
        "max_position_embeddings",
    ):
        if field_name not in checkpoint_config:
            continue
        value = checkpoint_config[field_name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"Checkpoint config field {field_name} must be a positive integer"
            )
        if value <= 0:
            raise ValueError(f"Checkpoint config field {field_name} must be positive")
        return value

    return 4096


def _validate_checkpoint_dir(checkpoint_dir: str) -> None:
    """Ensure a checkpoint directory exists and contains the expected files."""
    path = Path(checkpoint_dir)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    config_path = path / "config.json"
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config: {config_path}")
    if not any(path.glob("*.safetensors")):
        raise FileNotFoundError(
            f"Checkpoint directory has no .safetensors files: {checkpoint_dir}"
        )


def _label_placeholder(label: str) -> str:
    """Convert a span label into the placeholder token shown to users."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", label.upper()).strip("_")
    if not normalized:
        normalized = "REDACTED"
    return f"<{normalized}>"


def _select_non_overlapping_spans(spans: Sequence[DetectedSpan]) -> list[DetectedSpan]:
    """Keep a left-to-right non-overlapping subset of detected spans."""
    ordered = sorted(
        spans, key=lambda span: (span.start, -(span.end - span.start), span.label)
    )
    kept: list[DetectedSpan] = []
    cursor = 0
    for span in ordered:
        if span.start < cursor:
            continue
        if span.end <= span.start:
            continue
        kept.append(span)
        cursor = span.end
    return kept


def _resolve_window_batch_size(
    *,
    device: torch.device,
    requested: int | None,
) -> int:
    """Resolve the inference window batch size from args/env/defaults."""
    if requested is not None:
        if requested <= 0:
            raise ValueError("window_batch_size must be positive")
        return requested
    raw = os.environ.get("OPF_WINDOW_BATCH_SIZE")
    if raw is not None and raw.strip():
        try:
            value = int(raw.strip())
        except ValueError as exc:
            raise ValueError(
                f"OPF_WINDOW_BATCH_SIZE must be an integer (got {raw!r})"
            ) from exc
        if value <= 0:
            raise ValueError("OPF_WINDOW_BATCH_SIZE must be positive")
        return value
    return 16 if device.type == "cuda" else 1


def _resolve_viterbi_cuda_batch_size() -> int:
    """Read and validate the CUDA Viterbi batch size from the environment."""
    raw = os.environ.get("OPF_VITERBI_CUDA_BATCH_SIZE", "512").strip()
    try:
        batch_size = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"OPF_VITERBI_CUDA_BATCH_SIZE must be an integer (got {raw!r})"
        ) from exc
    if batch_size <= 0:
        raise ValueError("OPF_VITERBI_CUDA_BATCH_SIZE must be positive")
    return batch_size


def _resolve_decoder_device(
    runtime: InferenceRuntime,
    decoder: ViterbiCRFDecoder | None,
) -> torch.device | None:
    """Return the optional CUDA device used for batched Viterbi decoding."""
    if decoder is None:
        return None
    if runtime.device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        return None
    if not get_env_bool("OPF_VITERBI_ON_CUDA", default=True):
        return None
    return runtime.device


def _build_prediction_result(
    runtime: InferenceRuntime,
    *,
    text: str,
    token_ids: Sequence[int],
    predicted_labels_by_index: dict[int, int],
) -> PredictionResult:
    """Build one structured prediction result from decoded token labels."""
    predicted_token_spans = labels_to_spans(
        predicted_labels_by_index, runtime.label_info
    )

    decoded_text, char_starts, char_ends = decode_text_with_offsets(
        token_ids, runtime.encoding
    )
    decoded_mismatch = decoded_text != text
    source_text = decoded_text if decoded_mismatch else text

    predicted_char_spans = token_spans_to_char_spans(
        predicted_token_spans, char_starts, char_ends
    )
    if runtime.trim_span_whitespace:
        predicted_char_spans = trim_char_spans_whitespace(
            predicted_char_spans, source_text
        )
    if runtime.discard_overlapping_predicted_spans:
        predicted_char_spans = discard_overlapping_spans_by_label(predicted_char_spans)

    detected: list[DetectedSpan] = []
    for label_idx, start, end in predicted_char_spans:
        if not (0 <= start < end <= len(source_text)):
            continue
        label = (
            str(runtime.label_info.span_class_names[label_idx])
            if 0 <= int(label_idx) < len(runtime.label_info.span_class_names)
            else f"label_{label_idx}"
        )
        detected.append(
            DetectedSpan(
                label=label,
                start=int(start),
                end=int(end),
                text=source_text[start:end],
                placeholder=_label_placeholder(label),
            )
        )

    display_spans = _apply_output_mode_to_detected_spans(
        _select_non_overlapping_spans(detected),
        output_mode=runtime.output_mode,
    )
    return PredictionResult(
        text=source_text,
        spans=tuple(display_spans),
        decoded_mismatch=decoded_mismatch,
    )


def load_inference_runtime(
    *,
    checkpoint: str,
    device_name: str,
    n_ctx_override: int | None = None,
    trim_span_whitespace: bool,
    discard_overlapping_predicted_spans: bool,
    output_mode: str,
) -> InferenceRuntime:
    """Load model, tokenizer, label space, and runtime metadata for inference."""
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"Unsupported output_mode: {output_mode!r}")
    _validate_checkpoint_dir(checkpoint)
    normalized_device_name = device_name.strip().lower()
    if normalized_device_name == "gpu":
        normalized_device_name = "cuda"
    elif normalized_device_name.startswith("gpu:"):
        normalized_device_name = f"cuda:{normalized_device_name.split(':', 1)[1]}"
    device = torch.device(normalized_device_name)
    checkpoint_config = _load_checkpoint_config(checkpoint)
    n_ctx = _resolve_n_ctx(checkpoint_config, n_ctx_override, device)
    encoding_name = checkpoint_config.get("encoding")
    if not isinstance(encoding_name, str) or not encoding_name:
        raise ValueError("Checkpoint config field encoding must be a non-empty string")
    encoding = tiktoken.get_encoding(encoding_name)
    pad_token_id = int(encoding.eot_token)
    config_context = str(Path(checkpoint) / "config.json")
    resolved_category_version, _span_class_names, resolved_ner_class_names = (
        resolve_label_space_from_config(checkpoint_config, context=config_context)
    )
    label_info = build_label_info(resolved_ner_class_names)
    model = Transformer.from_checkpoint(
        checkpoint,
        device=device,
    )
    bidirectional_context = checkpoint_config["bidirectional_context"]
    model.eval()
    if get_env_bool("OPF_TORCH_COMPILE"):
        compile_mode = os.environ.get("OPF_TORCH_COMPILE_MODE", "default")
        model = torch.compile(model, mode=compile_mode)
    return InferenceRuntime(
        checkpoint=checkpoint,
        model=model,
        encoding=encoding,
        label_info=label_info,
        device=device,
        n_ctx=n_ctx,
        trim_span_whitespace=trim_span_whitespace,
        discard_overlapping_predicted_spans=discard_overlapping_predicted_spans,
        output_mode=output_mode,
        active_encoding_name=encoding_name,
        pad_token_id=pad_token_id,
        bidirectional_context=bidirectional_context,
        category_version=resolved_category_version,
    )


@torch.inference_mode()
def predict_text(
    runtime: InferenceRuntime,
    text: str,
    *,
    decoder: ViterbiCRFDecoder | None,
) -> PredictionResult:
    """Run one text through the model and return decoded detected spans."""
    return predict_texts(runtime, [text], decoder=decoder, window_batch_size=1)[0]


@torch.inference_mode()
def predict_texts(
    runtime: InferenceRuntime,
    texts: Sequence[str],
    *,
    decoder: ViterbiCRFDecoder | None,
    window_batch_size: int | None = None,
) -> tuple[PredictionResult, ...]:
    """Run multiple texts through the model and decode them in batches."""
    if not texts:
        return ()

    resolved_window_batch_size = _resolve_window_batch_size(
        device=runtime.device,
        requested=window_batch_size,
    )
    background = int(runtime.label_info.background_token_label)
    example_order: list[str] = []
    example_texts: dict[str, str] = {}
    example_token_ids: dict[str, tuple[int, ...]] = {}
    aggregated_examples: dict[str, ExampleAggregation] = {}
    pending_windows: list[object] = []

    def process_window_batch(windows: Sequence[object]) -> None:
        if not windows:
            return
        max_window_len = max(len(window.tokens) for window in windows)
        if max_window_len <= 0:
            return
        token_rows: list[list[int]] = []
        mask_rows: list[list[bool]] = []
        for window in windows:
            window_tokens = list(window.tokens)
            if not window_tokens:
                raise ValueError("Window batch contains an empty window")
            pad_count = max_window_len - len(window_tokens)
            token_rows.append(window_tokens + ([runtime.pad_token_id] * pad_count))
            mask_rows.append(([True] * len(window_tokens)) + ([False] * pad_count))

        tokens_t = torch.tensor(token_rows, device=runtime.device, dtype=torch.int32)
        attention_mask_t = torch.tensor(
            mask_rows, device=runtime.device, dtype=torch.bool
        )
        logits = runtime.model(tokens_t, attention_mask=attention_mask_t)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        if log_probs.dim() != 3 or int(log_probs.shape[0]) != len(windows):
            raise ValueError(
                "Batched logprob output shape mismatch: got %s expected (%d,%d,*)"
                % (tuple(log_probs.shape), len(windows), max_window_len)
            )

        for batch_idx, window in enumerate(windows):
            window_log_probs = log_probs[batch_idx].cpu()
            for token_pos, is_valid in enumerate(window.mask):
                if not bool(is_valid):
                    continue
                token_idx = int(window.offsets[token_pos])
                if token_idx < 0:
                    continue
                example_id = window.token_example_ids[token_pos]
                if example_id is None:
                    continue
                aggregation = aggregated_examples[example_id]
                aggregation.ensure_capacity(token_idx)
                score_vec = window_log_probs[token_pos]
                existing = aggregation.logprob_logsumexp[token_idx]
                if existing is None:
                    aggregation.logprob_logsumexp[token_idx] = score_vec.clone()
                else:
                    aggregation.logprob_logsumexp[token_idx] = torch.logaddexp(
                        existing, score_vec
                    )
                aggregation.counts[token_idx] += 1
                aggregation.record_token_id(
                    token_idx, int(window.tokens[token_pos]), example_id
                )
                aggregation.length = max(aggregation.length, token_idx + 1)

    def enqueue_window(window: object) -> None:
        if not window.tokens:
            return
        pending_windows.append(window)
        if len(pending_windows) >= resolved_window_batch_size:
            process_window_batch(tuple(pending_windows))
            pending_windows.clear()

    for idx, text in enumerate(texts):
        example_id = f"predict-example-{idx}"
        token_ids = tuple(
            int(tok) for tok in runtime.encoding.encode(text, allowed_special="all")
        )
        example_order.append(example_id)
        example_texts[example_id] = text
        example_token_ids[example_id] = token_ids
        aggregated_examples[example_id] = ExampleAggregation(
            logprob_logsumexp=[], counts=[], labels=[], token_ids=[]
        )
        if not token_ids:
            continue
        example = TokenizedExample(
            tokens=token_ids,
            labels=tuple(background for _ in token_ids),
            example_id=example_id,
            text=text,
        )
        for window in example_to_windows(example, runtime.n_ctx):
            enqueue_window(window)

    if pending_windows:
        process_window_batch(tuple(pending_windows))

    states_with_scores: list[str] = []
    score_tensors: list[torch.Tensor] = []
    token_positions_by_example: dict[str, list[int]] = {}
    decoded_labels_by_example: dict[str, list[int]] = {}

    for example_id in example_order:
        aggregation = aggregated_examples[example_id]
        token_positions: list[int] = []
        token_score_vectors: list[torch.Tensor] = []
        for token_idx in range(aggregation.length):
            if token_idx >= len(aggregation.logprob_logsumexp):
                continue
            score_sum = aggregation.logprob_logsumexp[token_idx]
            count = aggregation.counts[token_idx]
            if score_sum is None or count <= 0:
                continue
            avg_logprob = score_sum - math.log(float(count))
            token_positions.append(token_idx)
            token_score_vectors.append(avg_logprob)
        token_positions_by_example[example_id] = token_positions
        if token_score_vectors:
            states_with_scores.append(example_id)
            score_tensors.append(torch.stack(token_score_vectors, dim=0))

    if decoder is not None and score_tensors:
        decode_device = _resolve_decoder_device(runtime, decoder)
        decoded_many = decoder.decode_many(
            score_tensors,
            device=decode_device,
            max_batch_size=_resolve_viterbi_cuda_batch_size(),
        )
        if len(decoded_many) != len(states_with_scores):
            raise RuntimeError(
                "Decoder returned unexpected number of decoded sequences: "
                f"{len(decoded_many)} != {len(states_with_scores)}"
            )
        for example_id, decoded in zip(states_with_scores, decoded_many):
            decoded_labels_by_example[example_id] = list(decoded)
    else:
        for example_id, stacked_scores in zip(states_with_scores, score_tensors):
            decoded_labels_by_example[example_id] = stacked_scores.argmax(dim=1).tolist()

    results: list[PredictionResult] = []
    for example_id in example_order:
        text = example_texts[example_id]
        token_ids = example_token_ids[example_id]
        if not token_ids:
            results.append(PredictionResult(text=text, spans=(), decoded_mismatch=False))
            continue
        token_positions = token_positions_by_example[example_id]
        decoded_labels = decoded_labels_by_example.get(example_id, [])
        if len(decoded_labels) != len(token_positions):
            score_vectors = [
                aggregated_examples[example_id].logprob_logsumexp[token_idx]
                for token_idx in token_positions
                if aggregated_examples[example_id].logprob_logsumexp[token_idx] is not None
            ]
            if score_vectors:
                stacked_scores = torch.stack(score_vectors, dim=0)
                decoded_labels = stacked_scores.argmax(dim=1).tolist()
        predicted_labels_by_index = {
            token_idx: int(label)
            for token_idx, label in zip(token_positions, decoded_labels)
        }
        results.append(
            _build_prediction_result(
                runtime,
                text=text,
                token_ids=token_ids,
                predicted_labels_by_index=predicted_labels_by_index,
            )
        )
    return tuple(results)
