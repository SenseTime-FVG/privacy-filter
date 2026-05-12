"""Sequence decoders and Viterbi bias-resolution helpers."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
import threading
from typing import Mapping, Sequence

import torch

from .sequence_labeling import LabelInfo

# Large negative score used to mask impossible CRF transitions.
_NEG_INF = -1e9

# Calibration artifact schema and default on-disk filename.
VITERBI_BIAS_KEYS: tuple[str, ...] = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)
DEFAULT_VITERBI_CALIBRATION_FILENAME = "viterbi_calibration.json"


def require_float(value: float | int | object, *, field_name: str) -> float:
    """Validate a numeric value and return it as `float`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    return float(value)


def zero_viterbi_transition_biases() -> dict[str, float]:
    """Return the default all-zero Viterbi transition-bias map."""
    return {key: 0.0 for key in VITERBI_BIAS_KEYS}


def discover_default_viterbi_calibration_path(
    checkpoint_dir: str | None,
) -> str | None:
    """Return the default calibration artifact path if it exists."""
    if checkpoint_dir is None:
        return None
    candidate = Path(checkpoint_dir) / DEFAULT_VITERBI_CALIBRATION_FILENAME
    if candidate.exists() and candidate.is_file():
        return str(candidate)
    return None


def load_viterbi_calibration_artifact(path: str) -> Mapping[str, object]:
    """Load the Viterbi calibration artifact."""
    artifact_path = Path(path)
    if not artifact_path.exists() or not artifact_path.is_file():
        raise FileNotFoundError(f"viterbi_calibration_path not found: {artifact_path}")
    with artifact_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Calibration artifact at {artifact_path} must contain a JSON object"
        )
    return payload


def _validate_exact_keys(
    value: Mapping[str, object],
    *,
    expected: set[str],
    field_name: str,
) -> None:
    """Validate that a mapping contains exactly the expected string keys."""
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{field_name} must contain exactly {sorted(expected)} "
            f"(missing={missing}, extra={extra})"
        )


def resolve_viterbi_biases_from_calibration_path(
    viterbi_calibration_path: str,
) -> dict[str, float]:
    """Resolve Viterbi transition biases from the default calibration point."""
    artifact = load_viterbi_calibration_artifact(viterbi_calibration_path)
    _validate_exact_keys(
        artifact,
        expected={"operating_points"},
        field_name="Calibration artifact",
    )

    operating_points = artifact["operating_points"]
    if not isinstance(operating_points, Mapping):
        raise ValueError("Calibration artifact missing operating_points")
    _validate_exact_keys(
        operating_points,
        expected={"default"},
        field_name="operating_points",
    )

    default_entry = operating_points["default"]
    if not isinstance(default_entry, Mapping):
        raise ValueError("operating_points.default must be an object")
    _validate_exact_keys(
        default_entry,
        expected={"biases"},
        field_name="operating_points.default",
    )

    raw_biases = default_entry["biases"]
    if not isinstance(raw_biases, Mapping):
        raise ValueError("operating_points.default.biases missing")
    _validate_exact_keys(
        raw_biases,
        expected=set(VITERBI_BIAS_KEYS),
        field_name="operating_points.default.biases",
    )

    resolved: dict[str, float] = {}
    for key in VITERBI_BIAS_KEYS:
        resolved[key] = require_float(
            raw_biases.get(key),
            field_name=f"operating_points.default.biases.{key}",
        )
    return resolved


@dataclass
class ViterbiCRFDecoder:
    """Decode boundary-aware token labels with CRF-style Viterbi constraints."""

    label_info: LabelInfo
    transition_bias_background_stay: float = 0.0
    transition_bias_background_to_start: float = 0.0
    transition_bias_inside_to_continue: float = 0.0
    transition_bias_inside_to_end: float = 0.0
    transition_bias_end_to_background: float = 0.0
    transition_bias_end_to_start: float = 0.0

    def __post_init__(self) -> None:
        """Precompute CRF transition/start/end score tables for this label space."""
        num_classes = len(self.label_info.token_to_span_label)
        self._start_scores = torch.full((num_classes,), _NEG_INF, dtype=torch.float32)
        self._end_scores = torch.full((num_classes,), _NEG_INF, dtype=torch.float32)
        self._transition_scores = torch.full(
            (num_classes, num_classes), _NEG_INF, dtype=torch.float32
        )
        self._score_cache: dict[
            tuple[str, int, torch.dtype],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        self._score_cache_lock = threading.Lock()

        background_token_idx = self.label_info.background_token_label
        background_span_idx = self.label_info.background_span_label
        token_boundary_tags = self.label_info.token_boundary_tags
        token_to_span_label = self.label_info.token_to_span_label

        for idx in range(num_classes):
            tag = token_boundary_tags.get(idx)
            span_label = token_to_span_label.get(idx)
            if tag in {"B", "S"} or idx == background_token_idx:
                self._start_scores[idx] = 0.0
            if tag in {"E", "S"} or idx == background_token_idx:
                self._end_scores[idx] = 0.0

            for next_idx in range(num_classes):
                next_tag = token_boundary_tags.get(next_idx)
                next_span_label = token_to_span_label.get(next_idx)
                if self._is_valid_transition(
                    prev_tag=tag,
                    prev_span=span_label,
                    next_tag=next_tag,
                    next_span=next_span_label,
                    background_token_idx=background_token_idx,
                    background_span_idx=background_span_idx,
                    next_idx=next_idx,
                ):
                    self._transition_scores[idx, next_idx] = self._transition_bias(
                        prev_tag=tag,
                        prev_span=span_label,
                        next_tag=next_tag,
                        next_span=next_span_label,
                        background_token_idx=background_token_idx,
                        background_span_idx=background_span_idx,
                        prev_idx=idx,
                        next_idx=next_idx,
                    )

    def _transition_bias(
        self,
        *,
        prev_tag: str | None,
        prev_span: int | None,
        next_tag: str | None,
        next_span: int | None,
        background_token_idx: int,
        background_span_idx: int,
        prev_idx: int,
        next_idx: int,
    ) -> float:
        """Return the transition bias for one allowed CRF edge."""
        prev_is_background = (
            prev_span == background_span_idx or prev_idx == background_token_idx
        )
        next_is_background = (
            next_span == background_span_idx or next_idx == background_token_idx
        )

        if prev_is_background:
            if next_is_background:
                return self.transition_bias_background_stay
            if next_tag in {"B", "S"}:
                return self.transition_bias_background_to_start
            return 0.0

        if prev_tag in {"B", "I"}:
            if next_tag == "I" and prev_span == next_span:
                return self.transition_bias_inside_to_continue
            if next_tag == "E" and prev_span == next_span:
                return self.transition_bias_inside_to_end
            return 0.0

        if prev_tag in {"E", "S"}:
            if next_is_background:
                return self.transition_bias_end_to_background
            if next_tag in {"B", "S"}:
                return self.transition_bias_end_to_start
            return 0.0

        return 0.0

    @staticmethod
    def _is_valid_transition(
        *,
        prev_tag: str | None,
        prev_span: int | None,
        next_tag: str | None,
        next_span: int | None,
        background_token_idx: int,
        background_span_idx: int,
        next_idx: int,
    ) -> bool:
        """Return whether one CRF transition is structurally valid."""
        next_is_background = (
            next_span == background_span_idx or next_idx == background_token_idx
        )
        if (next_span is None or next_tag is None) and not next_is_background:
            return False

        if prev_span is None or prev_tag is None:
            return next_is_background or next_tag in {"B", "S"}

        prev_is_background = prev_span == background_span_idx

        if prev_is_background:
            return next_is_background or next_tag in {"B", "S"}

        if prev_tag in {"E", "S"}:
            return next_is_background or next_tag in {"B", "S"}

        if prev_tag in {"B", "I"}:
            same_span = prev_span == next_span
            return same_span and next_tag in {"I", "E"}

        return False

    def decode(self, token_logprobs: torch.Tensor) -> list[int]:
        """Decode one `[seq_len, num_classes]` logprob tensor into label ids."""
        if token_logprobs.ndim != 2:
            raise ValueError("token_logprobs must have shape [seq_len, num_classes]")

        seq_len, num_classes = token_logprobs.shape
        if seq_len == 0:
            return []

        device = token_logprobs.device
        dtype = token_logprobs.dtype
        if self._start_scores.device == device and self._start_scores.dtype == dtype:
            start_scores = self._start_scores
            end_scores = self._end_scores
            transition_scores = self._transition_scores
        else:
            device_index = device.index if device.index is not None else -1
            cache_key = (device.type, device_index, dtype)
            with self._score_cache_lock:
                cached_scores = self._score_cache.get(cache_key)
                if cached_scores is None:
                    cached_scores = (
                        self._start_scores.to(device=device, dtype=dtype),
                        self._end_scores.to(device=device, dtype=dtype),
                        self._transition_scores.to(device=device, dtype=dtype),
                    )
                    self._score_cache[cache_key] = cached_scores
            start_scores, end_scores, transition_scores = cached_scores

        scores = token_logprobs[0] + start_scores
        backpointers = torch.empty(
            (seq_len - 1, num_classes), device=device, dtype=torch.int64
        )

        for idx in range(1, seq_len):
            transitions = scores.unsqueeze(1) + transition_scores
            best_scores, best_paths = transitions.max(dim=0)
            scores = best_scores + token_logprobs[idx]
            backpointers[idx - 1] = best_paths

        if not torch.isfinite(scores).any():
            return token_logprobs.argmax(dim=1).tolist()

        scores = scores + end_scores
        last_label = scores.argmax()
        path = torch.empty((seq_len,), device=device, dtype=torch.int64)
        path[-1] = last_label
        for idx in range(seq_len - 2, -1, -1):
            last_label = backpointers[idx, last_label]
            path[idx] = last_label
        return path.tolist()

    def decode_many(
        self,
        token_logprobs_list: Sequence[torch.Tensor],
        *,
        device: torch.device | None = None,
        max_batch_size: int = 128,
    ) -> list[list[int]]:
        """Decode multiple sequences, optionally batching on CUDA."""
        if not token_logprobs_list:
            return []
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if device is None or device.type != "cuda":
            return [self.decode(scores) for scores in token_logprobs_list]

        lengths = [int(scores.shape[0]) for scores in token_logprobs_list]
        if any(scores.ndim != 2 for scores in token_logprobs_list):
            raise ValueError("decode_many expects [seq_len, num_classes] tensors")
        if any(length <= 0 for length in lengths):
            return [self.decode(scores) for scores in token_logprobs_list]

        order = sorted(
            range(len(token_logprobs_list)), key=lambda idx: lengths[idx], reverse=True
        )
        results: list[list[int] | None] = [None] * len(token_logprobs_list)
        for start in range(0, len(order), max_batch_size):
            batch_indices = order[start : start + max_batch_size]
            batch_scores = [token_logprobs_list[idx] for idx in batch_indices]
            batch_lengths = [lengths[idx] for idx in batch_indices]
            decoded_batch = self._decode_many_cuda_batch(
                batch_scores,
                batch_lengths,
                device=device,
            )
            for original_idx, decoded in zip(batch_indices, decoded_batch):
                results[original_idx] = decoded

        output: list[list[int]] = []
        for decoded in results:
            if decoded is None:
                raise RuntimeError(
                    "Internal decode_many failure: missing decoded sequence"
                )
            output.append(decoded)
        return output

    def _decode_many_cuda_batch(
        self,
        token_logprobs_batch: Sequence[torch.Tensor],
        lengths: Sequence[int],
        *,
        device: torch.device,
    ) -> list[list[int]]:
        """Decode one same-device CUDA batch of emission tensors."""
        batch_size = len(token_logprobs_batch)
        if batch_size == 0:
            return []
        num_classes = int(token_logprobs_batch[0].shape[1])
        max_len = int(max(lengths))
        dtype = token_logprobs_batch[0].dtype
        for scores in token_logprobs_batch:
            if int(scores.shape[1]) != num_classes:
                raise ValueError(
                    "All decode_many tensors must share the same class dimension"
                )

        emissions = torch.full(
            (batch_size, max_len, num_classes),
            -float("inf"),
            device=device,
            dtype=dtype,
        )
        for row, (scores, length) in enumerate(zip(token_logprobs_batch, lengths)):
            if scores.device != device or scores.dtype != dtype:
                scores = scores.to(device=device, dtype=dtype)
            emissions[row, :length] = scores

        lengths_t = torch.tensor(lengths, device=device, dtype=torch.long)
        device_index = device.index if device.index is not None else -1
        cache_key = (device.type, device_index, dtype)
        with self._score_cache_lock:
            cached_scores = self._score_cache.get(cache_key)
            if cached_scores is None:
                cached_scores = (
                    self._start_scores.to(device=device, dtype=dtype),
                    self._end_scores.to(device=device, dtype=dtype),
                    self._transition_scores.to(device=device, dtype=dtype),
                )
                self._score_cache[cache_key] = cached_scores
        start_scores, end_scores, transition_scores = cached_scores

        scores = emissions[:, 0, :] + start_scores[None, :]
        backpointer_dtype = torch.int16 if num_classes <= 32767 else torch.int32
        backpointers = torch.zeros(
            (max_len - 1, batch_size, num_classes),
            device=device,
            dtype=backpointer_dtype,
        )
        batch_arange = torch.arange(batch_size, device=device, dtype=torch.long)

        for step in range(1, max_len):
            active = lengths_t > step
            if not bool(active.any().item()):
                break
            active_idx = batch_arange[active]
            transitions = scores[active_idx].unsqueeze(2) + transition_scores
            best_scores, best_paths = transitions.max(dim=1)
            scores[active_idx] = best_scores + emissions[active_idx, step, :]
            backpointers[step - 1, active_idx] = best_paths.to(backpointer_dtype)

        bad_rows = ~torch.isfinite(scores).any(dim=1)
        scores = scores + end_scores[None, :]
        last_labels = scores.argmax(dim=1)
        paths = torch.zeros((batch_size, max_len), device=device, dtype=torch.long)
        paths[batch_arange, lengths_t - 1] = last_labels
        for step in range(max_len - 2, -1, -1):
            active = lengths_t > (step + 1)
            if not bool(active.any().item()):
                continue
            active_idx = batch_arange[active]
            next_labels = paths[active_idx, step + 1]
            prev = backpointers[step, active_idx, next_labels].to(torch.long)
            paths[active_idx, step] = prev

        if bool(bad_rows.any().item()):
            fallback_paths = emissions.argmax(dim=2)
            bad_idx = batch_arange[bad_rows]
            for idx in bad_idx.tolist():
                length = int(lengths_t[idx].item())
                paths[idx, :length] = fallback_paths[idx, :length]

        output: list[list[int]] = []
        for row, length in enumerate(lengths):
            output.append(paths[row, :length].tolist())
        return output


def resolve_viterbi_transition_biases(
    *,
    viterbi_calibration_path: str | None,
    checkpoint_dir: str | None = None,
) -> dict[str, float]:
    """Resolve the effective transition-bias map for one decode configuration."""
    resolved_calibration_path = (
        str(viterbi_calibration_path).strip()
        if viterbi_calibration_path is not None
        else None
    )
    if resolved_calibration_path == "":
        resolved_calibration_path = None
    if resolved_calibration_path is None:
        resolved_calibration_path = discover_default_viterbi_calibration_path(
            checkpoint_dir
        )
    if resolved_calibration_path is None:
        return zero_viterbi_transition_biases()
    return resolve_viterbi_biases_from_calibration_path(resolved_calibration_path)


def build_sequence_decoder(
    *,
    decode_mode: str,
    label_info: LabelInfo,
    viterbi_calibration_path: str | None,
    checkpoint_dir: str | None = None,
) -> tuple[ViterbiCRFDecoder | None, dict[str, float] | None]:
    """Build the requested decoder and return any resolved Viterbi biases."""
    if decode_mode == "argmax":
        return None, None
    if decode_mode != "viterbi":
        raise ValueError(
            f"Unsupported decode_mode {decode_mode!r}; expected 'viterbi' or 'argmax'"
        )
    resolved_biases = resolve_viterbi_transition_biases(
        viterbi_calibration_path=viterbi_calibration_path,
        checkpoint_dir=checkpoint_dir,
    )
    return ViterbiCRFDecoder(label_info=label_info, **resolved_biases), resolved_biases
