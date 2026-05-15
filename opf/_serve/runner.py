# pyright: reportMissingImports=false

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Sequence

from .._api import OPF, RedactionResult


@dataclass(frozen=True)
class ServiceSettings:
    """Validated runtime settings for the OPF HTTP service."""

    checkpoint: str | None
    context_window_length: int | None
    trim_whitespace: bool
    device: str
    output_mode: str
    decode_mode: str
    discard_overlapping_predicted_spans: bool
    viterbi_calibration_path: str | None
    host: str
    port: int
    log_level: str
    max_batch_size: int
    batch_timeout_ms: float
    max_queue_size: int
    window_batch_size: int | None
    warmup: bool


@dataclass
class _QueuedRequest:
    """One pending HTTP request waiting for batched inference."""

    text: str
    future: asyncio.Future[dict[str, Any]]


class BatchedRedactionService:
    """Serialize model access while batching queued single-text requests."""

    def __init__(
        self,
        redactor: OPF,
        *,
        max_batch_size: int,
        batch_timeout_ms: float,
        max_queue_size: int,
        window_batch_size: int | None,
    ) -> None:
        self._redactor = redactor
        self._max_batch_size = int(max_batch_size)
        self._batch_timeout_s = float(batch_timeout_ms) / 1000.0
        self._window_batch_size = (
            int(window_batch_size)
            if window_batch_size is not None
            else None
        )
        self._queue: asyncio.Queue[_QueuedRequest] = asyncio.Queue(maxsize=max_queue_size)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="opf-serve")
        self._worker_task: asyncio.Task[None] | None = None
        self._started = False

    @property
    def queue_size(self) -> int:
        """Return the current number of queued requests."""
        return int(self._queue.qsize())

    @property
    def max_batch_size(self) -> int:
        """Return the configured request batch size."""
        return self._max_batch_size

    @property
    def max_queue_size(self) -> int:
        """Return the configured maximum queue size."""
        return int(self._queue.maxsize)

    @property
    def batch_timeout_ms(self) -> float:
        """Return the configured micro-batch timeout in milliseconds."""
        return self._batch_timeout_s * 1000.0

    async def start(self) -> None:
        """Start the background batch worker."""
        if self._started:
            return
        self._worker_task = asyncio.create_task(self._worker_loop(), name="opf-serve-batcher")
        self._started = True

    async def stop(self) -> None:
        """Stop the batch worker and shut down the executor."""
        worker_task = self._worker_task
        self._worker_task = None
        self._started = False
        if worker_task is not None:
            worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await worker_task
        self._executor.shutdown(wait=True, cancel_futures=True)

    async def submit(self, text: str) -> dict[str, Any]:
        """Queue one text for batched inference and await its result."""
        return (await self.submit_many([text]))[0]

    async def submit_many(self, texts: Sequence[str]) -> list[dict[str, Any]]:
        """Queue multiple texts and await their results with shared backpressure."""
        if not texts:
            return []
        if len(texts) > self.max_queue_size:
            raise RuntimeError(
                "batch request is too large for the configured queue; reduce client batch size"
            )
        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future[dict[str, Any]]] = []
        for text in texts:
            future: asyncio.Future[dict[str, Any]] = loop.create_future()
            await self._queue.put(_QueuedRequest(text=text, future=future))
            futures.append(future)
        return list(await asyncio.gather(*futures))

    def _run_redact_many(
        self,
        texts: Sequence[str],
    ) -> tuple[str | RedactionResult, ...]:
        """Run synchronous batched inference on the shared OPF instance."""
        return self._redactor.redact_many(
            texts,
            window_batch_size=self._window_batch_size,
        )

    @staticmethod
    def _serialize_result(result: str | RedactionResult) -> dict[str, Any]:
        """Convert one inference result into the HTTP JSON payload."""
        if isinstance(result, RedactionResult):
            return result.to_dict()
        return {"redacted_text": str(result)}

    async def _worker_loop(self) -> None:
        """Drain the queue, form micro-batches, and dispatch inference."""
        loop = asyncio.get_running_loop()
        while True:
            first = await self._queue.get()
            batch = [first]
            deadline = loop.time() + self._batch_timeout_s
            try:
                while len(batch) < self._max_batch_size:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        queued = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    except TimeoutError:
                        break
                    batch.append(queued)

                results = await loop.run_in_executor(
                    self._executor,
                    self._run_redact_many,
                    [item.text for item in batch],
                )
                if len(results) != len(batch):
                    raise RuntimeError(
                        "Batched inference returned an unexpected number of results"
                    )
                for item, result in zip(batch, results):
                    if not item.future.cancelled():
                        item.future.set_result(self._serialize_result(result))
            except Exception as exc:
                for item in batch:
                    if not item.future.cancelled():
                        item.future.set_exception(exc)
            finally:
                for _item in batch:
                    self._queue.task_done()


def settings_from_args(args: argparse.Namespace) -> ServiceSettings:
    """Build immutable service settings from parsed CLI args."""
    return ServiceSettings(
        checkpoint=args.checkpoint,
        context_window_length=args.n_ctx,
        trim_whitespace=bool(args.trim_span_whitespace),
        device=str(args.device),
        output_mode=str(args.output_mode),
        decode_mode=str(args.decode_mode),
        discard_overlapping_predicted_spans=bool(
            args.discard_overlapping_predicted_spans
        ),
        viterbi_calibration_path=args.viterbi_calibration_path,
        host=str(args.host),
        port=int(args.port),
        log_level=str(args.log_level),
        max_batch_size=int(args.max_batch_size),
        batch_timeout_ms=float(args.batch_timeout_ms),
        max_queue_size=int(args.max_queue_size),
        window_batch_size=(
            int(args.window_batch_size)
            if args.window_batch_size is not None
            else None
        ),
        warmup=bool(args.warmup),
    )


def build_redactor(settings: ServiceSettings) -> OPF:
    """Construct the shared redactor used by the HTTP service."""
    redactor = OPF(
        model=settings.checkpoint,
        context_window_length=settings.context_window_length,
        trim_whitespace=settings.trim_whitespace,
        device=settings.device,
        output_mode=settings.output_mode,
        decode_mode=settings.decode_mode,
        discard_overlapping_predicted_spans=settings.discard_overlapping_predicted_spans,
        output_text_only=False,
    )
    if settings.decode_mode == "viterbi":
        return redactor.set_viterbi_decoder(
            calibration_path=settings.viterbi_calibration_path,
        )
    return redactor.set_decode_mode("argmax")


def create_app(settings: ServiceSettings) -> Any:
    """Create the FastAPI application for the OPF service."""
    try:
        from fastapi import Body, FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OPF service dependencies are missing. Run `pip install -e .` again "
            "or install `fastapi` and `uvicorn`."
        ) from exc

    redactor = build_redactor(settings)
    service = BatchedRedactionService(
        redactor,
        max_batch_size=settings.max_batch_size,
        batch_timeout_ms=settings.batch_timeout_ms,
        max_queue_size=settings.max_queue_size,
        window_batch_size=settings.window_batch_size,
    )

    class RedactRequest(BaseModel):
        text: str = Field(..., description="One input text to redact.")

    class BatchRedactRequest(BaseModel):
        texts: list[str] = Field(
            default_factory=list,
            description="Multiple input texts to redact in one HTTP call.",
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.warmup:
            await asyncio.to_thread(redactor.get_prediction_components)
        await service.start()
        app.state.redactor = redactor
        app.state.batched_redaction_service = service
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(
        title="OPF Service",
        summary="OpenAI Privacy Filter HTTP service with micro-batched inference.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "queue_size": service.queue_size,
            "max_queue_size": service.max_queue_size,
            "max_batch_size": service.max_batch_size,
            "batch_timeout_ms": service.batch_timeout_ms,
            "window_batch_size": settings.window_batch_size,
            "device": settings.device,
            "output_mode": settings.output_mode,
            "decode_mode": settings.decode_mode,
        }

    @app.post("/redact")
    async def redact(payload: RedactRequest = Body(...)) -> dict[str, Any]:
        try:
            return await service.submit(payload.text)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/redact/batch")
    async def redact_batch(
        payload: BatchRedactRequest = Body(...),
    ) -> list[dict[str, Any]]:
        try:
            return await service.submit_many(payload.texts)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def run(args: argparse.Namespace) -> None:
    """Run the OPF HTTP service with uvicorn."""
    settings = settings_from_args(args)
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OPF service dependencies are missing. Run `pip install -e .` again "
            "or install `fastapi` and `uvicorn`."
        ) from exc

    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )
