"""Pipeline processor for running a single pipeline in a thread."""

import logging
import queue
import threading
import time
from collections import deque
from typing import Any

import torch

from scope.core.pipelines.controller import parse_ctrl_input
from scope.core.pipelines.wan2_1.vace import VACEEnabledPipeline

from .kafka_publisher import publish_event
from .pipeline_manager import PipelineNotAvailableException
from .pipeline_throttler import PipelineThrottler

logger = logging.getLogger(__name__)

# Multiply the # of output frames from pipeline by this to get the max size of the output queue
OUTPUT_QUEUE_MAX_SIZE_FACTOR = 3

SLEEP_TIME = 0.01

# FPS calculation constants
MIN_FPS = 1.0  # Minimum FPS to prevent division by zero
MAX_FPS = 60.0  # Maximum FPS cap
OUTPUT_FPS_SAMPLE_SIZE = 30
OUTPUT_FPS_MIN_SAMPLES = 2


class PipelineProcessor:
    """Processes frames through a single pipeline in a dedicated thread."""

    def __init__(
        self,
        pipeline: Any,
        pipeline_id: str,
        initial_parameters: dict = None,
        session_id: str | None = None,
        user_id: str | None = None,
        connection_id: str | None = None,
        connection_info: dict | None = None,
    ):
        """Initialize a pipeline processor.

        Args:
            pipeline: Pipeline instance to process frames with
            pipeline_id: ID of the pipeline (used for logging)
            initial_parameters: Initial parameters for the pipeline
            session_id: Session ID for event tracking
            user_id: User ID for event tracking
            connection_id: Connection ID from fal.ai WebSocket for event correlation
            connection_info: Connection metadata (gpu_type, region, etc.)
        """
        self.pipeline = pipeline
        self.pipeline_id = pipeline_id
        self.session_id = session_id
        self.user_id = user_id
        self.connection_id = connection_id
        self.connection_info = connection_info

        # Each processor creates its own queues
        self.input_queue = queue.Queue(maxsize=30)
        self.output_queue = queue.Queue(maxsize=8)
        # Lock to protect input_queue assignment for thread-safe reference swapping
        self.input_queue_lock = threading.Lock()

        # Current parameters used by processing thread
        self.parameters = initial_parameters or {}
        # Queue for parameter updates from external threads
        self.parameters_queue = queue.Queue(maxsize=8)

        self.worker_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()
        self.running = False

        self.is_prepared = False

        # Output FPS tracking — measured at batch granularity, not per-frame.
        # Per-frame tracking produces artificially high FPS because all frames in a
        # chunk are enqueued in a tight burst (microseconds apart), hiding the true
        # inter-chunk cycle time. Batch-level tracking gives the real sustained rate:
        #   fps = num_frames / (wall-clock time since previous batch completed)
        self._last_batch_complete_time: float | None = None
        self._sustained_fps_ema: float = MAX_FPS  # EMA; starts high, converges quickly
        self.current_output_fps = MAX_FPS
        self.output_fps_lock = threading.Lock()
        # Keep legacy fields so existing callers don't break
        self.output_frame_deltas: deque = deque(maxlen=OUTPUT_FPS_SAMPLE_SIZE)
        self._last_frame_time: float | None = None

        self.paused = False
        # Input mode is signaled by the frontend at stream start
        self._video_mode = (initial_parameters or {}).get("input_mode") == "video"

        # Reference to next processor in chain (if chained)
        # Used to update next processor's input_queue when output_queue is reassigned
        self.next_processor: PipelineProcessor | None = None

        # Route based on frontend's VACE intent (not pipeline.vace_enabled which is lazy-loaded)
        # This fixes the chicken-and-egg problem where VACE isn't enabled until vace_input_frames arrives
        self.vace_enabled = (initial_parameters or {}).get("vace_enabled", False)
        self.vace_use_input_video = (initial_parameters or {}).get(
            "vace_use_input_video", True
        )

        # Cache VACE support check to avoid isinstance on every chunk
        self._pipeline_supports_vace = isinstance(pipeline, VACEEnabledPipeline)

        # Throttler for controlling processing rate in chained pipelines
        # Throttling is applied when this pipeline produces frames faster than
        # the next pipeline in the chain can consume them
        self.throttler = PipelineThrottler()

    def _resize_output_queue(self, target_size: int):
        """Resize the output queue to the target size, transferring existing frames.

        Args:
            target_size: The desired maximum size for the output queue
        """
        if self.output_queue is None:
            return

        if self.output_queue.maxsize < target_size:
            logger.info(
                f"Increasing output queue size to {target_size}, current size {self.output_queue.maxsize}"
            )

            # Transfer frames from old queue to new queue
            old_queue = self.output_queue
            self.output_queue = queue.Queue(maxsize=target_size)
            while not old_queue.empty():
                try:
                    frame = old_queue.get_nowait()
                    self.output_queue.put_nowait(frame)
                except queue.Empty:
                    break

            # Update next processor's input_queue to point to the new output_queue
            # Use lock to ensure thread-safe reference swapping
            if self.next_processor is not None:
                with self.next_processor.input_queue_lock:
                    self.next_processor.input_queue = self.output_queue

    def set_next_processor(self, next_processor: "PipelineProcessor"):
        """Set the next processor in the chain and update output queue size accordingly.

        Args:
            next_processor: The next pipeline processor in the chain
        """
        self.next_processor = next_processor

        # Set throttler's reference to next processor for throttling decisions
        self.throttler.set_next_processor(next_processor)

        # Calculate output queue size based on next processor's requirements
        next_pipeline = next_processor.pipeline
        if hasattr(next_pipeline, "prepare"):
            requirements = next_pipeline.prepare(video=True)
            input_size = requirements.input_size
            target_size = max(8, input_size * OUTPUT_QUEUE_MAX_SIZE_FACTOR)
            self._resize_output_queue(target_size)

        # Update next processor's input_queue to point to this output_queue
        # Use lock to ensure thread-safe reference swapping
        with next_processor.input_queue_lock:
            next_processor.input_queue = self.output_queue

    def start(self):
        """Start the pipeline processor thread."""
        if self.running:
            return

        self.running = True
        self.shutdown_event.clear()

        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

        logger.info(f"PipelineProcessor started for pipeline: {self.pipeline_id}")

    def stop(self):
        """Stop the pipeline processor thread."""
        if not self.running:
            return

        self.running = False
        self.shutdown_event.set()
        self.throttler.interrupt()

        if self.worker_thread and self.worker_thread.is_alive():
            if threading.current_thread() != self.worker_thread:
                self.worker_thread.join(timeout=5.0)

        # Clear queues
        with self.input_queue_lock:
            input_queue_ref = self.input_queue
        if input_queue_ref:
            while not input_queue_ref.empty():
                try:
                    input_queue_ref.get_nowait()
                except queue.Empty:
                    break

        if self.output_queue:
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    break

        logger.info(f"PipelineProcessor stopped for pipeline: {self.pipeline_id}")

    def update_parameters(self, parameters: dict[str, Any]):
        """Update parameters that will be used in the next pipeline call."""
        try:
            self.parameters_queue.put_nowait(parameters)
        except queue.Full:
            logger.info(
                f"Parameter queue full for {self.pipeline_id}, dropping parameter update"
            )
            return False

    def worker_loop(self):
        """Main worker loop that processes frames."""
        logger.info(f"Worker thread started for pipeline: {self.pipeline_id}")

        while self.running and not self.shutdown_event.is_set():
            try:
                self.process_chunk()

            except PipelineNotAvailableException as e:
                logger.debug(
                    f"Pipeline {self.pipeline_id} temporarily unavailable: {e}"
                )
                # Sleep briefly and continue
                self.shutdown_event.wait(SLEEP_TIME)
                continue
            except Exception as e:
                if self._is_recoverable(e):
                    logger.error(
                        f"Error in worker loop for {self.pipeline_id}: {e}",
                        exc_info=True,
                    )
                    continue
                else:
                    logger.error(
                        f"Non-recoverable error in worker loop for {self.pipeline_id}: {e}, stopping"
                    )
                    # Publish error event for pipeline processing failure
                    publish_event(
                        event_type="error",
                        session_id=self.session_id,
                        connection_id=self.connection_id,
                        pipeline_ids=[self.pipeline_id],
                        user_id=self.user_id,
                        error={
                            "error_type": "pipeline_processing_failed",
                            "message": str(e),
                            "exception_type": type(e).__name__,
                            "recoverable": False,
                        },
                        connection_info=self.connection_info,
                    )
                    break

        logger.info(f"Worker thread stopped for pipeline: {self.pipeline_id}")

    def prepare_chunk(
        self,
        input_queue_ref: queue.Queue,
        chunk_size: int,
        frame_strategy: str = "uniform",
    ) -> list[torch.Tensor]:
        """
        Pull chunk_size frames from the queue according to frame_strategy.

        Strategies
        ----------
        "sequential"
            Take the oldest chunk_size frames (FIFO). Chunks are temporally
            consecutive with no gaps. Any surplus frames remain in the queue
            for the next chunk. Best for pre-recorded video files.

        "realtime"
            Discard all frames except the newest chunk_size. Minimises
            input latency at the cost of skipping intermediate frames.
            Best for live camera streams.

        "uniform"  (default / legacy)
            Uniformly sample chunk_size frames across the full queue and
            discard everything up to the last sampled frame. Balances
            temporal coverage with queue draining.

        Args:
            input_queue_ref: Reference to the input queue (caller holds lock).
            chunk_size: Number of frames to return.
            frame_strategy: One of "sequential", "realtime", or "uniform".

        Returns:
            List of tensor frames, each (1, H, W, C).
        """
        if frame_strategy == "sequential":
            # FIFO: take exactly the first chunk_size frames, leave the rest.
            frames = []
            for _ in range(chunk_size):
                frames.append(input_queue_ref.get_nowait())
            return frames

        if frame_strategy == "realtime":
            # Drain everything, keep only the last chunk_size frames.
            all_frames = []
            while not input_queue_ref.empty():
                try:
                    all_frames.append(input_queue_ref.get_nowait())
                except queue.Empty:
                    break
            return all_frames[-chunk_size:]

        # Default: "uniform" — legacy behaviour.
        step = input_queue_ref.qsize() / chunk_size
        indices = [round(i * step) for i in range(chunk_size)]
        frames = []
        last_idx = indices[-1]
        for i in range(last_idx + 1):
            frame = input_queue_ref.get_nowait()
            if i in indices:
                frames.append(frame)
        return frames

    def process_chunk(self):
        """Process a single chunk of frames."""
        # Check if there are new parameters
        try:
            new_parameters = self.parameters_queue.get_nowait()
            if new_parameters != self.parameters:
                # Clear stale transition when new prompts arrive without transition
                if (
                    "prompts" in new_parameters
                    and "transition" not in new_parameters
                    and "transition" in self.parameters
                ):
                    self.parameters.pop("transition", None)

                # Update video mode if input_mode parameter changes
                if "input_mode" in new_parameters:
                    self._video_mode = new_parameters.get("input_mode") == "video"

                # Accumulate ctrl_input: keys = latest, mouse = sum
                if "ctrl_input" in new_parameters:
                    if "ctrl_input" in self.parameters:
                        existing = self.parameters["ctrl_input"]
                        new_ctrl = new_parameters["ctrl_input"]
                        new_parameters["ctrl_input"] = {
                            "button": new_ctrl.get("button", []),
                            "mouse": [
                                existing.get("mouse", [0, 0])[0]
                                + new_ctrl.get("mouse", [0, 0])[0],
                                existing.get("mouse", [0, 0])[1]
                                + new_ctrl.get("mouse", [0, 0])[1],
                            ],
                        }

                # Merge new parameters with existing ones
                self.parameters = {**self.parameters, **new_parameters}
        except queue.Empty:
            pass

        # Pause or resume the processing
        paused = self.parameters.pop("paused", None)
        if paused is not None and paused != self.paused:
            # Reset so the next FPS delta doesn't span the pause/unpause gap
            self._last_frame_time = None
            self.paused = paused
        if self.paused:
            self.shutdown_event.wait(SLEEP_TIME)
            return

        # Prepare pipeline
        reset_cache = self.parameters.pop("reset_cache", None)
        lora_scales = self.parameters.pop("lora_scales", None)

        # Handle reset_cache: clear this processor's cache
        if reset_cache:
            logger.info(f"Clearing cache for pipeline processor: {self.pipeline_id}")
            # Mark as unprepared so init_cache=True reaches the pipeline even if
            # this iteration returns early (e.g. waiting for enough video frames).
            # Without this, reset_cache is consumed here but the early-return path
            # causes the next iteration to call the pipeline with init_cache=False,
            # silently skipping the state reset.
            self.is_prepared = False
            # Clear output queue
            if self.output_queue:
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        break

        requirements = None
        if hasattr(self.pipeline, "prepare"):
            prepare_params = dict(self.parameters.items())
            if self._video_mode:
                # Signal to prepare() that video input is expected
                prepare_params["video"] = True
            requirements = self.pipeline.prepare(**prepare_params)

        video_input = None
        input_frame_count = 0
        if requirements is not None:
            current_chunk_size = requirements.input_size

            # Ensure the input queue is large enough to hold a full chunk.
            # This is needed when pipelines request more frames than the default
            # queue capacity (e.g. SDEdit requests 33 frames for the WAN VAE).
            with self.input_queue_lock:
                if self.input_queue.maxsize < current_chunk_size:
                    old_q = self.input_queue
                    new_q = queue.Queue(maxsize=current_chunk_size + 10)
                    while not old_q.empty():
                        try:
                            new_q.put_nowait(old_q.get_nowait())
                        except queue.Full:
                            break
                    self.input_queue = new_q
                    logger.info(
                        "Expanded input queue to %d (chunk_size=%d)",
                        new_q.maxsize, current_chunk_size,
                    )
                input_queue_ref = self.input_queue

            # Check if queue has enough frames before consuming them
            if input_queue_ref.qsize() < current_chunk_size:
                # Not enough frames in queue, sleep briefly and try again next iteration
                self.shutdown_event.wait(SLEEP_TIME)
                return

            # Use prepare_chunk to sample frames from the queue
            video_input = self.prepare_chunk(
                input_queue_ref,
                current_chunk_size,
                frame_strategy=requirements.frame_strategy,
            )
            input_frame_count = len(video_input) if video_input else 0

        try:
            # Pass parameters (excluding prepare-only parameters)
            call_params = dict(self.parameters.items())

            # Pass reset_cache as init_cache to pipeline
            call_params["init_cache"] = not self.is_prepared
            if reset_cache is not None:
                call_params["init_cache"] = reset_cache

            # Pass lora_scales only when present
            if lora_scales is not None:
                call_params["lora_scales"] = lora_scales

            # Extract ctrl_input, parse it, and reset mouse for next frame
            if "ctrl_input" in self.parameters:
                ctrl_data = self.parameters["ctrl_input"]
                call_params["ctrl_input"] = parse_ctrl_input(ctrl_data)
                # Reset mouse accumulator, keep key state
                self.parameters["ctrl_input"]["mouse"] = [0.0, 0.0]

            # Route video input based on VACE status
            # Don't overwrite if preprocessor already provided vace_input_frames
            if video_input is not None and "vace_input_frames" not in call_params:
                if (
                    self._pipeline_supports_vace
                    and self.vace_enabled
                    and self.vace_use_input_video
                ):
                    call_params["vace_input_frames"] = video_input
                else:
                    call_params["video"] = video_input

            processing_start = time.time()
            output_dict = self.pipeline(**call_params)
            processing_time = time.time() - processing_start

            # Extract video from the returned dictionary
            output = output_dict.get("video")
            if output is None:
                return

            # Forward extra params to downstream pipeline (dual-output pattern)
            # Preprocessors return {"video": frames, "vace_input_frames": ..., "vace_input_masks": ...}
            extra_params = {k: v for k, v in output_dict.items() if k != "video"}
            if extra_params and self.next_processor is not None:
                self.next_processor.update_parameters(extra_params)

            # Clear one-shot parameters after use to prevent sending them on subsequent chunks
            # These parameters should only be sent when explicitly provided in parameter updates
            one_shot_params = [
                "vace_ref_images",
                "images",
                "first_frame_image",
                "last_frame_image",
            ]
            for param in one_shot_params:
                if param in call_params and param in self.parameters:
                    self.parameters.pop(param, None)

            # Clear transition when complete
            if "transition" in call_params and "transition" in self.parameters:
                transition_active = False
                if hasattr(self.pipeline, "state"):
                    transition_active = self.pipeline.state.get(
                        "_transition_active", False
                    )

                transition = call_params.get("transition")
                if not transition_active or transition is None:
                    self.parameters.pop("transition", None)

            num_frames = output.shape[0]

            # Record batch timing for throttling calculations
            if input_frame_count > 0:
                self.throttler.record_input_batch(input_frame_count, processing_time)
            if num_frames > 0:
                self.throttler.record_output_batch(num_frames, processing_time)

            # Normalize to [0, 255] and convert to uint8
            # Keep frames on GPU - frame_processor handles CPU transfer for streaming
            output = (
                (output * 255.0)
                .clamp(0, 255)
                .to(dtype=torch.uint8)
                .contiguous()
                .detach()
            )

            # Resize output queue to meet target max size
            target_output_queue_max_size = num_frames * OUTPUT_QUEUE_MAX_SIZE_FACTOR
            self._resize_output_queue(target_output_queue_max_size)

            # Put frames in output queue
            # For intermediate pipelines, output goes to next pipeline's input
            # For last pipeline, output goes to frame_processor's output_queue
            # Output frames are [H, W, C], convert to [1, H, W, C] for consistency
            for frame in output:
                frame = frame.unsqueeze(0)
                try:
                    self.output_queue.put_nowait(frame)
                except queue.Full:
                    logger.info(
                        f"Output queue full for {self.pipeline_id}, dropping processed frame"
                    )
                    continue

            # Update sustained-FPS estimate at batch granularity.
            # Measuring here (once per chunk) avoids the burst-timing artifact that
            # makes per-frame tracking report 60 fps even when chunks take seconds.
            self._track_output_batch(num_frames, processing_time)

            # Apply throttling if this pipeline is producing faster than next can consume
            # Only throttle if: (1) has video input, (2) has next processor
            if video_input is not None and self.next_processor is not None:
                self.throttler.throttle()

        except Exception as e:
            if self._is_recoverable(e):
                logger.error(
                    f"Error processing chunk for {self.pipeline_id}: {e}", exc_info=True
                )
            else:
                raise e

        self.is_prepared = True

    def _track_output_batch(self, num_frames: int, generation_time: float = 0.0):
        """Update the sustained-FPS estimate from one completed output batch.

        Uses wall-clock time between successive batch completions so that the
        full generate → enqueue cycle is captured, not just the burst-enqueue
        time within a single chunk (which would give artificially high FPS).

        For the first batch there is no previous completion time, so we bootstrap
        the EMA from the pipeline's own generation time to avoid the cold-start
        problem where the EMA starts at MAX_FPS and takes many chunks to converge.

        Args:
            num_frames: Number of frames produced in this batch.
            generation_time: Wall-clock seconds the pipeline call itself took.
                             Only used for bootstrapping the first-chunk estimate.
        """
        now = time.time()
        with self.output_fps_lock:
            if self._last_batch_complete_time is not None and num_frames > 0:
                # Normal case: measure the full chunk-to-chunk cycle time.
                cycle_time = now - self._last_batch_complete_time
                if cycle_time > 0:
                    batch_fps = num_frames / cycle_time
                    batch_fps = max(MIN_FPS, min(MAX_FPS, batch_fps))
                    # EMA with α=0.4: responds quickly but smooths single-chunk outliers
                    alpha = 0.4
                    self._sustained_fps_ema = (
                        alpha * batch_fps + (1 - alpha) * self._sustained_fps_ema
                    )
                    self.current_output_fps = self._sustained_fps_ema
                    logger.debug(
                        "Batch FPS update: frames=%d cycle=%.2fs "
                        "batch_fps=%.2f ema_fps=%.2f",
                        num_frames, cycle_time, batch_fps, self._sustained_fps_ema,
                    )
            elif self._last_batch_complete_time is None and generation_time > 0 and num_frames > 0:
                # First batch: bootstrap from the pipeline's own generation time.
                # Without this the EMA starts at MAX_FPS (60) and takes many chunks
                # to converge, causing a burst-then-stall on the first few chunks.
                initial_fps = num_frames / generation_time
                initial_fps = max(MIN_FPS, min(MAX_FPS, initial_fps))
                self._sustained_fps_ema = initial_fps
                self.current_output_fps = initial_fps
                logger.debug(
                    "Batch FPS bootstrap: frames=%d gen_time=%.2fs initial_fps=%.2f",
                    num_frames, generation_time, initial_fps,
                )
            self._last_batch_complete_time = now

    def get_fps(self) -> float:
        """Get the current dynamically calculated pipeline FPS.

        Returns the FPS based on how fast frames are produced into the output queue,
        adjusted for queue fill level to prevent buildup.
        """
        with self.output_fps_lock:
            output_fps = self.current_output_fps
        return min(MAX_FPS, output_fps)

    @staticmethod
    def _is_recoverable(error: Exception) -> bool:
        """Check if an error is recoverable."""
        if isinstance(error, torch.cuda.OutOfMemoryError):
            return False
        return True
