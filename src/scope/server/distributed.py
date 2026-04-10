"""Distributed multi-GPU coordination for torchrun-launched processes.

Usage (two GPUs):

    torchrun --nproc_per_node=2 -m scope.server.app [server args]
    # or equivalently:
    torchrun --nproc_per_node=2 $(which daydream-scope) [server args]

Architecture:
- Rank-0 runs the HTTP/WebRTC server as normal.
- Ranks 1..N-1 run run_worker_loop(), which blocks waiting for broadcast
  commands from rank-0.
- Before each inference call, rank-0 broadcasts CMD_CALL + kwargs so that
  all ranks participate in the NCCL collectives inside the transformer.
- Before loading a distributed-capable pipeline, rank-0 broadcasts CMD_LOAD
  so all ranks build the pipeline on their own device simultaneously.
- The NCCL broadcasts serve as natural synchronisation barriers: rank-0 cannot
  advance past broadcast_command() until all worker ranks have called
  receive_command(), which means workers are always ready before inference
  starts.

Currently supported pipelines: helios, helios-sdedit
"""

import logging
import os
import pickle

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Module-level state (set once by init_distributed)
_rank: int = 0
_world_size: int = 1
_device: torch.device | None = None

# Worker command codes broadcast as int64 scalar tensors.
CMD_STOP = 0
CMD_LOAD = 1
CMD_CALL = 2

# Pipelines that support distributed context-parallel inference.
DISTRIBUTED_PIPELINE_IDS: frozenset[str] = frozenset({"helios", "helios-sdedit"})


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def init_distributed() -> bool:
    """Initialise the NCCL process group when launched via torchrun.

    Reads the RANK / WORLD_SIZE / LOCAL_RANK environment variables that
    torchrun injects automatically.  Returns True if a multi-GPU group was
    initialised, False on single-GPU / CPU runs.
    """
    global _rank, _world_size, _device

    if not dist.is_available() or "RANK" not in os.environ:
        return False

    dist.init_process_group(backend="nccl")
    _rank = dist.get_rank()
    _world_size = dist.get_world_size()
    _device = torch.device("cuda", _rank % torch.cuda.device_count())
    torch.cuda.set_device(_device)

    logger.info(
        "Distributed mode active: rank=%d world_size=%d device=%s",
        _rank,
        _world_size,
        _device,
    )
    return True


def is_distributed() -> bool:
    return _world_size > 1


def get_rank() -> int:
    return _rank


def get_world_size() -> int:
    return _world_size


def get_device() -> torch.device:
    """Return the CUDA device for this rank (or a default single-GPU device)."""
    if _device is not None:
        return _device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_main_rank() -> bool:
    """True for rank-0 (the process that runs the HTTP server)."""
    return _rank == 0


# ---------------------------------------------------------------------------
# Broadcast helpers
# ---------------------------------------------------------------------------


def _broadcast_bytes_from_rank0(data: bytes) -> None:
    """Rank-0: broadcast arbitrary bytes to every rank."""
    buf = torch.frombuffer(data, dtype=torch.uint8).clone().to(_device)
    size = torch.tensor([len(buf)], dtype=torch.int64, device=_device)
    dist.broadcast(size, src=0)
    dist.broadcast(buf, src=0)


def _receive_bytes_from_rank0() -> bytes:
    """Non-rank-0: receive a byte payload sent by rank-0."""
    size = torch.zeros(1, dtype=torch.int64, device=_device)
    dist.broadcast(size, src=0)
    buf = torch.zeros(int(size.item()), dtype=torch.uint8, device=_device)
    dist.broadcast(buf, src=0)
    return bytes(buf.cpu().numpy())


def broadcast_command(cmd: int, payload: object | None = None) -> None:
    """Rank-0: send a command + optional picklable payload to all ranks.

    Blocks until every rank has received the command tensor (NCCL semantics).
    """
    cmd_t = torch.tensor([cmd], dtype=torch.int64, device=_device)
    dist.broadcast(cmd_t, src=0)
    if payload is not None:
        _broadcast_bytes_from_rank0(pickle.dumps(payload))


def receive_command() -> tuple[int, object | None]:
    """Non-rank-0: block until the next command arrives from rank-0.

    Returns (cmd_code, payload).  payload is None for CMD_STOP.
    """
    cmd_t = torch.zeros(1, dtype=torch.int64, device=_device)
    dist.broadcast(cmd_t, src=0)
    cmd = int(cmd_t.item())
    payload: object | None = None
    if cmd in (CMD_LOAD, CMD_CALL):
        payload = pickle.loads(_receive_bytes_from_rank0())
    return cmd, payload


# ---------------------------------------------------------------------------
# Distributed pipeline wrapper (used by rank-0 only)
# ---------------------------------------------------------------------------


class DistributedPipelineWrapper:
    """Wraps a real pipeline; broadcasts every inference call to worker ranks.

    The wrapper is transparent: attribute access and prepare() are forwarded
    to the underlying pipeline.  __call__ broadcasts CMD_CALL before delegating
    to the real pipeline so that all ranks participate in the NCCL collectives
    inside the transformer.
    """

    def __init__(self, pipeline, pipeline_id: str) -> None:
        # Use object.__setattr__ to avoid triggering __getattr__ during init.
        object.__setattr__(self, "_pipeline", pipeline)
        object.__setattr__(self, "_pipeline_id", pipeline_id)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_pipeline"), name)

    def prepare(self, **kwargs):
        pipeline = object.__getattribute__(self, "_pipeline")
        if hasattr(pipeline, "prepare"):
            return pipeline.prepare(**kwargs)
        return None

    def __call__(self, **kwargs):
        pipeline_id = object.__getattribute__(self, "_pipeline_id")
        pipeline = object.__getattribute__(self, "_pipeline")
        broadcast_command(CMD_CALL, {"_pipeline_id": pipeline_id, **kwargs})
        return pipeline(**kwargs)


# ---------------------------------------------------------------------------
# Worker loop (runs on ranks 1..N-1)
# ---------------------------------------------------------------------------


def run_worker_loop() -> None:
    """Entry point for non-rank-0 processes.

    Loads pipelines and participates in inference collectives on request from
    rank-0.  Blocks until CMD_STOP is received, then returns so the caller can
    clean up the process group.
    """
    from scope.server.pipeline_manager import PipelineManager

    manager = PipelineManager()
    pipelines: dict[str, object] = {}

    logger.info("Worker rank %d ready, entering loop", _rank)

    while True:
        cmd, payload = receive_command()

        if cmd == CMD_STOP:
            logger.info("Worker rank %d received STOP, exiting loop", _rank)
            break

        elif cmd == CMD_LOAD:
            pipeline_id: str = payload["pipeline_id"]
            load_params: dict = payload.get("load_params") or {}
            logger.info("Worker rank %d loading pipeline %s", _rank, pipeline_id)
            try:
                pipeline = manager._load_pipeline_implementation(pipeline_id, load_params)
                pipelines[pipeline_id] = pipeline
                logger.info("Worker rank %d loaded %s", _rank, pipeline_id)
            except Exception:
                logger.exception(
                    "Worker rank %d failed to load pipeline %s", _rank, pipeline_id
                )
            finally:
                # Must match the dist.barrier() call on rank-0 that follows
                # broadcast_command(CMD_LOAD, ...) in pipeline_manager.py.
                dist.barrier()

        elif cmd == CMD_CALL:
            pipeline_id = payload.pop("_pipeline_id")
            pipeline = pipelines.get(pipeline_id)
            if pipeline is None:
                logger.error(
                    "Worker rank %d: pipeline %s not loaded — cannot run inference",
                    _rank,
                    pipeline_id,
                )
                # We cannot safely continue participating in NCCL collectives
                # if the pipeline is missing.  Log and let the NCCL timeout
                # surface the problem rather than silently hang.
                continue
            try:
                pipeline(**payload)  # result discarded on non-rank-0
            except Exception:
                logger.exception(
                    "Worker rank %d: error during inference for %s", _rank, pipeline_id
                )
