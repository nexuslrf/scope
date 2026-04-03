"""
Test script for HeliosSDEditPipeline (SDEdit, FlowEdit, FlowAlign).

Mirrors the structure of test_vace.py.

Usage:
    Edit the CONFIG dictionary below, then:
    python -m scope.core.pipelines.helios.test_sdedit
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.config import get_models_dir

# ============================= CONFIGURATION =============================

CONFIG = {
    # ===== EDIT TYPE =====
    # "sdedit"    — noise-based editing (one prompt)
    # "flowedit"  — differential velocity, 4 calls/step (source + target prompts)
    # "flowalign" — DIFS alignment, 3 calls/step (source + target prompts)
    "edit_type": "sdedit",

    # ===== INPUT =====
    "source_video": "frontend/public/assets/test.mp4",

    # ===== PROMPTS =====
    "target_prompt": "A silver Porsche 911 car speeds along a curving mountain road.",
    "source_prompt": "",   # Required for flowedit / flowalign
    "negative_prompt": (
        "Bright tones, overexposed, static, blurred details, subtitles, style, "
        "works, paintings, images, static, overall gray, worst quality, low quality, "
        "JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
        "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
        "still picture, messy background, three legs, many people in the background, "
        "walking backwards"
    ),

    # ===== EDIT STRENGTH =====
    # SDEdit/FlowEdit edit_stage:
    #   0.0 = maximum edit (pure noise start)
    #   1.0 = start of pyramid stage 1 (default, good balance)
    #   2.0 = start of pyramid stage 2 (minimum edit, fine details preserved)
    "edit_stage": 1.0,

    # FlowEdit/FlowAlign guidance scales (1.0 = distilled, no CFG needed)
    "source_guidance_scale": 1.0,
    "target_guidance_scale": 1.0,
    "zeta_scale": 1e-3,   # FlowAlign only

    # ===== GENERATION PARAMETERS =====
    "num_chunks": 3,
    "height": 384,
    "width": 640,
    "pyramid_steps": [4, 4, 4],
    "amplify_first_chunk": True,

    # ===== OUTPUT =====
    "output_dir": "output",
}

# ========================= END CONFIGURATION =========================

_LATENT_FRAMES_PER_CHUNK = 9
_PIXEL_FRAMES_PER_CHUNK = (_LATENT_FRAMES_PER_CHUNK - 1) * 4 + 1  # 33


def resolve_path(path_str: str, relative_to: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (relative_to / p).resolve()


def load_video_tensor(path: str, height: int, width: int, max_frames: int | None = None) -> torch.Tensor:
    """Load a video and return [T, H, W, C] float32 in [0, 1]."""
    from torchvision.io import read_video

    video, _, _ = read_video(str(path), pts_unit="sec", output_format="TCHW")
    video = video.float() / 255.0
    if max_frames is not None and video.shape[0] > max_frames:
        video = video[:max_frames]
    T, C, H, W = video.shape
    if (H, W) != (height, width):
        video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)
    return video.permute(0, 2, 3, 1).contiguous()  # [T, H, W, C]


def main():
    print("=" * 70)
    print("  Helios SDEdit / FlowEdit / FlowAlign Test")
    print("=" * 70)

    cfg = CONFIG
    edit_type = cfg["edit_type"]
    height, width = cfg["height"], cfg["width"]
    num_chunks = cfg["num_chunks"]
    total_pixel_frames = num_chunks * _PIXEL_FRAMES_PER_CHUNK

    print(f"\nEdit type:   {edit_type}")
    print(f"Edit stage:  {cfg['edit_stage']}")
    print(f"Target:      {cfg['target_prompt']!r}")
    if edit_type in ("flowedit", "flowalign"):
        print(f"Source:      {cfg['source_prompt']!r}")
    print(f"Chunks:      {num_chunks} x {_PIXEL_FRAMES_PER_CHUNK} frames")
    print(f"Resolution:  {width}x{height}\n")

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent.parent.parent
    output_dir = resolve_path(cfg["output_dir"], script_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Initialize pipeline
    # ------------------------------------------------------------------
    models_dir = get_models_dir()
    pipeline_cfg = OmegaConf.create(
        {
            "model_dir": str(models_dir),
            "num_latent_frames_per_chunk": _LATENT_FRAMES_PER_CHUNK,
            "history_sizes": [16, 2, 1],
            "pyramid_steps": cfg["pyramid_steps"],
            "amplify_first_chunk": cfg["amplify_first_chunk"],
            "guidance_scale": 1.0,
            "height": height,
            "width": width,
            "base_seed": 42,
            "edit_type": edit_type,
            "edit_stage": cfg["edit_stage"],
            "source_prompt": cfg["source_prompt"],
            "source_guidance_scale": cfg["source_guidance_scale"],
            "target_guidance_scale": cfg["target_guidance_scale"],
            "zeta_scale": cfg["zeta_scale"],
        }
    )

    print("Initializing HeliosSDEditPipeline …")
    from scope.core.pipelines.helios.pipeline_sdedit import HeliosSDEditPipeline

    pipeline = HeliosSDEditPipeline(pipeline_cfg, device=device, dtype=torch.bfloat16)
    print("Pipeline ready.\n")

    # ------------------------------------------------------------------
    # Load and pre-encode source video
    # ------------------------------------------------------------------
    src_path = resolve_path(cfg["source_video"], project_root)
    if not src_path.exists():
        raise FileNotFoundError(f"Source video not found: {src_path}")

    print(f"Loading source video: {src_path}")
    src_frames = load_video_tensor(str(src_path), height, width, total_pixel_frames)  # [T, H, W, C]
    print(f"  Loaded {src_frames.shape[0]} frames → {num_chunks} chunks\n")

    # Pre-encode full source video
    print("Pre-encoding source video to latents …")
    pipeline.prepare(video=src_frames.to(device))
    if pipeline._src_latents_full is not None:
        print(f"  Source latents: {pipeline._src_latents_full.shape}\n")

    # Save source video preview
    export_to_video(src_frames.numpy(), str(output_dir / "source_video.mp4"), fps=24)

    # ------------------------------------------------------------------
    # Generate chunks
    # ------------------------------------------------------------------
    print("=== Generating ===")
    outputs = []
    latencies = []

    for chunk_idx in range(num_chunks):
        t0 = time.time()

        result = pipeline(
            prompts=[{"text": cfg["target_prompt"], "weight": 100}],
            negative_prompt=cfg["negative_prompt"],
            pyramid_steps=cfg["pyramid_steps"],
            amplify_first_chunk=cfg["amplify_first_chunk"],
            init_cache=(chunk_idx == 0),
        )
        frames = result["video"]  # [T, H, W, C] float32

        latency = time.time() - t0
        fps = frames.shape[0] / latency
        latencies.append(latency)
        print(f"  Chunk {chunk_idx}: {frames.shape[0]} frames, "
              f"latency={latency:.2f}s, fps={fps:.1f}")
        outputs.append(frames.cpu())

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    output_video = np.concatenate([f.numpy() for f in outputs], axis=0)
    output_video = np.clip(output_video, 0.0, 1.0)

    output_path = output_dir / f"helios_{edit_type}_stage{cfg['edit_stage']}.mp4"
    export_to_video(output_video, str(output_path), fps=24)
    print(f"\nSaved: {output_path}")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    print("\n=== Performance ===")
    print(f"Latency  avg={sum(latencies)/len(latencies):.2f}s  "
          f"min={min(latencies):.2f}s  max={max(latencies):.2f}s")
    total_frames = output_video.shape[0]
    total_time = sum(latencies)
    print(f"Overall  {total_frames} frames in {total_time:.1f}s ({total_frames/total_time:.1f} fps)")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
