"""
Test script for HeliosVACEPipeline.

Supports multiple modes (same as reward_forcing/test_vace.py):
- R2V: Reference-to-Video using reference images (condition only, don't appear in output)
- Depth guidance: Structural guidance using depth maps
- Inpainting: Masked video-to-video generation
- Extension: Temporal extension with reference frames in output
  * firstframe: first_frame_image at start of first chunk
  * lastframe:  last_frame_image at end of last chunk
  * firstlastframe: first_frame_image at start AND last_frame_image at end

Modes can be combined (e.g. R2V + Depth, Extension + Inpainting, etc.).

Key distinction - R2V vs Extension:
- R2V (ref_images): Reference images condition the video but DON'T appear in output
- Extension (first_frame_image/last_frame_image): Reference frames ARE in output video

Usage:
    Edit the CONFIG dictionary below, then:
    python -m scope.core.pipelines.helios.test_vace
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
    # ===== MODE SELECTION =====
    # At least one mode must be enabled.  Modes can be combined.
    "use_r2v": False,         # Reference-to-Video: condition on reference images
    "use_depth": False,       # Depth guidance: structural control via depth maps
    "use_inpainting": True,  # Inpainting: masked video-to-video generation
    "use_extension": False,   # Extension: temporal generation from anchor frames

    # ===== INPUT PATHS =====
    # R2V: reference image paths (condition entire video, don't appear in output)
    "ref_images": [
        "tests/fixtures/cardboard_man_desk.jpg",
    ],
    # Depth: path to depth map video (grayscale or RGB)
    "depth_video": "tests/fixtures/white_square_moving.mp4",
    # Inpainting: input video and mask video
    "input_video": "frontend/public/assets/test.mp4",
    "mask_video": "tests/fixtures/static_mask_half_white_half_black.mp4",
    # Extension: frame images (appear in output video)
    "first_frame_image": "tests/fixtures/cardboard_man_desk.jpg",
    "last_frame_image": "tests/fixtures/blue_electric_lava.webp",
    "extension_mode": "firstframe",  # "firstframe", "lastframe", or "firstlastframe"

    # ===== GENERATION PARAMETERS =====
    "prompt": None,               # Override all mode prompts, or None to use defaults
    "prompt_r2v": "",
    "prompt_depth": "a cat walking towards the camera",
    "prompt_inpainting": "a fireball",
    "prompt_extension": "",
    "num_chunks": 5,
    "height": 384,
    "width": 640,
    "vace_context_scale": 1.0,

    # Pyramid denoising schedule (steps per stage)
    "pyramid_steps": [2, 2, 2],
    "amplify_first_chunk": True,

    # ===== INPAINTING SPECIFIC =====
    "mask_threshold": 0.5,   # Threshold for binarizing mask (0–1)
    "mask_value": 127,       # Gray value for masked regions (0–255)

    # ===== OUTPUT =====
    "output_dir": "output",
}

# ========================= END CONFIGURATION =========================

# ============================= CONSTANTS =============================

# WAN VAE temporal compression: T_pix = (T_lat - 1) * 4 + 1
# Default num_latent_frames_per_chunk = 9 → 33 pixel frames per chunk
_LATENT_FRAMES_PER_CHUNK = 9
_PIXEL_FRAMES_PER_CHUNK = (_LATENT_FRAMES_PER_CHUNK - 1) * 4 + 1  # 33

# ============================= UTILITIES =============================


def resolve_path(path_str: str, relative_to: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (relative_to / p).resolve()


def load_video_tensor(
    path: str,
    height: int,
    width: int,
    max_frames: int | None = None,
) -> torch.Tensor:
    """Load a video and return [T, H, W, C] float32 in [0, 1]."""
    from torchvision.io import read_video

    video, _, _ = read_video(str(path), pts_unit="sec", output_format="TCHW")
    # video: [T, C, H, W] uint8
    video = video.float() / 255.0

    if max_frames is not None and video.shape[0] > max_frames:
        video = video[:max_frames]

    T, C, H, W = video.shape
    if (H, W) != (height, width):
        video = F.interpolate(
            video, size=(height, width), mode="bilinear", align_corners=False
        )
    return video.permute(0, 2, 3, 1).contiguous()  # [T, H, W, C]


def load_image_tensor(path: str, height: int, width: int) -> torch.Tensor:
    """Load an image and return [1, H, W, C] float32 in [0, 1]."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img = img.resize((width, height))
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1, H, W, C]


def video_to_vace_tensor(
    frames: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Convert [T, H, W, C] float [0,1] → [1, C, T, H, W] float [-1,1] on device."""
    # [T, H, W, C] → [T, C, H, W] → [1, C, T, H, W]
    v = frames.permute(0, 3, 1, 2).unsqueeze(0)
    return (v * 2.0 - 1.0).to(device=device, dtype=torch.float32)


def extract_chunk(tensor: torch.Tensor, chunk_idx: int, chunk_size: int) -> torch.Tensor:
    """Slice chunk *chunk_idx* from [1, C, T, H, W]; pad last chunk if needed."""
    T = tensor.shape[2]
    start = min(chunk_idx * chunk_size, T - 1)  # clamp so we always have ≥1 frame to pad from
    end = start + chunk_size
    chunk = tensor[:, :, start:min(end, T), :, :]
    if chunk.shape[2] < chunk_size:
        pad = chunk[:, :, -1:, :, :].repeat(1, 1, chunk_size - chunk.shape[2], 1, 1)
        chunk = torch.cat([chunk, pad], dim=2)
    return chunk


def make_mask_tensor(
    mask_np: np.ndarray,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert binary mask [T, H, W] in {0,1} → [1, 1, T, H, W] on device."""
    t = torch.from_numpy(mask_np).float()  # [T, H, W]
    # Resize spatially if needed
    if t.shape[1:] != (height, width):
        t = t.unsqueeze(1)  # [T, 1, H, W]
        t = F.interpolate(t, size=(height, width), mode="nearest")
        t = t.squeeze(1)    # [T, H, W]
    return t.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T, H, W]


def create_mask_from_video(
    mask_video_path: Path,
    num_frames: int,
    height: int,
    width: int,
    threshold: float,
) -> np.ndarray:
    """Load a mask video and return binary mask [F, H, W] in {0, 1}."""
    from torchvision.io import read_video

    video, _, _ = read_video(str(mask_video_path), pts_unit="sec", output_format="TCHW")
    mask_gray = video.float().mean(dim=1) / 255.0  # [T, H, W]

    # Resize spatially
    if (mask_gray.shape[1], mask_gray.shape[2]) != (height, width):
        mask_gray = F.interpolate(
            mask_gray.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(1)

    binary = (mask_gray > threshold).float().numpy()

    # Loop or truncate to num_frames
    T = binary.shape[0]
    if T < num_frames:
        reps = (num_frames // T) + 1
        binary = np.tile(binary, (reps, 1, 1))[:num_frames]
    else:
        binary = binary[:num_frames]
    return binary


def load_depth_video(
    path: Path,
    height: int,
    width: int,
    max_frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Load depth video → [1, 3, F, H, W] in [-1, 1] (3-channel, replicated)."""
    from torchvision.io import read_video

    video, _, _ = read_video(str(path), pts_unit="sec", output_format="TCHW")
    video = video.float() / 255.0  # [T, C, H, W]
    if video.shape[0] > max_frames:
        video = video[:max_frames]
    if (video.shape[2], video.shape[3]) != (height, width):
        video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)
    # Use first channel as grayscale depth, replicate to 3ch
    depth = video[:, 0:1, :, :].repeat(1, 3, 1, 1)  # [T, 3, H, W]
    depth = depth * 2.0 - 1.0
    return depth.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, 3, T, H, W]


# ============================= MAIN =============================


def main():
    print("=" * 70)
    print("  Helios VACE Test")
    print("=" * 70)

    cfg = CONFIG
    use_r2v = cfg["use_r2v"]
    use_depth = cfg["use_depth"]
    use_inpainting = cfg["use_inpainting"]
    use_extension = cfg["use_extension"]

    if not (use_r2v or use_depth or use_inpainting or use_extension):
        raise ValueError("At least one mode must be enabled in CONFIG")

    # Select prompt
    if cfg["prompt"] is not None:
        prompt = cfg["prompt"]
    elif use_extension:
        prompt = cfg["prompt_extension"]
    elif use_inpainting:
        prompt = cfg["prompt_inpainting"]
    elif use_depth:
        prompt = cfg["prompt_depth"]
    else:
        prompt = cfg["prompt_r2v"]

    height, width = cfg["height"], cfg["width"]
    num_chunks = cfg["num_chunks"]
    total_pixel_frames = num_chunks * _PIXEL_FRAMES_PER_CHUNK

    print(f"\nModes:   R2V={use_r2v}  Depth={use_depth}  "
          f"Inpainting={use_inpainting}  Extension={use_extension}")
    if use_extension:
        print(f"  Extension mode: {cfg['extension_mode']}")
    print(f"Prompt:  {prompt!r}")
    print(f"Chunks:  {num_chunks} x {_PIXEL_FRAMES_PER_CHUNK} frames "
          f"({_LATENT_FRAMES_PER_CHUNK} latent frames each)")
    print(f"Resolution: {width}x{height}")
    print(f"VACE scale: {cfg['vace_context_scale']}\n")

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
            "vace_context_scale": cfg["vace_context_scale"],
            "height": height,
            "width": width,
            "base_seed": 42,
        }
    )

    print("Initializing HeliosVACEPipeline …")
    from scope.core.pipelines.helios.pipeline_vace import HeliosVACEPipeline

    pipeline = HeliosVACEPipeline(pipeline_cfg, device=device, dtype=torch.bfloat16)
    print("Pipeline ready.\n")

    # ------------------------------------------------------------------
    # Prepare inputs
    # ------------------------------------------------------------------

    # R2V: tile reference image across all chunks
    r2v_video: torch.Tensor | None = None
    if use_r2v:
        print("=== Preparing R2V Inputs ===")
        r2v_frames = []
        for img_path in cfg["ref_images"]:
            resolved = resolve_path(img_path, project_root)
            if not resolved.exists():
                print(f"  Warning: ref image not found: {resolved}")
                continue
            img = load_image_tensor(str(resolved), height, width)  # [1, H, W, C]
            r2v_frames.append(img)
            print(f"  Loaded: {resolved}")
        if r2v_frames:
            # Tile first ref image to full video length
            ref_frame = r2v_frames[0]  # [1, H, W, C]
            r2v_video = ref_frame.repeat(total_pixel_frames, 1, 1, 1)  # [T, H, W, C]
            r2v_video = video_to_vace_tensor(r2v_video, device)  # [1, 3, T, H, W]
            print(f"  R2V tensor shape: {r2v_video.shape}\n")
        else:
            print("  No valid ref images — disabling R2V\n")
            use_r2v = False

    # Depth video
    depth_video: torch.Tensor | None = None
    if use_depth:
        print("=== Preparing Depth Inputs ===")
        depth_path = resolve_path(cfg["depth_video"], project_root)
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth video not found: {depth_path}")
        depth_video = load_depth_video(depth_path, height, width, total_pixel_frames, device)
        print(f"  Depth tensor shape: {depth_video.shape}\n")

    # Inpainting video + mask
    input_video: torch.Tensor | None = None
    mask_np: np.ndarray | None = None
    if use_inpainting:
        print("=== Preparing Inpainting Inputs ===")
        input_path = resolve_path(cfg["input_video"], project_root)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")
        mask_path = resolve_path(cfg["mask_video"], project_root)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask video not found: {mask_path}")

        src = load_video_tensor(str(input_path), height, width, total_pixel_frames)
        # [T, H, W, C] float [0,1]

        mask_np = create_mask_from_video(mask_path, total_pixel_frames, height, width,
                                         cfg["mask_threshold"])

        # Create masked video: fill controlled region with gray
        mask_value_f = cfg["mask_value"] / 255.0
        mask_exp = torch.from_numpy(mask_np).float().unsqueeze(-1)  # [T, H, W, 1]
        masked_src = torch.where(mask_exp > 0.5, torch.full_like(src, mask_value_f), src)

        input_video = video_to_vace_tensor(masked_src, device)  # [1, 3, T, H, W]
        print(f"  Input video shape: {input_video.shape}")
        print(f"  Mask shape: {mask_np.shape}, mean={mask_np.mean():.3f}\n")

        # Save masked preview
        preview_path = output_dir / "input_masked_video.mp4"
        export_to_video(masked_src.numpy(), str(preview_path), fps=24)
        print(f"  Saved masked preview: {preview_path}")

    # Extension: load anchor frame images
    first_frame_tensor: torch.Tensor | None = None
    last_frame_tensor: torch.Tensor | None = None
    if use_extension:
        print("=== Preparing Extension Inputs ===")
        ext_mode = cfg["extension_mode"]
        if ext_mode in ("firstframe", "firstlastframe"):
            p = resolve_path(cfg["first_frame_image"], project_root)
            if not p.exists():
                raise FileNotFoundError(f"First frame image not found: {p}")
            first_frame_tensor = load_image_tensor(str(p), height, width)  # [1, H, W, C]
            print(f"  First frame: {p}")
        if ext_mode in ("lastframe", "firstlastframe"):
            p = resolve_path(cfg["last_frame_image"], project_root)
            if not p.exists():
                raise FileNotFoundError(f"Last frame image not found: {p}")
            last_frame_tensor = load_image_tensor(str(p), height, width)  # [1, H, W, C]
            print(f"  Last frame: {p}")
        print()

    # ------------------------------------------------------------------
    # Prepare pipeline VACE state
    # ------------------------------------------------------------------
    if use_r2v and r2v_video is not None and not (use_inpainting or use_extension):
        # R2V-only (or R2V + depth): pre-encode full reference tiling offline
        # For combinations with inpainting/extension we use per-chunk path below.
        print("Pre-encoding R2V reference video …")
        ref_tchw = r2v_video[0].permute(1, 2, 3, 0)  # [T, H, W, C] in [-1,1] → need [0,1]
        ref_tchw = (ref_tchw + 1.0) / 2.0
        pipeline.prepare(vace_video=ref_tchw.to(device))
        print(f"  VACE latents: {pipeline._vace_latents_full.shape}\n")
    else:
        pipeline.prepare()

    # ------------------------------------------------------------------
    # Generate chunks
    # ------------------------------------------------------------------
    print("=== Generating ===")
    outputs = []
    latencies = []

    for chunk_idx in range(num_chunks):
        t0 = time.time()
        is_first = chunk_idx == 0
        is_last = chunk_idx == num_chunks - 1

        kwargs: dict = {
            "prompts": [{"text": prompt, "weight": 100}],
            "pyramid_steps": cfg["pyramid_steps"],
            "amplify_first_chunk": cfg["amplify_first_chunk"],
            "init_cache": is_first,
        }

        # Determine VACE input for this chunk (per-chunk modes)
        # Priority: inpainting > depth > r2v (per-chunk fallback) > extension
        # Extension is composited on top of whatever else is active.

        vace_frames: torch.Tensor | None = None
        vace_mask: torch.Tensor | None = None

        if use_depth and use_inpainting:
            # Composite: depth in masked region, source video in unmasked region
            depth_chunk = extract_chunk(depth_video, chunk_idx, _PIXEL_FRAMES_PER_CHUNK)
            input_chunk = extract_chunk(input_video, chunk_idx, _PIXEL_FRAMES_PER_CHUNK)
            T = _PIXEL_FRAMES_PER_CHUNK
            start = chunk_idx * T
            mask_slice = mask_np[start:start + T]
            if mask_slice.shape[0] < T:
                pad_rows = np.tile(mask_slice[-1:], (T - mask_slice.shape[0], 1, 1))
                mask_slice = np.concatenate([mask_slice, pad_rows], axis=0)
            m = make_mask_tensor(mask_slice, height, width, device)  # [1, 1, T, H, W]
            m_exp = m.expand_as(input_chunk)
            vace_frames = torch.where(m_exp > 0.5, depth_chunk, input_chunk)
            vace_mask = m

        elif use_depth:
            vace_frames = extract_chunk(depth_video, chunk_idx, _PIXEL_FRAMES_PER_CHUNK)
            # no mask → full-frame depth control

        elif use_inpainting:
            T = _PIXEL_FRAMES_PER_CHUNK
            start = chunk_idx * T
            mask_slice = mask_np[start:start + T]
            if mask_slice.shape[0] < T:
                pad_rows = np.tile(mask_slice[-1:], (T - mask_slice.shape[0], 1, 1))
                mask_slice = np.concatenate([mask_slice, pad_rows], axis=0)
            vace_frames = extract_chunk(input_video, chunk_idx, _PIXEL_FRAMES_PER_CHUNK)
            vace_mask = make_mask_tensor(mask_slice, height, width, device)

        elif use_r2v and r2v_video is not None and (use_inpainting or use_extension):
            # R2V per-chunk fallback (when combined with inpainting/extension)
            vace_frames = extract_chunk(r2v_video, chunk_idx, _PIXEL_FRAMES_PER_CHUNK)

        # Extension: overlay anchor frames on vace_frames
        if use_extension:
            ext_mode = cfg["extension_mode"]
            T = _PIXEL_FRAMES_PER_CHUNK

            apply_first = ext_mode in ("firstframe", "firstlastframe") and is_first
            apply_last = ext_mode in ("lastframe", "firstlastframe") and is_last

            if apply_first or apply_last:
                # Build extension video and mask for this chunk
                ext_video = torch.zeros(1, 3, T, height, width, device=device)
                # Mask: 1 = generate freely, 0 = reference frame (preserved)
                ext_mask = torch.ones(1, 1, T, height, width, device=device)

                if apply_first and first_frame_tensor is not None:
                    # [1, H, W, C] → [1, 3, 1, H, W] in [-1,1]
                    ff = (first_frame_tensor.permute(0, 3, 1, 2) * 2.0 - 1.0).to(device)
                    ext_video[:, :, 0:1, :, :] = ff
                    ext_mask[:, :, 0:1, :, :] = 0.0  # preserve first frame

                if apply_last and last_frame_tensor is not None:
                    lf = (last_frame_tensor.permute(0, 3, 1, 2) * 2.0 - 1.0).to(device)
                    ext_video[:, :, -1:, :, :] = lf
                    ext_mask[:, :, -1:, :, :] = 0.0  # preserve last frame

                if vace_frames is not None:
                    # Blend: use extension anchor where mask=0, existing vace_frames elsewhere
                    vace_frames = torch.where(ext_mask.expand_as(vace_frames) < 0.5,
                                              ext_video, vace_frames)
                    if vace_mask is None:
                        vace_mask = ext_mask
                    else:
                        # Intersect: 0 (preserve) wins
                        vace_mask = torch.min(vace_mask, ext_mask)
                else:
                    vace_frames = ext_video
                    vace_mask = ext_mask

        if vace_frames is not None:
            kwargs["vace_input_frames"] = vace_frames
        if vace_mask is not None:
            kwargs["vace_input_masks"] = vace_mask

        result = pipeline(**kwargs)
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

    mode_parts = []
    if use_r2v:
        mode_parts.append("r2v")
    if use_depth:
        mode_parts.append("depth")
    if use_inpainting:
        mode_parts.append("inpainting")
    if use_extension:
        mode_parts.append(f"extension_{cfg['extension_mode']}")
    suffix = "_".join(mode_parts)

    output_path = output_dir / f"helios_vace_{suffix}.mp4"
    export_to_video(output_video, str(output_path), fps=24)
    print(f"\nSaved: {output_path}")

    # ------------------------------------------------------------------
    # Save visualizations
    # ------------------------------------------------------------------
    if use_depth and depth_video is not None:
        depth_vis = depth_video[0, 0, :output_video.shape[0]].cpu().numpy()
        depth_vis = ((depth_vis + 1.0) / 2.0)
        depth_vis_rgb = np.stack([depth_vis, depth_vis, depth_vis], axis=-1)
        export_to_video(depth_vis_rgb, str(output_dir / "depth_maps.mp4"), fps=24)
        print(f"Saved: {output_dir / 'depth_maps.mp4'}")

    if use_inpainting and mask_np is not None:
        mask_vis = np.stack([mask_np, mask_np, mask_np], axis=-1)
        export_to_video(mask_vis, str(output_dir / "mask_visualization.mp4"), fps=24)
        print(f"Saved: {output_dir / 'mask_visualization.mp4'}")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    print("\n=== Performance ===")
    print(f"Latency  avg={sum(latencies)/len(latencies):.2f}s  "
          f"min={min(latencies):.2f}s  max={max(latencies):.2f}s")
    total_frames = output_video.shape[0]
    total_time = sum(latencies)
    print(f"Overall  {total_frames} frames in {total_time:.1f}s ({total_frames/total_time:.1f} fps)")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
