import argparse
import time
from pathlib import Path

import torch
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

from scope.core.config import get_models_dir
from scope.core.pipelines.utils import parse_jsonl_prompts, print_statistics

from .pipeline import HeliosPipeline

# Helios-Distilled generates 9 latent frames per chunk; with temporal compression
# factor 4 (causal VAE) the decoder produces ~33 pixel frames per chunk at 24 fps.
OUTPUT_FPS = 24
DEFAULT_NUM_CHUNKS = 5  # ~165 frames (~7 s of video)


def generate_video(
    pipeline: HeliosPipeline,
    prompt_texts: list[str],
    output_path: Path,
    num_chunks: int = DEFAULT_NUM_CHUNKS,
) -> tuple[list[float], list[float]]:
    """Generate a video from a sequence of prompts.

    Each prompt is used for `num_chunks` autoregressive chunks. On the first
    chunk of a new prompt the history cache is reset (init_cache=True).

    Args:
        pipeline: The HeliosPipeline instance.
        prompt_texts: List of prompt strings; each drives `num_chunks` chunks.
        output_path: Path to save the output MP4.
        num_chunks: Number of chunks to generate per prompt.

    Returns:
        Tuple of (latency_measures, fps_measures) across all chunks.
    """
    outputs = []
    latency_measures = []
    fps_measures = []

    for prompt_text in prompt_texts:
        for chunk_idx in range(num_chunks):
            start = time.time()

            output_dict = pipeline(
                prompt=prompt_text,
                init_cache=(chunk_idx == 0),
            )
            output = output_dict["video"]  # THWC, float32 [0, 1]

            elapsed = time.time() - start
            num_frames = output.shape[0]
            fps = num_frames / elapsed

            print(
                f"[prompt={prompt_text[:40]!r} chunk={chunk_idx}] "
                f"{num_frames} frames, latency={elapsed:.2f}s, fps={fps:.2f}"
            )

            latency_measures.append(elapsed)
            fps_measures.append(fps)
            outputs.append(output.detach().cpu())

    output_video = torch.cat(outputs)  # (total_T, H, W, C)
    print(f"Total output shape: {tuple(output_video.shape)}")
    export_to_video(output_video.numpy(), str(output_path), fps=OUTPUT_FPS)

    return latency_measures, fps_measures


def main():
    parser = argparse.ArgumentParser(description="Test Helios pipeline")
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Path to a JSONL file containing prompt sequences",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=DEFAULT_NUM_CHUNKS,
        help=f"Number of autoregressive chunks per prompt (default: {DEFAULT_NUM_CHUNKS})",
    )
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    config = OmegaConf.create(
        {
            "model_dir": str(get_models_dir()),
            "height": args.height,
            "width": args.width,
            # Remaining fields use HeliosConfig defaults.
            "num_latent_frames_per_chunk": 9,
            "history_sizes": [16, 2, 1],
            "pyramid_steps": [2, 2, 2],
            "amplify_first_chunk": True,
            "guidance_scale": 1.0,
            "base_seed": 42,
        }
    )

    device = torch.device("cuda")
    pipeline = HeliosPipeline(config, device=device, dtype=torch.bfloat16)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    if args.prompts:
        prompt_sequences = parse_jsonl_prompts(args.prompts)
        print(f"Loaded {len(prompt_sequences)} prompt sequences from {args.prompts}")

        all_latency = []
        all_fps = []

        for i, prompt_texts in enumerate(prompt_sequences):
            print(f"\n=== Video {i} ({len(prompt_texts)} prompt(s)) ===")
            output_path = output_dir / f"output_{i}.mp4"
            lat, fps = generate_video(pipeline, prompt_texts, output_path, args.num_chunks)
            all_latency.extend(lat)
            all_fps.extend(fps)
            print(f"Saved {output_path}")

        print_statistics(all_latency, all_fps)
    else:
        prompt_texts = [
            "A serene aerial view of a misty mountain valley at dawn, golden light breaking through the clouds.",
            "The mist slowly parts as the sun climbs higher, revealing a winding river far below.",
        ]

        output_path = output_dir / "output.mp4"
        lat, fps = generate_video(pipeline, prompt_texts, output_path, args.num_chunks)
        print(f"Saved {output_path}")
        print_statistics(lat, fps)


if __name__ == "__main__":
    main()
