"""Helios autoregressive video diffusion pipeline for scope.

Generates one ~33-pixel-frame chunk per __call__ invocation, maintaining
autoregressive history across calls for temporal coherence.
"""

import math
import os
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from ..base_schema import BasePipelineConfig
from ..interface import Pipeline
from ..process import postprocess_chunk
from .schema import HeliosConfig

if TYPE_CHECKING:
    pass


def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Compute shift mu for dynamic timestep shifting (from Flux/Helios pipeline)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class HeliosPipeline(Pipeline):
    """Scope pipeline wrapper for Helios-Distilled autoregressive video generation.

    Generates video chunk-by-chunk using pyramid denoising (stage2 / DMD).
    T2V only — no video input is read; history is maintained internally.
    """

    @classmethod
    def get_config_class(cls) -> type[BasePipelineConfig]:
        return HeliosConfig

    def __init__(
        self,
        config,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        from diffusers.models import AutoencoderKLWan

        from ._vendor.scheduling_helios import HeliosScheduler
        from ._vendor.transformer_helios import HeliosTransformer3DModel
        from ..wan2_1.components.text_encoder import WanTextEncoderWrapper
        from ..wan2_1.modules.tokenizers import HuggingfaceTokenizer

        self.device = device or torch.device("cpu")
        self.dtype = dtype
        self._config = config

        model_dir = getattr(config, "model_dir", None)
        if model_dir:
            helios_path = os.path.join(model_dir, "Helios-Distilled")
            wan_base_path = os.path.join(model_dir, "Wan2.1-T2V-1.3B")
        else:
            helios_path = "BestWishYsh/Helios-Distilled"
            wan_base_path = "Wan-AI/Wan2.1-T2V-1.3B"

        # --- Transformer ---
        self.transformer = HeliosTransformer3DModel.from_pretrained(
            helios_path, subfolder="transformer", torch_dtype=dtype
        ).to(self.device)
        self.transformer.eval().requires_grad_(False)
        # try:
        #     self.transformer.set_attention_backend("_flash_3_hub")
        # except Exception:
        self.transformer.set_attention_backend("flash_hub")

        # --- VAE (float32 for numerical stability during encode/decode) ---
        self.vae = AutoencoderKLWan.from_pretrained(
            helios_path, subfolder="vae", torch_dtype=torch.float32
        ).to(self.device)
        self.vae.eval().requires_grad_(False)

        # --- Scheduler ---
        self.scheduler = HeliosScheduler.from_pretrained(helios_path, subfolder="scheduler")

        # --- Text encoder: reuse WAN BF16 pth + 226-token tokenizer ---
        self._text_encoder = WanTextEncoderWrapper(
            model_name="Wan2.1-T2V-1.3B",
            model_dir=model_dir,
        ).to(self.device)
        tokenizer_path = (
            os.path.join(wan_base_path, "google", "umt5-xxl")
            if model_dir
            else "google/umt5-xxl"
        )
        self._tokenizer = HuggingfaceTokenizer(tokenizer_path, seq_len=226, clean="whitespace")

        # --- VAE normalization constants ---
        self._latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(self.device, torch.float32)
        )
        self._latents_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(self.device, torch.float32)
        )

        # --- Pre-compute fixed RoPE index tensors (keep_first_frame=True) ---
        # history_sizes sorted big→small: [16, 2, 1]
        history_sizes = sorted(config.history_sizes, reverse=True)
        T = config.num_latent_frames_per_chunk
        # Total positions: 1 (anchor) + sum(history_sizes) + T (current)
        indices = torch.arange(0, 1 + sum(history_sizes) + T)
        idx_prefix, idx_long, idx_mid, idx_short_1x, idx_hidden = indices.split(
            [1, *history_sizes, T], dim=0
        )
        self._indices_hidden_states = idx_hidden.unsqueeze(0).to(self.device)
        self._indices_history_short = (
            torch.cat([idx_prefix, idx_short_1x], dim=0).unsqueeze(0).to(self.device)
        )
        self._indices_history_mid = idx_mid.unsqueeze(0).to(self.device)
        self._indices_history_long = idx_long.unsqueeze(0).to(self.device)

        self._history_sizes = history_sizes  # [16, 2, 1] sorted
        self._num_history_frames = sum(history_sizes)  # 19

        # --- Initialize autoregressive state ---
        self._reset_state()
        self.first_call = True

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------

    def prepare(self, **kwargs) -> None:
        """T2V only — no video input required."""
        return None

    def __call__(self, **kwargs) -> dict:
        # Accept either the server's `prompts` list or a plain `prompt` string.
        # Server sends: prompts=[{"text": "...", "weight": N}, ...]
        raw_prompts = kwargs.get("prompts", None)
        if raw_prompts:
            first = raw_prompts[0]
            prompt = first.get("text", "") if isinstance(first, dict) else str(first)
        else:
            prompt = kwargs.get("prompt", "")
        negative_prompt = kwargs.get("negative_prompt", "")
        init_cache = kwargs.get("init_cache", False)
        pyramid_steps = kwargs.get("pyramid_steps", self._config.pyramid_steps)
        amplify_first_chunk = kwargs.get("amplify_first_chunk", self._config.amplify_first_chunk)

        is_first_chunk = self.first_call or init_cache

        if is_first_chunk:
            self._reset_state()
        self.first_call = False

        # Re-encode prompt only when it changes
        if prompt != self._cached_prompt:
            self._cached_prompt_embeds, self._cached_neg_embeds = self._encode_prompt(
                prompt, negative_prompt
            )
            self._cached_prompt = prompt

        # Generate chunk latents via pyramid denoising
        latents = self._generate_chunk(
            self._cached_prompt_embeds,
            self._cached_neg_embeds,
            is_first_chunk=is_first_chunk,
            pyramid_steps=pyramid_steps,
            amplify_first_chunk=amplify_first_chunk,
        )  # [1, C, T, H//8, W//8]

        # Save the first latent frame as the persistent anchor
        if is_first_chunk:
            self._image_latents = latents[:, :, 0:1, :, :].to(torch.float32)

        # Append to rolling history and cap at num_history_frames
        self._history_latents = torch.cat(
            [self._history_latents, latents.to(torch.float32)], dim=2
        )
        if self._history_latents.shape[2] > self._num_history_frames:
            self._history_latents = self._history_latents[:, :, -self._num_history_frames :]

        self._total_generated += latents.shape[2]

        # Decode to pixels
        current_latents = latents.to(torch.float32) / self._latents_std + self._latents_mean
        current_latents = current_latents.to(self.vae.dtype)
        video = self.vae.decode(current_latents, return_dict=False)[0]  # [1, C, T, H, W] BCTHW

        # BCTHW → BTCHW → THWC [0, 1]
        return {"video": postprocess_chunk(video.permute(0, 2, 1, 3, 4))}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Zero-initialize all autoregressive history state."""
        C = self.transformer.config.in_channels
        H = self._config.height // 8
        W = self._config.width // 8
        self._history_latents = torch.zeros(
            1, C, self._num_history_frames, H, W,
            device=self.device, dtype=torch.float32,
        )
        self._image_latents = None
        self._total_generated = 0
        self._cached_prompt = None
        self._cached_prompt_embeds = None
        self._cached_neg_embeds = None

    @torch.no_grad()
    def _encode_prompt(
        self, prompt: str, negative_prompt: str = ""
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize at max_length=226 and encode with UMT5-XXL."""
        ids_pos, mask_pos = self._tokenizer(
            [prompt], return_mask=True, add_special_tokens=True
        )
        ids_neg, mask_neg = self._tokenizer(
            [negative_prompt or ""], return_mask=True, add_special_tokens=True
        )

        ids_pos = ids_pos.to(self.device)
        mask_pos = mask_pos.to(self.device)
        ids_neg = ids_neg.to(self.device)
        mask_neg = mask_neg.to(self.device)

        seq_lens_pos = mask_pos.gt(0).sum(dim=1).long()
        seq_lens_neg = mask_neg.gt(0).sum(dim=1).long()

        # Forward through the underlying UMT5 encoder (seq_len=226)
        embeds_pos = self._text_encoder.text_encoder(ids_pos, mask_pos)  # [1, 226, 4096]
        embeds_neg = self._text_encoder.text_encoder(ids_neg, mask_neg)

        # Zero out padding positions
        for u, v in zip(embeds_pos, seq_lens_pos):
            u[v:] = 0.0
        for u, v in zip(embeds_neg, seq_lens_neg):
            u[v:] = 0.0

        return embeds_pos.to(torch.bfloat16), embeds_neg.to(torch.bfloat16)

    def _sample_block_noise(
        self,
        B: int,
        C: int,
        T: int,
        H: int,
        W: int,
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ) -> torch.Tensor:
        """Sample spatially-correlated block noise for pyramid stage transitions."""
        gamma = self.scheduler.config.gamma
        _, ph, pw = patch_size
        block_size = ph * pw

        cov = (
            torch.eye(block_size, device=self.device) * (1 + gamma)
            - torch.ones(block_size, block_size, device=self.device) * gamma
        )
        cov += torch.eye(block_size, device=self.device) * 1e-6
        dist = torch.distributions.MultivariateNormal(
            torch.zeros(block_size, device=self.device), covariance_matrix=cov
        )
        block_number = B * C * T * (H // ph) * (W // pw)
        noise = dist.sample((block_number,))  # [block_number, block_size]
        noise = noise.view(B, C, T, H // ph, W // pw, ph, pw)
        noise = noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(B, C, T, H, W)
        return noise

    @torch.no_grad()
    def _generate_chunk(
        self,
        prompt_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
        is_first_chunk: bool,
        pyramid_steps: list[int],
        amplify_first_chunk: bool,
    ) -> torch.Tensor:
        """Run 3-stage pyramid denoising to produce one latent chunk.

        Returns:
            Latents [1, C, T, H//8, W//8] in float32.
        """
        T = self._config.num_latent_frames_per_chunk
        H = self._config.height // 8
        W = self._config.width // 8
        C = self.transformer.config.in_channels
        B = 1
        num_stages = len(pyramid_steps)  # 3

        # Build current history slices (all at full H//8, W//8)
        history = self._history_latents[:, :, -self._num_history_frames :]
        lat_long, lat_mid, lat_short_1x = history.split(self._history_sizes, dim=2)

        # Anchor frame: first latent frame of the very first chunk (zeros before that)
        if self._image_latents is None:
            anchor = torch.zeros(1, C, 1, H, W, device=self.device, dtype=torch.float32)
        else:
            anchor = self._image_latents
        lat_short = torch.cat([anchor, lat_short_1x], dim=2)  # [1, C, 2, H, W]

        # Sample fresh noise for the current chunk
        generator = torch.Generator(device=self.device).manual_seed(
            self._config.base_seed + self._total_generated
        )
        latents = torch.randn(
            1, C, T, H, W, generator=generator, device=self.device, dtype=torch.float32
        )

        # Downsample to the coarsest pyramid level (2 bilinear halvings for 3 stages)
        latents_2d = latents.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        h, w = H, W
        for _ in range(num_stages - 1):
            h //= 2
            w //= 2
            latents_2d = F.interpolate(latents_2d, size=(h, w), mode="bilinear") * 2
        latents = latents_2d.reshape(B, T, C, h, w).permute(0, 2, 1, 3, 4)

        # For DMD distilled models, we keep start-point latents per stage
        start_point_list = [latents]

        is_amplify = is_first_chunk and amplify_first_chunk
        patch_size = self.transformer.config.patch_size

        for i_s in range(num_stages):
            # Shift parameter for dynamic timestep scaling
            image_seq_len = (latents.shape[2] * latents.shape[3] * latents.shape[4]) // (
                patch_size[0] * patch_size[1] * patch_size[2]
            )
            mu = _calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            self.scheduler.set_timesteps(
                pyramid_steps[i_s],
                i_s,
                device=self.device,
                mu=mu,
                is_amplify_first_chunk=is_amplify,
            )

            if i_s > 0:
                # Upsample 2× (nearest neighbour, matching original stage2_sample)
                h *= 2
                w *= 2
                latents_2d = latents.permute(0, 2, 1, 3, 4).reshape(
                    B * T, C, h // 2, w // 2
                )
                latents_2d = F.interpolate(latents_2d, size=(h, w), mode="nearest")
                latents = latents_2d.reshape(B, T, C, h, w).permute(0, 2, 1, 3, 4)

                # Block-noise correction to fix upsample artifacts
                ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]
                gamma = self.scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)
                noise = self._sample_block_noise(B, C, T, h, w).to(
                    device=self.device, dtype=torch.float32
                )
                latents = alpha * latents + beta * noise
                start_point_list.append(latents)

            # Denoising loop for this pyramid stage
            for idx, t in enumerate(self.scheduler.timesteps):
                timestep = t.expand(B).to(torch.int64)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents.to(self.dtype),
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        indices_hidden_states=self._indices_hidden_states,
                        indices_latents_history_short=self._indices_history_short,
                        indices_latents_history_mid=self._indices_history_mid,
                        indices_latents_history_long=self._indices_history_long,
                        latents_history_short=lat_short.to(self.dtype),
                        latents_history_mid=lat_mid.to(self.dtype),
                        latents_history_long=lat_long.to(self.dtype),
                        return_dict=False,
                    )[0]

                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    return_dict=False,
                    cur_sampling_step=idx,
                    dmd_noisy_tensor=start_point_list[i_s],
                    dmd_sigmas=self.scheduler.sigmas,
                    dmd_timesteps=self.scheduler.timesteps,
                    all_timesteps=self.scheduler.timesteps,
                )[0]

        return latents  # [1, C, T, H//8, W//8] at full resolution
