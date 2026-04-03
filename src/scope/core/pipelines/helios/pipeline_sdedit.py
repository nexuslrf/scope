"""Helios SDEdit / FlowEdit / FlowAlign pipeline for scope.

Video-to-video editing pipeline that wraps Helios-Distilled with three editing algorithms:

  SDEdit:
    Injects noise into source video latents at a chosen pyramid stage/step,
    then denoises toward the target prompt using the standard DMD denoising loop.
    Edit strength is controlled by `edit_stage` (lower = more editing).

  FlowEdit:
    Differential velocity editing: at each denoising step, subtracts the source
    velocity from the target velocity (Euler integration, 4 transformer calls/step).

  FlowAlign:
    3-call variant of FlowEdit using DIFS alignment (vp - vq) with optional
    zeta correction term for structure preservation.

Streaming design:
  - One chunk (33 pixel frames → 9 latent frames) processed per __call__.
  - Autoregressive history conditioning is maintained identically to the base
    HeliosPipeline for temporal coherence.
  - Source video can be pre-encoded in prepare() or provided live per-chunk.
"""

import logging
import math

import torch
import torch.nn.functional as F

from .pipeline import HeliosPipeline, _calculate_shift
from .schema_sdedit import HeliosSDEditConfig
from ..interface import Requirements
from ..process import postprocess_chunk, preprocess_chunk

logger = logging.getLogger(__name__)

# VAE temporal compression factor: T_pix = (T_lat - 1) * VAE_TEMPORAL + 1
_VAE_TEMPORAL = 4


# ---------------------------------------------------------------------------
# Pyramid helpers (match prepare_stage2_clean_input training formula)
# ---------------------------------------------------------------------------


def _build_latent_pyramid(x0: torch.Tensor, num_stages: int) -> list[torch.Tensor]:
    """Downsample clean latents to each pyramid stage resolution.

    Returns list of length *num_stages*, index 0 = coarsest, -1 = full resolution.
    Bilinear downsampling without ×2 scaling (signal, not noise).

    Args:
        x0: [B, C, T, H, W] float32 normalized latents at full resolution.
        num_stages: Number of pyramid stages (typically 3).
    """
    B, C, T, H, W = x0.shape
    pyramid = [x0]
    cur = x0
    for _ in range(num_stages - 1):
        h_new = cur.shape[-2] // 2
        w_new = cur.shape[-1] // 2
        cur_2d = cur.permute(0, 2, 1, 3, 4).reshape(B * T, C, cur.shape[-2], cur.shape[-1])
        cur_2d = F.interpolate(cur_2d, size=(h_new, w_new), mode="bilinear", align_corners=False)
        cur = cur_2d.reshape(B, T, C, h_new, w_new).permute(0, 2, 1, 3, 4)
        pyramid.append(cur)
    # pyramid[0]=full, pyramid[-1]=coarsest → reverse to [coarsest, ..., full]
    return list(reversed(pyramid))


def _build_noise_pyramid(eps: torch.Tensor, num_stages: int) -> list[torch.Tensor]:
    """Downsample noise to each pyramid stage resolution with ×2 scaling per level.

    Returns list of length *num_stages*, index 0 = coarsest, -1 = full resolution.
    Matches noise_list in prepare_stage2_clean_input.

    Args:
        eps: [B, C, T, H, W] float32 noise at full resolution.
        num_stages: Number of pyramid stages.
    """
    B, C, T, H, W = eps.shape
    pyramid = [eps]
    cur = eps
    for _ in range(num_stages - 1):
        h_new = cur.shape[-2] // 2
        w_new = cur.shape[-1] // 2
        cur_2d = cur.permute(0, 2, 1, 3, 4).reshape(B * T, C, cur.shape[-2], cur.shape[-1])
        cur_2d = F.interpolate(cur_2d, size=(h_new, w_new), mode="bilinear", align_corners=False) * 2
        cur = cur_2d.reshape(B, T, C, h_new, w_new).permute(0, 2, 1, 3, 4)
        pyramid.append(cur)
    return list(reversed(pyramid))


def _compute_sdedit_input(
    scheduler,
    lat_pyr: list[torch.Tensor],
    noise_pyr: list[torch.Tensor],
    num_stages: int,
    stage_idx: int,
    step_idx: int,
    T_stage: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SDEdit noisy input x_t and DMD anchor start_point at stage_idx.

    Mirrors prepare_stage2_clean_input training formula:
      stage 0:  start_point = noise_pyr[0]
      stage k:  start_point = start_sigma[k]*noise_k + (1-start_sigma[k])*upsample(x0_{k-1})
      end_point (not last) = end_sigma[k]*noise_k + (1-end_sigma[k])*x0_k
      end_point (last)     = x0_k
      x_t = sigma_t * start_point + (1 - sigma_t) * end_point

    Returns:
        (x_t, start_point) both at stage_idx resolution, in lat_pyr[stage_idx].dtype.
    """
    device = lat_pyr[stage_idx].device
    dtype = lat_pyr[stage_idx].dtype
    B, C, T = lat_pyr[stage_idx].shape[:3]

    # start_point
    if stage_idx == 0:
        start_point = noise_pyr[0].to(device=device, dtype=torch.float32)
    else:
        prev_clean = lat_pyr[stage_idx - 1].to(device=device, dtype=torch.float32)
        tgt_h, tgt_w = lat_pyr[stage_idx].shape[-2], lat_pyr[stage_idx].shape[-1]
        B_p, C_p, T_p = prev_clean.shape[:3]
        flat = prev_clean.permute(0, 2, 1, 3, 4).reshape(B_p * T_p, C_p, prev_clean.shape[-2], prev_clean.shape[-1])
        flat = F.interpolate(flat, size=(tgt_h, tgt_w), mode="nearest")
        prev_up = flat.reshape(B_p, T_p, C_p, tgt_h, tgt_w).permute(0, 2, 1, 3, 4)
        start_sigma_k = scheduler.start_sigmas[stage_idx]
        noise_k = noise_pyr[stage_idx].to(device=device, dtype=torch.float32)
        start_point = start_sigma_k * noise_k + (1 - start_sigma_k) * prev_up

    # end_point
    x0_k = lat_pyr[stage_idx].to(device=device, dtype=torch.float32)
    noise_k_ep = noise_pyr[stage_idx].to(device=device, dtype=torch.float32)
    if stage_idx == num_stages - 1:
        end_point = x0_k
    else:
        end_sigma_k = scheduler.end_sigmas[stage_idx]
        end_point = end_sigma_k * noise_k_ep + (1 - end_sigma_k) * x0_k

    # sigma at step_idx in the full T_stage linspace
    s0 = float(scheduler.sigmas_per_stage[stage_idx][0].item())  # ≈ 0.999
    s1 = float(scheduler.sigmas_per_stage[stage_idx][-1].item())  # ≈ 0
    sigma_t = s0 if T_stage <= 1 else s0 + (s1 - s0) * step_idx / (T_stage - 1)

    x_t = sigma_t * start_point + (1 - sigma_t) * end_point
    return x_t.to(dtype), start_point.to(dtype)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class HeliosSDEditPipeline(HeliosPipeline):
    """Helios pipeline for video-to-video editing via SDEdit, FlowEdit, or FlowAlign.

    Extends HeliosPipeline with:
    - prepare(): pre-encode source video or request live per-chunk frames
    - __call__(): routes to SDEdit or FlowEdit denoising based on edit_type config
    - Maintains full autoregressive history for temporal coherence across chunks
    """

    @classmethod
    def get_config_class(cls):
        return HeliosSDEditConfig

    def __init__(
        self,
        config,
        device=None,
        dtype=torch.bfloat16,
        enable_context_parallel: bool = False,
        cp_backend: str = "ulysses",
    ):
        super().__init__(
            config,
            device=device,
            dtype=dtype,
            enable_context_parallel=enable_context_parallel,
            cp_backend=cp_backend,
        )

        # Source video state (set by prepare())
        self._src_latents_full: torch.Tensor | None = None  # [1, C, T_total_lat, H_lat, W_lat]
        self._src_chunk_counter: int = 0

        # Source prompt embedding cache (FlowEdit / FlowAlign)
        self._cached_source_prompt: str | None = None
        self._cached_src_embeds: torch.Tensor | None = None
        self._cached_src_neg_embeds: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------

    def prepare(self, **kwargs):
        """Prepare source video for editing.

        Modes:
          - prepare(video=<[T,H,W,C] float [0,1] tensor>): pre-encode full source video.
          - prepare(video=True): live mode — server provides frames per chunk.
          - prepare(): no source video, T2V fallback.

        Called by the server on every chunk iteration.  We intentionally do NOT
        reset _src_latents_full here so a pre-encoded source video persists across
        the whole streaming session.  Only an explicit new tensor resets it.

        Returns:
          Requirements(input_size=9) in live mode (≤ queue maxsize=30), None otherwise.
        """
        src_video = kwargs.get("video", None)

        if src_video is True:
            # Live mode: request 9 frames (well below queue maxsize=30).
            # _get_source_latents will temporally interpolate to the VAE's required
            # pixel frame count before encoding.
            return Requirements(input_size=9)

        if src_video is not None and src_video is not False:
            # Explicit tensor: pre-encode full source video [T, H, W, C] float [0, 1]
            pixel = src_video.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, H, W]
            pixel = (pixel * 2.0 - 1.0).to(device=self.device, dtype=torch.float32)
            with torch.no_grad():
                latents_btchw = self.vae.encode_to_latent(pixel, use_cache=False)  # [1, T_lat, C, H, W]
            self._src_latents_full = latents_btchw.permute(0, 2, 1, 3, 4).to(torch.float32)
            self._src_chunk_counter = 0
            logger.info(
                "HeliosSDEditPipeline: pre-encoded source video, "
                "latents shape=%s", self._src_latents_full.shape
            )
        elif src_video is None and not kwargs.get("video", False):
            # No video at all — clear any pre-encoded source (T2V mode)
            self._src_latents_full = None

        return None

    def __call__(self, **kwargs) -> dict:
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
            self._src_chunk_counter = 0
        self.first_call = False

        if prompt != self._cached_prompt:
            self._cached_prompt_embeds, self._cached_neg_embeds = self._encode_prompt(
                prompt, negative_prompt
            )
            self._cached_prompt = prompt

        # Get source latents for this chunk
        video_frames = kwargs.get("video", None)
        x0_src = self._get_source_latents(self._src_chunk_counter, video_frames)

        edit_type = kwargs.get("edit_type", getattr(self._config, "edit_type", "sdedit"))
        edit_stage = float(kwargs.get("edit_stage", getattr(self._config, "edit_stage", 1.0)))
        target_gs = float(kwargs.get("target_guidance_scale", getattr(self._config, "target_guidance_scale", 1.0)))
        source_gs = float(kwargs.get("source_guidance_scale", getattr(self._config, "source_guidance_scale", 1.0)))
        zeta = float(kwargs.get("zeta_scale", getattr(self._config, "zeta_scale", 1e-3)))

        if x0_src is not None and edit_type == "sdedit":
            latents = self._generate_chunk_sdedit(
                x0_src, self._cached_prompt_embeds, self._cached_neg_embeds,
                is_first_chunk, pyramid_steps, amplify_first_chunk,
                edit_stage=edit_stage,
            )
        elif x0_src is not None and edit_type in ("flowedit", "flowalign"):
            src_prompt = kwargs.get("source_prompt", getattr(self._config, "source_prompt", ""))
            if src_prompt != self._cached_source_prompt:
                self._cached_src_embeds, self._cached_src_neg_embeds = self._encode_prompt(
                    src_prompt, negative_prompt
                )
                self._cached_source_prompt = src_prompt
            latents = self._generate_chunk_flowedit(
                x0_src,
                self._cached_prompt_embeds,
                self._cached_neg_embeds,
                self._cached_src_embeds,
                self._cached_src_neg_embeds,
                is_first_chunk,
                pyramid_steps,
                is_flowalign=(edit_type == "flowalign"),
                edit_stage=edit_stage,
                target_gs=target_gs,
                source_gs=source_gs,
                zeta=zeta,
            )
        else:
            # Fallback: standard T2V (no source video provided)
            logger.debug("HeliosSDEditPipeline: no source video, falling back to T2V")
            latents = self._generate_chunk(
                self._cached_prompt_embeds, self._cached_neg_embeds,
                is_first_chunk, pyramid_steps, amplify_first_chunk,
            )

        self._src_chunk_counter += 1

        if is_first_chunk:
            self._image_latents = latents[:, :, 0:1, :, :].to(torch.float32)

        self._history_latents = torch.cat(
            [self._history_latents, latents.to(torch.float32)], dim=2
        )
        if self._history_latents.shape[2] > self._num_history_frames:
            self._history_latents = self._history_latents[:, :, -self._num_history_frames:]
        self._total_generated += latents.shape[2]

        latents_btchw = latents.to(torch.float32).permute(0, 2, 1, 3, 4)
        video = self.vae.decode_to_pixel(latents_btchw, use_cache=False)
        return {"video": postprocess_chunk(video)}

    # ------------------------------------------------------------------
    # Source video helpers
    # ------------------------------------------------------------------

    def _get_source_latents(
        self,
        chunk_idx: int,
        video_frames=None,
    ) -> torch.Tensor | None:
        """Return source latents [1, C, T_lat, H_lat, W_lat] float32 for this chunk.

        Priority: pre-encoded full video (from prepare()) > live per-chunk input.
        Returns None when no source video is available.
        """
        T = self._config.num_latent_frames_per_chunk

        if self._src_latents_full is not None:
            start = chunk_idx * T
            end = start + T
            total = self._src_latents_full.shape[2]
            if start >= total:
                # Beyond source video → zero tensor (equivalent to T2V)
                return torch.zeros(
                    1, self._src_latents_full.shape[1], T,
                    self._src_latents_full.shape[3], self._src_latents_full.shape[4],
                    device=self.device, dtype=torch.float32,
                )
            chunk = self._src_latents_full[:, :, start:min(end, total), :, :]
            if chunk.shape[2] < T:
                # Pad last chunk by repeating the final frame
                pad = chunk[:, :, -1:, :, :].expand(-1, -1, T - chunk.shape[2], -1, -1)
                chunk = torch.cat([chunk, pad], dim=2)
            return chunk.to(self.device)

        if video_frames is not None:
            # Encode live camera / uploaded frames
            pixel = preprocess_chunk(
                video_frames,
                device=self.device,
                dtype=torch.float32,
                height=self._config.height,
                width=self._config.width,
            )  # [1, C, T_in, H, W] in [-1, 1]
            # The WAN VAE needs exactly T_pix = (T_lat - 1)*4 + 1 frames to produce
            # T_lat latent frames.  Interpolate temporally if the server sent fewer.
            target_T_pix = (self._config.num_latent_frames_per_chunk - 1) * _VAE_TEMPORAL + 1
            if pixel.shape[2] != target_T_pix:
                pixel = F.interpolate(
                    pixel, size=(target_T_pix, pixel.shape[3], pixel.shape[4]),
                    mode="nearest",
                )
            with torch.no_grad():
                latents_btchw = self.vae.encode_to_latent(pixel, use_cache=False)  # [1, T_lat, C, H, W]
            return latents_btchw.permute(0, 2, 1, 3, 4).to(torch.float32)  # [1, C, T_lat, H, W]

        return None

    # ------------------------------------------------------------------
    # SDEdit denoising
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_chunk_sdedit(
        self,
        x0_src: torch.Tensor,
        prompt_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
        is_first_chunk: bool,
        pyramid_steps: list[int],
        amplify_first_chunk: bool,
        edit_stage: float = 1.0,
    ) -> torch.Tensor:
        """SDEdit: inject noise then denoise toward target prompt.

        Args:
            x0_src: [1, C, T, H_lat, W_lat] float32 normalized source latents.

        Returns:
            [1, C, T, H_lat, W_lat] float32 denoised latents.
        """
        B, C, T, H, W = x0_src.shape
        num_stages = len(pyramid_steps)
        patch_size = self.transformer.config.patch_size

        # Parse edit_stage → (start_stage, start_step)
        start_stage = min(int(edit_stage), num_stages - 1)
        step_frac = edit_stage - int(edit_stage)
        T_stage = pyramid_steps[start_stage]
        start_step = int(step_frac * T_stage)

        # Build pyramids
        lat_pyr = _build_latent_pyramid(x0_src, num_stages)
        generator = torch.Generator(device=self.device).manual_seed(
            self._config.base_seed + self._total_generated
        )
        eps_full = torch.randn(x0_src.shape, generator=generator, device=self.device, dtype=torch.float32)
        noise_pyr = _build_noise_pyramid(eps_full, num_stages)

        # Compute SDEdit noisy init and DMD anchor at start_stage resolution
        x_t, start_point = _compute_sdedit_input(
            self.scheduler, lat_pyr, noise_pyr, num_stages, start_stage, start_step, T_stage
        )
        x_t = x_t.to(device=self.device, dtype=torch.float32)
        start_point = start_point.to(device=self.device, dtype=torch.float32)

        # Build autoregressive history slices
        history = self._history_latents[:, :, -self._num_history_frames:]
        lat_long, lat_mid, lat_short_1x = history.split(self._history_sizes, dim=2)
        anchor = (
            self._image_latents
            if self._image_latents is not None
            else torch.zeros(1, C, 1, H, W, device=self.device, dtype=torch.float32)
        )
        lat_short = torch.cat([anchor, lat_short_1x], dim=2)

        is_amplify = is_first_chunk and amplify_first_chunk

        latents = x_t
        start_point_list = [start_point]  # indexed relative to start_stage

        h, w = lat_pyr[start_stage].shape[-2], lat_pyr[start_stage].shape[-1]

        for i_s in range(start_stage, num_stages):
            image_seq_len = (T * h * w) // (patch_size[0] * patch_size[1] * patch_size[2])
            mu = _calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )

            if i_s == start_stage and start_step > 0:
                # Set full schedule then truncate to start from the right sigma
                self.scheduler.set_timesteps(
                    T_stage, i_s, device=self.device, mu=mu,
                    is_amplify_first_chunk=is_amplify,
                )
                self.scheduler.timesteps = self.scheduler.timesteps[start_step:]
                self.scheduler.sigmas = self.scheduler.sigmas[start_step:]
            else:
                self.scheduler.set_timesteps(
                    pyramid_steps[i_s], i_s, device=self.device, mu=mu,
                    is_amplify_first_chunk=is_amplify,
                )

            if i_s > start_stage:
                # Upsample 2× (nearest, matching base pipeline)
                h *= 2
                w *= 2
                latents_2d = latents.permute(0, 2, 1, 3, 4).reshape(B * T, C, h // 2, w // 2)
                latents_2d = F.interpolate(latents_2d, size=(h, w), mode="nearest")
                latents = latents_2d.reshape(B, T, C, h, w).permute(0, 2, 1, 3, 4)

                # Block-noise correction (identical to base pipeline)
                ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]
                gamma = self.scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)
                block_noise_gen = torch.Generator(device=self.device).manual_seed(
                    self._config.base_seed + self._total_generated + i_s
                )
                noise = self._sample_block_noise(B, C, T, h, w, generator=block_noise_gen).to(
                    device=self.device, dtype=torch.float32
                )
                latents = alpha * latents + beta * noise
                start_point_list.append(latents.clone())

            spl_idx = i_s - start_stage
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
                    dmd_noisy_tensor=start_point_list[spl_idx],
                    dmd_sigmas=self.scheduler.sigmas,
                    dmd_timesteps=self.scheduler.timesteps,
                    all_timesteps=self.scheduler.timesteps,
                )[0]

        return latents  # [1, C, T, H, W] at full resolution

    # ------------------------------------------------------------------
    # FlowEdit / FlowAlign denoising
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _generate_chunk_flowedit(
        self,
        x0_src: torch.Tensor,
        tar_prompt_embeds: torch.Tensor,
        tar_neg_embeds: torch.Tensor,
        src_prompt_embeds: torch.Tensor,
        src_neg_embeds: torch.Tensor,
        is_first_chunk: bool,
        pyramid_steps: list[int],
        is_flowalign: bool = False,
        edit_stage: float = 1.0,
        target_gs: float = 1.0,
        source_gs: float = 1.0,
        zeta: float = 1e-3,
    ) -> torch.Tensor:
        """FlowEdit / FlowAlign: differential velocity editing.

        FlowEdit (is_flowalign=False):
          velocity = Vt_tar - Vt_src  (up to 4 transformer calls/step)

        FlowAlign (is_flowalign=True):
          velocity = vp - vq with optional DIFS correction (3 transformer calls/step)

        Args:
            x0_src: [1, C, T, H_lat, W_lat] float32 normalized source latents.
            tar_*: Target prompt embeddings.
            src_*: Source prompt embeddings.

        Returns:
            [1, C, T, H_lat, W_lat] float32 edited latents.
        """
        B, C, T, H, W = x0_src.shape
        num_stages = len(pyramid_steps)
        patch_size = self.transformer.config.patch_size

        start_stage = min(int(edit_stage), num_stages - 1)
        step_frac = edit_stage - int(edit_stage)
        start_step = int(step_frac * pyramid_steps[start_stage])

        if not is_flowalign:
            zeta = 0.0

        # Build pyramids (once for all stages)
        lat_pyr = _build_latent_pyramid(x0_src, num_stages)
        generator = torch.Generator(device=self.device).manual_seed(
            self._config.base_seed + self._total_generated
        )
        eps_full = torch.randn(x0_src.shape, generator=generator, device=self.device, dtype=torch.float32)
        noise_pyr = _build_noise_pyramid(eps_full, num_stages)

        # Build autoregressive history slices
        history = self._history_latents[:, :, -self._num_history_frames:]
        lat_long, lat_mid, lat_short_1x = history.split(self._history_sizes, dim=2)
        anchor = (
            self._image_latents
            if self._image_latents is not None
            else torch.zeros(1, C, 1, H, W, device=self.device, dtype=torch.float32)
        )
        lat_short = torch.cat([anchor, lat_short_1x], dim=2)

        # Initialize Zt_edit at start_stage resolution (= clean source at that stage)
        latents = lat_pyr[start_stage].to(device=self.device, dtype=torch.float32)

        for i_s in range(start_stage, num_stages):
            x0_k = lat_pyr[i_s].to(device=self.device, dtype=torch.float32)
            noise_k = noise_pyr[i_s].to(device=self.device, dtype=torch.float32)

            # Compute start_point for this stage (training-consistent DMD anchor)
            if i_s == 0:
                start_point = noise_k
            else:
                x0_prev = lat_pyr[i_s - 1].to(device=self.device, dtype=torch.float32)
                tgt_h, tgt_w = x0_k.shape[-2], x0_k.shape[-1]
                Bp, Cp, Tp = x0_prev.shape[:3]
                flat = x0_prev.permute(0, 2, 1, 3, 4).reshape(Bp * Tp, Cp, x0_prev.shape[-2], x0_prev.shape[-1])
                x0_prev_up = (
                    F.interpolate(flat, size=(tgt_h, tgt_w), mode="nearest")
                    .reshape(Bp, Tp, Cp, tgt_h, tgt_w)
                    .permute(0, 2, 1, 3, 4)
                )
                start_sigma_k = self.scheduler.start_sigmas[i_s]
                start_point = start_sigma_k * noise_k + (1 - start_sigma_k) * x0_prev_up

            # end_point
            if i_s < num_stages - 1:
                end_sigma_k = self.scheduler.end_sigmas[i_s]
                end_point = end_sigma_k * noise_k + (1 - end_sigma_k) * x0_k
            else:
                end_point = x0_k

            # Set timesteps for this stage
            h, w = x0_k.shape[-2], x0_k.shape[-1]
            image_seq_len = (T * h * w) // (patch_size[0] * patch_size[1] * patch_size[2])
            mu = _calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            self.scheduler.set_timesteps(
                pyramid_steps[i_s], i_s, device=self.device, mu=mu,
                is_amplify_first_chunk=False,  # FlowEdit is ODE; amplification not meaningful
            )

            # Inter-stage: carry edit delta to the next (finer) resolution
            if i_s > start_stage:
                x0_prev_carry = lat_pyr[i_s - 1].to(device=self.device, dtype=torch.float32)
                fe_B, fe_C, fe_T, h_prev, w_prev = latents.shape
                delta_edit = latents.float() - x0_prev_carry.float()
                flat_d = delta_edit.permute(0, 2, 1, 3, 4).reshape(fe_B * fe_T, fe_C, h_prev, w_prev)
                delta_up = (
                    F.interpolate(flat_d, size=(h, w), mode="bilinear", align_corners=False)
                    .reshape(fe_B, fe_T, fe_C, h, w)
                    .permute(0, 2, 1, 3, 4)
                )
                latents = (x0_k + delta_up).to(torch.float32)

            for idx, t in enumerate(self.scheduler.timesteps):
                timestep = t.expand(B).to(torch.int64)
                sigma_t = float(self.scheduler.sigmas[idx])
                sigma_next = float(self.scheduler.sigmas[idx + 1])

                # Apply edit only from (start_stage, start_step) onward
                _do_edit = (i_s > start_stage) or (i_s == start_stage and idx >= start_step)
                if not _do_edit:
                    continue

                # Stage2-consistent source trajectory
                Zt_src = (sigma_t * start_point + (1 - sigma_t) * end_point).float()
                # Target trajectory = Zt_src + accumulated edit
                Zt_tar = (latents.float() + Zt_src - x0_k.float())

                def _call_transformer(hidden, embeds, ctx):
                    with self.transformer.cache_context(ctx):
                        return self.transformer(
                            hidden_states=hidden.to(self.dtype),
                            timestep=timestep,
                            encoder_hidden_states=embeds,
                            indices_hidden_states=self._indices_hidden_states,
                            indices_latents_history_short=self._indices_history_short,
                            indices_latents_history_mid=self._indices_history_mid,
                            indices_latents_history_long=self._indices_history_long,
                            latents_history_short=lat_short.to(self.dtype),
                            latents_history_mid=lat_mid.to(self.dtype),
                            latents_history_long=lat_long.to(self.dtype),
                            return_dict=False,
                        )[0].float()

                if not is_flowalign:
                    # FlowEdit: velocity = Vt_tar - Vt_src (up to 4 calls)
                    v_tar_cond = _call_transformer(Zt_tar, tar_prompt_embeds, "fe_tar_cond")
                    if target_gs > 1.0:
                        v_tar_uncond = _call_transformer(Zt_tar, tar_neg_embeds, "fe_tar_uncond")
                        Vt_tar = v_tar_uncond + target_gs * (v_tar_cond - v_tar_uncond)
                    else:
                        Vt_tar = v_tar_cond
                    del Zt_tar

                    v_src_cond = _call_transformer(Zt_src, src_prompt_embeds, "fe_src_cond")
                    if source_gs > 1.0:
                        v_src_uncond = _call_transformer(Zt_src, src_neg_embeds, "fe_src_uncond")
                        Vt_src = v_src_uncond + source_gs * (v_src_cond - v_src_uncond)
                    else:
                        Vt_src = v_src_cond
                    del Zt_src

                    velocity = Vt_tar - Vt_src
                    difs = None
                else:
                    # FlowAlign: 3-call DIFS alignment
                    vq = _call_transformer(Zt_src, src_prompt_embeds, "fa_vq")
                    vp_tar = _call_transformer(Zt_tar, tar_prompt_embeds, "fa_vp_tar")
                    del Zt_tar

                    if target_gs > 1.0:
                        vp_src = _call_transformer(Zt_src, src_prompt_embeds, "fa_vp_src")
                        vp = vp_src + target_gs * (vp_tar - vp_src)
                    else:
                        vp = vp_tar

                    difs = None
                    if zeta > 0:
                        Zt_tar_for_difs = latents.float() + Zt_src - x0_k.float()
                        difs = (Zt_src - sigma_t * vq) - (Zt_tar_for_difs - sigma_t * vp)
                    del Zt_src

                    velocity = vp - vq

                # Pure Euler integration (no scheduler.step — FlowEdit is an ODE)
                dt = sigma_next - sigma_t  # negative: sigma is decreasing
                if difs is not None:
                    latents = (latents.float() + dt * velocity + zeta * difs).to(torch.float32)
                else:
                    latents = (latents.float() + dt * velocity).to(torch.float32)

        return latents  # [1, C, T, H, W] at full resolution
