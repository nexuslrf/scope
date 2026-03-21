"""Helios VACE pipeline — Helios with ControlNet-style video conditioning.

Extends HeliosPipeline to accept a reference/control video that guides
generation chunk-by-chunk via the VACE attention-injection mechanism.

Usage:
    pipeline = HeliosVACEPipeline(config, device=device)

    # Provide control video once per sequence
    pipeline.prepare(vace_video=video_tensor)  # [T, H, W, C] float32 [0,1]

    # Then call per-chunk as usual
    for chunk_idx in range(num_chunks):
        out = pipeline(prompt="...", init_cache=(chunk_idx == 0))
        frames = out["video"]  # [T, H, W, C]
"""

import logging
import math
import os

import torch
import torch.nn.functional as F

from .pipeline import HeliosPipeline, _calculate_shift
from .schema_vace import HeliosVACEConfig
from ..wan2_1.vace import VACEEnabledPipeline

logger = logging.getLogger(__name__)


class HeliosVACEPipeline(HeliosPipeline, VACEEnabledPipeline):
    """Helios pipeline with VACE video conditioning.

    Inherits all generation logic from HeliosPipeline.  The only
    differences are:
    - loads HeliosVACETransformer3DModel (adds VACE blocks + vace_patch_embedding)
    - prepare() encodes a control video to 96-channel VACE latents
    - _generate_chunk() passes the per-chunk VACE slice to the transformer
    """

    @classmethod
    def get_config_class(cls):
        return HeliosVACEConfig

    def __init__(self, config, device=None, dtype=torch.bfloat16):
        from ._vendor.transformer_helios_vace import HeliosVACETransformer3DModel
        from ._vendor.transformer_helios import HeliosTransformer3DModel

        # Call HeliosPipeline.__init__ which loads base transformer.
        # We then replace it with the VACE-extended model.
        super().__init__(config, device=device, dtype=dtype)

        model_dir = getattr(config, "model_dir", None)
        if model_dir:
            helios_path = os.path.join(model_dir, "Helios-Distilled")
            vace_path = os.path.join(
                model_dir, "WanVideo_comfy",
                "Wan2_1-VACE_module_14B_bf16.safetensors",
            )
        else:
            helios_path = "BestWishYsh/Helios-Distilled"
            vace_path = None  # will be resolved by model_dir later

        # Replace base transformer with VACE-extended version
        base_transformer: HeliosTransformer3DModel = self.transformer
        base_cfg = dict(base_transformer.config)

        vace_transformer = HeliosVACETransformer3DModel(**base_cfg).to(dtype)
        vace_transformer.load_state_dict(base_transformer.state_dict(), strict=False)
        del base_transformer

        # Load VACE module weights (if available)
        if vace_path and os.path.exists(vace_path):
            vace_transformer.load_vace_module(vace_path)
        else:
            # Try to locate in model_dir with the standard Kijai layout
            if model_dir:
                candidate = os.path.join(
                    model_dir, "Wan2_1-VACE_module_14B_bf16.safetensors"
                )
                if os.path.exists(candidate):
                    vace_transformer.load_vace_module(candidate)
                else:
                    import logging
                    logging.getLogger(__name__).warning(
                        "VACE module weights not found — VACE blocks will use random init. "
                        "Download Wan2_1-VACE_module_14B_bf16.safetensors to your model dir."
                    )

        self.transformer = vace_transformer.to(self.device)
        self.transformer.eval().requires_grad_(False)
        try:
            self.transformer.set_attention_backend("_flash_3")
        except Exception:
            self.transformer.set_attention_backend("flash_hub")

        # VACE state (set by prepare())
        self._vace_latents_full: torch.Tensor | None = None
        self._vace_chunk_counter: int = 0

        # VACEEnabledPipeline interface — tells the server to route video as
        # vace_input_frames instead of a regular video input.
        self.vace_enabled = True

    # ------------------------------------------------------------------
    # Pipeline interface overrides
    # ------------------------------------------------------------------

    def prepare(self, **kwargs):
        """Encode a control video into VACE latents, or request live frames.

        Two modes:
        - Offline: pass vace_video=[T,H,W,C] float32 [0,1] to pre-encode the
          full control video once; __call__ will slice per chunk.
        - Live: pass video=True (set by the server in video-input mode) to
          signal that per-chunk frames will arrive via __call__'s
          vace_input_frames / video kwargs.  Returns Requirements so the
          server knows to fetch camera frames from the input queue.
        """
        from ..interface import Requirements

        vace_video = kwargs.get("vace_video", None)
        if vace_video is not None:
            # Pre-encode the full control video (offline / batch use case).
            video = vace_video.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, H, W]
            video = video * 2.0 - 1.0
            video = video.to(self.device)
            self._vace_latents_full = self._prepare_vace_latents(video)
            self._vace_chunk_counter = 0
            return None

        # No full video provided — reset pre-encoded latents.
        self._vace_latents_full = None
        self._vace_chunk_counter = 0

        # In live video-input mode the server sets video=True.  Return
        # Requirements so it fetches frames from the camera queue each chunk.
        # Use 9 pixel frames (→ 3 latent frames via WAN VAE 4x causal compression).
        # The transformer trilinearly resizes VACE latents to match the chunk
        # size (9 latent frames), so fewer input frames still provide structural
        # conditioning.  Must stay well below the 30-frame input_queue maxsize.
        if kwargs.get("video", False):
            return Requirements(input_size=9)

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_vace_chunk(self, chunk_idx: int) -> torch.Tensor | None:
        """Return the VACE control slice for chunk *chunk_idx*.

        Returns None if no VACE latents have been prepared.
        Returns a zero-padded slice if the control video is shorter than
        the requested chunk window.
        """
        if self._vace_latents_full is None:
            return None

        T = self._config.num_latent_frames_per_chunk
        start = chunk_idx * T
        end = start + T
        total_T = self._vace_latents_full.shape[2]

        if start >= total_T:
            # Beyond control video → zero conditioning
            return torch.zeros(
                self._vace_latents_full.shape[0],
                self._vace_latents_full.shape[1],
                T,
                self._vace_latents_full.shape[3],
                self._vace_latents_full.shape[4],
                device=self.device,
                dtype=self._vace_latents_full.dtype,
            )

        chunk = self._vace_latents_full[:, :, start:end, :, :]
        if chunk.shape[2] < T:
            # Last chunk — zero-pad remaining frames
            pad = torch.zeros(
                chunk.shape[0], chunk.shape[1], T - chunk.shape[2],
                chunk.shape[3], chunk.shape[4],
                device=self.device, dtype=chunk.dtype,
            )
            chunk = torch.cat([chunk, pad], dim=2)
        return chunk

    @torch.no_grad()
    def _prepare_vace_latents(
        self,
        video: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a pixel video to a 96-channel VACE latent tensor.

        The 96 channels are structured as:
          [inactive (16ch) | reactive (16ch) | mask_encoding (64ch)]

        Args:
            video: Pixel video [B, C, T_pix, H_pix, W_pix] in [-1, 1], float32.
            mask: Optional pixel mask [B, 1, T_pix, H_pix, W_pix] in [0, 1].
                  1 = controlled/reactive region, 0 = preserved/inactive region.
                  If None, all-ones (full-frame control): inactive=zeros, reactive=video.

        Returns:
            VACE latents [B, 96, T_lat, H_lat, W_lat].
        """
        B, C, T_pix, H_pix, W_pix = video.shape
        vae = self.vae  # float32 AutoencoderKLWan

        if mask is None:
            # Full-frame control: inactive = zeros, reactive = full video
            inactive_input = torch.zeros_like(video)
            reactive_input = video
        else:
            mask_float = mask.to(device=self.device, dtype=torch.float32)
            # Resize to match video dims if needed (5D nearest interpolation)
            if mask_float.shape[2:] != (T_pix, H_pix, W_pix):
                mask_float = F.interpolate(
                    mask_float, size=(T_pix, H_pix, W_pix), mode="nearest"
                )
            inactive_input = video * (1.0 - mask_float)
            reactive_input = video * mask_float

        # Reactive: encode the controlled region
        reactive_raw = vae.encode(reactive_input).latent_dist.mode()  # [B, 16, T_lat, H_lat, W_lat]
        _, C_lat, T_lat, H_lat, W_lat = reactive_raw.shape

        # Inactive: encode the preserved region
        inactive_raw = vae.encode(inactive_input).latent_dist.mode()  # [B, 16, T_lat, H_lat, W_lat]

        # Do NOT normalize VACE latents.  The VACE checkpoint (Wan2_1-VACE_module)
        # was trained on raw (unnormalized) VAE latents.  The main denoising path
        # normalizes via (x - mean) / std, but the VACE conditioning path uses raw
        # latents as input to vace_patch_embedding, which is a Conv3d that maps
        # 96 raw channels → transformer inner dim.  Normalizing here would shift the
        # distribution and break the learned conditioning signal.

        # 64-channel mask encoding
        if mask is None:
            mask_enc = self._encode_mask_ones(B, T_lat, H_lat, W_lat)
        else:
            mask_enc = self._encode_mask(mask_float, T_lat, H_lat, W_lat)

        vace_latents = torch.cat([inactive_raw, reactive_raw, mask_enc], dim=1)
        return vace_latents.to(self.dtype)

    def _encode_mask_ones(
        self, B: int, T_lat: int, H_lat: int, W_lat: int
    ) -> torch.Tensor:
        """Return the 64-channel mask encoding for an all-ones mask.

        An all-ones pixel mask (every region is 'controlled') maps to
        an all-ones 64-channel tensor after the pixel-to-channel
        rearrangement used in VACE.
        """
        return torch.ones(B, 64, T_lat, H_lat, W_lat, device=self.device, dtype=torch.float32)

    def _encode_mask(
        self,
        mask_pixels: torch.Tensor,
        T_lat: int,
        H_lat: int,
        W_lat: int,
    ) -> torch.Tensor:
        """Encode an arbitrary pixel mask to 64-channel latent mask.

        Implements the 8×8 pixel-to-channel rearrangement from VACE
        (identical to vace_encode_masks in wan2_1/vace/utils/encoding.py).

        Args:
            mask_pixels: [B, 1, T_pix, H_pix, W_pix] values in [0, 1].
            T_lat, H_lat, W_lat: target latent dimensions.

        Returns:
            [B, 64, T_lat, H_lat, W_lat].
        """
        B, _, T_pix, H_pix, W_pix = mask_pixels.shape
        p_h, p_w = 8, 8  # VAE spatial stride

        encoded = []
        for b in range(B):
            m = mask_pixels[b, 0]  # [T_pix, H_pix, W_pix]
            h_lat_raw = H_pix // p_h
            w_lat_raw = W_pix // p_w
            # Pixel-to-channel: group 8×8 spatial blocks → 64 channels
            m = m.view(T_pix, h_lat_raw, p_h, w_lat_raw, p_w)
            m = m.permute(2, 4, 0, 1, 3)         # [p_h, p_w, T_pix, h, w]
            m = m.reshape(p_h * p_w, T_pix, h_lat_raw, w_lat_raw)  # [64, T_pix, h, w]
            # Interpolate temporal dim to latent length
            m = F.interpolate(
                m.unsqueeze(0), size=(T_lat, H_lat, W_lat), mode="nearest-exact"
            ).squeeze(0)  # [64, T_lat, H_lat, W_lat]
            encoded.append(m)

        return torch.stack(encoded, dim=0)  # [B, 64, T_lat, H_lat, W_lat]

    @torch.no_grad()
    def _encode_vace_input_frames(
        self,
        vace_input_frames: list | torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode per-chunk live video input to VACE conditioning latents.

        Args:
            vace_input_frames: Either a list of raw frame tensors (each [1, H, W, C]
                uint8 from the camera queue) or an already-preprocessed tensor
                [B, C, T, H, W] in [-1, 1].
            mask: Optional pixel mask [B, 1, T, H, W] in [0, 1].
                  1=controlled, 0=preserved.  If None, full-frame control.

        Returns:
            VACE latents [1, 96, T_lat, H_lat, W_lat].
        """
        from ..process import preprocess_chunk

        if isinstance(vace_input_frames, list):
            # Raw frames from camera queue — preprocess to [1, C, T, H, W] in [-1, 1]
            video = preprocess_chunk(
                vace_input_frames,
                device=self.device,
                dtype=torch.float32,
                height=self._config.height,
                width=self._config.width,
            )
        else:
            video = vace_input_frames.to(device=self.device, dtype=torch.float32)

        return self._prepare_vace_latents(video, mask=mask)

    @torch.no_grad()
    def _generate_chunk(
        self,
        prompt_embeds: torch.Tensor,
        neg_embeds: torch.Tensor,
        is_first_chunk: bool,
        pyramid_steps: list[int],
        amplify_first_chunk: bool,
        vace_control: torch.Tensor | None = None,
        vace_scale: float = 1.0,
    ) -> torch.Tensor:
        """Pyramid denoising loop, optionally with VACE control.

        Identical to base _generate_chunk but passes control_hidden_states
        and control_hidden_states_scale to every transformer call.
        """
        T = self._config.num_latent_frames_per_chunk
        H = self._config.height // 8
        W = self._config.width // 8
        C = self.transformer.config.in_channels
        B = 1
        num_stages = len(pyramid_steps)

        # Build history slices
        history = self._history_latents[:, :, -self._num_history_frames:]
        lat_long, lat_mid, lat_short_1x = history.split(self._history_sizes, dim=2)

        if self._image_latents is None:
            anchor = torch.zeros(1, C, 1, H, W, device=self.device, dtype=torch.float32)
        else:
            anchor = self._image_latents
        lat_short = torch.cat([anchor, lat_short_1x], dim=2)

        generator = torch.Generator(device=self.device).manual_seed(
            self._config.base_seed + self._total_generated
        )
        latents = torch.randn(
            1, C, T, H, W, generator=generator, device=self.device, dtype=torch.float32
        )

        # Downsample to coarsest pyramid level
        latents_2d = latents.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        h, w = H, W
        for _ in range(num_stages - 1):
            h //= 2
            w //= 2
            latents_2d = F.interpolate(latents_2d, size=(h, w), mode="bilinear") * 2
        latents = latents_2d.reshape(B, T, C, h, w).permute(0, 2, 1, 3, 4)

        start_point_list = [latents]
        is_amplify = is_first_chunk and amplify_first_chunk
        patch_size = self.transformer.config.patch_size

        for i_s in range(num_stages):
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
                pyramid_steps[i_s], i_s, device=self.device, mu=mu,
                is_amplify_first_chunk=is_amplify,
            )

            if i_s > 0:
                h *= 2
                w *= 2
                latents_2d = latents.permute(0, 2, 1, 3, 4).reshape(B * T, C, h // 2, w // 2)
                latents_2d = F.interpolate(latents_2d, size=(h, w), mode="nearest")
                latents = latents_2d.reshape(B, T, C, h, w).permute(0, 2, 1, 3, 4)

                ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]
                gamma = self.scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)
                block_noise_seed = self._config.base_seed + self._total_generated + i_s
                block_noise_generator = torch.Generator(device=self.device).manual_seed(
                    block_noise_seed
                )
                noise = self._sample_block_noise(B, C, T, h, w, generator=block_noise_generator).to(
                    device=self.device, dtype=torch.float32
                )
                latents = alpha * latents + beta * noise
                start_point_list.append(latents)

            for idx, t in enumerate(self.scheduler.timesteps):
                timestep = t.expand(B).to(torch.int64)

                transformer_kwargs = dict(
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
                )
                if vace_control is not None:
                    transformer_kwargs["control_hidden_states"] = vace_control.to(self.dtype)
                    transformer_kwargs["control_hidden_states_scale"] = vace_scale

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(**transformer_kwargs)[0]

                latents = self.scheduler.step(
                    noise_pred, t, latents,
                    return_dict=False,
                    cur_sampling_step=idx,
                    dmd_noisy_tensor=start_point_list[i_s],
                    dmd_sigmas=self.scheduler.sigmas,
                    dmd_timesteps=self.scheduler.timesteps,
                    all_timesteps=self.scheduler.timesteps,
                )[0]

        return latents

    # Override __call__ to wire VACE control through to _generate_chunk
    def __call__(self, **kwargs) -> dict:
        from ..process import postprocess_chunk

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
        vace_scale = getattr(self._config, "vace_context_scale", 1.0)

        is_first_chunk = self.first_call or init_cache

        if is_first_chunk:
            self._reset_state()
            self._vace_chunk_counter = 0
        self.first_call = False

        if prompt != self._cached_prompt:
            self._cached_prompt_embeds, self._cached_neg_embeds = self._encode_prompt(
                prompt, negative_prompt
            )
            self._cached_prompt = prompt

        # Get VACE control for this chunk.
        # Priority: pre-encoded full-video latents (from prepare()) > per-chunk live input.
        vace_control = self._get_vace_chunk(self._vace_chunk_counter)

        if vace_control is None:
            # Fall back to per-chunk encoding from live video input.
            # Check vace_input_frames first (server VACE path), then video (fallback).
            _fi = kwargs.get("vace_input_frames")
            live_input = _fi if _fi is not None else kwargs.get("video")
            if live_input is not None:
                live_mask = kwargs.get("vace_input_masks")
                try:
                    vace_control = self._encode_vace_input_frames(live_input, mask=live_mask)
                    logger.info(
                        f"HeliosVACEPipeline: encoded VACE control, shape={vace_control.shape}"
                    )
                except Exception:
                    logger.exception("HeliosVACEPipeline: VACE encoding failed")
                    raise

        logger.info(
            f"HeliosVACEPipeline.__call__: chunk={self._vace_chunk_counter}, "
            f"vace={'shape=' + str(vace_control.shape) if vace_control is not None else 'none (T2V mode)'}"
        )
        latents = self._generate_chunk(
            self._cached_prompt_embeds,
            self._cached_neg_embeds,
            is_first_chunk=is_first_chunk,
            pyramid_steps=pyramid_steps,
            amplify_first_chunk=amplify_first_chunk,
            vace_control=vace_control,
            vace_scale=vace_scale,
        )

        logger.info(
            f"HeliosVACEPipeline: chunk {self._vace_chunk_counter} generated, latents={latents.shape}"
        )
        self._vace_chunk_counter += 1

        if is_first_chunk:
            self._image_latents = latents[:, :, 0:1, :, :].to(torch.float32)

        self._history_latents = torch.cat(
            [self._history_latents, latents.to(torch.float32)], dim=2
        )
        if self._history_latents.shape[2] > self._num_history_frames:
            self._history_latents = self._history_latents[:, :, -self._num_history_frames:]

        self._total_generated += latents.shape[2]

        current_latents = latents.to(torch.float32) / self._latents_std + self._latents_mean
        current_latents = current_latents.to(self.vae.dtype)
        video = self.vae.decode(current_latents, return_dict=False)[0]

        return {"video": postprocess_chunk(video.permute(0, 2, 1, 3, 4))}
