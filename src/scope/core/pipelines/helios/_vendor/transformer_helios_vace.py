"""VACE transformer extensions for Helios autoregressive video generation.

Adds HeliosVACETransformerBlock and HeliosVACETransformer3DModel which extend
the base Helios transformer with ControlNet-style VACE conditioning.

Design follows VACE_INTEGRATION.md:
- 8 VACE control blocks at layers [0, 5, 10, 15, 20, 25, 30, 35]
- Hints collected from VACE stream, reversed, injected into main stream
- Hints injected only into current-chunk token positions (not history)
- Control latents resized per pyramid stage to avoid token count mismatch
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import logging
from safetensors.torch import load_file

from .transformer_helios import (
    HeliosAttention,
    HeliosAttnProcessor,
    HeliosTransformer3DModel,
    HeliosOutputNorm,
    pad_for_3d_conv,
    center_down_sample_3d,
)

logger = logging.get_logger(__name__)

_VACE_LAYERS = [0, 5, 10, 15, 20, 25, 30, 35]


class HeliosVACETransformerBlock(nn.Module):
    """VACE control-stream transformer block for Helios.

    Mirrors HeliosTransformerBlock but processes VACE control tokens.
    At layer 0 (has_proj_in=True) the main stream is fused into the
    control stream via proj_in.  Every layer emits an additive hint
    via proj_out which is later injected into the main stream.

    Critical: norm3 uses elementwise_affine=True because the WAN2.1-VACE
    checkpoint stores learned norm3.weight / norm3.bias for all blocks.
    """

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        eps: float = 1e-6,
        has_proj_in: bool = False,
    ):
        super().__init__()

        self.has_proj_in = has_proj_in
        if has_proj_in:
            self.proj_in = nn.Linear(dim, dim)

        # 1. Self-attention on control tokens
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = HeliosAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            processor=HeliosAttnProcessor(),
        )

        # 2. Cross-attention with text embeddings
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn2 = HeliosAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            processor=HeliosAttnProcessor(),
        )

        # 3. FFN — elementwise_affine=True to match WAN2.1-VACE checkpoint
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=True)
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")

        # Modulation table (same shape as base block)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        # Hint output projection
        self.proj_out = nn.Linear(dim, dim)

    def forward(
        self,
        ctrl: torch.Tensor,
        main_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one VACE control block.

        Args:
            ctrl: Control stream tokens [B, ctrl_len, D].
            main_hidden_states: Current-chunk main stream tokens [B, ctrl_len, D]
                (used only at layer 0 for proj_in).
            encoder_hidden_states: Projected text tokens [B, text_len, D].
            temb: Per-token timestep projection [B, ctrl_len, 6, D] (4-D).
            rotary_emb: RoPE embeddings for control tokens [B, ctrl_len, rope_dim].

        Returns:
            (updated_ctrl, hint): both [B, ctrl_len, D].
        """
        # Layer 0 only: project the control stream and fuse with the main stream.
        # proj_in transforms the VACE control tokens; the main stream is added
        # as a conditioning signal (matching the reference implementation).
        if self.has_proj_in:
            projected = self.proj_in(ctrl)
            if projected.shape[1] == main_hidden_states.shape[1]:
                ctrl = projected + main_hidden_states
            else:
                ctrl = projected

        # Modulation gates from per-token timestep embedding (4-D)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.unsqueeze(0) + temb.float()
        ).chunk(6, dim=2)
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        c_shift_msa = c_shift_msa.squeeze(2)
        c_scale_msa = c_scale_msa.squeeze(2)
        c_gate_msa = c_gate_msa.squeeze(2)

        # 1. Self-attention
        norm_ctrl = (self.norm1(ctrl.float()) * (1 + scale_msa) + shift_msa).type_as(ctrl)
        attn_out = self.attn1(norm_ctrl, None, None, rotary_emb, None)
        ctrl = (ctrl.float() + attn_out * gate_msa).type_as(ctrl)

        # 2. Cross-attention with text
        norm_ctrl = self.norm2(ctrl.float()).type_as(ctrl)
        attn_out = self.attn2(norm_ctrl, encoder_hidden_states, None, None, None)
        ctrl = ctrl + attn_out

        # 3. FFN
        norm_ctrl = (self.norm3(ctrl.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(ctrl)
        ff_out = self.ffn(norm_ctrl)
        ctrl = (ctrl.float() + ff_out.float() * c_gate_msa).type_as(ctrl)

        # Emit conditioning hint
        hint = self.proj_out(ctrl)
        return ctrl, hint


class HeliosVACETransformer3DModel(HeliosTransformer3DModel):
    """Helios transformer extended with VACE ControlNet-style conditioning.

    Adds a parallel VACE control stream (8 blocks) that injects additive
    hints into the main transformer at selected layer indices.

    New parameters vs base model:
    - vace_patch_embedding: Conv3d(96, inner_dim, patch_size)
    - vace_blocks: 8 HeliosVACETransformerBlocks at vace_layers

    Loading:
        # 1. Load base checkpoint normally
        base = HeliosTransformer3DModel.from_pretrained(path, ...)
        # 2. Instantiate VACE model (no meta device)
        transformer = HeliosVACETransformer3DModel(**base.config, vace_layers=_VACE_LAYERS)
        # 3. Copy base weights (VACE keys stay random)
        transformer.load_state_dict(base.state_dict(), strict=False)
        del base
        # 4. Load VACE module with key remapping
        transformer.load_vace_module("path/to/Wan2_1-VACE_module_14B_bf16.safetensors")
    """

    def __init__(self, *args, vace_layers: list[int] | None = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.vace_layers = vace_layers or _VACE_LAYERS
        num_vace_blocks = len(self.vace_layers)
        inner_dim = self.inner_dim
        p_t, p_h, p_w = self.config.patch_size

        # 96-channel VACE patch embedding
        self.vace_patch_embedding = nn.Conv3d(
            96,
            inner_dim,
            kernel_size=(p_t, p_h, p_w),
            stride=(p_t, p_h, p_w),
        )

        # VACE control blocks
        self.vace_blocks = nn.ModuleList(
            [
                HeliosVACETransformerBlock(
                    dim=inner_dim,
                    ffn_dim=self.config.ffn_dim,
                    num_heads=self.config.num_attention_heads,
                    eps=self.config.eps,
                    has_proj_in=(i == 0),
                )
                for i in range(num_vace_blocks)
            ]
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        # history / index arguments (same as base)
        indices_hidden_states=None,
        indices_latents_history_short=None,
        indices_latents_history_mid=None,
        indices_latents_history_long=None,
        latents_history_short=None,
        latents_history_mid=None,
        latents_history_long=None,
        # VACE conditioning
        control_hidden_states: torch.Tensor | None = None,
        control_hidden_states_scale: float = 1.0,
        return_dict: bool = True,
        attention_kwargs=None,
    ):
        """Forward pass identical to base, with optional VACE conditioning.

        If control_hidden_states is None, behaves exactly like base model.
        Otherwise runs the VACE stream and injects hints into main blocks.
        """
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        batch_size = hidden_states.shape[0]
        p_t, p_h, p_w = self.config.patch_size

        # ------------------------------------------------------------------ #
        # 1.  Patch-embed current chunk
        # ------------------------------------------------------------------ #
        hidden_states = self.patch_embedding(hidden_states)
        _, _, post_patch_num_frames, post_patch_height, post_patch_width = hidden_states.shape

        if indices_hidden_states is None:
            indices_hidden_states = (
                torch.arange(0, post_patch_num_frames).unsqueeze(0).expand(batch_size, -1)
            )

        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        rotary_emb = self.rope(
            frame_indices=indices_hidden_states,
            height=post_patch_height,
            width=post_patch_width,
            device=hidden_states.device,
        )
        rotary_emb = rotary_emb.flatten(2).transpose(1, 2)
        original_context_length = hidden_states.shape[1]

        # ------------------------------------------------------------------ #
        # 2.  Short history
        # ------------------------------------------------------------------ #
        H1 = W1 = None
        if latents_history_short is not None and indices_latents_history_short is not None:
            latents_history_short = latents_history_short.to(hidden_states)
            latents_history_short = self.patch_short(latents_history_short)
            _, _, _, H1, W1 = latents_history_short.shape
            latents_history_short = latents_history_short.flatten(2).transpose(1, 2)

            rotary_emb_history_short = self.rope(
                frame_indices=indices_latents_history_short,
                height=H1,
                width=W1,
                device=latents_history_short.device,
            )
            rotary_emb_history_short = rotary_emb_history_short.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([latents_history_short, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_history_short, rotary_emb], dim=1)

        # ------------------------------------------------------------------ #
        # 3.  Mid history
        # ------------------------------------------------------------------ #
        if latents_history_mid is not None and indices_latents_history_mid is not None:
            latents_history_mid = latents_history_mid.to(hidden_states)
            latents_history_mid = pad_for_3d_conv(latents_history_mid, (2, 4, 4))
            latents_history_mid = self.patch_mid(latents_history_mid)
            latents_history_mid = latents_history_mid.flatten(2).transpose(1, 2)

            rotary_emb_history_mid = self.rope(
                frame_indices=indices_latents_history_mid,
                height=H1,
                width=W1,
                device=latents_history_mid.device,
            )
            rotary_emb_history_mid = pad_for_3d_conv(rotary_emb_history_mid, (2, 2, 2))
            rotary_emb_history_mid = center_down_sample_3d(rotary_emb_history_mid, (2, 2, 2))
            rotary_emb_history_mid = rotary_emb_history_mid.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([latents_history_mid, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_history_mid, rotary_emb], dim=1)

        # ------------------------------------------------------------------ #
        # 4.  Long history
        # ------------------------------------------------------------------ #
        if latents_history_long is not None and indices_latents_history_long is not None:
            latents_history_long = latents_history_long.to(hidden_states)
            latents_history_long = pad_for_3d_conv(latents_history_long, (4, 8, 8))
            latents_history_long = self.patch_long(latents_history_long)
            latents_history_long = latents_history_long.flatten(2).transpose(1, 2)

            rotary_emb_history_long = self.rope(
                frame_indices=indices_latents_history_long,
                height=H1,
                width=W1,
                device=latents_history_long.device,
            )
            rotary_emb_history_long = pad_for_3d_conv(rotary_emb_history_long, (4, 4, 4))
            rotary_emb_history_long = center_down_sample_3d(rotary_emb_history_long, (4, 4, 4))
            rotary_emb_history_long = rotary_emb_history_long.flatten(2).transpose(1, 2)

            hidden_states = torch.cat([latents_history_long, hidden_states], dim=1)
            rotary_emb = torch.cat([rotary_emb_history_long, rotary_emb], dim=1)

        history_context_length = hidden_states.shape[1] - original_context_length

        # ------------------------------------------------------------------ #
        # 5.  Timestep + text embeddings
        # ------------------------------------------------------------------ #
        if indices_hidden_states is not None and self.zero_history_timestep:
            timestep_t0 = torch.zeros((1,), dtype=timestep.dtype, device=timestep.device)
            temb_t0, timestep_proj_t0, _ = self.condition_embedder(
                timestep_t0, encoder_hidden_states, is_return_encoder_hidden_states=False
            )
            temb_t0 = temb_t0.unsqueeze(1).expand(batch_size, history_context_length, -1)
            timestep_proj_t0 = (
                timestep_proj_t0.unflatten(-1, (6, -1))
                .view(1, 6, 1, -1)
                .expand(batch_size, -1, history_context_length, -1)
            )

        temb, timestep_proj, encoder_hidden_states = self.condition_embedder(
            timestep, encoder_hidden_states
        )
        timestep_proj = timestep_proj.unflatten(-1, (6, -1))

        if indices_hidden_states is not None and not self.zero_history_timestep:
            main_repeat_size = hidden_states.shape[1]
        else:
            main_repeat_size = original_context_length
        temb = temb.view(batch_size, 1, -1).expand(batch_size, main_repeat_size, -1)
        timestep_proj = (
            timestep_proj.view(batch_size, 6, 1, -1).expand(batch_size, 6, main_repeat_size, -1)
        )

        if indices_hidden_states is not None and self.zero_history_timestep:
            temb = torch.cat([temb_t0, temb], dim=1)
            timestep_proj = torch.cat([timestep_proj_t0, timestep_proj], dim=2)

        if timestep_proj.ndim == 4:
            timestep_proj = timestep_proj.permute(0, 2, 1, 3)
        # timestep_proj: [B, total_seq_len, 6, D]

        # ------------------------------------------------------------------ #
        # 6.  VACE control stream
        # ------------------------------------------------------------------ #
        hints = []
        vace_layer_set = set(self.vace_layers)

        if control_hidden_states is not None:
            # a. Resize control to match current pyramid stage resolution
            target_T = post_patch_num_frames * p_t
            target_H = post_patch_height * p_h
            target_W = post_patch_width * p_w
            if control_hidden_states.shape[2:] != (target_T, target_H, target_W):
                control_hidden_states = F.interpolate(
                    control_hidden_states.float(),
                    size=(target_T, target_H, target_W),
                    mode="trilinear",
                    align_corners=False,
                ).to(control_hidden_states.dtype)

            # b. Patch-embed control → [B, ctrl_len, D]
            ctrl = self.vace_patch_embedding(control_hidden_states.to(hidden_states.dtype))
            ctrl = ctrl.flatten(2).transpose(1, 2)

            # c. Chunk slices for VACE blocks (current chunk only)
            temb_chunk = timestep_proj[:, -original_context_length:, :, :]
            rotary_emb_chunk = rotary_emb[:, -original_context_length:, :]
            main_chunk = hidden_states[:, -original_context_length:, :]

            # d. Run VACE blocks sequentially, collect hints
            for vace_block in self.vace_blocks:
                ctrl, hint = vace_block(
                    ctrl, main_chunk, encoder_hidden_states, temb_chunk, rotary_emb_chunk
                )
                hints.append(hint)

            # e. Reverse hint list (ControlNet-style: deepest VACE block → earliest main layer)
            hints = hints[::-1]

        # ------------------------------------------------------------------ #
        # 7.  Main transformer blocks with hint injection
        # ------------------------------------------------------------------ #
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        rotary_emb = rotary_emb.contiguous()

        for block_idx, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    original_context_length,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    original_context_length,
                )

            # Inject VACE hint at the designated layers (current-chunk positions only)
            if block_idx in vace_layer_set:
                hint = hints.pop()
                # Slice to avoid in-place mutation (clone only the chunk slice)
                chunk = hidden_states[:, -original_context_length:, :] + hint * control_hidden_states_scale
                hidden_states = torch.cat(
                    [hidden_states[:, :-original_context_length, :], chunk], dim=1
                )

        # ------------------------------------------------------------------ #
        # 8.  Output norm + projection + unpatchify
        # ------------------------------------------------------------------ #
        hidden_states = self.norm_out(hidden_states, temb, original_context_length)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    # ------------------------------------------------------------------
    # VACE module loading
    # ------------------------------------------------------------------

    def load_vace_module(self, path: str) -> tuple[list[str], list[str]]:
        """Load VACE module weights from a safetensors file with key remapping.

        Args:
            path: Path to Wan2_1-VACE_module_14B_bf16.safetensors

        Returns:
            (missing_keys, unexpected_keys) from load_state_dict
        """
        raw_sd = load_file(path)
        remapped: dict[str, torch.Tensor] = {}
        for key, value in raw_sd.items():
            new_key = self._remap_vace_module_key(key)
            if new_key is not None:
                remapped[new_key] = value

        missing, unexpected = self.load_state_dict(remapped, strict=False)
        vace_missing = [k for k in missing if "vace" in k]
        if vace_missing:
            logger.warning(
                f"load_vace_module: {len(vace_missing)} VACE keys not found in checkpoint "
                f"(first 10: {vace_missing[:10]})"
            )
        return missing, unexpected

    @staticmethod
    def _remap_vace_module_key(key: str) -> str | None:
        """Map a WAN2.1-VACE checkpoint key to the model state-dict key.

        Returns None for keys that should be skipped (e.g. metadata).
        """
        # Skip metadata entries
        if key.startswith("model_type."):
            return None

        # vace_patch_embedding (may appear under several names in different checkpoints)
        for prefix in ("vace_patch_emb.", "patch_embedding_vace."):
            if key.startswith(prefix):
                return "vace_patch_embedding." + key[len(prefix):]
        if key.startswith("vace_patch_embedding."):
            return key  # already correct

        # vace_blocks remapping — apply in order of specificity
        def _sub(pattern, replacement):
            nonlocal key
            key = re.sub(pattern, replacement, key)

        # proj_in / proj_out
        _sub(r"^(vace_blocks\.\d+\.)before_proj\.", r"\1proj_in.")
        _sub(r"^(vace_blocks\.\d+\.)after_proj\.", r"\1proj_out.")

        # self_attn sub-keys
        _sub(r"^(vace_blocks\.\d+\.)self_attn\.q\.", r"\1attn1.to_q.")
        _sub(r"^(vace_blocks\.\d+\.)self_attn\.k\.", r"\1attn1.to_k.")
        _sub(r"^(vace_blocks\.\d+\.)self_attn\.v\.", r"\1attn1.to_v.")
        _sub(r"^(vace_blocks\.\d+\.)self_attn\.o\.", r"\1attn1.to_out.0.")
        _sub(r"^(vace_blocks\.\d+\.)self_attn\.", r"\1attn1.")  # fallback (norms etc.)

        # cross_attn sub-keys
        _sub(r"^(vace_blocks\.\d+\.)cross_attn\.q\.", r"\1attn2.to_q.")
        _sub(r"^(vace_blocks\.\d+\.)cross_attn\.k\.", r"\1attn2.to_k.")
        _sub(r"^(vace_blocks\.\d+\.)cross_attn\.v\.", r"\1attn2.to_v.")
        _sub(r"^(vace_blocks\.\d+\.)cross_attn\.o\.", r"\1attn2.to_out.0.")
        _sub(r"^(vace_blocks\.\d+\.)cross_attn\.", r"\1attn2.")  # fallback

        # ffn
        _sub(r"^(vace_blocks\.\d+\.)ffn\.0\.", r"\1ffn.net.0.proj.")
        _sub(r"^(vace_blocks\.\d+\.)ffn\.2\.", r"\1ffn.net.2.")

        # modulation → scale_shift_table
        _sub(r"^(vace_blocks\.\d+\.)modulation$", r"\1scale_shift_table")

        return key
