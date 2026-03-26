from pydantic import Field

from ..artifacts import HuggingfaceRepoArtifact
from ..base_schema import BasePipelineConfig, ModeDefaults, ui_field_config
from ..common_artifacts import (
    LIGHTTAE_ARTIFACT,
    LIGHTVAE_ARTIFACT,
    TAE_ARTIFACT,
    UMT5_ENCODER_ARTIFACT,
    WAN_1_3B_ARTIFACT,
)
from ..enums import Quantization
from ..utils import VaeType

HELIOS_DISTILLED_ARTIFACT = HuggingfaceRepoArtifact(
    repo_id="BestWishYsh/Helios-Distilled",
    files=[
        "model_index.json",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model*.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ],
)


class HeliosConfig(BasePipelineConfig):
    pipeline_id = "helios"
    pipeline_name = "Helios"
    pipeline_description = (
        "Helios is an autoregressive long-form video generation model from PKU-YuanGroup. "
        "This pipeline runs the Helios-Distilled 14B variant (pyramid refinement, "
        "6 denoising steps per chunk, no classifier-free guidance) to generate "
        "video in 33-frame chunks (~1.4s at 24fps). Quality is high; generation "
        "is not real-time."
    )
    estimated_vram_gb = 24.0
    artifacts = [
        UMT5_ENCODER_ARTIFACT,
        WAN_1_3B_ARTIFACT,
        HELIOS_DISTILLED_ARTIFACT,
        LIGHTVAE_ARTIFACT,
        TAE_ARTIFACT,
        LIGHTTAE_ARTIFACT,
    ]
    supports_cache_management = True
    supports_quantization = True
    modified = True

    height: int = Field(
        default=384,
        ge=1,
        json_schema_extra=ui_field_config(order=1, component="resolution", is_load_param=True),
    )
    width: int = Field(
        default=640,
        ge=1,
        json_schema_extra=ui_field_config(order=1, component="resolution", is_load_param=True),
    )
    base_seed: int = Field(
        default=42,
        ge=0,
        json_schema_extra=ui_field_config(order=2, is_load_param=True, label="Seed"),
    )
    vae_type: VaeType = Field(
        default=VaeType.WAN,
        description="VAE type to use. 'wan' is the full VAE, 'lightvae' is 75% pruned (faster but lower quality).",
        json_schema_extra=ui_field_config(order=3, is_load_param=True, label="VAE"),
    )
    num_latent_frames_per_chunk: int = Field(
        default=9,
        ge=1,
        json_schema_extra=ui_field_config(order=4, is_load_param=True, label="Latent frames/chunk"),
    )
    history_sizes: list[int] = Field(
        default=[16, 2, 1],
        json_schema_extra=ui_field_config(order=5, is_load_param=True, label="History sizes"),
    )
    pyramid_steps: list[int] = Field(
        default=[2, 2, 2],
        json_schema_extra=ui_field_config(order=6, label="Pyramid steps"),
    )
    amplify_first_chunk: bool = Field(
        default=True,
        json_schema_extra=ui_field_config(order=7, label="Amplify first chunk"),
    )
    guidance_scale: float = Field(
        default=1.0,
        ge=0.0,
        json_schema_extra=ui_field_config(order=8, label="Guidance scale"),
    )
    enable_compile: bool = Field(
        default=False,
        json_schema_extra=ui_field_config(order=9, is_load_param=True, label="torch.compile"),
    )
    text_encoder_quantization: Quantization | None = Field(
        default=None,
        description="Quantization method for the text encoder (fp8_e4m3fn uses the pre-quantized FP8 checkpoint).",
        json_schema_extra=ui_field_config(
            order=10, component="quantization", is_load_param=True, label="Text Encoder Quantization"
        ),
    )

    modes = {"text": ModeDefaults(default=True)}
