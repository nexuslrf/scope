from pydantic import Field

from ..base_schema import ModeDefaults, ui_field_config
from .schema import (
    HELIOS_DISTILLED_ARTIFACT,
    UMT5_ENCODER_ARTIFACT,
    WAN_1_3B_ARTIFACT,
    HeliosConfig,
)


class HeliosSDEditConfig(HeliosConfig):
    pipeline_id = "helios-sdedit"
    pipeline_name = "Helios SDEdit"
    pipeline_description = (
        "Helios video-to-video editing using SDEdit, FlowEdit, or FlowAlign. "
        "SDEdit injects noise into the source video latents then denoises toward the target prompt. "
        "FlowEdit/FlowAlign use differential velocity fields to transform source to target. "
        "Provide a source video via prepare() or live video input."
    )
    estimated_vram_gb = 24.0
    artifacts = [UMT5_ENCODER_ARTIFACT, WAN_1_3B_ARTIFACT, HELIOS_DISTILLED_ARTIFACT]
    supports_cache_management = True
    supports_quantization = True
    modified = True

    modes = {
        "video": ModeDefaults(default=True),
    }

    pyramid_steps: list[int] = Field(
        default=[2, 2, 2],
        description="Denoising steps per pyramid stage (coarse → medium → full resolution). Higher = better quality, slower.",
        json_schema_extra=ui_field_config(order=6, label="Pyramid steps"),
    )

    edit_type: str = Field(
        default="sdedit",
        description="Editing algorithm: 'sdedit' (noise-based), 'flowedit' (differential velocity, 4 calls/step), 'flowalign' (DIFS alignment, 3 calls/step).",
        json_schema_extra=ui_field_config(order=8, label="Edit type"),
    )
    edit_stage: float = Field(
        default=1.0,
        ge=0.0,
        le=2.99,
        description=(
            "Edit entry point in pyramid schedule. Integer part = pyramid stage (0-2), "
            "fractional part = fraction into that stage. "
            "Lower = more editing; higher = more source structure preserved. "
            "E.g. 0.0=max edit (pure noise), 1.0=start of stage 1, 2.0=start of stage 2."
        ),
        json_schema_extra=ui_field_config(order=9, label="Edit stage"),
    )
    source_prompt: str = Field(
        default="",
        description="Source video description (required for FlowEdit and FlowAlign).",
        json_schema_extra=ui_field_config(order=10, label="Source prompt"),
    )
    source_guidance_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="CFG scale for source velocity (FlowEdit only; 1.0 = no CFG).",
        json_schema_extra=ui_field_config(order=11, label="Source guidance scale"),
    )
    target_guidance_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="CFG scale for target velocity (FlowEdit/FlowAlign; 1.0 = no CFG).",
        json_schema_extra=ui_field_config(order=12, label="Target guidance scale"),
    )
    zeta_scale: float = Field(
        default=1e-3,
        ge=0.0,
        description="DIFS alignment term weight (FlowAlign only).",
        json_schema_extra=ui_field_config(order=13, label="Zeta scale (FlowAlign)"),
    )
