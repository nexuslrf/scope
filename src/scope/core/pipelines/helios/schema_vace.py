from pydantic import Field

from ..base_schema import ModeDefaults, ui_field_config
from ..common_artifacts import VACE_14B_ARTIFACT
from .schema import HeliosConfig, HELIOS_DISTILLED_ARTIFACT, UMT5_ENCODER_ARTIFACT, WAN_1_3B_ARTIFACT


class HeliosVACEConfig(HeliosConfig):
    pipeline_id = "helios-vace"
    pipeline_name = "Helios VACE"
    pipeline_description = (
        "Helios with VACE video conditioning. "
        "Generates autoregressive video guided by a reference/control video "
        "via ControlNet-style hint injection (8 VACE blocks). "
        "Provide a control video through prepare() to enable V2V conditioning."
    )
    estimated_vram_gb = 28.0
    artifacts = [UMT5_ENCODER_ARTIFACT, WAN_1_3B_ARTIFACT, HELIOS_DISTILLED_ARTIFACT, VACE_14B_ARTIFACT]
    supports_vace = True
    modified = True

    modes = {
        "text": ModeDefaults(default=True),
        "video": ModeDefaults(),
    }

    vace_context_scale: float = Field(
        default=1.0,
        ge=0.0,
        le=2.0,
        json_schema_extra=ui_field_config(order=8, label="VACE scale"),
    )
