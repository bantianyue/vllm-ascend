# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate auxiliary-output configurations on the Ascend platform.

Upstream ``VllmConfig._verify_aux_output_compatibility`` only enforces the
Model Runner V2 requirement for in-tree platforms; out-of-tree platforms are
expected to validate auxiliary-output support themselves. vllm-ascend ships
Model Runner V2 only, so mirror the in-tree requirement here: an auxiliary
output configuration resolved without the V2 runner is rejected instead of
being silently accepted.
"""

from vllm.config.vllm import VllmConfig

_ORIGINAL_ATTR = "_ascend_original_verify_aux_output_compatibility"

_AUX_OUTPUT_V2_REQUIREMENT = (
    "AuxOutput Connector requires Model Runner V2; set VLLM_USE_V2_MODEL_RUNNER=1."
)


def _ascend_verify_aux_output_compatibility(self) -> None:
    aux_output_config = getattr(self, "aux_output_config", None)
    if (
        aux_output_config is not None
        and aux_output_config.enabled
        and not getattr(self, "use_v2_model_runner", False)
    ):
        raise ValueError(_AUX_OUTPUT_V2_REQUIREMENT)
    getattr(VllmConfig, _ORIGINAL_ATTR)(self)


# Keep the patch idempotent: re-importing must not wrap the wrapper.
if not hasattr(VllmConfig, _ORIGINAL_ATTR):
    setattr(VllmConfig, _ORIGINAL_ATTR, VllmConfig._verify_aux_output_compatibility)
VllmConfig._verify_aux_output_compatibility = _ascend_verify_aux_output_compatibility
