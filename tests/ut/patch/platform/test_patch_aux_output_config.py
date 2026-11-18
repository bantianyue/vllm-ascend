# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from vllm.config.aux_output import AuxOutputConfig
from vllm.config.vllm import VllmConfig

# Importing the module applies the Ascend-side AuxOutput validation.
from vllm_ascend.patch.platform import patch_aux_output_config  # noqa: F401


def _config(*, use_v2: bool) -> SimpleNamespace:
    return SimpleNamespace(
        aux_output_config=AuxOutputConfig(enable_return_routed_experts=True),
        use_v2_model_runner=use_v2,
        model_config=SimpleNamespace(runner_type="generate", is_moe=True),
        cache_config=SimpleNamespace(enable_prefix_caching=True),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        kv_transfer_config=None,
    )


def test_aux_output_requires_model_runner_v2():
    with pytest.raises(ValueError, match="requires Model Runner V2"):
        VllmConfig._verify_aux_output_compatibility(_config(use_v2=False))


def test_aux_output_keeps_the_upstream_checks():
    config = _config(use_v2=True)
    config.model_config.is_moe = False
    with pytest.raises(ValueError, match="only supports MoE models"):
        VllmConfig._verify_aux_output_compatibility(config)


def test_guard_is_inactive_when_capture_is_disabled():
    config = SimpleNamespace(
        aux_output_config=AuxOutputConfig(),
        use_v2_model_runner=False,
        model_config=SimpleNamespace(),
        cache_config=SimpleNamespace(),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        kv_transfer_config=None,
    )
    VllmConfig._verify_aux_output_compatibility(config)
