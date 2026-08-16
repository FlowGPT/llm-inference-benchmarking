import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


MODULE_PATH = (
    Path(__file__).parents[1] / "scripts" / "build_parallel_draft_proxy.py"
)


def load_proxy_module():
    spec = importlib.util.spec_from_file_location("build_parallel_draft_proxy", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_acceptance_schedules_have_requested_means():
    proxy = load_proxy_module()
    for k in (3, 7):
        schedules = proxy.acceptance_schedules(k)
        assert set(schedules) == {50, 55, 60}
        for target, rates in schedules.items():
            assert len(rates) == k
            assert rates == sorted(rates, reverse=True)
            assert sum(rates) / k == pytest.approx(target / 100)


def test_eagle3_proxy_uses_three_aux_hidden_states():
    proxy = load_proxy_module()
    spec = proxy.proxy_spec("eagle3")
    assert spec.num_speculative_tokens == 3
    assert spec.config["architectures"] == ["Eagle3LlamaForCausalLM"]
    assert spec.config["num_aux_hidden_states"] == 3
    assert spec.tensor_shapes["fc.weight"] == (5120, 15360)


def test_dspark_keeps_native_block_size():
    proxy = load_proxy_module()
    spec = proxy.proxy_spec("dspark")
    assert spec.num_speculative_tokens == 7
    assert spec.config["architectures"] == ["Qwen3DSparkModel"]
    assert spec.config["dspark_block_size"] == 7
    assert spec.config["n_predict"] == 7


def test_dflash_proxy_has_mask_and_parallel_block_metadata():
    proxy = load_proxy_module()
    spec = proxy.proxy_spec("dflash")
    assert spec.num_speculative_tokens == 3
    assert spec.config["architectures"] == ["DFlashDraftModel"]
    assert spec.config["dflash_config"]["mask_token_id"] == 131071
    assert spec.config["n_predict"] == 3


def test_dry_run_writes_config_and_tensor_manifest_only(tmp_path):
    proxy = load_proxy_module()
    output = tmp_path / "eagle3"
    proxy.write_dry_run(proxy.proxy_spec("eagle3"), output)
    assert sorted(path.name for path in output.iterdir()) == [
        "config.json",
        "tensor-manifest.json",
    ]
    config = json.loads((output / "config.json").read_text())
    manifest = json.loads((output / "tensor-manifest.json").read_text())
    assert config["architectures"] == ["Eagle3LlamaForCausalLM"]
    assert manifest["method"] == "eagle3"
    assert manifest["tensors"][0]["shape"] == [5120, 15360]


def test_dry_run_cli_writes_selected_method(tmp_path):
    output = tmp_path / "proxy"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--method",
            "dflash",
            "--dry-run",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads((output / "tensor-manifest.json").read_text())["method"] == (
        "dflash"
    )
