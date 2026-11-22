import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from sam3_lora.adapters import (
    AdapterInjector,
    LoRAConfig,
    LoRALinear,
    adapter_state_dict,
    apply_lora_to_model,
    load_adapter_state,
    load_lora_weights,
    merge_lora_linears,
    restore_multihead_attention,
    save_lora_weights,
)
from tests.helpers import adapted, tiny_model


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_zero_adapter_is_identity(self):
        base = tiny_model(dtype=torch.float64)
        model = copy.deepcopy(base)
        apply_lora_to_model(model, LoRAConfig(rank=2, target_modules=("proj",)))
        x = torch.randn(2, 2, dtype=torch.float64)
        torch.testing.assert_close(model(x), base(x))
        self.assertEqual(model.proj.lora.lora_A.dtype, torch.float64)
        self.assertFalse(model.proj.original_layer.weight.requires_grad)

    def test_attention_masks_and_layout(self):
        for batch_first in (True, False):
            for mask_kind in ("none", "bool", "float", "padding_float", "per_head"):
                with self.subTest(batch_first=batch_first, mask=mask_kind):
                    base = tiny_model(True, torch.float64)
                    base.attn.batch_first = batch_first
                    base.eval()
                    model = copy.deepcopy(base)
                    apply_lora_to_model(model, LoRAConfig(rank=2, target_modules=("all",)))
                    q = torch.randn((2, 3, 8) if batch_first else (3, 2, 8), dtype=torch.float64)
                    mask = torch.ones(3, 3, dtype=torch.bool).triu(1)
                    args = {}
                    if mask_kind in ("bool", "float"):
                        args["attn_mask"] = (
                            mask
                            if mask_kind == "bool"
                            else torch.zeros(3, 3, dtype=q.dtype).masked_fill(mask, float("-inf"))
                        )
                    if mask_kind == "padding_float":
                        args["key_padding_mask"] = torch.tensor(
                            [[0.0, 0.0, -float("inf")], [0.0, 0.0, 0.0]], dtype=q.dtype
                        )
                    if mask_kind == "per_head":
                        args["attn_mask"] = mask.expand(4, -1, -1)
                    expected = base.attn(
                        q, q, q, need_weights=True, average_attn_weights=False, **args
                    )
                    actual = model.attn(
                        q, q, q, need_weights=True, average_attn_weights=False, **args
                    )
                    for a, b in zip(expected, actual):
                        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)

    def test_cross_attention_and_causal(self):
        base = tiny_model(True)
        base.eval()
        model = copy.deepcopy(base)
        apply_lora_to_model(model, LoRAConfig(rank=2, target_modules=("all",)))
        q = torch.randn(2, 3, 8)
        k = torch.randn(2, 5, 8)
        torch.testing.assert_close(
            model.attn(q, k, k)[0], base.attn(q, k, k, need_weights=False)[0]
        )
        mask = torch.ones(3, 3, dtype=torch.bool).triu(1)
        torch.testing.assert_close(
            model.attn(q, q, q, is_causal=True)[0],
            base.attn(q, q, q, attn_mask=mask, need_weights=False)[0],
        )

    def test_unbatched_fails_explicitly(self):
        model = adapted(True)
        with self.assertRaisesRegex(ValueError, "3-D"):
            model.attn(*(torch.zeros(3, 8),) * 3)

    def test_lora_gradients_and_base_freezing(self):
        model = adapted(True)
        q = torch.randn(2, 3, 8)
        model.attn(q, q, q)[0].square().sum().backward()
        for name, p in model.named_parameters():
            if "lora_B" in name:
                self.assertGreater(float(p.grad.abs().sum()), 0)
            elif ".lora." not in name:
                self.assertIsNone(p.grad)

    def test_merge_parity_state_keys_dtype(self):
        for attention in (False, True):
            with self.subTest(attention=attention):
                original = tiny_model(attention, torch.float64)
                keys = set(original.state_dict())
                apply_lora_to_model(original, LoRAConfig(rank=2, target_modules=("all",)))
                original.eval()
                for n, p in original.named_parameters():
                    if "lora_B" in n:
                        nn.init.normal_(p, std=0.1)
                x = (
                    torch.randn(2, 3, 8, dtype=torch.float64)
                    if attention
                    else torch.randn(2, 2, dtype=torch.float64)
                )
                expected = original.attn(x, x, x)[0] if attention else original(x)
                merge_lora_linears(original)
                restore_multihead_attention(original)
                actual = original.attn(x, x, x, need_weights=False)[0] if attention else original(x)
                torch.testing.assert_close(expected, actual, atol=1e-12, rtol=1e-10)
                self.assertEqual(set(original.state_dict()), keys)
                self.assertTrue(all(p.dtype == torch.float64 for p in original.parameters()))
                self.assertFalse(original.attn.training if attention else original.proj.training)

    def test_flat_checkpoint_roundtrip(self):
        model = adapted()
        model.proj.lora.lora_B.data.fill_(0.5)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "weights.pt"
            save_lora_weights(model, path)
            other = adapted()
            load_lora_weights(other, path)
            self.assertEqual(
                set(adapter_state_dict(other)), {"proj.lora.lora_A", "proj.lora.lora_B"}
            )
            for key, value in adapter_state_dict(model).items():
                torch.testing.assert_close(value, adapter_state_dict(other)[key])

    def test_checkpoint_preflight_prevents_partial_mutation(self):
        model = adapted()
        before = adapter_state_dict(model)
        bad = {k: torch.ones_like(v) for k, v in before.items()}
        bad["proj.lora.lora_B"] = torch.ones(99, 1)
        with self.assertRaisesRegex(ValueError, "shape"):
            load_adapter_state(model, bad)
        for k, v in before.items():
            torch.testing.assert_close(v, adapter_state_dict(model)[k])
        with self.assertRaisesRegex(ValueError, "keys"):
            load_adapter_state(model, {})

    def test_injection_rejects_double_and_no_target(self):
        with self.assertRaisesRegex(ValueError, "twice"):
            apply_lora_to_model(adapted(), LoRAConfig())
        with self.assertRaisesRegex(ValueError, "No LoRA targets"):
            apply_lora_to_model(tiny_model(), LoRAConfig(target_modules=("missing",)))

    def test_component_flags_and_fused_qkv(self):
        model = nn.Module()
        model.vision_backbone = nn.Module()
        model.language_backbone = nn.Module()
        model.vision_backbone.qkv = nn.Linear(8, 24)
        model.language_backbone.qkv = nn.Linear(8, 24)
        apply_lora_to_model(model, LoRAConfig(rank=2, target_modules=("qkv",)))
        self.assertIsInstance(model.vision_backbone.qkv, LoRALinear)
        self.assertIsInstance(model.language_backbone.qkv, nn.Linear)
        self.assertEqual(model.sam3_lora_report.linear_modules, ("vision_backbone.qkv",))

    def test_unsupported_attention_rejected(self):
        model = nn.Module()
        model.attn = nn.MultiheadAttention(8, 2, kdim=4)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            AdapterInjector(LoRAConfig(target_modules=("all",))).apply(model)

    def test_invalid_config(self):
        for options in (
            {"rank": 0},
            {"rank": True},
            {"dropout": 1},
            {"alpha": float("nan")},
            {"target_modules": ()},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                LoRAConfig(**options)
