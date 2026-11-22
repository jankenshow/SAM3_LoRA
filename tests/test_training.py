import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from sam3_lora.adapters import adapter_state_dict
from sam3_lora.training.checkpoint import restore_training_state, save_training_state
from sam3_lora.training.engine import EpochEngine
from sam3_lora.training.runner import EvaluationSampler, TrainingApplication
from tests.helpers import adapted


def mse(model, batch):
    x, y = batch
    return (model(x) - y).square().mean()


def batches():
    return [(torch.tensor([[float(i), 1.0]]), torch.tensor([[float(i) / 2]])) for i in range(1, 6)]


class TrainingTests(unittest.TestCase):
    def test_accumulation_last_partial_group_matches_manual(self):
        torch.manual_seed(4)
        model = nn.Linear(2, 1)
        reference = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        manual = torch.optim.SGD(reference.parameters(), lr=0.01)
        engine = EpochEngine(model, optimizer, mse, "cpu", accumulation=2, max_grad_norm=None)
        engine.run(batches(), training=True)
        for start in range(0, 5, 2):
            manual.zero_grad()
            group = batches()[start : start + 2]
            sum(mse(reference, b) / len(group) for b in group).backward()
            manual.step()
        self.assertEqual(engine.optimizer_steps, 3)
        for a, b in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b)

    def test_validation_no_update_and_loss_mean(self):
        model = nn.Linear(2, 1)
        before = copy.deepcopy(model.state_dict())
        engine = EpochEngine(model, torch.optim.SGD(model.parameters(), lr=0.01), mse, "cpu")
        expected = sum(float(mse(model, b).detach()) for b in batches()) / 5
        self.assertAlmostEqual(engine.run(batches(), training=False), expected)
        self.assertFalse(model.training)
        self.assertEqual(engine.optimizer_steps, 0)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k])

    def test_resume_optimizer_adapter_scheduler_rng(self):
        model = adapted()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        engine = EpochEngine(
            model,
            optimizer,
            mse,
            "cpu",
            max_grad_norm=None,
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9),
        )
        engine.run(batches()[:2], training=True)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "state.pt"
            save_training_state(path, model, engine, 1, 2.0)
            expected_random = torch.rand(2)
            engine.run(batches()[2:], training=True)
            expected = adapter_state_dict(model)
            other = adapted()
            opt = torch.optim.AdamW([p for p in other.parameters() if p.requires_grad], lr=0.001)
            resumed = EpochEngine(
                other,
                opt,
                mse,
                "cpu",
                max_grad_norm=None,
                scheduler=torch.optim.lr_scheduler.StepLR(opt, 1, 0.9),
            )
            state = restore_training_state(path, other, resumed)
            torch.testing.assert_close(torch.rand(2), expected_random)
            resumed.run(batches()[2:], training=True)
            for k, v in expected.items():
                torch.testing.assert_close(v, adapter_state_dict(other)[k])
            self.assertEqual(state["epoch"], 1)
            self.assertEqual(resumed.optimizer_steps, 5)
            self.assertEqual(resumed.scheduler.get_last_lr(), engine.scheduler.get_last_lr())

    def test_nonfinite_loss_and_empty_loader_fail(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        engine = EpochEngine(model, optimizer, lambda m, b: m(b).sum() * float("nan"), "cpu")
        with self.assertRaises(FloatingPointError):
            engine.run([torch.ones(1, 2)], training=True)
        with self.assertRaisesRegex(ValueError, "empty"):
            engine.run([], training=True)

    def test_invalid_fp16_cpu_fails(self):
        model = nn.Linear(2, 1)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            EpochEngine(
                model, torch.optim.SGD(model.parameters(), lr=0.1), mse, "cpu", precision="fp16"
            )

    def test_validation_sampler_no_padding(self):
        partitions = [list(EvaluationSampler(range(5), rank, 3)) for rank in range(3)]
        flat = [i for partition in partitions for i in partition]
        self.assertEqual(sorted(flat), list(range(5)))
        self.assertEqual(len(flat), len(set(flat)))

    def test_workflow_writes_artifacts_and_resumes_with_tiny_model(self):
        with tempfile.TemporaryDirectory() as temp:
            config = {
                "lora": {"rank": 2, "target_modules": ["proj"]},
                "training": {
                    "data_dir": "unused",
                    "num_epochs": 1,
                    "batch_size": 1,
                    "num_workers": 0,
                    "lr_scheduler": "constant",
                    "learning_rate": 0.001,
                },
                "output": {"output_dir": temp},
            }
            options = {
                "model_factory": lambda c, d: adapted(),
                "loss_factory": lambda d: mse,
                "dataset_factory": lambda c, d, s: batches()[:2],
                "collate": lambda samples: samples[0],
            }
            first = TrainingApplication(config, **options).run("cpu")
            self.assertEqual(first["optimizer_steps"], 2)
            for name in [
                "config.yaml",
                "history.jsonl",
                "training_state.pt",
                "best_lora_weights.pt",
                "last_lora_weights.pt",
                "injection.json",
            ]:
                self.assertTrue((Path(temp) / name).is_file(), name)
            config["training"]["num_epochs"] = 2
            second = TrainingApplication(config, **options).run(
                "cpu", Path(temp) / "training_state.pt"
            )
            self.assertEqual(second["optimizer_steps"], 4)
            self.assertEqual(len((Path(temp) / "history.jsonl").read_text().splitlines()), 2)
