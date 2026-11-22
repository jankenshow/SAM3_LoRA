import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sam3_lora.config import load_config, validate_training_config

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_every_config_validates_without_model_load(self):
        for path in (ROOT / "configs").glob("*.yaml"):
            with self.subTest(path=path.name):
                validate_training_config(load_config(path))

    def test_help_and_dry_run_do_not_import_torch_or_sam3(self):
        for args in (
            ["--help"],
            ["train", "--help"],
            ["infer", "--help"],
            ["evaluate", "--help"],
            ["merge", "--help"],
            ["compare", "--help"],
            ["prepare", "--help"],
            ["train", "--config", "configs/base_config.yaml", "--dry-run"],
        ):
            with self.subTest(args=args):
                script = """import sys
class BlockHeavy:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','sam3'):
   raise RuntimeError('Heavy import: '+fullname)
sys.meta_path.insert(0,BlockHeavy())
from sam3_lora.cli.main import main
main(sys.argv[1:])
"""
                result = subprocess.run(
                    [sys.executable, "-c", script, *args], cwd=ROOT, text=True, capture_output=True
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_config_rejected(self):
        config = load_config(ROOT / "configs/base_config.yaml")
        for key, value in [
            ("batch_size", 0),
            ("learning_rate", float("nan")),
            ("mixed_precision", "invalid"),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_training_config({**config, "training": {**config["training"], key: value}})
        with self.assertRaises(ValueError):
            validate_training_config({**config, "lora": {"rank": 0}})

    def test_tools_entrypoint_help(self):
        for name in (
            "train_sam3_lora_native.py",
            "infer_sam.py",
            "inference_lora.py",
            "validate_sam3_lora.py",
            "merge_lora_weights.py",
            "convert_coco_to_odvg.py",
            "convert_roboflow_to_coco.py",
            "convert_refcoco_to_coco.py",
        ):
            with self.subTest(name=name):
                result = subprocess.run(
                    [sys.executable, str(ROOT / "tools" / name), "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_tools_module_entrypoint_help(self):
        for path in (ROOT / "tools").glob("*.py"):
            if path.name.startswith("_"):
                continue
            with self.subTest(tool=path.name):
                result = subprocess.run(
                    [sys.executable, "-m", f"tools.{path.stem}", "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_tool_dry_run_from_other_directory_without_heavy_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            # sitecustomize guards the actual script process before imports start.
            Path(temp, "sitecustomize.py").write_text("""import sys
class BlockHeavy:
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('torch', 'sam3'):
   raise RuntimeError('Heavy import: '+fullname)
sys.meta_path.insert(0, BlockHeavy())
""")
            environment = {**os.environ, "PYTHONPATH": temp}
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/train_sam3_lora_native.py"),
                    "--config",
                    str(ROOT / "configs/base_config.yaml"),
                    "--dry-run",
                ],
                cwd=temp,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"model_loaded": false', result.stdout)
