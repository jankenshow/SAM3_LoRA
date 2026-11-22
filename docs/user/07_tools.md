# 第7章 toolsの一覧・使い方

前章：[マージ](06_merge.md) · [目次](../README.md)

## 7.1 配置と責務

学習・推論・評価・重みマージ・データ変換のコマンド用スクリプトは、プロジェクト直下の `tools/` に置いています。各スクリプトは共通CLIへの薄い入口で、処理本体を複製しません。引数・設定・出力の意味は各機能の章と共通です。

| ツール | 対応する統一CLI | 手順 |
|---|---|---|
| [train_sam3_lora_native.py](../../tools/train_sam3_lora_native.py) | train | [第3章：学習・再開](03_training.md) |
| [infer_sam.py](../../tools/infer_sam.py) | infer | [第4章：画像PCS推論](04_inference.md) |
| [inference_lora.py](../../tools/inference_lora.py) | infer（旧名の別入口） | infer_samと同じ処理 |
| [validate_sam3_lora.py](../../tools/validate_sam3_lora.py) | evaluate | [第5章：評価](05_evaluation.md) |
| [merge_lora_weights.py](../../tools/merge_lora_weights.py) | merge | [第6章：マージ](06_merge.md) |
| [convert_coco_to_odvg.py](../../tools/convert_coco_to_odvg.py) | prepare coco-to-odvg | [第2章：COCO→ODVG](02_datasets.md#26-odvg他形式) |
| [convert_refcoco_to_coco.py](../../tools/convert_refcoco_to_coco.py) | prepare refcoco | [第2章：RefCOCO](02_datasets.md#26-odvg他形式) |
| [convert_roboflow_to_coco.py](../../tools/convert_roboflow_to_coco.py) | prepare roboflow | [第2章：画像単位JSON→COCO](02_datasets.md#26-odvg他形式) |

`prepare inspect`、`prepare split`、`compare` は統一CLIから使います。すべてのサブコマンドの一覧は `python -m sam3_lora --help` で確認できます。

## 7.2 計算せずに確認する

以下は `src/SAM3_LoRA` を作業ディレクトリとする例です。事前のeditable installなしでもソース上から実行できます。

```sh
python tools/train_sam3_lora_native.py --help
python tools/train_sam3_lora_native.py --config configs/base_config.yaml --dry-run
python tools/infer_sam.py --help
python tools/convert_coco_to_odvg.py --help
```

helpとtrainのdry-runでは、torch・SAM3のimport、モデル構築、重み取得、学習・推論を行いません。dry-runは設定値の検査であり、データの存在・GPU対応・精度は確認しません。

プロジェクトrootではモジュールとしての起動も可能です。

```sh
python -m tools.train_sam3_lora_native --config configs/base_config.yaml --dry-run
```

## 7.3 実際に利用する

以下は計算資源のある実行先で使う例です。今回このPCでは実行していません。

```sh
# 学習
python tools/train_sam3_lora_native.py --config configs/base_config.yaml --device cuda

# 画像PCS推論
python tools/infer_sam.py --config configs/base_config.yaml \
  --weights outputs/sam3_lora/best_lora_weights.pt \
  --image data/test/image003.jpg --prompt 'dog' \
  --device cuda --output outputs/inference/result.png

# 評価
python tools/validate_sam3_lora.py --config configs/base_config.yaml \
  --weights outputs/sam3_lora/best_lora_weights.pt \
  --data-dir data/test --device cuda --output outputs/evaluation/lora

# COCOからODVGへの変換（推論なし）
python tools/convert_coco_to_odvg.py --data-dir data
```

マージ・他形式変換の引数は各章を参照してください。`tools/inference_lora.py` は `tools/infer_sam.py` と同じ処理へ委譲します。元モデルとLoRAの比較には `python -m sam3_lora compare ...` を使います。

## 7.4 パスとインストール後の使い方

`tools/_bootstrap.py` はソースパッケージの探索先を追加しますが、**作業ディレクトリを変更しません**。別の場所からスクリプトの絶対パスで実行した場合も、設定内の相対パスと出力先は実行時の作業ディレクトリ基準です。設定ファイルの位置やtoolsの位置を基準にはしません。通常はプロジェクトrootから実行し、別の場所から使う場合は設定・データ・重み・出力のパスを明示してください。

wheelへtoolsは含めません。パッケージのインストール先では `sam3-lora ...` または `python -m sam3_lora ...` を利用します。CLI全体のオプション（例：`--verbose`）を付ける場合も `python -m sam3_lora --verbose train ...` の形を使えます。

ルートにあった旧スクリプトはtoolsへ移動し、旧場所に転送ファイルは残していません。既存の起動スクリプト・ノートブックのパスを更新してください。旧資料・実験コードの扱いは [移行案内](../developer/migration.md) を参照してください。
