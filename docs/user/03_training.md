# 第3章 LoRA学習・設定・再開

前章：[データ](02_datasets.md) · [目次](../README.md) · 次章：[推論](04_inference.md)

## 3.1 LoRAをどこへ入れるか

元の重みを固定し、Linearの出力に低ランク差分を加えます。この実装では入力 `x` に対し `base(x) + dropout(x) @ A @ B * (alpha / rank)`。Aは `[in_features, rank]`、Bは `[rank, out_features]` です。

Image encoderのViTにある `qkv` は、入力特徴からQ・K・Vをまとめて作る**投影Linearのモジュール名**です。Q/K/Vテンソル自体をレイヤーとして置換する意味ではありません。`nn.MultiheadAttention` は別の手順でQ/K/V投影と出力投影を明示するモジュールに置換し、そのLinearへLoRAを挿入します。この2つの処理を区別します。

設定の `target_modules` は末尾のモジュール名への部分一致（`all` は全Linear）です。コンポーネントの除外設定を先に適用します。

| 設定 | SAM3上の対象 |
|---|---|
| apply_to_vision_encoder | Image encoder／vision backbone |
| apply_to_text_encoder | Text encoder／language backbone |
| apply_to_geometry_encoder | Exemplar encoder（コードのgeometry encoder） |
| apply_to_detr_encoder | Fusion encoder（transformer.encoder） |
| apply_to_detr_decoder | Detector decoder（transformer.decoder） |
| apply_to_mask_decoder | PCSのmask head等、名前にsegmentation_head/mask_decoderを持つ部分 |

Text encoderは既定で対象外です。実際に選ばれた名前は `injection.json` に残します。対象ゼロや二重挿入はエラーです。標準と異なるkdim/vdim、bias_kv、add_zero_attn付きMHAは明示的に拒否します。明示MHAは3次元のバッチ入力を対象にしています。

## 3.2 設定を確認する

```sh
python -m sam3_lora train --config configs/base_config.yaml --dry-run
```

`base_config.yaml` はbatch=1、workers=0、10epoch、AMPなしの出発点です。full/light/minimal/crackは対象層やランク等が異なる設定例であり、精度・メモリを保証するレシピではありません。既存設定との互換性のため、各例のランク等は保持しています。Image encoderを有効にした例には投影Linear `qkv` を明記しました。

重要な設定は次の通りです。

| 設定 | 動作 |
|---|---|
| data_dir / data_format | 第2章のデータ配置とCOCO/ODVG選択 |
| batch_size | 1プロセスの1ミニバッチ当たり画像数 |
| gradient_accumulation_steps | この数のバッチごとにoptimizer更新。最後の端数は実際の数で平均 |
| mixed_precision | no/fp32、fp16、bf16。fp16学習はCUDAのみ |
| max_grad_norm | 更新前の勾配クリップ。0なら無効 |
| lr_scheduler / warmup_steps | cosine/linear/constant。warmupの単位はoptimizer更新回数 |
| num_epochs | エポック上限 |
| early_stopping_patience | 任意。改善しなかったエポック数の上限 |
| early_stopping_min_delta | best判定に必要な損失の減少量 |
| seed | Python・NumPy・PyTorchの初期seed |

LoRAの保存形式は従来の `module.lora.lora_A/B` を保持します。旧経路では一部設定が無視されていたため、同じYAMLでも学習軌跡が変わり得ます。[移行案内](../developer/migration.md)を参照してください。

## 3.3 実際の学習（実行先で行う）

```sh
python -m sam3_lora train --config configs/base_config.yaml --device cuda
```

複数GPUはtorchrunまたは互換のGPU番号指定を使います。

```sh
torchrun --standalone --nproc-per-node=2 -m sam3_lora train \
  --config configs/base_config.yaml --device cuda
# 互換の起動方法
python -m sam3_lora train --config configs/base_config.yaml --device 0 1
```

分散経路はNCCL/CUDAを対象とします。実効的な画像数の目安は `batch_size × 蓄積数 × GPU数` ですが、画像ごとのクエリ数・物体数と損失正規化も学習に影響します。train samplerは長さを揃えるため一部の画像を重複させる場合があります。valid samplerは重複させず、全rankのバッチ損失の合計と件数を集約します。

損失はSAM3のmatcherとloss wrapperへ分離しています。bbox/GIoU、分類・Presence、マスク・Dice、補助出力・one-to-manyを扱います。学習・検証で同じ損失経路を使用します。表示されるepoch lossは**バッチ損失の平均**であり、物体数やクエリ数での再加重平均ではありません。

## 3.4 保存物と再開

| 保存物 | 内容 |
|---|---|
| config.yaml / injection.json | 実行設定・実際の挿入箇所 |
| history.jsonl | epoch、train/valid loss、optimizer更新回数 |
| best_lora_weights.pt | 改善した時点の差分重み |
| last_lora_weights.pt | 最後のエポックの差分重み |
| training_state.pt | 最後の差分重み・optimizer・AMP scaler・scheduler・epoch・best・乱数状態 |

```sh
python -m sam3_lora train --config configs/base_config.yaml --device cuda \
  --resume outputs/sam3_lora/training_state.pt
```

差分重みだけではoptimizer等を再開できません。`--resume` には `training_state.pt` を渡し、元モデル・LoRA構造・データ・学習設定を揃えます。再開なしで既存の学習履歴へ書き込もうとすると停止します。別試行には別output_dirを使います。

再開はエポック境界です。Python、NumPy、CPU/CUDAの乱数状態を保存しますが、分散時はrank0の状態であり、worker別乱数やエポック途中の状態までは保持しません。複数worker・複数GPUでのビット単位の完全再現は保証していません。総epoch数などを変更するとschedulerの将来計画も変わります。

## 3.5 問題の切り分け

設定はdry-run、データはinspect、挿入先はinjection.json、学習経過はhistory.jsonlの順に見ます。欠損したvalid、不正な注釈、非有限loss/クリップ時の勾配はエラーにします。`python -m sam3_lora --verbose train ...` は設定等のエラーもトレースバック表示します。今回このPCでは上記の実学習コマンドを実行していません。
