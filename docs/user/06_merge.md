# 第6章 LoRAのマージとモデル保存

前章：[評価](05_evaluation.md) · [目次](../README.md) · 次章：[ツール](07_tools.md)

## 6.1 差分重みと統合重み

差分重みはA/Bだけなので、元のSAM3重みと同じLoRA構造が必要です。マージは `delta_W = (A @ B).T * alpha / rank` を元Linearの重みに加算し、推論時に別のLoRA行列積を行わなくてよい形へ戻す処理です。

明示的なQ/K/V投影へ置換した `nn.MultiheadAttention` は、投影をマージした後、元の融合 `in_proj_weight` を持つ標準MHAへ戻します。ViTの `qkv` LinearはLinearのまま統合します。デバイス・重みdtype・train/eval状態を維持します。推論比較ではeval状態を使います。

## 6.2 コマンド

以下も元モデル全体を構築・読込する処理です。今回このPCでは行っていません。

```sh
python -m sam3_lora merge --config configs/base_config.yaml \
  --weights outputs/sam3_lora/best_lora_weights.pt \
  --output outputs/merged/sam3_lora_merged.pt
```

CPUで統合し、標準SAM3構造のstate_dictを保存します。`--device` は互換引数として受け付けますが、マージ処理はCPU固定です。出力ファイルが既に存在する場合は停止します。差分のキーまたはshapeが異なる場合も事前検証で停止し、部分的に読み込みません。

## 6.3 検証と限界

軽量テストでは、差分を非ゼロにしたLinear/MHAについて、マージ前後の出力、元のstate_dictキー、FP64 dtypeを確認しています。実SAM3での精度同等性は未検証です。実運用前には同じ入力・後処理でマージ前後のテンソル誤差と評価指標を確認してください。有限精度の丸め差は発生し得ます。

統合後は差分だけの再学習・差分の切替とは扱いが異なります。元モデルと差分重みは残してください。このコマンドはCore ML/CoreAI/GGUF等への変換ではありません。その先の変換工程へは、統合した標準SAM3 state_dictを入力します。
