# 第5章 評価と指標の読み方

前章：[推論](04_inference.md) · [目次](../README.md) · 次章：[マージ](06_merge.md)

## 5.1 評価の単位

**1画像 × 1プロンプト** を1単位としてCOCO形式へ変換します。同じ画像のdogとcatは別単位です。各単位に連番IDを振り直し、元のCOCO画像IDとデータセット内の位置は `source_id / source_index` として別々に記録します。

GTの正規化cxcywhからCOCOのピクセルxywhへ明示変換します。GT・予測マスクは288×288へnearestで揃え、RLEで保存します。入力画像1008×1008、評価マスク288×288、推論結果の元画像サイズは別の意味です。

```sh
python -m sam3_lora evaluate --config configs/base_config.yaml \
  --weights outputs/sam3_lora/best_lora_weights.pt \
  --data-dir data/test --device cuda --threshold 0.3 --nms-iou 0.7 \
  --output outputs/evaluation/lora
```

`--data-dir` には **split自体** を渡します。学習のdata_dir（train/validを含む親）とは指定の意味が異なります。元モデルの評価は同じconfigと `--use-base-model` を使い、差分重みを省略します。

```sh
python -m sam3_lora evaluate --config configs/base_config.yaml --use-base-model \
  --data-dir data/test --device cuda --output outputs/evaluation/base
```

`--num-samples N` は先頭N画像だけの確認用で、無作為標本ではありません。少数結果をデータ全体の性能として扱わないでください。実行時は1画像ずつ入力し、全画像の画像テンソルを保持しません。評価用のRLE注釈・予測は結果集計のため保持します。

## 5.2 指標

| 指標 | 意味 |
|---|---|
| mAP / mAP50 / mAP75 | マスクのインスタンス対応に基づくCOCO AP。IoU 0.50:0.95平均／0.50／0.75 |
| cgF1 | 同梱SAM3のCGF1Evaluatorが利用可能な場合の指標辞書 |
| pixel_iou | 各単位内でGT・予測マスクをそれぞれunionし、全単位のTP/FP/FN画素を合算したIoU |
| pixel_precision / pixel_recall | 同じ合算方法による画素precision/recall |
| mean_unit_iou | GTまたは予測が非空の各単位のIoU平均 |
| negative_prompt_fpr | 負例単位のうち1件以上検出した割合。負例がなければnull |

予測ゼロでも有効なCOCO結果を作り、GTがある場合のAPは0になります。GTも予測もない単位はmean_unit_iouから除きます。評価範囲にGTがまったくなければCOCOのAPは-1（未定義）になります。cgF1の依存が欠ける場合は `cgF1_unavailable` に理由を残し、0点と取り違えません。壊れた入力など、依存不足以外のエラーは隠しません。

## 5.3 負例と後処理の比較条件

通常の不在プロンプト追加は `--include-negatives` で有効にします。物体のない画像がもともと持つbackgroundクエリは、このフラグを省略しても残ります。負例数・生成条件を揃えて比較してください。存在する物体の注釈漏れはFPRを歪めます。

スコアとNMSは推論と共通です。`--no-presence` はPresenceを後処理で掛けない比較用です。`--merge` はNMSの代わりに重複マスクをunionする診断用で、`--merge-iou` や `--merge-dilate` を変更できます。dilateの画素単位はモデル出力マスクの解像度です。マージでは最大スコアの候補のbboxを代表として残します。通常結果との比較では、後処理条件を必ず併記してください。

## 5.4 保存物・旧版との比較

`ground_truth.json`、`predictions.json`、`metrics.json` を出力します。別試行には別のoutputを使ってください。旧版のID対応、bbox座標、マスク補間、Presence計算に不整合があったため、旧数値と現行数値の差をそのままモデルの改善・劣化と解釈しないでください。今回、実SAM3モデルでのAPやcgF1は再測定していません。
