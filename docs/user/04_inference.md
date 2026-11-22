# 第4章 画像推論・比較・出力

前章：[学習](03_training.md) · [目次](../README.md) · 次章：[評価](05_evaluation.md)

## 4.1 入力とセッション

設定のローカルSAM3重みと、学習済みLoRA差分重みを組み合わせて画像PCSを行います。実モデルの実行コマンドなので、十分な計算資源のある実行先で使用します。

```sh
python -m sam3_lora infer --config configs/base_config.yaml \
  --weights outputs/sam3_lora/best_lora_weights.pt \
  --image data/test/image003.jpg --prompt 'dog' 'cat' \
  --device cuda --threshold 0.3 --nms-iou 0.7 \
  --output outputs/inference/result.png
```

各引用文字列が1つのプロンプトです。Python APIは `sam3_lora.inference.session.InferenceSession` を使います。1セッションでモデルを保持して複数画像を処理できます。現在の `predict` は各プロンプトについてforwardを行います。画像・テキスト特徴キャッシュによる最適化は追加していません。

`--weights` を省略すると設定のoutput_dir内のbest差分重みを使います。入力解像度は `--resolution 1008` が標準です。別解像度に対応するかは使用モデル側の制約を確認してください。`auto` はCUDAがあればCUDA、なければCPUです。実モデルをCPUで快適に動かせるという意味ではありません。

## 4.2 共通後処理

物体ごとの分類logitをsigmoidし、Presence logitがあればそのsigmoidを掛けます。既定の同梱画像モデルは未結合の分類スコアを出す構成です。その後、スコア閾値、マスクIoUによるNMS、スコア降順で最大100件に制限します。マスクlogitは0より大きい画素を対象とします。推論と評価は同じ `postprocess` を使います。

この経路により、旧推論でPresenceを考慮しなかった場合とは検出数が変わることがあります。独自モデルで分類スコアへPresenceを事前結合している場合は、呼び出し側で重ねて結合しない設計にする必要があります。

## 4.3 出力形式

| ファイル | 内容 |
|---|---|
| result.png | 元画像にマスク等を重ねた結果 |
| result.json | プロンプトごとのboxes・scores・マスクPNG名 |
| result_prompt0_mask0.png 等 | 元画像サイズの二値マスク（0/255） |

公開推論結果のbboxは元画像のピクセル単位 **xyxy** です。内部の `Prediction.boxes` とデータセットのbboxは正規化 **cxcywh** です。`pixel_output` が明示的に変換し、画像範囲へclipします。内部マスクから元画像への変換はnearestです。

`--boundingbox true`、`--score false`、`--no-masks` は重ね描画を調整します。`--no-masks` を付けても個別マスクPNGとJSONは保存します。表示設定と推論・結果保存を混同しないようにしています。

## 4.4 元モデルとの比較

```sh
python -m sam3_lora compare --config configs/base_config.yaml \
  --weights outputs/sam3_lora/best_lora_weights.pt \
  --image data/test/image003.jpg --prompt 'dog' \
  --device cuda --output outputs/comparison
```

同じ前処理・スコア・NMS・表示設定で、base.png、lora.png、comparison.pngと各JSON・マスクを保存します。baseとLoRAは順にモデルを構築するため、モデル2個の同時保持を避けます。比較画像だけでは精度は確定できません。次章の独立した評価データと合わせて判断します。
