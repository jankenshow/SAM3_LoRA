# 開発者編1 設計・依存方向・拡張方法

[目次](../README.md) · 次：[テスト](testing.md)

## 1. 設計の境界

標準実装は `sam3_lora/` のみです。CLIは引数をワークフローへ渡し、ワークフローはデータ・モデル・損失・出力の各責務を組み合わせます。SAM3のimport・重み取得は `modeling.py` を境界にし、CLI helpと設定確認から切り離します。

```text
cli/parser → cli/commands → TrainingApplication / InferenceSession / evaluate
                                ↓                 ↓              ↓
                           EpochEngine      共通postprocess ← PromptCOCOBuilder
                                ↓                 ↓
                           SAM3LossStep        data/bridge
                                └──── modeling ───┘
                                         ↓
                                  vendor/sam3

adapters と data/records・prompts・transforms は SAM3 に依存しない
```

「SAM3に依存しない」はPyTorchにも依存しないという意味ではありません。テンソル演算を行うモジュールはtorchを使います。`cli/parser.py`、設定、LoRAConfigはモデルを読み込まず検査できます。

| モジュール／型 | 責務・テスト境界 |
|---|---|
| adapters/LoRAConfig | 対象・ランク・係数の検証。軽量import |
| adapters/AdapterInjector | コンポーネント選択、MHA置換、LinearのLoRA挿入。InjectionReportを生成 |
| adapters/checkpoint・merge | 差分だけの保存／事前検証／統合。SAM3を構築しない |
| data/Instance・PromptQuery・Sample | 正規化cxcywh、物体ID、画像IDと位置を明示した共通契約 |
| data/PromptResolver・NegativePromptPolicy | テキスト優先順位と負例選択 |
| data/COCOSegmentDataset・ODVGDataset | 入力形式を共通Sampleへ変換 |
| data/transforms | 画像正規化・bbox変換・二値マスク補間 |
| data/bridge | Sampleから公式Datapoint／collatorへの接続、最終出力の取得 |
| training/EpochEngine | モデル非依存の勾配蓄積・AMP・更新・検証 |
| training/SAM3LossStep | SAM3固有のmatcher・損失・補助出力 |
| training/TrainingApplication | seed、DataLoader、分散、保存、early stoppingの進行 |
| inference/InferenceSession | モデルの保持、画像入力、クエリごとの予測 |
| inference/postprocess | Presence、閾値、マスクNMS、出力座標変換 |
| inference/visualization | Pillowによる画像・二値マスク・JSON出力 |
| evaluation/PromptCOCOBuilder | 画像×プロンプト単位のGT・予測を作成 |
| evaluation/coco・masks | COCO AP、unionした画素指標、診断用マージ |
| cli/main | ログ・例外の表示。verboseならトレースバック |

## 2. LoRA実装を1つにする

本体は `adapters/layers.py` と `attention.py` にのみあります。公開APIは `sam3_lora.adapters` です。exptの参照もこのパッケージへ揃え、ルートの再exportファイルは削除しました。expt内に残る旧APIの委譲も、同じ実装を利用します。コマンド用スクリプトは `tools/` に置き、共通の `_bootstrap.py` から統一CLIへ委譲します。A/Bの向きと `module.lora.lora_A/B` の保存キーを固定します。loadは全キーとshapeを検証した後に適用し、不一致で一部だけ変わる状態を避けます。

MHA置換は内部関数が `out_proj.weight` を直接読む経路でLoRA forwardを迂回しないための処理です。通常のLinear置換と同一視しません。明示MHAは上流同様のmatmul/softmax経路を維持し、bool/floatマスクとcausal指定を扱います。高速化を測定した変更ではありません。

## 3. 学習の拡張

`TrainingApplication` は `model_factory / dataset_factory / loss_factory / collate` を注入できます。テストでは数個のパラメータのモデルに差し替え、SAM3の構築なしで保存・再開まで確認します。

`EpochEngine.loss_step(model, batch)` は微分可能なスカラーlossを返す契約です。新しい損失はこの契約を守り、最適化ループへデータ形式の分岐を持ち込まないでください。学習と検証の差は勾配・optimizer更新の有無です。最後の端数蓄積の除数、DDPのno_sync境界、scheduler更新回数を回帰テストで保護しています。

分散の開始・集約・終了は `training/distributed.py` に置きます。validは重複のないSamplerとunwrappedモデルで処理し、不揃いなrank長でDDP buffer同期を待たない構成にしています。実GPUでの分散動作は今回未検証です。

## 4. 新しい入力形式・後処理

新データ形式はSampleを返すloaderを追加し、`build_dataset` の選択へ接続します。座標変換・プロンプト解決を複製しないでください。正規化bboxはrecords内でcxcywh、ユーザー向け画像出力はpixel xyxy、COCOはpixel xywhという境界を守ります。

推論・評価で後処理を変える場合は `postprocess` を共通変更し、同一の人工logitでテストします。`evaluate` はsession_factoryも注入可能なので、実モデルなしで集計の回帰を確認できます。可視化は結果dictだけを受け取り、モデルをimportしません。

## 5. 試作・同梱コード

新たな実験は `expt/<意味のある名前>/` へ置き、標準経路へは完成した共通機能だけを取り込みます。実験から標準モジュールをimportする方向に揃えます。標準コードからexptをimportしません。

`vendor/sam3/` は公式SAM3の固定コミットを参照するsubmoduleです。Pythonパッケージはその中の `sam3/`、BPE語彙は `assets/` にあります。移行前の176ファイルは公式コミットと一致しています。参照版の更新が必要になった場合は、LoRA側の変更と分けて理由・互換条件を記録してください。重みのダウンロードは既定で無効で、helpや通常importでは行いません。

## 6. ツールの配置

`tools/` はソースツリー用のコマンド入口です。CLIの引数定義・処理本体は `sam3_lora/cli` に集約し、ツールへコピーしません。`_bootstrap.run` はプロジェクトrootをPythonの探索パスに追加するだけで、作業ディレクトリは変更しません。直接のスクリプト実行と `python -m tools.<名前>` の双方を扱います。

help/dry-runではSAM3・torchをimportしません。tools自体はwheelへ含めず、インストール先では `sam3-lora` または `python -m sam3_lora` を使います。新しいコマンドはまず統一CLIへ実装し、必要な場合だけ薄い入口をtoolsへ追加してください。利用手順は [ユーザー編第7章](../user/07_tools.md) にまとめています。

## 7. source checkoutと配布物の資源配置

`modeling.ensure_sam3_path` は初期化済みの公式リポジトリrootを探索パスへ追加します。未初期化のsource checkoutには初期化コマンドを示すエラーを返します。wheelでは通常の `sam3` パッケージをimportします。

`find_bpe_path` はwheel内の `sam3/assets/`、公式checkoutの兄弟ディレクトリ `assets/` の順に語彙を探します。`setup.py` のBuildPyは、公式ソースを変更せずwheel生成時だけ語彙を所定の位置へコピーします。MANIFEST.inはsdistにも元の語彙を含めます。ライセンス文書はpyprojectのlicense-filesで配布物へ収録します。
