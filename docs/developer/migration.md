# 開発者編3 移行案内・挙動変更

[目次](../README.md) · [設計](architecture.md) · [検証](testing.md)

## 1. 配置と標準入口

初期のコード整理はGit操作を行わず実施しました。その後、明示的な承認により公式SAM3のsubmodule登録を行いました。commit・pushは行っていません。移動一覧は [moves.json](moves.json) に記録しています。

| 旧入口・配置 | 現在 |
|---|---|
| train_sam3_lora_native.py の巨大Trainer | tools/train_sam3_lora_native.py → CLI train → training/runner・engine・losses |
| infer_sam.py / inference_lora.py | tools/内の同名スクリプト → CLI inferへの薄い互換入口 |
| validate_sam3_lora.py | tools/validate_sam3_lora.py → CLI evaluateへの薄い互換入口 |
| merge_lora_weights.py | tools/merge_lora_weights.py → CLI mergeへの薄い互換入口 |
| lora_layers.py と2系列のlora実装 | sam3_lora/adapters/に一本化。ルートの互換ファイルは削除 |
| dataset_formats.py、Trainer・評価側の別loader | sam3_lora/data/へ共通化。ルートの互換ファイルは削除 |
| convert_*_to_*.py | tools/内の同名スクリプト → CLI prepareへの薄い入口。実装はdata/conversion |
| src/、旧sam3_lora/model・train・lora | expt/prototypes/ |
| その他の学習・推論・診断・起動試行 | expt/training・legacy・diagnostics・launchers |
| sam3/ | vendor/sam3/の公式submoduleへ移行（元の176ファイルは一致） |
| rootに散在した旧Markdown | 現行マニュアルへ集約し、不要な旧資料を削除 |
| asset/、データ配布物 | 旧画像を削除し、ZIPの代わりにexamples/datasets/へ取得先を記載 |

旧スクリプト名の入口は残していますが、**全引数・旧Pythonクラスの完全互換ではありません**。`python tools/<script> --help` と現行マニュアルで対応を確認してください。旧SAM3TrainerNativeは公開APIとして残していません。設定で実際に使っていなかったlogging_steps/eval_steps/save_steps等は例から外しました。

## 2. Python APIと重み

推論は `InferenceSession`、学習は `TrainingApplication` を利用します。旧 `SAM3LoRAInference` 名は互換facadeとして残しています。LoRAの実装は `sam3_lora.adapters` からimportしてください。

COCO/ODVG loaderは、公式SAM3 Datapointの代わりにモデル非依存の **Sample** を返します。公式collatorへは `data.bridge.collate_samples` または `to_datapoint` を通します。画像ID・クエリ・座標を明示して個別テストできる境界に変えています。

従来のflatな `module.lora.lora_A/B` キーと行列の向きは保持します。ただし読込を厳密にし、キー欠落・余剰・shape不一致はエラーになります。旧設定と異なるtarget/flags/rankで読み込むと停止するため、**既存の差分重みには作成時の設定を使用**してください。今回の設定例ではImage encoder対象の `qkv` を明記したので、以前qkvを対象にしなかった差分に新しい例をそのまま当てないでください。

旧prototypeの別形式を無条件で変換する機能はありません。必要なら保存キーとA/B向きを検査する専用の移行処理を作成します。差分ファイルと、optimizer等を含むtraining_stateは別の形式です。

## 3. 整理に伴って修正した挙動

| 箇所 | 以前の問題 | 現在 |
|---|---|---|
| 学習ループ | 主要なnativeループで勾配蓄積・mixed_precision・scheduler・clipの設定が反映されていない | 共通EpochEngineで実行。端数蓄積も実数で平均 |
| 検証 | データ読込エラーを広く捕捉して検証を省略し得た | 欠損・不正なvalidを明示エラー |
| データ・評価ID | COCO image IDとデータセット位置を混用 | source_id/source_indexを分離 |
| bbox | 正規化cxcywhとxyxyを経路ごとに解釈 | 各境界で明示的に変換 |
| マスク | train/evalで別変換、不正注釈のskipによる対応ずれ | 共通nearest補間、異常は停止 |
| 推論スコア | Presenceを掛けない推論と、掛ける評価の不一致 | 共通postprocess |
| MHAのマスク | float key padding mask／causalの扱いが不十分 | bool/floatとcausalを実装、未対応入力は明示エラー |
| LoRA merge | 標準MHAへ戻す際のdtype/device・モード | 明示的に保持 |
| Roboflow変換 | 複数カテゴリをobjectへ潰す | クラス名を保持。不明なクラスを推測しない |
| 設定・依存 | 未使用設定、複数の依存一覧、テンプレートの著者・URL | 対応設定へ整理、pyprojectへ集約、未確認の著者・URLを除去 |
| checkpoint | 差分のみでoptimizer等を再開できない | 再開用stateを別保存、既存履歴への無指定上書きを拒否 |

これらはファイル移動だけではなく実際の挙動修正です。旧設定と同じ学習軌跡や、旧評価値と同じ数値になるとは限りません。軽量テストの範囲は [testing.md](testing.md) に記載しました。実SAM3の再学習や精度比較は今回行っていません。

## 4. 旧実験・元ファイル

旧試作は参考用としてexptに保持しています。同一だったtoy datasetは1実装へ集約し、旧LoRA実装はcanonical実装への委譲に置換しました。不要な旧実装を再び標準経路へコピーしないでください。旧資料は削除しました。実験コードの実行手順は現行保証の対象外です。

移動前のソースとハッシュは作業記録 `tmp/sam3_lora_refactor/` にも保存しています。同梱SAM3については移動前の全ファイルのSHA-256と照合し、内容の保持を検証しています。生成したwheel・検査ログも同じ作業記録へ置き、プロジェクトのrootにビルド生成物を残さない構成にしています。

## 5. ツールをtoolsへ配置

追加の整理で、ルートに残っていた学習・推論・評価・マージ・データ変換の8入口を `tools/` へ移動しました。旧ルートパスへの転送ファイルは残していません。シェルやノートブックの参照は `python tools/<旧ファイル名>` または統一CLIへ変更してください。infer_samのPython APIを利用していた場合は `sam3_lora.inference.session` からのimportを推奨します。

ルートのPython API互換ファイルは次節の通り削除しました。`setup.py` はビルド互換入口としてルートに残します。参考用exptには移動前のパスが含まれます。現行のツール一覧は [第7章](../user/07_tools.md) を参照してください。

## 6. ルートのPython API互換ファイルを削除

`lora_layers.py` と `dataset_formats.py` は正式パッケージの再exportだけを行っていたため、削除しました。expt内の5ファイルの `from lora_layers import ...` は `from sam3_lora.adapters import ...` へ変更済みです。`dataset_formats` を参照するPythonコードはプロジェクト内にありませんでした。

外部のスクリプト・ノートブックで旧importを使っている場合は、用途ごとに次へ変更してください。

```python
from sam3_lora.adapters import LoRAConfig, apply_lora_to_model, load_lora_weights
from sam3_lora.data.dataset import COCOSegmentDataset, ODVGDataset, build_dataset
from sam3_lora.data.prompts import PromptResolver, first_text, load_descriptions_sidecar
from sam3_lora.data.transforms import decode_segmentation
```

データローダの返り値は既述のSampleです。import先の変更は、旧expt全体の実行互換性を保証するものではありません。exptは引き続き参考用で、実モデルの実行は今回行っていません。旧コード例を含むdocs/archiveは公開前の整理で削除しました。

## 7. 公開対象の整理

不要な `docs/archive/` と `docs/assets/` を削除しました。データセットZIPは公開リポジトリから外し、`examples/datasets/README.md` に取得元と形式を記載しています。Machineryと一致しなかった舗装データセットの説明文も削除しました。旧資料の作者連絡先・古いAPI例は公開対象から除外されています。

過去のGit履歴やindexの内容は、このファイル整理だけでは変わりません。commit・pushや履歴の変更は別の操作です。

## 8. 公式SAM3をsubmoduleへ移行

公式コミット `757bbb0206a0b68bee81b17d7eb4877177025b2f` は、移行前の176ファイルすべてと一致します。最新mainではAttention等が変更されているため、今回の移行にモデル実装の更新は含めていません。

登録先は `vendor/sam3`、その中のPythonパッケージは `sam3/` です。読み込み・BPE探索・パッケージ配布・軽量テストの参照位置を調整しました。初回cloneでは `--recurse-submodules`、既存checkoutでは `git submodule update --init --recursive` を使ってください。登録に伴い.gitmodulesとgitlinkはステージされますが、その他の変更は自動ステージせず、commit・pushも行いません。
