# 実験・旧実装

標準の学習・推論・評価には [統一CLI](../README.md) を使います。このディレクトリは過去の試行を失わずに分離するための参考資料です。通常のテスト・パッケージ配布には含めません。

| 場所 | 内容 |
|---|---|
| training/ | 旧Trainer・学習ライブラリ／起動方法の別案 |
| prototypes/ | toy model、旧src構成、旧パッケージ構成、旧設定 |
| legacy/ | 旧推論・比較・データ準備 |
| diagnostics/ | 実モデルを必要とする旧調査スクリプト |
| launchers/ | 環境固有パスを持つ旧シェル／YAML起動例 |

これらを一括実行しないでください。重いモデル構築や学習、古いAPI・パスを含みます。移動したためそのままでは動かないものもあり、現行CLIの動作保証には含めません。特に旧 `SAM3TrainerNative` をimportする診断コードは、そのクラスを新しいroot入口から取得できません。再利用する場合は現行のTrainingApplication・data/bridgeへ用途別に移植します。

LoRA層の実装はここに複製せず `sam3_lora.adapters` へ委譲し、重複していたtoyデータセットも `prototypes/toy_dataset.py` だけにまとめています。実験を復活させる場合もこの依存方向を保ってください。旧APIのcheckpoint形式や層選択の意味が現行と同じかは個別に確認が必要です。

ルートの `lora_layers.py` と `dataset_formats.py` は削除しました。旧コードのLoRA importは `sam3_lora.adapters` を直接参照します。新しい実験でも正式パッケージを使い、ルートの互換ファイルへ依存しないでください。
