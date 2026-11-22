# SAM3 LoRA

SAM3の画像PCSを対象とするLoRA学習・推論・評価プロジェクトです。標準経路を `sam3_lora/` に統一し、旧実装・試作は `expt/`、SAM3本体は `vendor/sam3/` の公式submoduleとして管理しています。

**最初に [マニュアル目次](docs/README.md) を読んでください。** データ準備 → 学習 → 推論 → 評価 → マージの順に説明しています。[開発者向け設計](docs/developer/architecture.md)、[旧構成からの移行](docs/developer/migration.md)、[検証内容](docs/developer/testing.md)もあります。

## 計算せずに確認する

この2つのコマンドはSAM3やPyTorchをimportせず、モデル構築・重み取得・学習を行いません。YAMLの読込にはPyYAMLが必要です。

```sh
cd src/SAM3_LoRA
python -m sam3_lora --help
python -m sam3_lora train --config configs/base_config.yaml --dry-run
```

実際のSAM3学習・推論は十分なメモリを持つ実行環境で行ってください。今回このPCで実施したのは、小さなダミーモデルと人工データによるCPUテストのみです。実モデルでの精度・速度・CUDA分散実行は未検証です。

## SAM3の取得

```sh
# 既存のcheckoutで公式SAM3を初期化する
git submodule update --init --recursive
```

`vendor/sam3` は公式 [facebookresearch/sam3](https://github.com/facebookresearch/sam3) のコミット `757bbb0206a0b68bee81b17d7eb4877177025b2f` に固定しています。clone時は `--recurse-submodules` を指定できます。モデル重みは別途用意します。helpと設定のdry-runにはsubmoduleの初期化は不要です。

## 構成

```text
sam3_lora/     標準ライブラリ・統一CLI
  adapters/    LoRA設定・挿入・保存・マージ
  data/        COCO/ODVG・プロンプト・座標変換・SAM3接続
  training/    学習進行・損失・最適化・再開
  inference/   セッション・共通後処理・結果保存
  evaluation/  プロンプト単位のCOCO評価・画素指標
  cli/         引数・コマンド・エラー処理
tools/        学習・推論・評価・データ変換コマンドの入口
configs/      実装に対応した設定例
tests/        軽量なCPU回帰テスト（実モデル不要）
docs/         ユーザー／開発者マニュアル
expt/         実験・旧実装の参考資料
vendor/sam3/  公式SAM3のsubmodule（固定コミット）
examples/     データセットの取得先案内
licenses/     同梱コードのライセンス文書
```

`tools/train_sam3_lora_native.py`、`tools/infer_sam.py` などは統一CLIへの薄い互換入口です。[ツール一覧と使い方](docs/user/07_tools.md)を参照してください。新しいコードは `sam3_lora` を利用してください。依存関係の定義は `pyproject.toml` に集約しています。

## 出典・ライセンス・改変について

本リポジトリは [Sompote/SAM3_LoRA](https://github.com/Sompote/SAM3_LoRA) を基にした改変版です。起点はコミット [`b0050de84d1e5268e3dcd6f53a60aa0dde88ed08`](https://github.com/Sompote/SAM3_LoRA/tree/b0050de84d1e5268e3dcd6f53a60aa0dde88ed08) です。

元リポジトリの[READMEのLicense節](https://github.com/Sompote/SAM3_LoRA/blob/b0050de84d1e5268e3dcd6f53a60aa0dde88ed08/README.md#license)では **Apache 2.0** と記載されています。ただし、確認した起点コミットにはLICENSE本文ファイルがないため、ライセンスの参照元を上流READMEとして明記しています。

SAM3本体は [Meta / facebookresearch/sam3](https://github.com/facebookresearch/sam3) の改変していない固定コミットをsubmoduleとして参照し、**SAM License** が適用されます。契約本文は [licenses/SAM3_LICENSE](licenses/SAM3_LICENSE)、適用範囲と出典は [licenses/README.md](licenses/README.md) を参照してください。第三者コードを含む全体を、一律のApache 2.0として扱うものではありません。

この改変版では、2026-11-22にパッケージ・ディレクトリ構成の整理、LoRA実装とデータ処理の共通化、学習設定・評価座標・スコア処理の修正、統一CLI・軽量テスト・日本語マニュアルの整備を行いました。SAM3本体を公式submoduleへ移行しました。データセットZIP・旧画像・旧資料は公開対象から除外し、データ取得先の案内へ置き換えています。原著者による公式の改変版ではありません。詳細は [移行案内](docs/developer/migration.md) に記録しています。
