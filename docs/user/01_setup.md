# 第1章 環境と重みの準備

[目次](../README.md) → 次章：[データセット](02_datasets.md)

## 1.1 実行環境を分ける

標準パッケージはPython 3.12以上を対象とします。実モデルの学習先はCUDA環境を想定しています。同梱SAM3にはTritonなどのGPU依存があるため、汎用の `torch.device('mps')` を受け付けることと、SAM3全体がMac/MPSで動作することは同義ではありません。今回Macで確認したのはCPUの軽量テストです。

設定確認だけなら、既存Python環境にPyYAMLがあれば次を実行できます。

```sh
python -m sam3_lora --help
python -m sam3_lora train --config configs/base_config.yaml --dry-run
```

`--dry-run` は設定の構文・値域を確認します。画像の読込、実モデルの構築、重みの読込・ダウンロード、学習・推論は行いません。データファイルの存在やGPUの対応までは検証しません。

## 1.2 公式SAM3の初期化と実行環境への導入

ソースから実モデルを使う前に、プロジェクトrootで公式SAM3を初期化します。

```sh
git submodule update --init --recursive
```

初回clone時には `git clone --recurse-submodules <forkのURL>` を使えます。`vendor/sam3` は公式リポジトリのコミット `757bbb0206a0b68bee81b17d7eb4877177025b2f` に固定しています。これはソースコードだけの取得で、モデル重みの取得ではありません。help/dry-runでは初期化を要求しません。


以下は実行先のマシンで行う手順です。今回このPCでは実施していません。

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[runtime]'
```

PyTorchとtorchvisionの配布物は、実行先のCUDA環境に対応する組み合わせを選んでください。依存関係は `pyproject.toml` の基本依存、`runtime`（同梱SAM3）、`experiment`（旧実験）、`dev`（整形・パッケージ検査）に分けています。`requirements.txt` は同じ定義への参照です。

`sam3-lora` と `python -m sam3_lora` は同じCLIです。ソース上では後者をそのまま使えます。公式SAM3のリポジトリは `vendor/sam3/`、Pythonパッケージは `vendor/sam3/sam3/` にあり、実モデルを使う時点で読み込まれます。

## 1.3 ローカル重み

設定の `model.checkpoint_path` に公式SAM3のPyTorchチェックポイント `sam3.pt` を指定します。LoRA差分重みとは別のファイルです。

```yaml
model:
  checkpoint_path: weights/sam3.pt
  allow_download: false
```

重みはコピー必須ではなく、既存の絶対パスでも構いません。指定したローカルファイルが存在しない場合は、モデルをimportする前にエラーになります。ダウンロードは既定で禁止し、ローカルパスを省略して `allow_download: true` を明示した場合だけ同梱SAM3へ委ねます。今回、重みの取得は行っていません。

BPE語彙は公式submoduleの `vendor/sam3/assets/bpe_simple_vocab_16e6.txt.gz` を利用します。wheelでは同じ語彙をパッケージ内のsam3/assetsへ収録します。異なる語彙を指定する場合のみ `model.bpe_path` を設定します。モデル・重みの利用条件は元の配布物に従います。この整理により新しいライセンスを付与していません。
