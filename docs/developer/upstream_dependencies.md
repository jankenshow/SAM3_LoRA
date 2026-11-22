# 開発者編4 依存元・ライセンス・submodule

[目次](../README.md) · [設計](architecture.md) · [移行案内](migration.md)

## 1. 出典

LoRA側の元リポジトリは [Sompote/SAM3_LoRA](https://github.com/Sompote/SAM3_LoRA)、起点コミットは `b0050de84d1e5268e3dcd6f53a60aa0dde88ed08` です。READMEにはApache 2.0と記載されていますが、当該版にLICENSE本文ファイルは確認できませんでした。出典・表記の扱いは [licenses/README.md](../../licenses/README.md) にまとめています。

SAM3本体は [facebookresearch/sam3](https://github.com/facebookresearch/sam3) の公式submoduleです。現在の固定コミットは [`757bbb0206a0b68bee81b17d7eb4877177025b2f`](https://github.com/facebookresearch/sam3/tree/757bbb0206a0b68bee81b17d7eb4877177025b2f) です。移行前の176ファイルを対応するパスへマッピングしてSHA-256で照合し、すべて一致しました。現在の公式mainを使う場合とは異なり、移行に伴うSAM3演算コードの変更はありません。

## 2. 初期化・固定

```sh
git submodule update --init --recursive
git submodule status -- vendor/sam3
```

親リポジトリのgitlinkは固定コミットを指します。通常の初期化では、その版を取得します。`git submodule update --remote` は参照版を動かす操作なので、通常のセットアップには使いません。デモ画像・動画等は公式リポジトリのcheckout内に存在しますが、親リポジトリでは通常ファイルとして重複管理しません。

## 3. 配置

| 場所 | 内容 |
|---|---|
| vendor/sam3/ | 公式リポジトリroot |
| vendor/sam3/sam3/ | Pythonパッケージ |
| vendor/sam3/assets/bpe_simple_vocab_16e6.txt.gz | 必要なBPE語彙 |
| vendor/sam3/LICENSE | 固定版のSAM License |
| licenses/SAM3_LICENSE | wheel/sdistにも収録する同一文書 |

語彙はwheel生成時に `sam3/assets/` へコピーします。公式checkoutは変更しません。標準の画像PCS経路から、公式agentの外部LLM送信機能は呼びません。

## 4. 参照版を更新する場合

1. 更新先コミットと変更内容を確認する。
2. LoRA対象のMHA/QKV投影、collator・データ型、損失API、正規化と語彙位置を確認する。
3. 軽量テスト、wheel/sdist生成、語彙・ライセンス同梱を確認する。
4. 実行環境で少数のforward/backwardと精度検証を行ってから、本格的な学習へ進む。
5. 承認したコミットを親リポジトリに記録し、README・ライセンス控え・検証記録を更新する。

今回このPCでは4の実モデル処理を行っていません。初期移行時は演算コードの一致と軽量テストで確認しています。
