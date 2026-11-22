# 開発者編2 軽量テストと検証範囲

[目次](../README.md) · [設計](architecture.md) · 次：[移行案内](migration.md)

## 1. 実モデルを使わないテスト

このPCの計算資源に合わせ、実SAM3の学習・推論・重み取得を避けています。標準テストは小さなCPUモデルと8×6程度の人工画像で実行します。

```sh
python -m unittest discover -s tests -v
```

今回使用した既存環境からの実行例です。新しく学習用の環境を構築したわけではありません。

```sh
../sam3_pytroch/.venv/bin/python -m unittest discover -s tests -v
```

| ファイル | 何を検証するか |
|---|---|
| test_adapters.py | 初期差分ゼロでの同等性、非ゼロのLoRA勾配、凍結、融合QKV投影、MHAのmask・causal・cross attention、マージの出力・キー・dtype、厳密checkpoint |
| test_data.py | 非連番ID、座標、COCO/ODVG同等性、マスクRLE/polygon、sidecar・負例、不正データ、元データを保つ分割、クラス維持変換、RefCOCOのsymlink配置 |
| test_training.py | 勾配蓄積の端数を含む手計算との一致、検証の非更新、保存・optimizer/scheduler再開、非有限値、重複のないvalid sampler、toyモデルでのワークフロー |
| test_postprocess.py | Presence・NMS・空予測・元画像座標・COCO単位・GT要求・可視化 |
| test_evaluation.py | 完全予測と予測ゼロのCOCO AP、人工出力での評価結果・負例FPR保存 |
| test_runtime_boundary.py | 公式collatorとの契約、負例のみのbatch、最終出力container、toy出力でのInferenceSession、重み不足時の事前停止、submodule未初期化時のエラー、source/wheelのBPE探索 |
| test_cli.py | 全設定例、toolsの直接／モジュール起動help、別ディレクトリからのdry-run、torch/SAM3 importを禁止したhelp/dry-run |

`runtime_fixtures.py` は同梱ソースからデータ型だけを抜き出し、公式collatorと出力containerを実行します。SAM3のmodel_builderはimportしません。これにより接続契約をテストしていますが、フルランタイムの動作確認とは区別します。

## 2. 整形・静的検査・配布

```sh
ruff check sam3_lora tools tests ./*.py
ruff format --check sam3_lora tools tests ./*.py
python -m build --wheel --no-isolation --outdir /path/to/artifacts
```

ruffの範囲は標準実装・互換入口・軽量テストです。公式vendorと参考用exptは除外します。パッケージには標準実装、同梱sam3、BPE語彙、同梱設定が入り、tools・expt・テスト・データ配布物は入りません。

今回の整形・配布検査用ツールは、ワークスペース内 `tmp/sam3_lora_refactor/tools-venv` にのみ導入しました。システムPythonやグローバル設定を変更していません。公式SAM3のsubmodule登録に必要なGit操作だけを明示承認のもと実施し、commit・pushは行っていません。

## 3. 今回の実行結果と限界

2026-11-22の結果は **54件成功**（テスト本体1.775秒）です。静的検査・整形チェック66ファイル、wheel生成、配布物のhelp/dry-runも成功しました。同梱SAM3の176ファイルは移動前とSHA-256が一致しました。詳細は [validation.json](validation.json) と [テストログ](test-results.txt) に記録しています。CPUテストは数秒程度の規模です。wheelの生成と、別ディレクトリに展開した配布物からのhelp/dry-run、BPEの同梱も検査します。

未検証の範囲は実SAM3での学習・推論・精度・速度・メモリ、CUDA AMP、複数GPUのcollectiveやcheckpoint再開、実データ全件の読込です。実行先では、まず少数サンプルのforward/backwardと保存・再読込、次に同一条件のbase/LoRA評価、最後に分散を確認してください。今回の軽量テストを実モデルの精度検証や速度測定として扱わないでください。

ルートの互換ファイル2本の削除後も52件が成功しました。標準・tools・exptを含むPython 110ファイルを構文解析し、削除したモジュールへのimportが残っていないことを確認しています。exptの実モデル構築・実行は行っていません。

公式submoduleへの移行後は54件が成功しました。sdistを生成し、そのsdistからwheelを生成する経路も確認済みです。wheel内のBPE語彙とSAM Licenseは公式submoduleのファイルと一致します。元のsource checkoutがない場所からhelp/dry-runとBPE探索を確認しました。データセットZIP・デモ画像・動画は配布物に含みません。submodule本体は変更なしで、親のindexの変更は.gitmodulesとvendor/sam3のgitlinkだけです。
