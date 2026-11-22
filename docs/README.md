# SAM3 LoRA マニュアル

このマニュアルは `src/SAM3_LoRA` 内で完結する標準実装を説明します。各章のコマンドは **`src/SAM3_LoRA` を作業ディレクトリ** とします。相対パスは設定ファイルの場所ではなく、コマンドを実行したディレクトリを基準に解決されます。

## ユーザーマニュアル

1. [環境と重みの準備](user/01_setup.md)
2. [データセットの意味・準備・変換](user/02_datasets.md)
3. [LoRA学習・設定・再開](user/03_training.md)
4. [画像推論・比較・出力](user/04_inference.md)
5. [評価と指標の読み方](user/05_evaluation.md)
6. [LoRAのマージとモデル保存](user/06_merge.md)
7. [toolsの一覧・使い方](user/07_tools.md)

## 開発者マニュアル

1. [責務分離・依存方向・拡張方法](developer/architecture.md)
2. [軽量テストと検証範囲](developer/testing.md)
3. [移行案内・挙動変更](developer/migration.md)
4. [依存元・ライセンス・submodule](developer/upstream_dependencies.md)
5. [ファイル移動一覧（JSON）](developer/moves.json)

[exptの位置付け](../expt/README.md)も参照してください。旧資料は削除し、現在の手順をこのマニュアルへ集約しています。

## 用語

SAM3の用語は本マニュアル全体で統一します。Image encoderは画像特徴抽出、Fusion encoderは画像・テキスト・プロンプト特徴の統合、Detector decoderは物体クエリの更新、Presence head／mask headは存在判定／マスク生成を担います。Detectorはそれらを含むPCS側の全体を指し、Fusion encoder単体とは区別します。

このプロジェクトの標準CLIは画像PCSを扱います。動画の状態管理やPVS用Trackerの学習・推論は追加していません。TrackerのSAM系Mask decoderと、PCSのmask headは別です。
