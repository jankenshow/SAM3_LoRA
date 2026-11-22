# 第2章 データセットの意味・準備・変換

前章：[環境](01_setup.md) · [目次](../README.md) · 次章：[学習](03_training.md)

## 2.1 画像・物体・プロンプトの関係

1画像には複数の物体と複数のテキストクエリを持たせます。例えば「dog」のクエリは、その画像内で該当する全物体のIDを参照します。「cat」が存在しない画像なら、対象物体が空の負例クエリとして扱えます。画像IDはCOCO上の値、配列添字はデータセット内の位置として区別します。10、42、74など連番でないIDも扱います。

各画像の注釈が対象プロンプトについて網羅的である前提です。未注釈の物体や、別表現で同じ物体を指すクエリを安易に負例にしないでください。単語の重なりで除外する汎用負例フィルタは、意味的な同義語までは判定しません。

## 2.2 COCO配置と座標

```text
data/
  train/
    _annotations.coco.json
    image001.jpg
  valid/
    _annotations.coco.json
    image002.jpg
  test/                       # 評価用。学習にはtrainとvalidを使う
    _annotations.coco.json
    image003.jpg
```

`images`、`annotations`、`categories` を持つCOCO JSONを使います。`file_name` は各splitからの相対パスです。bboxはピクセル単位の `[x, y, width, height]`。内部では正規化した `[center_x, center_y, width, height]` に統一します。polygon、圧縮RLE、非圧縮RLEのマスクを扱います。画像は正方形にリサイズし、RGBの各値を `pixel / 127.5 - 1` に正規化します。標準解像度は1008です。

マスクはnearestでリサイズし、二値を保ちます。bboxだけの注釈は学習データ構造上扱えますが、マスク損失はその物体を無効マスクとして扱います。マスク精度評価にはGTマスクが必要です。不正な注釈を黙って飛ばして物体IDがずれることは避け、読込エラーとして報告します。

```sh
python -m sam3_lora prepare inspect --split-dir data/train --check-images
```

この検査は注釈ID・参照・bbox、画像パスの存在を確認します。全画像のデコード検査ではありません。実際のデータローダは画像寸法と注釈寸法、不正なマスクなども確認します。

## 2.3 テキストの選択

`training.prompt_source: auto` の優先順は、注釈テキスト → 画像テキスト → カテゴリ説明 → カテゴリ名です。それぞれの段階で `_descriptions.json` の上書きを先に見ます。

```json
{
  "by_annotation": {"100": "a shiba inu"},
  "by_image": {"image001.jpg": "a dog sitting on the floor"},
  "by_category": {"dog": "a domestic dog"}
}
```

注釈では `caption / description / text / phrase / expression`、画像では `caption / description / text`、カテゴリでは `description / caption / instructions / prompt` を見ます。最終的なクエリは小文字化・前後空白除去します。

`prompt_source` を `caption`、`image_caption`、`category_description`、`category_name` に固定することもできます。指定フィールドがない場合はカテゴリ名に戻ります。`mix_class_prompt: true` は説明文に加えてカテゴリ名でも同じ物体を教師として与えます。

## 2.4 負例

同じsplitのプロンプト集合が50種類以下なら、画像内にないプロンプトをすべて負例にします。50種類を超える場合は `num_cross_negatives` 個までサンプリングします。さらに `num_negatives` 個まで汎用負例を加えます。小さな集合では `num_cross_negatives: 0` でも集合内の不在クエリは残る仕様です。既定の汎用候補は `data/prompts.py` に集約しています。

学習時は負例を再サンプリングし、検証時はseedと画像位置から再現可能にします。物体のない画像には `object` の負例クエリを最低1つ作ります。

## 2.5 split作成

```sh
python -m sam3_lora prepare split --split-dir data/all \
  --output data/split_v1 --valid-fraction 0.1 --test-fraction 0.1 --seed 42
```

画像単位に分割し、画像と対応注釈・sidecarを新しい場所へコピーします。元データは変更しません。出力先が既に存在する場合は停止します。件数は割合の切り捨てなので、小さいデータではvalid/testが空になり得ます。学習には空でないtrainとvalidが必要です。動画由来フレーム等のグループ分割は未実装なので、同一動画の漏洩を防ぐ分割は別途設計してください。

## 2.6 ODVG・他形式

COCOからマスク付きODVG JSONLへ変換できます。既存の出力は上書きしません。

```sh
python -m sam3_lora prepare coco-to-odvg --data-dir data
```

ODVGは `filename / height / width / grounding.regions` を持ち、regionに `phrase` とピクセル単位の **xyxy** bbox、任意の `segmentation` を置きます。`training.data_format: odvg` で選択します。`use_full_caption: true` は画像全体のcaptionに全regionを対応させる追加クエリを作ります。COCOと同条件を比較する場合はfalseにします。detection形式も `instances` 内にカテゴリ文字列・bboxがある形式を受け付けます。外部の整数ラベルマップを自動推定する機能はありません。

```sh
python -m sam3_lora prepare roboflow --split-dir data/train
python -m sam3_lora prepare refcoco --refcoco-dir input/refcoco \
  --images-dir input/images --output-dir data/refcoco --mode ref --sentences all
```

Roboflow変換は各画像のJSONに `image / annotations` とクラス名またはcategories表がある形式向けです。標準COCO exportは変換不要です。複数クラスを一律objectへ潰す旧挙動を修正しました。RefCOCOは `instances.json` と `refs*.p` を読み、`ref` モードでは文に対応する物体、`category` モードでは元カテゴリ単位の物体を使います。画像はbasenameで検索します。欠落画像・参照は集計して表示します。入力のpickleは出所が分かるものを使ってください。

データセットZIPは公開対象から外しました。[Machineryの取得先](../../examples/datasets/README.md)から配布元を確認し、取得したデータは `data/` など公開対象外の場所へ配置してください。旧ZIPに同梱されていたREADMEでは153画像のセマンティックマスク形式です。標準CLI向けのCOCO/ODVGへの準備は別途必要です。今回、データの新規取得や実データでの学習は行っていません。
