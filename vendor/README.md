# 公式SAM3 submodule

`sam3/` は [facebookresearch/sam3](https://github.com/facebookresearch/sam3) のGit submoduleです。

- URL: `https://github.com/facebookresearch/sam3.git`
- 固定コミット: `757bbb0206a0b68bee81b17d7eb4877177025b2f`（2025-11-29）
- 初期化: プロジェクトrootで `git submodule update --init --recursive`
- ライセンス: [SAM3本体のLICENSE](sam3/LICENSE)、配布物用の控えは [licenses/SAM3_LICENSE](../licenses/SAM3_LICENSE)

元のLoRAリポジトリにコピーされていた176ファイルすべてが一致する公式コミットを選びました。SAM3本体のファイルは改変していません。公式リポジトリの全体構成になったため、Pythonパッケージは `vendor/sam3/sam3/`、BPE語彙は `vendor/sam3/assets/` にあります。

親リポジトリはsubmoduleのURL・コミットを管理します。公式リポジトリ内のデモ画像・動画・agent等を親の通常ファイルとして重複登録しません。submoduleのcheckout自体にはそれらの公式ファイルが含まれます。SAM3が外部APIへ送信するagent機能は標準のLoRA学習・画像PCS推論から呼びません。

wheelでは実行に必要なPythonパッケージ、BPE語彙、SAM Licenseを含めます。デモのGIF・動画、公式のexamples・scripts・agent用テキストプロンプト等の追加データはwheelへ含めません。SAM3のPythonサブパッケージにはagentも含まれるため、外部APIを利用する機能を追加する場合は、その送信範囲を別途設計してください。

参照版を更新する際は、LoRAの対象MHA/Linear、データ契約、前処理、BPEの位置を確認してください。最新mainへ自動追従はしません。詳細は [依存元とsubmoduleの開発手順](../docs/developer/upstream_dependencies.md) を参照してください。
