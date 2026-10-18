# Affine Image Stabilizer

指定した背景点をテンプレートマッチングで追跡し、動画の平行移動・回転・一様な倍率変化を補正するツールです。
TkinterのGUIで設定する方法と、CSVを使ってCLIで実行する方法があります。

先頭フレームを基準に、軸ごとの3点二次補間で位置をサブピクセル単位で推定し、RANSACで相似変換を求めます。
固定透視変換も合成し、各フレームを1回で再サンプリングします。入力の全フレームを出力します。

## インストール

[uv](https://docs.astral.sh/uv/getting-started/installation/) を導入済みの環境で実行します。
Pythonの指定は `.python-version` の3.13、検証済みの版は3.13.5です。GUIにはTkが必要です。

```bash
git clone https://github.com/Mr-Kondo/templatematching_affine_stabilizer.git
cd templatematching_affine_stabilizer
uv sync --locked
```

macOS ARM環境でGUIを検証し、macOS・Linux・Windowsの動画入出力・CLIをCIで検証しています。
wheelのOS要件と未検証範囲は [依存関係の検証](docs/dependency-validation.md) を参照してください。

## 更新

ローカルの変更を確認し、保存したうえでmainを更新します。

```bash
git switch main
git pull --ff-only
uv sync --locked
```

## 実行

GUIを開き、先頭フレーム上で背景点を設定します。

```bash
uv run python app.py
```

動画とテンプレートCSVが `01_input/` にある場合、CLIでは新しい出力先を指定して実行できます。

```bash
uv run python app.py --nogui --video 01_input/input.mp4 --csv 01_input/Input.csv --outdir 03_output/new_run
```

GUIは `03_output/run_日時/`、CLIは `--outdir` に、補正動画・追跡CSV・診断CSV2種を保存します。
既存成果物は上書きしません。同じ出力先で再実行する場合は別のフォルダ名に変更してください。
従来の `02_code/affine_template_matching.py` からも実行できます。

## 詳しい操作・仕様

- [操作と入出力](docs/usage.md): GUI、CLI、CSV形式、入力条件、診断、エラーへの対処。
- [補正処理と限界](docs/algorithm.md): 照合・補間・推定・変換の順序と判断条件。
- [開発と検証](docs/development.md): コード構成、依存管理、テスト、性能評価、候補手法の採否。

推定に必要な最低点数は異なる中心2点です。品質判定では内点3点以上などを使います。
診断の合格は真の位置精度の保証ではありません。出力は固定FPSで、音声を含みません。

## ライセンス

ソースコードと付属文書は [MIT License](LICENSE) で公開します。依存ライブラリには、それぞれのライセンスが適用されます。
同梱動画・入力データ・生成動画は、このコードのライセンスによって利用権を付与する対象に含めません。
