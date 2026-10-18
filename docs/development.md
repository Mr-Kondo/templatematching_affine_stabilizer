# 開発と検証

[READMEへ戻る](../README.md)

## 入口とコード構成

| ファイル | 担当 |
|---|---|
| `app.py` | GUI・CLI共通の起動入口 |
| `02_code/stabilizer_cli.py` | 引数、動画選択、処理結果の表示 |
| `02_code/stabilizer_gui.py` | Tk画面、CSV編集、workerへの起動と通知の表示 |
| `02_code/stabilizer_core.py` | 入力検証、追跡、推定、補正、安全な出力公開 |
| `02_code/affine_template_matching.py` | 従来の直接実行とimport用の互換入口 |

GUI・CLIは `run_pipeline(video, csv_path, outdir, q=None)` を使います。
`TemplateMatcher` は追跡CSVを逐次保存し、`AffineCorrector` はフレーム索引と行列キャッシュを使います。
補正用のCSV本体・索引・推定診断・行列のメモリは、フレーム数に比例します。

`refine(mat, loc)` は軸ごとの二次補間を行います。
`crop(src, cx, cy, h)` は半辺hの正方形パッチを切り出します。
`crop_lrbt(src, cx, cy, left, right, top, bottom)` は方向別マージンのROIを切り出します。
`TemplateGUI` はTkinter画面、`main()` はGUI・CLIのエントリーポイントです。
境界処理はcrop_lrbtに共通化しています。GUIのPPM経由PhotoImage回避処理はmacOSで検証し、保持しています。

## 依存関係と通常のテスト

直接依存はNumPy、OpenCV headless、pandas、Pillow、tqdmです。
バージョンは [pyproject.toml](../pyproject.toml)、全依存の固定値は [uv.lock](../uv.lock) を参照してください。
[依存関係の検証](dependency-validation.md) にwheelと実動作の確認範囲を記載しています。

以下はルートで実行します。

```bash
uv sync --locked
uv run python -m unittest discover -s tests -v
uv run python app.py --help
uv run python 02_code/affine_template_matching.py --help
```

GitHub ActionsはPython 3.13.5のmacOS・Linux・Windowsで単体テスト・動画入出力・CLIを確認します。
対話GUIの全操作を3OSで保証するものではありません。macOS ARM環境では実Tkで保存から完了まで確認しています。

## 性能・品質の比較

```bash
uv run python benchmarks/run_benchmark.py --mode synth
uv run python benchmarks/run_benchmark.py --mode video --bg-eval
```

結果は日時付きの新規ディレクトリへ保存します。品質閾値・条件・指標の定義は
[ベンチマーク手順](../benchmarks/README.md) を参照してください。
同じ動画・CSV・seedで比較し、工程時間、資源、追跡誤差、独立背景点、推定成功と品質合格を分けて確認します。
[長時間入力の比較記録](../benchmarks/results/issue12-scaling.json) では追跡工程のメモリを比較しています。

## 調査とレビューの記録

- [予測探索・粗密探索](search-evaluation.md): 遮蔽・探索端での誤差と採否。
- [LKと再照合](lk-evaluation.md): 特徴点と中心の対応、復帰、速度と品質の比較。
- [位相相関・ECC](registration-evaluation.md): 背景領域・初期値・変換モデル・追加時間。
- [最終コードレビュー](code-review.md): 対象コミット、指摘、修正、残す例外と再検討条件。

比較用スクリプトはbenchmarks配下にあり、本番の探索方式を切り替えるものではありません。
元READMEにあったChatGPT o3を用いた開発の記載は、この履歴説明に移しています。
参照されていたprompt_chatgpto3.ymlは現リポジトリに存在しないため、取得可能なファイルとしては案内しません。
