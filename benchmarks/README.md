# Benchmarks — 性能・補正品質の基準データと回帰検証

既存パイプライン(テンプレートマッチング → RANSACアフィン推定 → warp)を
固定条件で実行し、**時間・資源・精度**を時系列比較できる形で記録します。
結果はすべてタイムスタンプ付きの新しいディレクトリに書き込まれ、既存の
動画・生成物を上書きすることはありません(誤って上書きが発生した場合は異常終了します)。

```
uv run python benchmarks/run_benchmark.py --mode synth      # 合成動画(正解既知)
uv run python benchmarks/run_benchmark.py --mode video --bg-eval   # 既存動画
uv run python benchmarks/run_benchmark.py --mode synth --repeat 3   # 3回繰り返し
```

## モード

### `synth` — 合成動画による正解検証

`benchmarks/synth.py` が、テクスチャ豊富な基準シーンに **並進・回転・倍率**
(合成中心は画像中心)の既知アフィン運動を与えた動画を生成します。
動画・テンプレートCSV・正解データ(ground_truth.json)はすべて実行ディレクトリに保存され、
同じパラメータ(`--seed` など)で再実行すれば同一入力を再現できます。

正解の座標規約:

- 動画フレーム `k`(1始まり、`k=1` が基準)。基準シーンの点 `p` は
  フレーム `k` で `A[k-1]·p` に現れる(`A[0]` は恒等)。
- ステビライザは `M_k ≈ A[k-1]⁻¹`(フレームk座標 → 基準座標)を推定するはず。

報告される精度指標(区別して報告):

| 指標 | 意味 |
|---|---|
| `matched_error_px` | 追跡点のマッチング位置と正解位置 `A[k-1]·(cx,cy)` の差 |
| `background_residual_px_not_tracked` | **追跡に使わない背景点**の`M_k·A[k-1]·p − p`。追跡点の残差では見えない破綻を暴露する |
| `affine_matrix_error_frobenius` | 推定行列と正解逆行列のFrobenius差 |
| `tracking_failures` | 誤差>2px(閾値)の追跡の割合 |

### `video` — 既存動画の固定条件比較

`--video` / `--csv` で入力とテンプレート定義を指定し、RNGを固定(`cv2.setRNGSeed`)
してから実行します。報告される精度指標:

| 指標 | 意味 |
|---|---|
| `tracked_point_residual_px` | 追跡点の RANSAC フィット残差 `|M_k·(mx,my) − (x,y)|` |
| `ransac_inlier_rate` / `outlier_point_rate` | フレームごとのRANSAC内点割合・外れ点率(追跡失敗率の代理) |
| `background_residual_px_not_tracked`(`--bg-eval`時) | テンプレートROI外のShi-Tomasi特徴点をLK+FBチェックで追跡し、`|M_k·q_k − q_1|` を評価。追跡点とは独立な残留ずれ |

## 記録内容(`report.json` / `summary.md`)

- **工程別時間**: `match`(マッチング)/ `split_png_dump`(PNG分解)/ `warp_and_write`(補正+書き出し)を別々に、`pipeline_total`(合計)も記録
- **全体スループット**: frames_per_second、入出力フレーム数
- **資源**: 最大メモリ(RSS。macOSはバイト、LinuxはKiBで取得したものをバイトに換算)、一時ファイル量(PNG中間ファイル)、出力サイズ
- **環境**: プラットフォーム、Python、numpy/OpenCV/pandas/Pillow/tqdm のバージョン、Gitコミット、全CLIパラメータ、シード

> メモ: `--repeat > 1` で同一プロセス内で複数回実行した場合、最大メモリは
> プロセス生涯のピーク値となるため各実行の上限目安です。厳密な1実行ごとの
> 値が必要なら`--repeat`を分けて実行してください。

## 既存パイプラインへの最小限の変更

- `AffineCorrector.split()`: 幅・高さ・FPSを `self._geom` に保存
- `AffineCorrector.correct(split=...)`: 分解済みフレームの再利用を許可(既定は従来通り)
- `AffineCorrector._M()`: 推定統計(フレーム番号・点数・内点数・成否・推定行列)を
  `self.estimate_stats` に記録

動作は従来と変わらず、GUIパスにも影響しません。

## 制限(スコープ外とするもの)

- 大規模な評価基盤(複数動画の自動SWEEP、DB保存、CI回帰ゲート等)は対象外。
- 合成動画の映像圧縮劣化は mp4v 固定。実行間の厳密なビット一致は保証しない。
- 真の変位>探索範囲のケース(パラメータ側の明示的エラーで停止)。
