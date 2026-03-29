# Affine Image Stabilizer

テンプレートマッチング（サブピクセル精度）を用いた動画ブレ補正ツール。  
複数の追跡点からアフィン変換行列を推定し、フレームごとに映像を安定化します。

---

## 機能特徴

- **GUI / CLI 両対応**: Tkinter による対話的なテンプレート設定 GUI と、バッチ処理用の CLI モードを提供
- **サブピクセル精度**: 3×3 二次近似によるサブピクセル精度のテンプレートマッチング
- **RANSAC 推定**: `cv2.estimateAffinePartial2D(RANSAC)` で外れ値に強いアフィン行列推定
- **プログレス表示**: GUI ではプログレスバー、CLI では tqdm による進捗表示
- **柔軟な探索範囲**: テンプレートごとに上下左右の探索マージンを個別設定可能

---

## ディレクトリ構成

```
templatematching_affine_stabilizer/
├── 01_input/                    # 入力ファイル
│   ├── input.mp4                # 入力動画（MP4）
│   └── Input.csv                # テンプレート定義CSV
├── 02_code/                     # ソースコード
│   └── affine_template_matching.py  # メインスクリプト（676行）
├── 03_output/                   # 出力ファイル
│   ├── Affined.mp4              # ブレ補正済み動画
│   └── Matched.csv              # マッチング結果
├── README.md                    # 本ドキュメント
├── pyproject.toml               # プロジェクト設定・依存パッケージ
├── uv.lock                      # 依存関係のロックファイル
├── .python-version              # Python バージョン指定
├── .venv/                       # Python仮想環境（uv が自動作成）
└── prompt_chatgpto3.yml         # ChatGPT o3用プロンプト
```

---

## インストール

### 1. uv をインストール（未インストールの場合）
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. リポジトリをクローン
```bash
git clone <repository-url>
cd templatematching_affine_stabilizer
```

### 3. 依存パッケージをインストール
```bash
uv sync
```

> **Note**: Python バージョン（3.13）は `.python-version` で管理されており、`uv` が自動的に適切なバージョンを使用します。

---

## 使い方

### GUI モード（デフォルト）

```bash
uv run python 02_code/affine_template_matching.py
```

または動画ファイルを明示的に指定:
```bash
uv run python 02_code/affine_template_matching.py --video 01_input/input.mp4
```

**GUI 操作方法**:
1. **テンプレート追加**: キャンバス上をクリックしてテンプレート位置を指定
2. **パラメータ設定**: 上部の Entry でテンプレートサイズ・探索範囲を入力
3. **ズーム**: `+` / `-` ボタンで画像を拡大縮小
4. **フレーム移動**: スライダーで任意フレームをプレビュー
5. **編集**: TreeView の行をダブルクリックで値を編集
6. **削除/元に戻す**: `Delete` / `Undo` ボタン
7. **実行**: `Execute` ボタンでマッチング＆補正を開始（テンプレートは3点以上必要）

### CLI モード（バッチ処理）

```bash
uv run python 02_code/affine_template_matching.py --nogui
```

既存の `Input.csv` を使用してバッチ処理を実行します。

**オプション一覧**:
| オプション | 説明 | デフォルト |
|-----------|------|-----------|
| `--video` | 入力動画ファイルのパス | `01_input/` 内の最初のMP4を自動検出 |
| `--csv` | テンプレート定義CSVのパス | `01_input/Input.csv` |
| `--nogui` | GUIを使わずバッチ処理 | (無効) |

---

## Input.csv フォーマット

テンプレート位置と探索範囲を定義するCSVファイル:

| カラム名 | 説明 |
|---------|------|
| `No.` | テンプレート番号（連番） |
| `x座標` | テンプレート中心のX座標（ピクセル） |
| `y座標` | テンプレート中心のY座標（ピクセル） |
| `テンプレートの大きさ（正方形）` | テンプレートの一辺サイズ（ピクセル） |
| `探索左` | 中心から左方向への探索マージン |
| `探索右` | 中心から右方向への探索マージン |
| `探索上` | 中心から上方向への探索マージン |
| `探索下` | 中心から下方向への探索マージン |

**例**:
```csv
No.,x座標,y座標,テンプレートの大きさ（正方形）,探索左,探索右,探索上,探索下
1,106,242,64,64,128,100,64
2,210,227,64,64,128,100,64
3,573,152,64,64,128,100,64
```

---

## 出力ファイル

### Matched.csv
各フレームでのテンプレートマッチング結果:
| カラム | 説明 |
|-------|------|
| `Frame_No` | フレーム番号 |
| `Template_No` | テンプレート番号 |
| `x`, `y` | 基準座標（1フレーム目のテンプレート中心） |
| `mx`, `my` | マッチング座標（サブピクセル精度） |

### Affined.mp4
アフィン変換によりブレ補正された出力動画。

---

## アルゴリズム概要

### 1. テンプレート抽出
1フレーム目から、指定された座標を中心に正方形のテンプレート画像を切り出し。

### 2. テンプレートマッチング
各フレームで `cv2.matchTemplate(TM_CCORR_NORMED)` を実行:
- 指定された探索範囲（ROI）内でマッチング
- ピーク位置を3×3の二次近似でサブピクセル化
- 結果を `Matched.csv` に保存

### 3. アフィン変換推定
フレームごとに `cv2.estimateAffinePartial2D(RANSAC)` で2×3アフィン行列を推定:
- 移動（translation）、回転（rotation）、スケール（scale）を補正
- RANSAC により外れ値の影響を低減

### 4. フレーム補正
`cv2.warpAffine` で各フレームを変換し、`Affined.mp4` に書き出し。

---

## ソースコード構成

**`02_code/affine_template_matching.py`** (約676行)

| クラス/関数 | 説明 |
|-----------|------|
| `refine(mat, loc)` | 二次近似によるサブピクセルピーク推定 |
| `crop(src, cx, cy, h)` | 正方形パッチの切り出し |
| `crop_lrbt(src, cx, cy, l, r, t, b)` | 非対称ROIの切り出し |
| `TemplateMatcher` | テンプレートマッチングを実行するクラス |
| `AffineCorrector` | アフィン補正を実行するクラス |
| `TemplateGUI` | Tkinter GUIクラス |
| `main()` | エントリーポイント（CLI引数解析） |

---

## 依存パッケージ

```
numpy==2.3.0
opencv-python-headless==4.11.0.86
pandas==2.3.0
pillow==11.2.1
tqdm==4.67.1
```

**Python バージョン**: 3.13（`.python-version` で管理）

---

## 開発ツール

本プロジェクトは ChatGPT o3 を活用して開発されました。  
プロンプトは `prompt_chatgpto3.yml` を参照してください。

---

## ライセンス

（必要に応じて追記）
