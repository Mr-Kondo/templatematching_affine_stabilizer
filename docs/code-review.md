# Code review: app.py / 02_code

## Summary

修正後はマージ可能です。対象は `c9d498331c120a3ff53b7db5d566f45a612d0045` の本番コードです。
初回対象 `7b4e1f2` と、改良前 `8fc3abc` からの差分、GUI・CLI・ベンチマークの呼出し元を確認しました。
研究用候補の採用判断・テストコードの品質レビューは、この本番コードレビューの範囲に含めません。

Counts: 0 critical, 0 important, 0 nits（修正後の未解決指摘）。
初回は1 critical、4 important、2 nitsでした。下の7件が対応します。

## Initial findings (resolved)

### Critical findings

- C1: `02_code/stabilizer_core.py:515`（初回対象）— correctness / 外部ライブラリ境界検証。
  `ok=M is not None` はNaN行列も成功にします。現在座標が一致する2点、異なる基準座標を与えると
  OpenCV 4.14はNaN行列と内点マスクを返します。失敗を期待するテストが修正前に失敗することを確認。
  有限性と正の行列式を検証し、不正行列は推定失敗として記録して停止する修正を行いました。

### Important findings

- I1: `02_code/stabilizer_core.py:481`（初回）— function size / SRP。
  `_M` の本体53行に推定・内点診断・行列差・品質条件・キャッシュが集中していました。
  `_estimate`、`_inlier_metrics`、`_motion_metrics`、`_quality_ok`へ役割を分け、キャッシュ管理は `_M` に残しました。
- I2: `02_code/stabilizer_cli.py:9`（初回）— function size / 抽象度の混在。
  main本体57行が引数・動画選択・処理・表示を行っていました。
  `_parser`、`_video_path`、`_print_report`へ分離し、既定値・失敗・表示を維持しました。
- I3: GUI描画・編集と切出しの既存API — function size / four-argument ceiling。
  長いTk宣言・描画トランザクション、方向別引数の保持に、判断理由と再検討条件がありませんでした。
  下記の例外をコード内に記録しました。数値合わせの設定クラスや互換ラッパーは追加しません。
- I4: `02_code/stabilizer_gui.py:305`（初回）— DRY。
  CSVヘッダーと行の列順が共有仕様 `TEMPLATE_FIELDS` と二重管理されていました。
  `_save_templates`を分離し、共有仕様からヘッダーと行を保存するようにしました。

### Nits

- N1: GUIのwidget節など — comment pollution。コードを繰り返す区分コメントを削除しました。
- N2: `_show` の旧Pillow用LANCZOS分岐 — dead compatibility code。
  Pillow 13へ固定した構成で不要な旧版分岐を削除しました。PPM経由のTk互換処理は維持し、実Tkで再検証しました。

## Documented exceptions

関数20行・引数4個の目安に対し、次を保持します。コード内の `clean-code exception` に理由と再検討条件があります。

- `crop_lrbt`、`_match_window`の方向別引数: 既存呼出しの座標契約を維持。探索幾何の変更時に見直します。
- `TemplateGUI._add_tpl`の引数順: 既存GUI呼出しを維持。項目追加時に行オブジェクトへの変更を検討します。
- GUI初期化: Tk終了までのCapture所有を一つにするため保持。キャンセル機能を追加するときに見直します。
- `_build_widgets`: 明示的なTk宣言の順序を保ちます。新パネル追加時に分割します。
- `_show`: PhotoImage参照の寿命を含む描画単位を保持。別ビュー導入時に分割します。
- `_edit`: 取消時に確定前の全変更を捨てる編集単位を維持。非モーダル編集に変えるときに分割します。
- `_match`: 元候補と再探索候補の採用条件を一か所で確認できるよう維持。新たな探索方式を採用するときに分割します。
- `_warp_frames`: 逐次読込・変換・進捗の単一ループを維持。別の出力先を追加するときに分割します。

## Coverage

- Section A (naming & functions): I1/I2を修正、I3は理由・再検討先を記録。数学の局所変数と既存API名は保持。
- Section B (comments & formatting): N1/N2を修正。PPM回避の理由、閾値の実測根拠、確定した例外コメントは保持。
- Section C (SOLID): I1/I2を修正。処理・GUI・CLIの分離を確認。不要なinterface、factory、継承は追加なし。
- Section D (DRY/KISS/YAGNI): I4を修正。GUI・CLIは同じrun_pipelineを使い、cropの境界処理も共通。新依存なし。
- Section E (AI failure modes): C1を修正。assertによる外部入力検証なし。GUI workerの広いcatchはエラー通知への転送で成功を偽装しない。
  例外時のCapture/Writer解放、専用一時領域、既存成果物の保持、CLI/GUIの入力検証を削除していないことを確認。

## Re-review and validation

修正後の対象コミットでA〜Eを再確認しました。29単体テストは成功し、退化2点テストは
修正前FAIL→修正後PASSです。実動画306フレームの回帰ベンチマークでCSV3種の値と
補正動画バイト列が変更前に一致し、品質診断の失敗は0件でした。
このMacの実TkでCSV保存→worker実行→4成果物作成→完了通知→Capture解放を確認しました。
クロスOSの対話GUI、強制終了・電源断の回復は未検証です。CIの単体テストはPRのChecksを参照してください。

レビューの完了は、未観測の動画での画質や正解精度を保証しません。
