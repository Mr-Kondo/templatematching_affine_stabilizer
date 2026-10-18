# 依存関係の検証（v0.12.0）

[READMEへ戻る](../README.md)

2026-10-18にPyPIの配布情報を確認し、Python 3.13.5でクリーン環境を作成しました。
直接使用する5パッケージを更新し、python-dateutil・pytz・six・tzdataの直接指定を削除しました。
必要な間接依存はuv.lockが管理します。OpenCVは5系への移行を含めず、4系の最新安定版を選びました。
Tk画面にOpenCVのGUI機能は使わないためheadless版を維持しています。

| パッケージ | 変更前 | 採用版 | Python 3.13向けwheelの確認 |
|---|---|---|---|
| NumPy | 2.3.0 | 2.5.4 | macOS ARM/Intel、Linux x86_64/ARM、Windows x86_64 |
| OpenCV headless | 4.11.0.86 | 4.14.0.94 | cp37-abi3: macOS ARM/Intel、Linux x86_64/ARM、Windows x86_64 |
| pandas | 2.3.0 | 3.0.6 | macOS ARM/Intel、Linux x86_64/ARM、Windows x86_64 |
| Pillow | 11.2.1 | 13.0.0 | macOS ARM/Intel、Linux x86_64/ARM、Windows x86_64 |
| tqdm | 4.67.1 | 4.70.1 | py3-none-any |

OpenCV wheelのmacOS最低要件はARMで13、Intelで14です。古いmacOSでのビルドや動作は未検証です。
全パッケージのPython最低要件は3.13.5を満たします。wheelが存在することだけでは実動作を証明しません。

## 確認内容

- このMac（ARM、Python 3.13.5）の新規環境で `uv sync --locked`、24単体テスト、両CLI入口を確認。
- 実動画306フレームで入出力数・追跡誤差の方式間差・品質診断・背景点の残留ずれを比較。
  生データは [issue14-dependencies.json](../benchmarks/results/issue14-dependencies.json)。
- Tk/Pillowは実TkでCSV読込、PPM PhotoImage、反復描画を確認。
- Linux/Windowsの単体テスト・動画入出力・CLIはGitHub Actionsで実施。各実行結果をPRのChecksで確認する。
  Linux/Windowsの対話的なGUI操作は未検証です。

処理時間は1回の測定であり、更新による一般的な高速化率を主張しません。

## 確認した配布情報

- [OpenCV headless](https://pypi.org/project/opencv-python-headless/4.14.0.94/)
- [NumPy](https://pypi.org/project/numpy/2.5.4/)
- [pandas](https://pypi.org/project/pandas/3.0.6/)
- [Pillow](https://pypi.org/project/pillow/13.0.0/)
- [tqdm](https://pypi.org/project/tqdm/4.70.1/)
