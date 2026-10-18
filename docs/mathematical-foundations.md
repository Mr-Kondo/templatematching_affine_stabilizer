# 手法の数学的根拠と成立条件

[READMEへ戻る](../README.md)

この文書は、本ツールが使う各手法の数式・成立条件・限界を、一次資料と現行コード(`02_code/stabilizer_core.py`)の対応箇所付きでまとめたものです。
手順の説明は [補正処理と限界](algorithm.md)、実測に基づく閾値と評価データは [ベンチマーク](../benchmarks/README.md) の「信頼度評価」を参照してください。

対応する依存の版: `opencv-python-headless 4.14.0.94`(OpenCV 4.14)。
出典の OpenCV 文書はこの版に合わせ、式の定義は同版タグのソース(Doxygen コメント)で確認しています。

前提: 以下は実装の挙動を説明するものであり、任意の入力に対する精度保証ではありません。
経験的な閾値や精度の実測値と、数学的に保証される性質は区別して書いています。

## 1. 座標系・記号・単位

- 画素座標: 画像左上を原点とし、$x$ は右向き、$y$ は下向き。単位は画素(px)。
- CSV の $(x, y)$ は 1 フレーム目(基準フレーム)上のテンプレート中心座標。$(mx, my)$ は現フレーム上の対応点。
- 行列は列ベクトル規約: $\begin{bmatrix} x' \\ y' \end{bmatrix} = A \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$。
- 基準フレームの番号は 1 から、推定はフレーム 2 以降に適用。
- 回転角の符号: $+x$ 方向の点が $+y$ 方向へ動く向き(表示座標では時計回り)を正とし、$\theta = \mathrm{atan2}(A_{21}, A_{11})$ に一致します。単位はラジアン(診断 CSV の `d_rot` のみ度)。

コード対応: `AffineCorrector._estimate` で `src=(mx, my)`、`dst=(x, y)` の順序で渡すため、推定される行列は「現フレーム座標 → 基準フレーム座標」の向きです(§6)。

## 2. 正規化相関によるテンプレートマッチング

### 2.1 定義式

実装は `cv2.matchTemplate` の 2 方式を同一ウィンドウ上で計算します(コード対応: `TemplateMatcher._match_window`)。
画像窓を $I$、テンプレートを $T$、窓の画素数を $w \cdot h$ として [OpenCV 4.14 の定義](https://docs.opencv.org/4.14.0/df/dfb/group__imgproc__object.html) は次の通りです。

TM_CCORR_NORMED(位置の採用に使うスコア):

$$
R_{\mathrm{c}}(x, y) = \frac{\sum_{x',y'} \left( T(x', y') \cdot I(x + x', y + y') \right)}{\sqrt{\sum_{x',y'} T(x', y')^2 \cdot \sum_{x',y'} I(x + x', y + y')^2}}
$$

TM_CCOEFF_NORMED(警告用の第 2 スコア。コード対応: `MATCH_CCOEFF_ALARM` との比較のみ):

$$
R_{\mathrm{ce}}(x, y) = \frac{\sum_{x',y'} \left( T'(x', y') \cdot I'(x + x', y + y') \right)}{\sqrt{\sum_{x',y'} T'(x', y')^2 \cdot \sum_{x',y'} I'(x + x', y + y')^2}}
$$

ここで $T'$、$I'$ はそれぞれテンプレート・窓の平均を引いたもの(平均除去)です。つまり $R_{\mathrm{ce}}$ は統計学のピアソン相関係数そのものです。

### 2.2 性質: 不変性と「確率ではない」こと

- **スコアは正答確率ではない。** どちらの値も二つの数値列間の相関量であって、位置が正しい確率を意味しません。確率的な較正(信頼区間への変換)はしていません。閾値 `MATCH_SCORE_MIN = 0.95` は実測分布から選んだ運用上の目安です。
- **TM_CCORR_NORMED の範囲。** 画素値が非負の通常の入力では分子 $\langle T, I \rangle \geq 0$・分母 $> 0$ なので $0 \leq R_{\mathrm{c}} \leq 1$。$R_{\mathrm{c}}$ はベクトル $T$ と $I$ のコサイン類似度です。
- **TM_CCORR_NORMED は窓の明るさ(オフセット)に対して不変でない。** 窓が一定値 $b$ で明るい構造 $I = b \cdot \mathbf{1} + r$ のとき、$b \to \infty$ で $R_{\mathrm{c}} \to \langle T, \mathbf{1} \rangle / (\lVert T \rVert \sqrt{wh}) > 0$、つまり全く関連しない明るい領域でも正値が出ます。一方で窓全体への乗法的変化 $I \mapsto \alpha I \; (\alpha > 0)$ には不変です(分子分母とも $\alpha$ で消える)。テンプレート完全遮蔽時に明るい領域でスコアが約 0.98 のまま残る実測は、このオフセット非不変性に由来します。
- **TM_CCOEFF_NORMED は窓の線形輝度変化 $I \mapsto \alpha I + \beta \; (\alpha > 0)$ に対して不変**(ピアソン相関の性質)。平均除去が $\beta$ を、分母の正規化が $\alpha$ を打ち消します。したがって照明の変化にはある程度頑健で、逆に内容が別物に置き換わった(遮蔽された)窓では急落します。実装はこれを独立の遮蔽アラームに使います(実測: 通常窓最低 0.819〜0.854、遮蔽 0.48、閾値 0.78。出典: [ベンチマーク](../benchmarks/README.md)「TM方式比較」)。位置の採用には使わないため、照明変化下での誤対応(実測差 66px など)を避けられます。

不変性は上記の窓全体の変換についての話です。窓の一部分だけが変わる(部分遮蔽・部分テクスチャ変化)場合、どちらの方式でも分母・分子の加算総数内の異常画素の比率に応じてスコアが変動し、頑健性は一般には保証されません。

### 2.3 ピーク比較と明瞭さ

`minMaxLoc` による全窓最大が採用位置です。実装は計測のためスコア面の 7×7 領域を抑圧した残りの最大値との差を「明瞭さ」(clearness)として記録します(コード対応: `TemplateMatcher._match_window`)。明瞭さは「次善の候補とどれだけ離れているか」の指標で、これも確率ではありません。反復模様・同種背景では本質的に小さくなります(実測と閾値 `MATCH_CLEAR_MIN = 0.0002` はベンチマーク文書参照)。

## 3. 相関面からのサブピクセル推定: 軸別 3 点放物線補間

整数ピーク位置 $(x_0, y_0)$ を中心に、水平・垂直それぞれ独立に 3 点を通る放物線の頂点を取ります(コード対応: `refine` / `_quad`)。

水平方向の 3 点 $(a, b, c) = (R(x_0-1, y_0),\ R(x_0, y_0),\ R(x_0+1, y_0))$ に対し:

$$
\delta_x = \frac{1}{2} \cdot \frac{a - c}{a - 2b + c}
$$

垂直方向も同様に上下の 3 点から $\delta_y$ を求めます。小数点位置は $(x_0 + \delta_x,\ y_0 + \delta_y)$ です。

数値例: $(a, b, c) = (0.90, 1.00, 0.70)$ のとき $a - 2b + c = -0.40$、$\delta_x = \tfrac{1}{2}(0.90 - 0.70) / (-0.40) = -0.25$、つまり頂点は整数ピークから $-0.25$ px の位置です。

性質と限界:

- **交差項を含まない。** 完全な 2 次元二次曲面 $R(x, y) \approx \alpha x^2 + \beta y^2 + \gamma xy + \delta x + \varepsilon y + \zeta$ の近似ではなく、$x$・$y$ 軸それぞれに独立の一次元放物線を当てはめたものです。2 次元ピーク形状の斜め偏りはこの近似の範囲でしか表されません。
- **範囲。** 整数ピークが行・列の厳密な最大値であれば、$a < b$ かつ $c < b$ から $a - 2b + c < 0$(上に凸)であり、$|\delta_x| < 0.5$ が自動的に成り立ちます。
- **平坦面。** $a - 2b + c = 0$(3 点が同値)では補正量 0 とします。
- **端。** ピークが相関面の端の行・列にあるとき 3 点が取れないため補正量 0(整数位置)とします(コード対応: `refine` の境界条件。`roi_edge` フラグで記録)。
- **近似不成立の取り扱い。** 分母が 0 以外でも 3 点が正しい放物線近似を成すとは限りません(ノイズ・隣接ピークの干渉)。この推定は稠密な曲面近似ではなく、真の相関面ピーク位置・真の位置精度を保証しません。合成動画ベンチの正解既知評価では、正答判定の誤差基準にサブピクセル補間固有のずれが影響しないよう、粗大誤差の判定に >8px を用いています(評価基準: [ベンチマーク](../benchmarks/README.md)「シナリオ評価」)。

出典: この補間は OpenCV 固有の API ではなく純数値の演算で、式は二次曲線の頂点公式です。相関面自体は上記 §2 の `matchTemplate` が生成します。

## 4. 相似変換の推定(4 自由度)

### 4.1 推定対象

`cv2.estimateAffinePartial2D`(コード対応: `AffineCorrector._estimate`)が返す 2×3 行列の定義は [OpenCV 4.14 の文書](https://docs.opencv.org/4.14.0/d9/d0c/group__calib3d.html) で次の形に定められています:

$$
A = \begin{bmatrix} \cos\theta \cdot s & -\sin\theta \cdot s & t_x \\ \sin\theta \cdot s & \cos\theta \cdot s & t_y \end{bmatrix}
$$

未知数は回転角 $\theta$、一様スケール $s$、並進 $t_x, t_y$ の 4 自由度です。左 2×2 ブロックは「回転と一様スケールの積」で、せん断・軸別スケール成分を持ちません。$s$ は 2×2 ブロックの SVD 特異値の平均で復元でき、$\theta$ は `atan2(A21, A11)` で復元できます(コード対応: `_motion_metrics` の `d_scale` / `d_rot`)。

### 4.2 最小サンプル 2 点と閉形式解

OpenCV の実装([`modules/calib3d/src/ptsetreg.cpp`](https://github.com/opencv/opencv/blob/4.14.0/modules/calib3d/src/ptsetreg.cpp) の `AffinePartial2DEstimatorCallback`)は

$$
x' = c \cdot X - s \cdot Y + t_1, \qquad y' = s \cdot X + c \cdot Y + t_2
$$

の未知数 $(c, s, t_1, t_2)$ を、2 点ペア $(X_i, Y_i) \to (x_i, y_i) \; (i = 1, 2)$ の 4 元連立一次方程式

$$
\begin{bmatrix} X_1 & -Y_1 & 1 & 0 \\ Y_1 & X_1 & 0 & 1 \\ X_2 & -Y_2 & 1 & 0 \\ Y_2 & X_2 & 0 & 1 \end{bmatrix}
\begin{bmatrix} c \\ s \\ t_1 \\ t_2 \end{bmatrix} =
\begin{bmatrix} x_1 \\ y_1 \\ x_2 \\ y_2 \end{bmatrix}
$$

を閉形式で解きます。分母は $(X_1 - X_2)^2 + (Y_1 - Y_2)^2$ で、これが条件数に相当します。

成立条件:

- **異なる 2 点ペアが最低限。** 2 点が一致すると分母が 0 になり解くことができません。実装は同じ中心座標の重複を入力検証ではじいています(コード対応: `validate_templates` の「異なる中心座標が 2 点以上」)。ごく接近した 2 点では解が数値的に不安定になります(分母の発散)。
- **補助の退化チェック。** OpenCV は一般アフィン(3 点サンプル)で `haveCollinearPoints` による共線チェックを挟みますが、2 点サンプルでは点数 3 を前提とした外積テストが発生せず、このチェックは何も棄却しません。したがって「重なり合う 2 点」のサンプルは RANSAC 側のチェックでは排除されず、閉形式の分母が 0 に向かう数値破綻(非有限値)は、上位実装の防御(`_estimate` の `np.isfinite` チェック)と入力検証(`validate_templates`)で処理します。
- **最小サンプル数と品質閾値は別物。** 「RANSAC が 2 点で変換を一意に決められる」こと(モデルの一意性)と、「品質判定で内点 3 点以上を要求する」こと(決定の冗長性: `EST_INLIER_COUNT_MIN = 3`)は別の概念です。2 点だけでは誤対応があっても検出できません。
- **精緻化(LM refine)。** OpenCV は `refineIters`(既定 10)回のレーヴェンバーグ・マルカート法による非線形最小化を内点だけで実行します(出典: `estimateAffinePartial2D` の引数 `refineIters` と実装)。ただし実装上の適用条件は `count > 2` のときのみで、対応点が 2 点だけのときは閉形式解がそのまま返ります。

### 4.3 数値例: 既知の並進・回転・拡大

$\theta = 0.1$ rad(約 5.7296°)、$s = 1.1$、$t = (3, 2)$ のとき($s\cos\theta = 1.0945$、$s\sin\theta = 0.1098$):

$$
A = \begin{bmatrix} 1.0945 & -0.1098 & 3 \\ 0.1098 & 1.0945 & 2 \end{bmatrix}
$$

- 点 $(100, 100)$ は $x' = 1.0945 \times 100 - 0.1098 \times 100 + 3 = 101.4688$、$y' = 0.1098 \times 100 + 1.0945 \times 100 + 2 = 122.4321$ に写ります。
- 逆復元: $s = \sqrt{1.0945^2 + 0.1098^2} \approx 1.1000$、$\theta = \mathrm{atan2}(0.1098,\ 1.0945) \approx 0.1$ rad。
- 純並進の例: $A = \begin{bmatrix} 1 & 0 & 12 \\ 0 & 1 & -5 \end{bmatrix}$ は全点を $(+12, -5)$ 動かし、このとき $\theta = 0$、$s = 1$ です。

### 4.4 一般アフィンとの違い

`estimateAffine2D`(一般アフィン、6 自由度、OpenCV の最小サンプルは 3 点)はせん断と軸別スケールを含む:

$$
A_{\mathrm{aff}} = \begin{bmatrix} a & b & t_x \\ d & e & t_y \end{bmatrix}
$$

本実装が 4 自由度を使う理由は、対象とする運動(並進・回転・ズーム)がその範囲に収まる想定であることと、少ない対応点数(実入力では 3〜5 点)で 6 自由度を安定に推定するには余裕がないためです。せん断・軸別スケール(レンズ歪みの一部、非正面投影など)を含む動画では、4 自由度は真の運動の近似にしかなりません(§8)。

## 5. RANSAC による頑健推定

### 5.1 内点判定

RANSAC(Fischler & Bolles, 1981)は最小サンプル(2 点ペア)を繰り返し無作為抽出し、各サンプルから閉形式解 $A^{*}$ を求め、最も多くの点を内点とする解を採用する手法です。OpenCV 4.14 の内点判定は再投影誤差

$$
e_i = \left\lVert \begin{bmatrix} x_i \\ y_i \end{bmatrix} - A^{*} \begin{bmatrix} X_i \\ Y_i \\ 1 \end{bmatrix} \right\rVert_2 \leq \text{ransacReprojThreshold}
$$

で行われます。本実装では `ransacReprojThreshold = 2`(px)。それ以下の点が内点、超える点が外点と扱われます。

実装パラメータ(コード対応: `AffineCorrector._estimate`。明示しない引数は OpenCV の既定):

- 最小サンプル点数は相似変換で 2(一般アフィンは 3)。出典: [`ptsetreg.cpp`](https://github.com/opencv/opencv/blob/4.14.0/modules/calib3d/src/ptsetreg.cpp) と [`fundam.cpp`](https://github.com/opencv/opencv/blob/4.14.0/modules/calib3d/src/fundam.cpp) の `estimateAffinePartial2D` 実装(`createRANSACPointSetRegistrator(cb, 2, ...)`)。
- `maxIters = 2000`、`confidence = 0.99`(OpenCV 既定)。
- 採用後に内点のみで LM refine による再最適化(§4.2)。
- 推定された行列が非有限または $\det A^{*}_{2 \times 2} \leq 0$ の場合は推定失敗として扱い、パイプラインは停止します(コード対応: `_estimate` の `np.isfinite` と行列式チェック、`est_ok = false` → 例外)。相似変換の閉形式では $s^2 = \det > 0$ が保たれるはずですが、浮動小数点演算・refine の副作用で非有限や符号崩れが発生した場合を除外する防御です。

### 5.2 反復回数の成功確率の前提

最小サンプル $m$ 点がすべて正しい対応である確率を内点率 $w$($w^m$ が 1 試行の全内点確率)とすると、$N$ 反復で少なくとも 1 回全内点サンプルを得る確率 $p$ は

$$
p = 1 - (1 - w^m)^N \quad \Longleftrightarrow \quad N \geq \frac{\log(1 - p)}{\log(1 - w^m)}
$$

です。$m = 2$、$p = 0.99$ のとき:

| 内点率 $w$ | 必要反復回数 $N$ |
|---|---|
| 0.6 | 11 |
| 0.4 | 27 |
| 0.2 | 113 |

実入力では対応点が 5 点程度なので 2 点サンプルの組み合わせは高々 10 通りであり、`maxIters = 2000` は余裕です。この意味では「内点率が低い場合に反復でどうカバーするか」よりも「少数点配置の中で唯一の全内点サンプルを少なくとも 1 度引き当てられるか」が支配的になります。対応点が 1 点・重複 2 点のような縮退配置では全内点サンプルがそもそも存在せず、RANSAC の確率的主張は適用できません(§4.2)。

### 5.3 限界

- 内点判定はサンプル解 $A^{*}$ に依存するため、「反復を終えた最大内点解が真の変換である」ことは外れ値分布の仮定に依存し、一般には証明できません。反復模様・同種背景では最大内点数の解が正答でないことが数学的に可能であり、実装側の防御は出力を止めずにスコア・明瞭さ・行列変化量などの診断(§7)を記録して後段と運用で判断できるようにすることです。
- 候補が多数同程度に並ぶ条件下では低信頼フラグ(診断 CSV)は警告を出しますが、低信頼でも停止・補間はしません。ROI 拡張再探索(`MATCH_EXPAND_FACTOR`)はトリガ条件で発動し、実測スコア改善 ≥ 0.005(`MATCH_RESEARCH_GAIN`)を満たす場合にのみ採用へ入れ替えます(§2.3・ベンチマーク文書)。

## 6. 変換の方向・同次座標・`H = P · A` と再サンプリング

### 6.1 変換の向きと同次座標

追跡対応は「現フレーム $(mx, my)$ → 基準フレーム $(x, y)$」の向きで推定されます(§1)。したがって推定された $A$ は、現フレーム上の点が基準フレームのどこに位置するかを与えます。
並進を含むため点は同次座標 $\begin{bmatrix} x & y & 1 \end{bmatrix}^{\top}$ で表し、推定後の 2×3 行列を 3×3 に拡張します(コード対応: `AffineCorrector._warp_frames` の `np.vstack([M, [0, 0, 1]])`)。

### 6.2 `H = P · A` と単一再サンプリング

出力は固定の透視変換 $P$(コード対応: `cv2.getPerspectiveTransform`: 四隅 $(0,0), (w-1,0), (w-1,h-1), (0,h-1)$ を $(10,5), (w-11,10), (w-6,h-11), (5,h-6)$ へ写す)と推定された相似変換 $A$ の合成

$$
H = P \cdot A
$$

を `cv2.warpPerspective(frame, H, (w, h))` に渡します(コード対応: `_warp_frames`)。

出力画素値はこの**合成された変換の逆写像で一度に決まります**: [OpenCV 4.14 の `warpPerspective`](https://docs.opencv.org/4.14.0/da/d54/group__imgproc__transform.html) は `WARP_INVERSE_MAP` 指定がない場合、渡された $H$ を内部で反転し、出力画素 $(x, y)$ に対応する入力画素

$$
(x', y') = \left( \frac{H^{-1}_{11}x + H^{-1}_{12}y + H^{-1}_{13}}{H^{-1}_{31}x + H^{-1}_{32}y + H^{-1}_{33}},\; \frac{H^{-1}_{21}x + H^{-1}_{22}y + H^{-1}_{23}}{H^{-1}_{31}x + H^{-1}_{32}y + H^{-1}_{33}} \right)
$$

の値を二次元双線形補間(`INTER_LINEAR`)で参照します(境界外は定数 0 で埋める `BORDER_CONSTANT`)。

**「1 回の再サンプリング」の意味:** 以前の 2 段方式はアフィンで 1 回・透視で 1 回の補間を含んでいました。本実装は $H = P \cdot A$ を 1 つの行列にまとめて渡すため、各フレームの画素は元映像から 1 回だけ補間されます。画素値の差はこれに伴って生じますが、実測(v0.4.0、306 フレーム平均差 ≤ 3.4/255、全体移動なし)は画角・対応の互換を確認するためのもので、全ての入力での上限を保証するものではありません。

### 6.3 フレーム 1 の恒等性

フレーム 1 は基準フレームなので $A = \mathrm{identity}$(推定対象外)で、出力は $H = P$ のみです(コード対応: `_warp_frames` の `np.eye(3)`)。出力全フレーム数は入力と等しく、1 対 1 対応を保ちます。

## 7. 品質診断が測るもの・測らないもの

実装は v0.6.0 以降 2 つの診断 CSV に数値を記録します(コード対応: `TemplateMatcher.run` → matcher 側 `Matched_diagnostics.csv`、`AffineCorrector.write_diagnostics` → 出力動画隣の `Affined_diagnostics.csv`)。各値が測る対象が異なるため混同してはなりません:

| 記録値(列) | 測るもの | 測らないもの |
|---|---|---|
| `inliers` / `inlier_ratio` | RANSAC 内点の件数・比率 | 対応点自体の正否、真の位置の正しさ |
| `residual_max` | 採用内点の再投影誤差 $\lVert \mathrm{dst} - A\,\mathrm{src} \rVert_2$ の最大(コード対応: `_inlier_metrics`) | 追跡点以外(背景画素)の残留誤差 |
| `spread_x` / `spread_y` | 内点座標の軸別標準偏差 | 数値的な退化判定(下記) |
| `d_t` / `d_scale` / `d_rot` | 前回推定フレーム(`delta_frame`)からの行列変化量(コード対応: `_motion_metrics`) | 行列が正しいかどうか(前回も誤っていれば小さくなる) |
| `est_ok` | `estimateAffinePartial2D` 成功 + 非有限/非正の行列式チェック通過 | 品質閾値の合格(`quality_ok` と独立) |
| `quality_ok` | 実測値に基づく品質閾値(`EST_*`)の全条件満足 | 真の位置精度・全画素の品質 |

**軸別標準偏差は退化判定の十分条件ではないことの明示:** `spread_x`・`spread_y` ≥ 10 px(`EST_SPREAD_MIN`)は「内点がそれなりの距離に散っている」ことの目安です。内点を 2 つ取るサンプルの条件数は 2 点間距離の逆数(§4.2)なので、点の散らばりは間接的な防御力の指標であり、十分条件でも必要条件でもありません(座標が散っていても特定の 2 点サンプルが接近し得る)。退化の数値的判定(条件数計算)は実装していないので、配置品質に関する数値の信頼性は閾値の実測根拠(ベンチマーク文書「信頼度評価」)に依ります。

残差・内点率・変化量・背景点誤差(追跡外の背景画素の残留ずれ)は別特性の評価です。追跡点の当てはめ残差が小さくても、不測の背景画素に残る真の補正誤差は必ずしも反映しません。診断合格は真の位置精度・全画素の品質の保証ではありません(手順の説明: [補正処理と限界](algorithm.md)、実測データ: [ベンチマーク](../benchmarks/README.md))。

matrix 収束評価・信頼区間の推定などの統計的な検証は実装していません。推定行列側の実装防御は `_estimate` の `np.isfinite`(`np.all`)と行列式検証です。

## 8. 適用の限界: 数学的限界と実測閾値の区別

下記は数式・モデル構造由来の限界です:

- **非剛体運動:** 4 自由度はせん断・非一様スケールを含めません(§4.4)。非剛体な変形には 1 つの相似変換で近似しかできません。
- **視差:** 深度の異なるシーン(実際の 3D 運動)には、単一平面の相似変換は近似にしかなりません。
- **遮蔽・部分テクスチャ変化:** TM_CCORR_NORMED は窓の明るさ(オフセット)に対して不変でない(§2.2)。全体の正規化は部分遮蔽のある窓の頑健性を一般に保証しません。
- **反復模様・同種背景:** 相関面の多峰に起因する誤対応は、明瞭さ指標では捉えられるが、閾値は実測に基づき数学的な保証はありません。
- **探索外移動:** ROI 内にテンプレート位置がなければマッチ不能。拡大再探索(`MATCH_EXPAND_FACTOR = 2.0`)は実測スコア改善 ≥ 0.005(`MATCH_RESEARCH_GAIN`)を満たす場合にのみ採用する制約を持ち(§2.3・ベンチマーク文書)、それが満たされなければ元の位置が維持されます。
- **黒パディング・低模様域:** 相関スコアは画素差のみで定義されるため、情報の乏しい領域では位置識別が構造的に弱い。

実測値と閾値と数学定理を区別してください。例えば:

- `MATCH_SCORE_MIN`・`MATCH_CCOEFF_ALARM`・`EST_*` 等の閾値は実入力・合成シナリオの分布から選ばれた運用上の目安で、理論から導かれる最低保証(信頼区間)を示すものではありません。
- 合格・不合格・低信頼フラグはフレーム単位の運用上の診断であり、いずれも推定行列の正しさを証明するものではありません。

## 9. 出典

- OpenCV 4.14.0(依存: `opencv-python-headless 4.14.0.94`)
  - [cv::TemplateMatchModes / matchTemplate](https://docs.opencv.org/4.14.0/df/dfb/group__imgproc__object.html) ・ソース [`modules/imgproc/include/opencv2/imgproc.hpp`](https://github.com/opencv/opencv/blob/4.14.0/modules/imgproc/include/opencv2/imgproc.hpp)
  - [calib3d(estimateAffinePartial2D を含む)](https://docs.opencv.org/4.14.0/d9/d0c/group__calib3d.html) ・ソース [`modules/calib3d/include/opencv2/calib3d.hpp`](https://github.com/opencv/opencv/blob/4.14.0/modules/calib3d/include/opencv2/calib3d.hpp)
  - [imgproc transform(warpPerspective を含む)](https://docs.opencv.org/4.14.0/da/d54/group__imgproc__transform.html)
  - [`modules/calib3d/src/ptsetreg.cpp`](https://github.com/opencv/opencv/blob/4.14.0/modules/calib3d/src/ptsetreg.cpp)(相似変換の 2 点閉形式解・最小サンプル点数)
  - [`modules/calib3d/src/fundam.cpp`](https://github.com/opencv/opencv/blob/4.14.0/modules/calib3d/src/fundam.cpp)(`estimateAffinePartial2D` の RANSAC/LMeDS 分岐・refine 適用条件)
- M. A. Fischler and R. C. Bolles, "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography", Communications of the ACM 24(6), 1981. [DOI 10.1145/358669.358692](https://doi.org/10.1145/358669.358692)(RANSAC の定義、内点率 $w$ と反復回数・成功確率の関係)
- R. Szeliski, "Computer Vision: Algorithms and Applications"([書籍サイト](https://szeliski.org/Book/))(2D 変換の分類・特徴点整列・RANSAC の体系的な解説の参考)
- 本実装の実測値・閾値・検証条件: [ベンチマーク](../benchmarks/README.md)「信頼度評価」、[コードレビュー](code-review.md)、[探索方式評価](search-evaluation.md)、[整列方式評価](registration-evaluation.md)
- 手順の概要と限界の説明: [補正処理と限界](algorithm.md)、[操作と入出力](usage.md)
