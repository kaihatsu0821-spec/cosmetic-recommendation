# Foundation Recommendation App

# Foundation Recommendation App

顔画像から肌色分析を行い、自然な仕上がりを重視したファンデーションを推薦するアプリです。

## Overview

メンズメイク初心者にとって、

- 自分に合うファンデーションの色が分からない
- 白浮きが不安

という課題があることに着目し開発しました。

顔画像から肌色を分析し、Lab色空間を用いて自然に見えるファンデーションを推薦します。

## Features

### 1. White Balance Correction

撮影環境による色ブレを抑えるため、画像内の白い紙領域をユーザーに指定してもらい、ホワイトバランス補正を実施します。

- 白領域の明るい画素を抽出
- median値を基準にRGBスケーリング

### 2. Skin Tone Extraction

MediaPipe FaceMeshを用いて頬領域を抽出し、肌色分析を行います。

工夫:
- 暗い画素除去
- 赤みの強い画素除去
- medianによるロバストな代表色推定

### 3. Beard Analysis

青髭領域を抽出し、肌色との差分をLab色空間上で解析します。

- beard mask生成
- z-scoreによる青み検出

### 4. Foundation Recommendation

Lab色空間上で候補ファンデーションとの距離を計算し推薦します。

評価関数:

d = sqrt(1.5 * dL² + da² + db²)

明度差(L値)を強めに評価することで、白浮きを抑えた推薦を実現しました。

### 5. Virtual Try-On

推薦色や任意色を顔領域へ適用し、疑似的に仕上がり確認できます。

## Tech Stack

- Python
- OpenCV
- MediaPipe
- PyTorch
- Streamlit

## Challenges

### 撮影環境による色ブレ

同一人物でも照明条件で肌色推定が大きく変化する問題がありました。

→ 白い紙を基準にしたホワイトバランス補正を導入。

### 白浮き判定

単純な色距離では自然さを表現できませんでした。

→ 明度(L値)を重視した独自距離関数を設計。

## Future Work

- Foundation DB拡張
- ブランド別推薦
- モバイル対応
- 自動白領域検出
- 青髭部分フィルター
- 「似合う色を探す」のページでの推薦