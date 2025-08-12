# 複数想起可能なモダンホップフィールドネットワーク

## 理論
### Modern Hopfield Network

エネルギー関数
```math
lse(\beta, x) = \beta^{-1} \ln \sum \exp( \beta x )
```
```math
E(x) = -lse(\beta, W^T x) + \frac{1}{2} x^T x + C
```
更新則
```math
x_i \leftarrow x_i + \nabla E = W softmax( \beta W^T x )
```
### 提案モデル
各粒子同士が近いほど反発力が生じる  
粒子 $x$ , 距離 $d(x_i, x_j) = |x|_2$ 
```math
F = \frac{1}{ N d(x_i, x_j) }
```
方向 × 平均反発力  
全ての粒子に対して行われるため計算量は $O(n^2)$ 
##### 更新則
```math
x_i \leftarrow x_i + \alpha \nabla E + 相互作用
```
```math
相互作用 = \frac{\gamma}{N} \sum_{j \neq i} \frac{ (x_i - x_j) }{ \|x_i - x_j\|^c }
```


## ディレクトリ構成
### コード
multistate_hnn: メイン. 多粒子MHNの実装.
toy: 2次元モデル
modern_hopfield_network: モダンホップフィールドネットワーク

### ディレクトリ
calc: 計算用のコード
plot: 可視化用のコード
config: 設定ファイル
resources: データ＆ローダー


# 以下詳細を書く
## 単純モデル
低次元空間での挙動を調べる
### 方法
エネルギー関数を定義したとき、エネルギーを最小にする状態を見つける。様々な初期値から始める。ただし、各粒子は相互作用する。  

### 可視化方法
二次元空間上で軌跡を表示