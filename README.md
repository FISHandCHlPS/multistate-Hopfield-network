# 単純モデル
あるエネルギー関数を定義したとき、エネルギーを最小にする状態を見つける  
様々な初期値から始める  
ただし、各粒子は相互作用する  

## 可視化方法
~~縦軸にエネルギー、横軸にxを取り軌跡を表示~~  
二次元空間上で軌跡を表示

## 相互作用
各粒子同士が近いほど反発力が生じる  
粒子 $x$ , 距離 $d(x_i, x_j) = |x|_2$ 
```math
F = \frac{1}{ N d(x_i, x_j) }
```
方向 × 平均反発力  
全ての粒子に対して行われるため計算量は $O(n^2)$ 

## 更新則

エネルギー関数
```math
lse(\beta, x) = \ln \sum \exp( \beta x )
```
```math
E(x) = -lse(\beta, W^T x) + \frac{1}{2} x^T x + C
```
更新則
```math
x_i \leftarrow x_i + \alpha \nabla E + \frac{1}{N} \sum_{j \neq i} \frac{ (x_i - x_j) }{ \|x_i - x_j\|^c }
```
$\alpha=1$のときCCCPと等価
