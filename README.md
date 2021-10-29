# bert-classfication-basemodel

## positive/negativeの２クラス分類を行う

- dataset
  - chABSAデータセット

## 「観点で分類を行う」

- class
  - company, business, product × sales, profit, amount, price, cost (15クラス分類)

## データセットについて

- 8:2で分割

## 機械学習手法

- BERT
  - 事前学習済みモデルとして日本語wikipediaを基にしたモデルを使用
- 学習率, doropout,　最適化関数等ハイパーパラメータ決定にoptunaを使用
