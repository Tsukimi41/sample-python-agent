# Sample AIWolf Python agent using aiwolf package
Sample AIWolf agent written in Python using [aiwolf package](https://github.com/AIWolfSharp/aiwolf-python).

## Explainable role estimation

5人人狼（村人2・占い師1・狂人1・人狼1）向けに、説明可能な可能世界推定器を実装しています。
`SampleVillager` と `SampleSeer` の投票を、全生存候補の人狼確率の比較で決定します。
同点はゲーム開始時のエージェント順で解消します。

- 他者の発言はソフト証拠、自己占いと確認された襲撃死はハード制約として扱います。
- 自己視点と公開視点を別々に再計算し、公開説明の数値や仮説順序にも私有結果を混ぜません。
- 各イベントの世界別尤度、信念履歴、推論グラフ、証拠除外時の再計算を参照できます。
- 構造化反論に対して、争点、代替仮説、感度分析を返します。
- SDK不要のJSON再生・評価器と、SDKを使う接続テストを用意しています。

仕様・数式・ファイルの役割・設計判断は [実装詳細](docs/implementation-completion.md) を参照してください。
資料は [PowerPoint](output/werewolf-ai.pptx) / [PDF](output/werewolf-ai.pdf) です。
5枚、約3分の原稿と出典をPowerPointの発表者ノートに収録しています。

## Prerequisites
* Python 3.10以上（検証環境: Python 3.11）
* 対戦接続時のみ [aiwolf package](https://github.com/AIWolfSharp/aiwolf-python)
You can install aiwolf package as follows,
```
pip install git+https://github.com/AIWolfSharp/aiwolf-python.git
```
## How to use
Suppose the AIWolf server at localhost is waiting a connection from an agent on port 10000.
You can connect this sample agent to the server as follows,
```
python start.py -h localhost -p 10000 -n name_you_like
```

## Tests

```console
python -m unittest discover -s tests -v
```

## サーバ不要の再現デモ

```console
python belief_evaluation.py examples/five_player.json --output output/evaluation.json
```

合成局面に対して、ハード制約のみ／ソフト証拠ありの両条件を比較します。
`role_accuracy`、自然対数の`log_loss`、多クラス`brier_score`、10区間の`ece`、
役職人数誤差、投票的中、推論グラフ、全時点のスナップショットをJSONで出力します。
正解役職表`truth`は評価専用で、推論には入力しません。

これは勝率実験ではありません。未学習の尤度、単一の合成局面から実戦性能を主張しません。
自由文説得・15人近似推論・一貫した嘘生成は、設計文書どおり次段階です。
狂人・人狼・15人村の既存サンプル方策には乱数が残っています。

## 発表資料の再生成

WindowsでMicrosoft PowerPointが利用できる場合:

```powershell
./presentation/build.ps1
```

編集可能な本文・表・発表者ノートを含むPPTXとPDFを`output/`へ出力します。
