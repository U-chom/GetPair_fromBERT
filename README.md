# GetPair_fromBERT

これは、SVMによりPとNの分類が行われた際に、分類に最も寄与した素性上位からBERTの分散表現を使用してP,Nの単語のコサイン類似度を求め、その度数が高いペアを抽出します。

BERTのモデルは京大のモデルをご使用ください。http://nlp.ist.i.kyoto-u.ac.jp/index.php?BERT%E6%97%A5%E6%9C%AC%E8%AA%9EPretrained%E3%83%A2%E3%83%87%E3%83%AB
