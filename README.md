# SGD
```
g++ -o SGD SGD.cpp
./SGD
```
SGD-ICPを実装してみました。

# SGD-ICPとは
確率的勾配降下法(SGD)を使用し、ICPアルゴリズムの処理を行うことです。
通常のICPは点群すべてを考慮し、最適化計算を反復手に行いますが、  
SGD-ICPはランダムに点群からいくつかとり、最適化計算を行います。
詳しい詳細はこちら→https://arxiv.org/pdf/1907.09133

# SGD-ICPの仕組みの説明
## 1.ミニバッチサイズの決定(Mini-batch size)
ミニバッチサイズとは点群探索を行う点の数のことです。
この数は任意です。(論文では1,10などが代表に挙げられています。)
```cpp
size_t m_batch_size = 22;
```
## 2. ミニバッチの作成(Mini-batch)
設定したミニバッチサイズを基に点群探索を行うミニバッチを作成します。
```cpp
void shuffle_data(std::vector<Point>& points) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(points.begin(), points.end(), g);
}

std::vector<Point> createBatch(std::vector<Point>& source, size_t& m_current_offset, size_t m_batch_size) {
    std::vector<Point> batch;
    auto target_offset = m_current_offset + m_batch_size;

    while (target_offset >= source.size()) {
        while (m_current_offset < source.size()) {
            batch.push_back(source[m_current_offset++]);
        }
        shuffle_data(source);
        m_current_offset = 0;
        target_offset = target_offset - source.size();
    }
    while (m_current_offset < target_offset) {
        batch.push_back(source[m_current_offset++]);
    }

    return batch;
}
```



