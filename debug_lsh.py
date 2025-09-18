#!/usr/bin/env python3
"""
调试LSH实现
"""
import json
from pathlib import Path
from atomic_v2.sketch_lsh import LSHIndex, char_jaccard

# 加载测试数据
DATA = json.loads(Path("tests/data/musique_sample.json").read_text(encoding="utf-8"))
PARAS = {p["idx"]: p for p in DATA["paragraphs"]}

def _text(idx): return PARAS[idx]["paragraph_text"]
def _title(idx): return PARAS[idx]["title"]

# 查看文档5、10、0的内容
print("文档5:")
print(f"标题: {_title(5)}")
print(f"内容: {_text(5)}")
print()

print("文档10:")
print(f"标题: {_title(10)}")
print(f"内容: {_text(10)}")
print()

print("文档0:")
print(f"标题: {_title(0)}")
print(f"内容: {_text(0)}")
print()

# 分析为什么文档0和10的相似度更高
text5 = _text(5)
text10 = _text(10)
text0 = _text(0)

# 使用测试中的方式计算相似度
j_10_5 = char_jaccard(text10, text5)
j_10_0 = char_jaccard(text10, text0)

print(f"测试中的相似度计算:")
print(f"j_10_5 (文档10和5): {j_10_5}")
print(f"j_10_0 (文档10和0): {j_10_0}")
print(f"j_10_5 >= j_10_0: {j_10_5 >= j_10_0}")
print()

# 分析n-gram重叠，使用更大的范围
def extract_char_ngrams(text: str, n_gram_range: tuple = (3, 10)):
    ngrams = set()
    text = text.lower()
    for n in range(n_gram_range[0], n_gram_range[1] + 1):
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i+n])
    return ngrams

ngrams_10 = extract_char_ngrams(text10)
ngrams_5 = extract_char_ngrams(text5)
ngrams_0 = extract_char_ngrams(text0)

# 分析Steve Hillage相关的n-gram
steve_ngrams_10 = {ng for ng in ngrams_10 if 'steve' in ng or 'hillage' in ng}
steve_ngrams_5 = {ng for ng in ngrams_5 if 'steve' in ng or 'hillage' in ng}
steve_common = steve_ngrams_10.intersection(steve_ngrams_5)

print(f"文档10中包含'steve'或'hillage'的n-gram: {sorted(steve_ngrams_10)}")
print(f"文档5中包含'steve'或'hillage'的n-gram: {sorted(steve_ngrams_5)}")
print(f"共同的包含'steve'或'hillage'的n-gram: {sorted(steve_common)}")
print(f"Steve Hillage相关共同n-gram数量: {len(steve_common)}")
print()

# 分析Green相关的n-gram
green_ngrams_10 = {ng for ng in ngrams_10 if 'green' in ng}
green_ngrams_0 = {ng for ng in ngrams_0 if 'green' in ng}
green_common = green_ngrams_10.intersection(green_ngrams_0)

print(f"文档10中包含'green'的n-gram: {sorted(green_ngrams_10)}")
print(f"文档0中包含'green'的n-gram: {sorted(green_ngrams_0)}")
print(f"共同的包含'green'的n-gram: {sorted(green_common)}")
print(f"Green相关共同n-gram数量: {len(green_common)}")
print()

print("结论:")
print(f"Green相关共同n-gram数量: {len(green_common)}")
print(f"Steve Hillage相关共同n-gram数量: {len(steve_common)}")
print("现在应该能看到更多Steve Hillage相关的n-gram")