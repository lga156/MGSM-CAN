import sys
import jieba
import hanlp
import random

# 加载预训练的词向量模型
word2vec = hanlp.load(hanlp.pretrained.word2vec.MERGE_SGNS_BIGRAM_CHAR_300_ZH)

# # 加载 HanLP 分词器
# tokenizer = hanlp.load(hanlp.pretrained.tok.SIGHAN2005_PKU_BERT_BASE_ZH)

def find_synonyms(word, topk=5):
    # 寻找近义词
    try:
        synonyms = word2vec.most_similar(word, topk=topk)
        print(synonyms.keys())
        return list(synonyms.keys())  # 返回近义词列表
    except KeyError:
        # 如果词不在词典中，则返回空列表
        return []

def synonym_replacement(sentence, n):
    words = list(jieba.cut(sentence, cut_all=False))
    new_words = words.copy()
    random.shuffle(words)
    num_replaced = 0

    for word in words:
        synonyms = find_synonyms(word)
        if num_replaced >= n or not synonyms:
            continue

        synonym = random.choice(synonyms)
        new_words = [synonym if w == word else w for w in new_words]
        num_replaced += 1

    return ''.join(new_words)

# 示例
sentence = "我喜欢吃苹果和菠萝和香蕉"
print(synonym_replacement(sentence, 2))
sys.exit()