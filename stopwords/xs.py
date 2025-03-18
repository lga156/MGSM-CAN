from gensim.models import KeyedVectors  # 导入gensim模块下的KeyedVectors类，用于导入数据
wv = KeyedVectors.load_word2vec_format('tencent-ailab-embedding-zh.txt', binary=False)  # 导入词向量数据
wv.most_similar("小孩", topn=3)  # 获取数据中和“小孩”最相近的前三个词语