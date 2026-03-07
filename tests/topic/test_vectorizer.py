"""话题向量化器单元测试"""

import numpy as np

from nonebot_plugin_wtfllm.vec import TopicVectorizer


class TestTopicVectorizer:
    """model2vec 语义向量化测试"""

    def test_chinese_text_produces_dense_output(self):
        vectorizer = TopicVectorizer()
        result = vectorizer.transform("今天天气真好")
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1
        assert result.shape[1] > 0

    def test_english_text_produces_dense_output(self):
        vectorizer = TopicVectorizer()
        result = vectorizer.transform("hello world")
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1

    def test_output_is_l2_normalized(self):
        vectorizer = TopicVectorizer()
        result = vectorizer.transform("归一化测试文本")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_consistent_results(self):
        vectorizer = TopicVectorizer()
        a = vectorizer.transform("测试一致性")
        b = vectorizer.transform("测试一致性")
        assert np.allclose(a, b)

    def test_batch_transform(self):
        vectorizer = TopicVectorizer()
        texts = ["第一条消息", "第二条消息", "第三条消息"]
        result = vectorizer.transform_batch(texts)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 3

    def test_similar_texts_closer_than_dissimilar(self):
        """语义相近的文本应距离更近"""
        vectorizer = TopicVectorizer()
        v1 = vectorizer.transform("今天吃了红烧肉很好吃")
        v2 = vectorizer.transform("今天吃了清蒸鱼很不错")
        v3 = vectorizer.transform("Python编程语言框架设计")

        dist_similar = np.linalg.norm(v1 - v2)
        dist_different = np.linalg.norm(v1 - v3)
        assert dist_similar < dist_different
