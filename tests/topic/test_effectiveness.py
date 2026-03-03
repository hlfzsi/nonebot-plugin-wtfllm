"""话题聚类效果测试

验证聚类不是随机分配，而是能有效按语义分离话题。
"""

import asyncio

import pytest

from nonebot_plugin_wtfllm.topic.manager import TopicManager


class TestTopicSeparationEffectiveness:
    """验证聚类能有效区分不同话题"""

    @pytest.fixture
    def manager(self):
        return TopicManager(
            maxsize=10,
            cluster_threshold=0.65,
            max_clusters=20,
            decay_seconds=7200,
        )

    @pytest.mark.asyncio
    async def test_food_vs_tech_separation(self, manager: TopicManager):
        """美食话题和技术话题应被分到不同簇"""
        food_msgs = [
            "今天中午吃了一碗热腾腾的牛肉面",
            "昨天晚餐做了红烧排骨特别好吃",
            "周末打算去那家新开的火锅店尝尝",
            "早餐吃了豆浆油条配小笼包",
            "推荐一下那个麻辣烫确实不错",
        ]
        tech_msgs = [
            "Python的异步编程使用asyncio库",
            "JavaScript的Promise和async关键字",
            "Linux服务器配置nginx反向代理",
            "Docker容器化部署微服务架构",
            "Git版本控制分支管理策略",
        ]

        food_labels = set()
        tech_labels = set()

        for i, msg in enumerate(food_msgs):
            label = await manager.ingest("a1", "g1", None, f"food_{i}", msg)
            food_labels.add(label)

        for i, msg in enumerate(tech_msgs):
            label = await manager.ingest("a1", "g1", None, f"tech_{i}", msg)
            tech_labels.add(label)

        assert food_labels != tech_labels, (
            f"美食和技术话题被分到完全相同的簇: food={food_labels}, tech={tech_labels}"
        )

    @pytest.mark.asyncio
    async def test_interleaved_topics_separation(self, manager: TopicManager):
        """分批交替发送的不同话题消息应被正确分离"""
        messages = [
            ("今天做了一道红烧肉特别入味", "food"),
            ("那个糖醋鱼的做法学会了吗", "food"),
            ("晚饭去吃日本拉面怎么样", "food"),
            ("Python的装饰器模式很实用", "tech"),
            ("React和Vue框架的性能对比", "tech"),
            ("Kubernetes集群管理配置", "tech"),
            ("周末打算去那家新开的火锅店", "food"),
            ("早餐吃了豆浆油条配小笼包", "food"),
            ("Docker容器化部署微服务架构", "tech"),
            ("Linux服务器配置nginx代理", "tech"),
        ]

        topic_labels: dict[str, set[int]] = {"food": set(), "tech": set()}
        for i, (msg, category) in enumerate(messages):
            label = await manager.ingest("a1", "g1", None, f"msg_{i}", msg)
            topic_labels[category].add(label)

        overlap = topic_labels["food"] & topic_labels["tech"]
        total = topic_labels["food"] | topic_labels["tech"]
        overlap_ratio = len(overlap) / max(len(total), 1)
        assert overlap_ratio < 1.0, (
            f"交替话题完全重叠: food={topic_labels['food']}, tech={topic_labels['tech']}"
        )

    @pytest.mark.asyncio
    async def test_same_topic_consistency(self, manager: TopicManager):
        """同一话题的连续消息应有较高的一致性"""
        msgs = [
            "今天去超市买了很多新鲜蔬菜水果",
            "买了西红柿黄瓜还有新鲜的草莓",
            "回来做了一个蔬菜沙拉配水果拼盘",
            "用买的蔬菜炒了一盘青椒肉丝",
            "那些水果蔬菜看起来非常新鲜",
        ]

        labels = []
        for i, msg in enumerate(msgs):
            label = await manager.ingest("a1", "g1", None, f"msg_{i}", msg)
            labels.append(label)

        from collections import Counter
        label_counts = Counter(labels)
        most_common_count = label_counts.most_common(1)[0][1]
        consistency_ratio = most_common_count / len(labels)
        assert consistency_ratio >= 0.4, (
            f"同话题一致性过低: {consistency_ratio:.2f}, labels={labels}"
        )


class TestQueryTopicConcurrency:
    """验证 query_topic 在模拟高并发场景下能正确定位到对应簇"""

    @pytest.mark.asyncio
    async def test_query_after_batch_ingest(self):
        """分批 ingest 不同话题后，query_topic 能正确匹配各自簇"""
        manager = TopicManager(
            maxsize=10, cluster_threshold=0.65, max_clusters=20,
        )

        for i, msg in enumerate([
            "今天中午吃了一碗热腾腾的牛肉面",
            "昨天晚餐做了红烧排骨特别好吃",
            "周末打算去那家新开的火锅店尝尝",
            "早餐吃了豆浆油条配小笼包",
            "推荐一下那个麻辣烫确实不错",
        ]):
            await manager.ingest("a1", "g1", None, f"food_{i}", msg)

        for i, msg in enumerate([
            "Python的异步编程使用asyncio库",
            "JavaScript的Promise和async关键字",
            "Linux服务器配置nginx反向代理",
            "Docker容器化部署微服务架构",
            "Git版本控制分支管理策略",
        ]):
            await manager.ingest("a1", "g1", None, f"tech_{i}", msg)

        food_query = "今天想去吃火锅配麻辣烫"
        await manager.ingest("a1", "g1", None, "food_query", food_query)
        food_label, food_ids = await manager.query_topic("a1", "g1", None, food_query)

        tech_query = "Python的asyncio异步编程框架"
        await manager.ingest("a1", "g1", None, "tech_query", tech_query)
        tech_label, tech_ids = await manager.query_topic("a1", "g1", None, tech_query)

        assert food_label != tech_label, (
            f"美食和技术查询被分到同一簇: food={food_label}, tech={tech_label}"
        )

    @pytest.mark.asyncio
    async def test_query_matches_earlier_topic_not_last(self):
        """最后 ingest 的是技术话题，但 query 美食时应返回美食簇"""
        manager = TopicManager(
            maxsize=10, cluster_threshold=0.65, max_clusters=20,
        )

        for i, msg in enumerate([
            "今天吃了红烧肉非常好吃",
            "晚餐吃了清蒸鱼很新鲜",
            "午饭吃了麻辣火锅过瘾",
            "早餐吃了小笼包配豆浆",
            "宵夜吃了烧烤串串真香",
        ]):
            await manager.ingest("a1", "g1", None, f"food_{i}", msg)

        for i, msg in enumerate([
            "Python编程语言很强大",
            "JavaScript框架更新了",
            "Linux系统管理配置",
            "Docker容器化部署好用",
            "Git版本控制必不可少",
        ]):
            await manager.ingest("a1", "g1", None, f"tech_{i}", msg)

        food_query = "今天想吃红烧排骨配小笼包"
        await manager.ingest("a1", "g1", None, "food_query", food_query)
        label, ids = await manager.query_topic("a1", "g1", None, food_query)
        food_ids_found = [mid for mid in ids if mid.startswith("food_")]
        assert len(food_ids_found) > 0, f"美食 query 未找到美食消息, ids={ids}"

    @pytest.mark.asyncio
    async def test_cross_session_query_isolation(self):
        """不同 session 的 query 互不干扰"""
        manager = TopicManager(maxsize=10, cluster_threshold=0.8)

        for i, msg in enumerate([
            "今天吃了红烧肉好吃", "晚餐做了糖醋排骨", "午餐吃了麻辣火锅",
        ]):
            await manager.ingest("a1", "g1", None, f"g1_msg_{i}", msg)

        for i, msg in enumerate([
            "Python异步编程好用", "Docker容器化部署", "Linux服务器配置",
        ]):
            await manager.ingest("a1", "g2", None, f"g2_msg_{i}", msg)

        _, g1_ids = await manager.query_topic("a1", "g1", None, "今天吃了什么")
        _, g2_ids = await manager.query_topic("a1", "g2", None, "Python编程")

        g1_all_local = all(mid.startswith("g1_") for mid in g1_ids)
        g2_all_local = all(mid.startswith("g2_") for mid in g2_ids)
        assert g1_all_local, f"g1 查询返回了 g2 的消息: {g1_ids}"
        assert g2_all_local, f"g2 查询返回了 g1 的消息: {g2_ids}"


class TestAsyncConcurrency:
    """异步并发 ingest + query_topic 竞态测试"""

    @pytest.mark.asyncio
    async def test_concurrent_ingest_does_not_corrupt_query(self):
        manager = TopicManager(
            maxsize=10, cluster_threshold=0.65, max_clusters=20,
        )

        for i, msg in enumerate([
            "今天吃了红烧肉非常好吃", "晚餐吃了清蒸鱼很新鲜",
            "午饭吃了麻辣火锅过瘾", "早餐吃了小笼包配豆浆", "宵夜吃了烧烤串串真香",
        ]):
            await manager.ingest("a1", "g1", None, f"food_init_{i}", msg)

        for i, msg in enumerate([
            "Python异步编程使用asyncio", "JavaScript框架React更新",
            "Linux服务器配置nginx代理", "Docker容器化部署微服务", "Git版本控制分支管理策略",
        ]):
            await manager.ingest("a1", "g1", None, f"tech_init_{i}", msg)

        errors: list[str] = []

        async def ingest_food():
            for i in range(20):
                await manager.ingest("a1", "g1", None, f"food_concurrent_{i}",
                                     f"并发美食消息第{i}碗面条好吃")

        async def ingest_tech():
            for i in range(20):
                await manager.ingest("a1", "g1", None, f"tech_concurrent_{i}",
                                     f"并发技术消息第{i}个Python模块")

        async def query_food():
            for _ in range(20):
                label, ids = await manager.query_topic("a1", "g1", None, "今天吃了什么面条")
                if label < 0:
                    continue
                for mid in ids:
                    if not mid.startswith(("food_", "tech_")):
                        errors.append(f"非法 message_id: {mid}")

        async def query_tech():
            for _ in range(20):
                label, ids = await manager.query_topic("a1", "g1", None, "Python编程开发框架")
                if label < 0:
                    continue
                for mid in ids:
                    if not mid.startswith(("food_", "tech_")):
                        errors.append(f"非法 message_id: {mid}")

        await asyncio.gather(
            ingest_food(), ingest_tech(), query_food(), query_tech(),
        )

        assert not errors, f"并发测试发现错误: {errors}"

    @pytest.mark.asyncio
    async def test_concurrent_ingest_across_sessions(self):
        manager = TopicManager(maxsize=10, cluster_threshold=0.8)
        errors: list[str] = []

        async def ingest_and_query_session(group_id: str, prefix: str, topic_text: str):
            for i in range(10):
                await manager.ingest("a1", group_id, None, f"{prefix}_{i}",
                                     f"{topic_text}第{i}条消息")
            for _ in range(10):
                label, ids = await manager.query_topic("a1", group_id, None, topic_text)
                for mid in ids:
                    if not mid.startswith(prefix):
                        errors.append(f"session {group_id} 返回了其他 session 的消息: {mid}")

        await asyncio.gather(
            ingest_and_query_session("g1", "g1", "今天吃了红烧肉好吃"),
            ingest_and_query_session("g2", "g2", "Python编程语言好用"),
            ingest_and_query_session("g3", "g3", "今天天气真好适合散步"),
        )

        assert not errors, f"跨 session 并发测试发现错误: {errors}"
