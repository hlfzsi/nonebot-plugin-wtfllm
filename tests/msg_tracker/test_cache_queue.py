"""msg_tracker/cache_queue.py 单元测试"""

import time

import pytest

from nonebot_plugin_wtfllm.msg_tracker.cache_queue import TTLDeque


class TestTTLDequeBasic:
    def test_append_and_get_all(self):
        dq = TTLDeque(ttl=60)
        dq.append("a")
        dq.append("b")
        assert dq.get_all() == ["a", "b"]

    def test_len(self):
        dq = TTLDeque(ttl=60)
        dq.append("x")
        dq.append("y")
        assert len(dq) == 2

    def test_empty(self):
        dq = TTLDeque(ttl=60)
        assert dq.get_all() == []
        assert len(dq) == 0


class TestTTLDequeExpiry:
    def test_expired_items_removed(self, monkeypatch):
        current_time = 1000.0
        monkeypatch.setattr(time, "time", lambda: current_time)

        dq = TTLDeque(ttl=10)
        dq.append("old")

        current_time = 1020.0
        monkeypatch.setattr(time, "time", lambda: current_time)

        dq.append("new")
        assert dq.get_all() == ["new"]

    def test_set_ttl_triggers_cleanup(self, monkeypatch):
        current_time = 1000.0
        monkeypatch.setattr(time, "time", lambda: current_time)

        dq = TTLDeque(ttl=100)
        dq.append("item1")
        dq.append("item2")

        current_time = 1005.0
        monkeypatch.setattr(time, "time", lambda: current_time)

        dq.set_ttl(2)
        assert dq.get_all() == []

    def test_non_expired_items_kept(self, monkeypatch):
        current_time = 1000.0
        monkeypatch.setattr(time, "time", lambda: current_time)

        dq = TTLDeque(ttl=10)
        dq.append("kept")

        current_time = 1005.0
        monkeypatch.setattr(time, "time", lambda: current_time)
        assert dq.get_all() == ["kept"]


class TestTTLDequeMaxlen:
    def test_maxlen_respected(self):
        dq = TTLDeque(maxlen=2, ttl=60)
        dq.append("a")
        dq.append("b")
        dq.append("c")
        assert dq.get_all() == ["b", "c"]


class TestTTLDequeRepr:
    def test_repr(self):
        dq = TTLDeque(maxlen=5, ttl=30)
        r = repr(dq)
        assert "TTLDeque" in r
        assert "30" in r
