"""msg_tracker/tracker.py 单元测试"""

import time

import pytest

from nonebot_plugin_wtfllm.msg_tracker.tracker import MsgTracker


class TestMsgTrackerStaticMethods:
    def test_get_main_key(self):
        assert MsgTracker._get_main_key("agent1", "user1") == "agent1:user1"

    def test_gat_second_key_with_group(self):
        assert MsgTracker._gat_second_key("group1") == "group1"

    def test_gat_second_key_none(self):
        assert MsgTracker._gat_second_key(None) == ""


class TestMsgTrackerTrackAndGet:
    def test_track_single_message(self):
        tracker = MsgTracker(ttl=60)
        tracker.track("a1", "u1", "g1", "msg1")
        result = tracker.get("u1", "a1")
        assert "g1" in result
        assert result["g1"] == ["msg1"]

    def test_track_multiple_messages(self):
        tracker = MsgTracker(ttl=60)
        tracker.track("a1", "u1", "g1", "msg1")
        tracker.track("a1", "u1", "g1", "msg2")
        result = tracker.get("u1", "a1")
        assert result["g1"] == ["msg1", "msg2"]

    def test_track_private_message(self):
        tracker = MsgTracker(ttl=60)
        tracker.track("a1", "u1", None, "pm1")
        result = tracker.get("u1", "a1")
        assert "" in result
        assert result[""] == ["pm1"]

    def test_track_different_groups(self):
        tracker = MsgTracker(ttl=60)
        tracker.track("a1", "u1", "g1", "msg_g1")
        tracker.track("a1", "u1", "g2", "msg_g2")
        result = tracker.get("u1", "a1")
        assert result["g1"] == ["msg_g1"]
        assert result["g2"] == ["msg_g2"]

    def test_get_empty(self):
        tracker = MsgTracker(ttl=60)
        result = tracker.get("nonexistent", "agent")
        assert result == {}

    def test_expired_messages_not_returned(self, monkeypatch):
        current_time = 1000.0
        monkeypatch.setattr(time, "time", lambda: current_time)

        tracker = MsgTracker(ttl=5)
        tracker.track("a1", "u1", "g1", "old_msg")

        current_time = 1010.0
        monkeypatch.setattr(time, "time", lambda: current_time)

        result = tracker.get("u1", "a1")
        assert result.get("g1", []) == []
