from unittest.mock import patch

from datetime import datetime

from nonebot_plugin_wtfllm.memory.items.note import Note


class TestNoteExpiryNormalization:
    def test_normalize_millisecond_expiry_on_create(self):
        note = Note.create(
            content="提醒",
            agent_id="agent_bot",
            user_id="user_1",
            expires_at=4102444800000,
        )

        assert note.expires_at == 4102444800

    def test_normalize_millisecond_expiry_on_assignment(self):
        note = Note.create(
            content="提醒",
            agent_id="agent_bot",
            user_id="user_1",
            expires_at=4102444800,
        )

        note.expires_at = 4102444800000

        assert note.expires_at == 4102444800

    def test_render_falls_back_for_out_of_range_expiry(self):
        note = Note.create(
            content="提醒",
            agent_id="agent_bot",
            user_id="user_1",
            expires_at=4102444800,
        )

        note.expires_at = 10**21

        assert note._render_expire_tag() == f"[expires {note.expires_at}]"

    @patch("nonebot_plugin_wtfllm.memory.items.note.time.time", return_value=1_700_000_000)
    def test_render_relative_expiry(self, mock_time):
        note = Note.create(
            content="提醒",
            agent_id="agent_bot",
            user_id="user_1",
            expires_at=1_700_001_800,
        )

        expire_text = datetime.fromtimestamp(note.expires_at).strftime("%Y-%m-%d %H:%M")
        assert note._render_expire_tag() == f"[expires in 30m at {expire_text}]"