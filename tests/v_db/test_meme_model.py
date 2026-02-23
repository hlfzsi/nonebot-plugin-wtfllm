"""v_db/models/meme.py 单元测试"""

from io import BytesIO

import pytest
from PIL import Image

from nonebot_plugin_wtfllm.v_db.models.meme import MemePayload


def _make_png_bytes() -> bytes:
    image = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_meme_payload_validators():
    payload = MemePayload(
        storage_id="m1",
        file_path="m1.webp",
        raw_message_id="raw_1",
        docs="  funny  ",
        tags=[" Fun ", "fun", "Meme"],
        uploader_id="user_1",
    )

    assert payload.docs == "funny"
    assert payload.tags == ["fun", "meme"]


def test_meme_payload_tags_helpers():
    payload = MemePayload(
        storage_id="m2",
        file_path="m2.webp",
        raw_message_id="raw_2",
        docs="doc",
        tags=["a", "b"],
        uploader_id="user_2",
    )

    assert payload.has_tag("A") is True
    assert payload.has_any_tag(["c", "b"]) is True
    assert payload.has_all_tags(["a", "b"]) is True

    added = payload.add_tags(["c", "A"])
    assert added.tags == ["a", "b", "c"]

    removed = added.remove_tags(["b"])
    assert removed.tags == ["a", "c"]


def test_convert_to_webp():
    data = _make_png_bytes()
    webp = MemePayload.convert_to_webp(data, quality=80)

    assert isinstance(webp, bytes)
    assert webp[:4] == b"RIFF"
