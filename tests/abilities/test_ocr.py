import base64
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from nonebot_plugin_wtfllm.abilities import ocr as ocr_module


class _FakeOutput:
    def __init__(self, *txts: str):
        self._txts = txts

    def to_markdown(self) -> str:
        return "\n".join(self._txts)


class _FakeEngine:
    def __init__(self):
        self.calls = []

    def __call__(self, source):
        self.calls.append(source)
        return _FakeOutput("第一行", "第二行")


class TestGetImageOcrText:
    @pytest.mark.asyncio
    async def test_empty_sources(self):
        result = await ocr_module.get_image_ocr_text([])
        assert result is None

    @pytest.mark.asyncio
    async def test_local_path_source(self, monkeypatch):
        engine = _FakeEngine()
        monkeypatch.setattr(ocr_module, "OCR_ENGINE", engine)
        monkeypatch.setattr(ocr_module, "RapidOCROutput", _FakeOutput)

        result = await ocr_module.get_image_ocr_text(Path("test.webp"))

        assert result == ["第一行\n第二行"]
        assert engine.calls == [Path("test.webp")]

    @pytest.mark.asyncio
    async def test_data_uri_source(self, monkeypatch):
        engine = _FakeEngine()
        monkeypatch.setattr(ocr_module, "OCR_ENGINE", engine)
        monkeypatch.setattr(ocr_module, "RapidOCROutput", _FakeOutput)

        payload = base64.b64encode(b"fake-image-bytes").decode("ascii")
        data_uri = f"data:image/webp;base64,{payload}"

        result = await ocr_module.get_image_ocr_text(data_uri)

        assert result == ["第一行\n第二行"]
        assert engine.calls == [b"fake-image-bytes"]

    @pytest.mark.asyncio
    async def test_url_source(self, monkeypatch):
        engine = _FakeEngine()
        monkeypatch.setattr(ocr_module, "OCR_ENGINE", engine)
        monkeypatch.setattr(ocr_module, "RapidOCROutput", _FakeOutput)
        monkeypatch.setattr(
            ocr_module,
            "_fetch_url_bytes",
            AsyncMock(return_value=b"downloaded-image"),
        )

        result = await ocr_module.get_image_ocr_text("https://example.com/test.webp")

        assert result == ["第一行\n第二行"]
        assert engine.calls == [b"downloaded-image"]