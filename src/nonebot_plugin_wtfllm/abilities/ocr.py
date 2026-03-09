__all__ = ["get_image_ocr_text"]

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias, cast

import httpx
import pybase64
from rapidocr import RapidOCR
from rapidocr.utils.output import RapidOCROutput

from ..utils import get_http_client, logger

OCRSource: TypeAlias = str | bytes | Path
NormalizedOCRSource: TypeAlias = bytes | Path

OCR_SEM = asyncio.Semaphore(4)
OCR_ENGINE = RapidOCR(params={"Global.log_level": "critical"})


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _is_data_uri(source: str) -> bool:
    return source.startswith("data:")


def _decode_data_uri(source: str) -> bytes:
    try:
        header, payload = source.split(",", 1)
    except ValueError as exc:
        raise ValueError("Invalid data URI image source.") from exc

    if ";base64" not in header:
        raise ValueError("Only base64 data URI is supported for OCR.")

    try:
        return pybase64.b64decode(payload)
    except ValueError as exc:
        raise ValueError("Invalid base64 payload in data URI image source.") from exc


def _coerce_sources(sources: Sequence[OCRSource] | OCRSource) -> list[OCRSource]:
    if isinstance(sources, str):
        return [sources]
    if isinstance(sources, bytes):
        return [sources]
    if isinstance(sources, Path):
        return [sources]
    return cast(list[OCRSource], list(sources))


async def _fetch_url_bytes(url: str) -> bytes:
    client: httpx.AsyncClient | None = None
    client = get_http_client()
    response = await client.get(url)
    response.raise_for_status()
    return response.content


async def _normalize_image_source(source: OCRSource) -> NormalizedOCRSource:
    if isinstance(source, bytes):
        return source

    if isinstance(source, Path):
        return source

    if isinstance(source, str):
        if _is_url(source):
            return await _fetch_url_bytes(source)
        if _is_data_uri(source):
            return _decode_data_uri(source)
        return Path(source)

    raise TypeError(f"Unsupported OCR source type: {type(source)!r}")


async def _ocr_single(source: OCRSource) -> str:
    async with OCR_SEM:
        normalized_source = await _normalize_image_source(source)
        result = await asyncio.to_thread(OCR_ENGINE, normalized_source)
    assert isinstance(result, RapidOCROutput), "Expected RapidOCROutput from OCR_ENGINE"
    return result.to_markdown().strip() or "[无OCR结果]"


async def get_image_ocr_text(
    sources: Sequence[OCRSource] | OCRSource,
) -> list[str | None] | None:
    """提取图片中的文字，返回结果顺序与输入顺序一致。"""
    source_list = _coerce_sources(sources)

    if not source_list:
        logger.warning("No image sources provided for OCR.")
        return None

    results = await asyncio.gather(
        *(_ocr_single(source) for source in source_list), return_exceptions=True
    )

    normalized_results: list[str | None] = []
    for source, result in zip(source_list, results):
        if isinstance(result, BaseException):
            logger.warning(f"OCR failed for source {source!r}: {result}")
            normalized_results.append(None)
            continue

        normalized_results.append(result)

    return normalized_results
