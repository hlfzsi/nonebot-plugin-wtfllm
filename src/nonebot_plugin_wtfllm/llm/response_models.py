__all__ = ["CHAT_OUTPUT"]

from abc import ABC, abstractmethod
import asyncio
from typing import List, Tuple, Type

from pydantic import BaseModel, Field
from nonebot_plugin_alconna import UniMessage
from nonebot_plugin_alconna.uniseg import Receipt, Image
from sqlalchemy.exc import SQLAlchemyError

from ._render import render_markdown_to_image
from ..utils import logger, ensure_msgid_from_receipt
from ..memory import ImageSegment
from ..db import tool_call_record_repo, thought_record_repo
from ..v_db import meme_repo
from .deps import AgentDeps
from ..proactive import topic_interest_store
from ..stream_processing import store_message_with_context


class SendableResponse(BaseModel, ABC):
    thought_of_chain: str = Field(
        ...,
        description="填写本次回复前的实际思考摘要，必须真实反映你的分析与决策，但保持简短。该字段仅供系统查询，不会直接发给用户。",
    )
    interested_topics: List[str] | None = Field(
        ...,
        description="预测用户接下来可能继续提及的主题线索列表。后续消息若与这些主题语义相关，可用于判断是否延续当前对话",
    )

    async def send(
        self,
        context: AgentDeps,
        extra_segments: UniMessage | None = None,
    ) -> None:
        run_id = context.tool_chain[-1].run_id if context.tool_chain else None
        if context.tool_chain:
            try:
                await tool_call_record_repo.save_batch_from_tool_call_info(
                    infos=context.tool_chain,
                    agent_id=context.ids.agent_id,
                    group_id=context.ids.group_id,
                    user_id=context.ids.user_id,
                )
            except (OSError, ValueError, RuntimeError, SQLAlchemyError) as e:
                logger.error(f"Failed to persist tool call records: {e}")
        else:
            try:
                await tool_call_record_repo.save_empty_record(
                    agent_id=context.ids.agent_id,
                    group_id=context.ids.group_id,
                    user_id=context.ids.user_id,
                )
            except (OSError, ValueError, RuntimeError, SQLAlchemyError) as e:
                logger.error(f"Failed to persist empty tool call record: {e}")

        try:
            await thought_record_repo.save_record(
                thought_of_chain=self.thought_of_chain,
                agent_id=context.ids.agent_id,
                group_id=context.ids.group_id,
                user_id=context.ids.user_id,
                run_id=run_id,
            )
        except (OSError, ValueError, RuntimeError, SQLAlchemyError) as e:
            logger.error(f"Failed to persist thought record: {e}")

        results = await self._perform_send(context, extra_segments)
        if results is None:
            return
        await self._post_send_messages(context, results)

    async def _post_send_messages(
        self, context: AgentDeps, results: List[Tuple[UniMessage, Receipt, bool]]
    ) -> None:
        """发送后处理逻辑"""
        if not context.nb_runtime:
            return

        for uni_msg, receipt, ingest_topic in results:
            msg_id = ensure_msgid_from_receipt(receipt, context.nb_runtime.session)
            await store_message_with_context(
                agent_id=context.ids.agent_id,
                uni_msg=uni_msg,
                sender=context.ids.agent_id,
                msg_id=msg_id,
                user_id=context.ids.user_id,
                group_id=context.ids.group_id,
                track_message=True,
                ingest_topic=ingest_topic,
            )

        if context.ids.user_id:
            logger.debug(
                f"Setting topic interests for user {context.ids.user_id}: {self.interested_topics}"
            )
            topic_interest_store.set_topics(
                agent_id=context.ids.agent_id,
                user_id=context.ids.user_id,
                group_id=context.ids.group_id,
                topics=self.interested_topics,
            )

    @abstractmethod
    async def _perform_send(
        self,
        context: AgentDeps,
        extra_segments: UniMessage | None = None,
    ) -> List[Tuple[UniMessage, Receipt, bool]] | None:
        """
        子类需返回一个列表，每个元素包含：
        (UniMessage对象, 发送回执Receipt, 是否执行ingest_topic)
        """
        ...


class VoiceResponse(SendableResponse):
    # TODO: 实现语音回复
    ...

    async def _perform_send(
        self,
        context: AgentDeps,
        extra_segments: UniMessage | None = None,
    ) -> List[Tuple[UniMessage, Receipt, bool]]:
        raise NotImplementedError("VoiceResponse is not implemented yet.")


class MarkdownResponse(SendableResponse):
    """发送Markdown内容给用户"""

    markdown_content: str = Field(..., description="Markdown内容, 支持LaTeX语法")

    summary: str = Field(..., description="这条回复的摘要,应少于50字, 用于后续检索")

    async def _perform_send(
        self,
        context: AgentDeps,
        extra_segments: UniMessage | None = None,
    ) -> List[Tuple[UniMessage, Receipt, bool]]:
        if not context.nb_runtime or not context.ids.user_id:
            raise ValueError("Context information missing")

        results: List[Tuple[UniMessage, Receipt, bool]] = []

        markdown_img = await render_markdown_to_image(self.markdown_content)
        msg1 = UniMessage(Image(raw=markdown_img, name="response.webp"))
        msg1[0].desc = self.summary  # pyright: ignore

        receipt1 = await msg1.send(target=context.nb_runtime.target)
        results.append((msg1, receipt1, False))

        if context.reply_segments or extra_segments:
            msg2 = UniMessage()
            if context.reply_segments:
                msg2 += context.reply_segments
            if extra_segments:
                msg2 += extra_segments

            receipt2 = await msg2.send(target=context.nb_runtime.target)
            results.append((msg2, receipt2, False))

        return results


class TextResponse(SendableResponse):
    """
    已经获得足够信息，需要直接给用户最终回复时使用。
    """

    mentions: List[str] = Field(
        default_factory=list,
        description="回复中需要特别提及的用户列表",
    )

    responses: List[str] = Field(
        ...,
        max_length=3,
        description="给用户的自然语言回复列表, 每个元素会被分割发送, 均为纯文本, 不包含@等特殊格式",
    )

    meme: str | None = Field(
        None,
        description="如果这条回复包含表情包, 请填写表情包的 UUID 或图片序号",
    )

    async def _perform_send(
        self,
        context: AgentDeps,
        extra_segments: UniMessage | None = None,
    ) -> List[Tuple[UniMessage, Receipt, bool]]:
        if not context.nb_runtime:
            raise ValueError("NonebotRuntime is required to send TextResponse")
        if not context.ids.user_id:
            raise ValueError("User ID is required to send TextResponse")

        texts = self.responses or [""]
        results: List[Tuple[UniMessage, Receipt, bool]] = []

        for idx, text in enumerate(texts):
            is_first = idx == 0
            is_last = idx == len(texts) - 1

            msg = UniMessage()

            if is_first:
                for mention in self.mentions:
                    mention = context.context.resolve_aliases(mention)
                    if mention:
                        msg.at(mention)
                if context.reply_segments:
                    msg += context.reply_segments

            msg.text(text)

            if is_last and self.meme:
                try:
                    _meme = context.context.resolve_media_ref(self.meme, ImageSegment)
                except ValueError:
                    _meme = None

                if _meme and _meme.available:
                    if _meme.local_path:
                        image_data = await _meme.get_bytes_async()
                        msg.image(raw=image_data)
                    elif _meme.url:
                        msg.image(url=_meme.url)
                else:
                    try:
                        _meme = await meme_repo.get_meme_by_id(self.meme)
                        if _meme:
                            _bytes = await _meme.get_bytes_async()
                            msg.image(raw=_bytes, name=f"{_meme.storage_id}.webp")
                        else:
                            logger.warning(
                                f"FinalResponse: Meme with UUID {self.meme} not found."
                            )
                            msg.text("\n哎呀图丢了")
                    except (OSError, ValueError, RuntimeError):
                        logger.exception("FinalResponse: Error fetching meme by UUID.")
                        msg.text("\n哎呀图丢了")

            if is_last and extra_segments:
                msg += extra_segments

            sent_message = await msg.send(target=context.nb_runtime.target)
            results.append((msg, sent_message, True))
            if not is_last:
                await asyncio.sleep(len(text) * 0.2)

        return results


class RejectResponse(SendableResponse):
    """
    无法获得足够信息，或出于维持人设或心情原因不想回答, 或者当前上下文不适合插嘴, 需要拒绝时使用。
    """

    reason: str = Field(..., description="真正拒绝的理由")

    message_to_user: str | None = Field(
        default=None,
        description="向用户显示的拒绝消息。如果为None，则不向用户显示",
    )

    async def _perform_send(
        self,
        context: AgentDeps,
        extra_segments: UniMessage | None = None,
    ) -> List[Tuple[UniMessage, Receipt, bool]] | None:
        if not context.nb_runtime:
            raise ValueError("NonebotRuntime is required to handle RejectResponse")
        if not context.ids.user_id:
            raise ValueError("User ID is required to handle RejectResponse")

        logger.info(
            f"Rejecting to respond to user {context.ids.user_id} for reason: {self.reason}"
        )

        msg = None
        results: List[Tuple[UniMessage, Receipt, bool]] = []

        if self.message_to_user:
            msg = UniMessage()
            msg.text(self.message_to_user)
            if context.reply_segments:
                msg += context.reply_segments

        if msg:
            if extra_segments:
                msg += extra_segments
            sent_message = await msg.send(target=context.nb_runtime.target)
            results.append((msg, sent_message, True))

        return results


CHAT_OUTPUT: Tuple[Type[SendableResponse], ...] = (
    TextResponse,
    MarkdownResponse,
    RejectResponse,
)
