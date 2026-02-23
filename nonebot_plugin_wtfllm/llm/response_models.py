__all__ = ["CHAT_OUTPUT"]

from abc import ABC, abstractmethod
from typing import List, Tuple, Type

from pydantic import BaseModel, Field
from nonebot_plugin_alconna import UniMessage
from nonebot_plugin_alconna.uniseg import Receipt, Image
from sqlalchemy.exc import SQLAlchemyError

from ._render import render_markdown_to_image
from ..utils import logger, ensure_msgid_from_receipt
from ..memory import ImageSegment
from ..db import tool_call_record_repo
from ..v_db import meme_repo
from .deps import AgentDeps
from ..stream_processing import convert_and_store_item
from ..msg_tracker import msg_tracker


class SendableResponse(BaseModel, ABC):
    async def send(
        self,
        context: AgentDeps,
        extra_segments: UniMessage | None = None,
    ) -> None:
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

        await self._perform_send(context, extra_segments)

    @abstractmethod
    async def _perform_send(
        self,
        context: AgentDeps,
        extra_segments: UniMessage | None = None,
    ) -> None: ...


class VoiceResponse(SendableResponse):
    # TODO: 实现语音回复
    ...

    async def _perform_send(
        self,
        context: AgentDeps,
        extra_segments: UniMessage | None = None,
    ) -> None:
        raise NotImplementedError("VoiceResponse is not implemented yet.")


class MarkdownResponse(SendableResponse):
    """发送Markdown内容给用户"""

    markdown_content: str = Field(..., description="Markdown内容, 支持LaTeX语法")

    summary: str = Field(..., description="这条回复的摘要,应少于50字, 用于后续检索")

    async def _perform_send(
        self,
        context: AgentDeps,
        extra_segments: UniMessage | None = None,
    ):
        if not context.nb_runtime:
            raise ValueError("NonebotRuntime is required to send MarkdownResponse")
        if not context.ids.user_id:
            raise ValueError("User ID is required to send MarkdownResponse")

        sent_messages_receipt: List[Receipt] = []
        sent_messages_content: List[UniMessage] = []

        markdown_img = await render_markdown_to_image(self.markdown_content)

        _image = Image(raw=markdown_img, name="response.webp")
        _image.desc = self.summary  # pyright: ignore[reportAttributeAccessIssue]

        msg1 = UniMessage()
        msg1 += _image
        sent_messages_receipt.append(await msg1.send(target=context.nb_runtime.target))
        sent_messages_content.append(msg1)

        if context.reply_segments or extra_segments:
            msg2 = UniMessage()
            if context.reply_segments:
                msg2 += context.reply_segments
            if extra_segments:
                msg2 += extra_segments
            sent_messages_receipt.append(
                await msg2.send(target=context.nb_runtime.target)
            )
            sent_messages_content.append(msg2)

        for content, receipt in zip(sent_messages_content, sent_messages_receipt):
            msg_id = ensure_msgid_from_receipt(receipt, context.nb_runtime.session)
            await convert_and_store_item(
                agent_id=context.ids.agent_id,
                user_id=context.ids.user_id,
                uni_msg=content,
                group_id=context.ids.group_id,
                sender=context.ids.agent_id,
                msg_id=msg_id,
            )
            msg_tracker.track(
                agent_id=context.ids.agent_id,
                user_id=context.ids.user_id,
                group_id=context.ids.group_id,
                msg_id=msg_id,
            )


class TextResponse(SendableResponse):
    """
    已经获得足够信息，需要直接给用户最终回复时使用。
    """

    mentions: List[str] = Field(
        default_factory=list,
        description="回复中需要特别提及的用户列表",
    )

    response: str = Field(..., description="给用户的自然语言回复, 纯文本")

    meme: str | None = Field(
        None,
        description="如果这条回复包含表情包, 请填写表情包的 UUID 或图片序号",
    )

    async def _perform_send(
        self,
        context: AgentDeps,
        extra_segments: UniMessage | None = None,
    ):
        if not context.nb_runtime:
            raise ValueError("NonebotRuntime is required to send TextResponse")
        if not context.ids.user_id:
            raise ValueError("User ID is required to send TextResponse")

        msg = UniMessage()
        for mention in self.mentions:
            mention = context.context.resolve_aliases(mention)
            if mention:
                msg.at(mention)

        msg.text(self.response)

        if context.reply_segments:
            msg += context.reply_segments

        if self.meme:
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

        if extra_segments:
            msg += extra_segments

        sent_message = await msg.send(target=context.nb_runtime.target)
        sent_msg_id = ensure_msgid_from_receipt(
            sent_message, context.nb_runtime.session
        )
        await convert_and_store_item(
            agent_id=context.ids.agent_id,
            user_id=context.ids.user_id,
            uni_msg=msg,
            group_id=context.ids.group_id,
            sender=context.ids.agent_id,
            msg_id=sent_msg_id,
        )
        msg_tracker.track(
            agent_id=context.ids.agent_id,
            user_id=context.ids.user_id,
            group_id=context.ids.group_id,
            msg_id=sent_msg_id,
        )


class RejectResponse(SendableResponse):
    """
    无法获得足够信息，或出于维持人设或心情原因不想回答, 或者当前上下文不适合插嘴, 需要拒绝时使用。
    """

    reason: str = Field(..., description="真正拒绝的理由")

    should_show_user: bool = Field(
        False, description="你希望你拒绝的原因被用户看见吗？"
    )

    message_to_show: str | None = Field(
        None, description="如果希望用户看到拒绝原因，可以自定义显示的内容"
    )

    async def _perform_send(
        self,
        context: AgentDeps,
        extra_segments: UniMessage | None = None,
    ):
        if not context.nb_runtime:
            raise ValueError("NonebotRuntime is required to handle RejectResponse")
        if not context.ids.user_id:
            raise ValueError("User ID is required to handle RejectResponse")

        logger.info(
            f"Rejecting to respond to user {context.ids.user_id} for reason: {self.reason}"
        )

        msg = None

        if self.should_show_user and self.message_to_show:
            msg = UniMessage()
            msg.text(self.message_to_show)
            if context.reply_segments:
                msg += context.reply_segments

        if msg:
            if extra_segments:
                msg += extra_segments
            sent_message = await msg.send(target=context.nb_runtime.target)

            sent_msg_id = ensure_msgid_from_receipt(
                sent_message, context.nb_runtime.session
            )

            await convert_and_store_item(
                agent_id=context.ids.agent_id,
                user_id=context.ids.user_id,
                uni_msg=msg,
                group_id=context.ids.group_id,
                sender=context.ids.agent_id,
                msg_id=sent_msg_id,
            )
            msg_tracker.track(
                agent_id=context.ids.agent_id,
                user_id=context.ids.user_id,
                group_id=context.ids.group_id,
                msg_id=sent_msg_id,
            )


CHAT_OUTPUT: Tuple[Type[SendableResponse], ...] = (
    TextResponse,
    MarkdownResponse,
    RejectResponse,
)
