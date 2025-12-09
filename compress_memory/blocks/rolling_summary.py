"""Rolling Summary Memory Block"""

import re
from typing import Any, List, Optional

from llama_index.core.base.llms.types import ChatMessage, TextBlock
from llama_index.core.bridge.pydantic import Field, ConfigDict
from llama_index.core.llms import LLM
from llama_index.core.memory.memory import BaseMemoryBlock

from ..prompts.compression import CHATBOT_COMPRESSION_PROMPT


class RollingSummaryBlock(BaseMemoryBlock[str]):
    """
    滚动摘要内存块

    接收从短期 Buffer 弹出的消息，使用 LLM 生成/更新结构化摘要。

    工作流程：
    1. _aput: 接收弹出的消息 → 旧摘要 + 新消息 → LLM → 新摘要
    2. _aget: 返回当前摘要
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="ConversationSummary")
    llm: LLM = Field(description="用于生成摘要的 LLM")
    priority: int = Field(default=0, description="0 = 永不截断")
    accept_short_term_memory: bool = Field(default=True)

    # 状态
    snapshot: str = Field(default="", description="当前摘要内容")

    # 配置
    compression_prompt: str = Field(
        default=CHATBOT_COMPRESSION_PROMPT,
        description="压缩提示词模板"
    )
    max_snapshot_tokens: int = Field(
        default=1500,
        description="摘要最大 token 数（提示用，不强制）"
    )

    async def _aget(
        self,
        messages: Optional[List[ChatMessage]] = None,
        **kwargs: Any,
    ) -> str:
        """返回当前摘要"""
        if not self.snapshot:
            return ""
        return self.snapshot

    async def _aput(self, messages: List[ChatMessage]) -> None:
        """
        接收弹出的消息，滚动更新摘要

        Args:
            messages: 从短期 Buffer 弹出的消息列表
        """
        if not messages:
            return

        # 格式化新消息
        new_messages_text = self._format_messages(messages)

        # 构建压缩 prompt
        prompt = self.compression_prompt.format(
            existing_snapshot=self.snapshot or "（这是首次压缩，无历史摘要）",
            new_messages=new_messages_text
        )

        # 调用 LLM
        response = await self.llm.acomplete(prompt)

        # 提取 <snapshot>...</snapshot>
        new_snapshot = self._extract_snapshot(response.text)

        if new_snapshot:
            self.snapshot = new_snapshot

    def _format_messages(self, messages: List[ChatMessage]) -> str:
        """格式化消息列表为文本"""
        lines = []
        for msg in messages:
            role = msg.role.value
            # 提取文本内容
            content = self._get_text_from_message(msg)
            if content:
                lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    def _get_text_from_message(self, message: ChatMessage) -> str:
        """从消息中提取文本内容"""
        text_parts = []
        for block in message.blocks:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
        return " ".join(text_parts)

    def _extract_snapshot(self, text: str) -> Optional[str]:
        """从 LLM 输出中提取 snapshot"""
        # 匹配 <snapshot>...</snapshot>
        match = re.search(
            r'<snapshot>(.*?)</snapshot>',
            text,
            re.DOTALL
        )
        if match:
            return f"<snapshot>{match.group(1).strip()}</snapshot>"

        # 如果没有标签，尝试清理后返回
        # 去掉 scratchpad 部分
        text = re.sub(r'<scratchpad>.*?</scratchpad>', '', text, flags=re.DOTALL)
        cleaned = text.strip()
        return cleaned if cleaned else None

    def reset(self) -> None:
        """重置摘要"""
        self.snapshot = ""
