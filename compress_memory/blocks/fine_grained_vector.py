"""
细粒度向量存储块 - 滑动窗口实现（以 QA 对为单位）

与 VectorMemoryBlock 的区别：
- VectorMemoryBlock: 每批消息 → 1 个 node (粗粒度，~30条消息一个node)
- FineGrainedVectorBlock: 滑动窗口 → 多个 node (细粒度，以QA对为单位)

关键设计：以 QA 对（user + assistant）为基本单位进行滑窗
- 先把消息配对成 turns: [(user, assistant), (user, assistant), ...]
- 再对 turns 进行滑动窗口

滑动窗口示例 (window_size=3, stride=2):
  turns: [turn0, turn1, turn2, turn3, turn4]
         (每个 turn = user + assistant)

  node_0: [turn0, turn1, turn2]    # 窗口起点 0
  node_1: [turn2, turn3, turn4]    # 窗口起点 2 (stride=2)

重叠的好处：
1. 保持上下文连贯性
2. 边界对话不会被孤立
3. QA 配对保证语义完整
"""

from typing import Any, List, Optional, Tuple

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.types import ChatMessage, TextBlock, MessageRole
from llama_index.core.bridge.pydantic import Field
from llama_index.core.memory.memory import BaseMemoryBlock
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)


# 类型别名：一个 Turn = (user_message, assistant_message)
Turn = Tuple[ChatMessage, Optional[ChatMessage]]


def get_default_embed_model() -> BaseEmbedding:
    return Settings.embed_model


class FineGrainedVectorBlock(BaseMemoryBlock[str]):
    """
    细粒度向量存储块 - 以 QA 对为单位的滑动窗口实现

    工作流程：
    1. 将消息配对成 turns: [(user, assistant), ...]
    2. 对 turns 进行滑动窗口切分
    3. 每个窗口存为一个 vector node
    """

    name: str = Field(default="FineGrainedHistory")

    # 向量存储
    vector_store: BasePydanticVectorStore = Field(
        description="向量存储后端"
    )
    embed_model: BaseEmbedding = Field(
        default_factory=get_default_embed_model,
        description="Embedding 模型"
    )

    # 滑动窗口参数（以 turn 为单位）
    window_size: int = Field(
        default=3,
        description="每个窗口包含的 turn 数（每个 turn = user + assistant）"
    )
    stride: int = Field(
        default=2,
        description="窗口滑动步长（stride < window_size 时有重叠）"
    )

    # 检索参数
    similarity_top_k: int = Field(
        default=5,
        description="向量检索返回的 top_k 数量"
    )
    retrieval_context_window: int = Field(
        default=3,
        description="用于构建查询的最近消息数"
    )
    node_postprocessors: List[BaseNodePostprocessor] = Field(
        default_factory=list,
        description="后处理器列表（如 Rerank）"
    )

    # 内部状态
    _node_count: int = 0
    _turn_count: int = 0

    def _get_text_from_message(self, message: ChatMessage) -> str:
        """从消息中提取文本"""
        text_parts = []
        for block in message.blocks:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
        return " ".join(text_parts)

    def _pair_messages_to_turns(self, messages: List[ChatMessage]) -> List[Turn]:
        """
        将消息配对成 turns

        策略：
        - 遇到 user 消息，开始新的 turn
        - 遇到 assistant 消息，配对到当前 turn
        - 处理边界情况：连续 user、连续 assistant、单独 user 等
        """
        turns: List[Turn] = []
        current_user: Optional[ChatMessage] = None

        for msg in messages:
            if msg.role == MessageRole.USER:
                # 如果有未配对的 user，先保存为单独的 turn
                if current_user is not None:
                    turns.append((current_user, None))
                current_user = msg

            elif msg.role == MessageRole.ASSISTANT:
                if current_user is not None:
                    # 配对成功
                    turns.append((current_user, msg))
                    current_user = None
                else:
                    # 没有配对的 user，单独保存 assistant（罕见情况）
                    # 创建一个空的 user placeholder
                    turns.append((
                        ChatMessage(role=MessageRole.USER, content=""),
                        msg
                    ))

        # 处理末尾未配对的 user
        if current_user is not None:
            turns.append((current_user, None))

        return turns

    def _format_turn(self, turn: Turn, turn_index: int) -> str:
        """格式化单个 turn"""
        user_msg, assistant_msg = turn

        user_text = self._get_text_from_message(user_msg) if user_msg else ""
        assistant_text = self._get_text_from_message(assistant_msg) if assistant_msg else ""

        lines = [f"<turn index='{turn_index}'>"]
        if user_text:
            lines.append(f"  <user>{user_text}</user>")
        if assistant_text:
            lines.append(f"  <assistant>{assistant_text}</assistant>")
        lines.append("</turn>")

        return "\n".join(lines)

    def _format_window(self, turns: List[Turn], start_turn_index: int) -> str:
        """格式化一个窗口的 turns"""
        parts = []
        for i, turn in enumerate(turns):
            parts.append(self._format_turn(turn, start_turn_index + i))
        return "\n\n".join(parts)

    async def _aput(self, messages: List[ChatMessage]) -> None:
        """
        存储消息到向量库（以 QA 对为单位的滑动窗口）

        Args:
            messages: 从短期 Buffer 弹出的消息列表
        """
        if not messages:
            return

        # 提取 session_id
        session_id = None
        for msg in messages:
            if "session_id" in msg.additional_kwargs:
                session_id = msg.additional_kwargs.get("session_id")
                break

        # 1. 将消息配对成 turns
        turns = self._pair_messages_to_turns(messages)
        if not turns:
            return

        # 2. 滑动窗口切分
        windows = []
        num_turns = len(turns)

        start = 0
        while start < num_turns:
            end = min(start + self.window_size, num_turns)
            window_turns = turns[start:end]

            if window_turns:
                windows.append((start, window_turns))

            if end >= num_turns:
                break

            start += self.stride

        if not windows:
            return

        # 3. 批量创建节点
        nodes = []
        texts_for_embedding = []

        for window_start, window_turns in windows:
            global_turn_start = self._turn_count + window_start
            text = self._format_window(window_turns, global_turn_start)

            if not text.strip():
                continue

            node = TextNode(
                text=text,
                metadata={
                    "session_id": session_id,
                    "window_index": self._node_count + len(nodes),
                    "turn_start": global_turn_start,
                    "turn_count": len(window_turns),
                    "type": "fine_grained_qa_window",
                }
            )
            nodes.append(node)
            texts_for_embedding.append(text)

        if not nodes:
            return

        # 4. 批量计算 embedding
        embeddings = await self.embed_model.aget_text_embedding_batch(
            texts_for_embedding,
            show_progress=False,
        )

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        # 5. 批量添加到向量存储
        await self.vector_store.async_add(nodes)

        self._node_count += len(nodes)
        self._turn_count += len(turns)

    async def _aget(
        self,
        messages: Optional[List[ChatMessage]] = None,
        session_id: Optional[str] = None,
        **block_kwargs: Any,
    ) -> str:
        """
        检索相关消息

        Args:
            messages: 当前对话消息（用于构建查询）
            session_id: 会话 ID（用于过滤）
        """
        if not messages or len(messages) == 0:
            return ""

        # 使用最近几条消息构建查询
        if len(messages) >= self.retrieval_context_window:
            context = messages[-self.retrieval_context_window:]
        else:
            context = messages

        # 提取查询文本
        query_parts = []
        for msg in context:
            text = self._get_text_from_message(msg)
            if text:
                query_parts.append(text)

        query_text = " ".join(query_parts)
        if not query_text:
            return ""

        # 构建过滤条件
        query_kwargs = {}
        if session_id is not None:
            query_kwargs["filters"] = MetadataFilters(
                filters=[MetadataFilter(key="session_id", value=session_id)]
            )

        # 计算查询 embedding
        query_embedding = await self.embed_model.aget_query_embedding(query_text)

        # 执行向量查询
        query = VectorStoreQuery(
            query_str=query_text,
            query_embedding=query_embedding,
            similarity_top_k=self.similarity_top_k,
            **query_kwargs,
        )

        results = await self.vector_store.aquery(query)

        if not results.nodes:
            return ""

        # 构建 NodeWithScore 列表
        nodes_with_scores = [
            NodeWithScore(node=node, score=score)
            for node, score in zip(results.nodes or [], results.similarities or [])
        ]

        # 应用后处理器（如 Rerank）
        if self.node_postprocessors:
            query_bundle = QueryBundle(query_str=query_text)
            for postprocessor in self.node_postprocessors:
                nodes_with_scores = postprocessor.postprocess_nodes(
                    nodes_with_scores,
                    query_bundle=query_bundle,
                )

        if not nodes_with_scores:
            return ""

        # 按 turn_start 排序，保持时间顺序
        nodes_with_scores.sort(
            key=lambda x: x.node.metadata.get("turn_start", 0)
        )

        # 格式化返回结果
        retrieved_texts = []
        for nws in nodes_with_scores:
            content = nws.node.get_content()
            score = nws.score or 0.0
            turn_start = nws.node.metadata.get("turn_start", "?")
            turn_count = nws.node.metadata.get("turn_count", "?")
            retrieved_texts.append(
                f"<!-- turns {turn_start}-{turn_start + turn_count - 1}, score={score:.3f} -->\n{content}"
            )

        return "\n\n".join(retrieved_texts)

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "node_count": self._node_count,
            "turn_count": self._turn_count,
            "window_size": self.window_size,
            "stride": self.stride,
            "overlap": self.window_size - self.stride,
        }
